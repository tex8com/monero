#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Config {
  std::string implementation = "radix13";
  std::string vector_path;
  std::string shader_prefix;
  std::uint32_t workgroup_size = 64;
  std::uint32_t doublings_per_dispatch = 4;
  std::uint32_t multiply_chunk_size = 0;
  std::uint32_t rounds = 1;
  std::uint32_t warmup_rounds = 1;
};

struct PushConstants {
  std::uint32_t count;
  std::uint32_t loop_bias;
  std::uint32_t step_index;
};

struct Corpus {
  std::vector<std::uint32_t> scalar;
  std::vector<std::uint32_t> points;
  std::vector<std::uint8_t> expected;
  std::uint64_t fingerprint = 0;
  std::uint32_t valid_count = 0;
};

struct Buffer {
  VkBuffer handle = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceSize size = 0;
  void *mapped = nullptr;
};

struct Timing {
  double host_seconds = 0;
  double gpu_seconds = 0;
  double decode_seconds = 0;
  double multiply_seconds = 0;
  double compress_seconds = 0;
};

[[noreturn]] void fail(const std::string &message, int code = 2) {
  std::cerr << "Vulkan radix13 benchmark error: " << message << '\n';
  std::exit(code);
}

void check(VkResult result, const char *operation) {
  if (result != VK_SUCCESS) {
    fail(std::string(operation) + " failed with VkResult " +
         std::to_string(static_cast<int>(result)));
  }
}

std::uint32_t positive(const char *text, const char *option) {
  char *end = nullptr;
  const unsigned long value = std::strtoul(text, &end, 10);
  if (end == text || *end != '\0' || value == 0 ||
      value > std::numeric_limits<std::uint32_t>::max()) {
    fail(std::string(option) + " requires a positive UInt32");
  }
  return static_cast<std::uint32_t>(value);
}

Config parse_args(int argc, char **argv) {
  Config config;
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    auto next = [&]() -> const char * {
      if (++index >= argc)
        fail(argument + " requires a value");
      return argv[index];
    };
    if (argument == "--implementation")
      config.implementation = next();
    else if (argument == "--vectors")
      config.vector_path = next();
    else if (argument == "--shader-prefix")
      config.shader_prefix = next();
    else if (argument == "--workgroup-size")
      config.workgroup_size = positive(next(), "--workgroup-size");
    else if (argument == "--doublings-per-dispatch")
      config.doublings_per_dispatch =
          positive(next(), "--doublings-per-dispatch");
    else if (argument == "--multiply-chunk-size")
      config.multiply_chunk_size =
          positive(next(), "--multiply-chunk-size");
    else if (argument == "--rounds")
      config.rounds = positive(next(), "--rounds");
    else if (argument == "--warmup-rounds")
      config.warmup_rounds = positive(next(), "--warmup-rounds");
    else if (argument == "--help" || argument == "-h") {
      std::cout << "Usage: wallet-vulkan-radix13-benchmark --vectors PATH "
                   "--shader-prefix PATH "
                   "[--implementation radix13|radix2625] "
                   "[--workgroup-size N] "
                   "[--doublings-per-dispatch 1|2|4] "
                   "[--multiply-chunk-size 4] "
                   "[--rounds N] [--warmup-rounds N]\n";
      std::exit(0);
    } else {
      fail("unknown argument " + argument);
    }
  }
  if (config.vector_path.empty() || config.shader_prefix.empty())
    fail("--vectors and --shader-prefix are required");
  if (config.implementation != "radix13" &&
      config.implementation != "radix2625")
    fail("--implementation must be radix13 or radix2625");
  if (config.doublings_per_dispatch != 1 &&
      config.doublings_per_dispatch != 2 &&
      config.doublings_per_dispatch != 4)
    fail("--doublings-per-dispatch must be 1, 2 or 4");
  if (config.multiply_chunk_size != 0 &&
      config.multiply_chunk_size != 4)
    fail("--multiply-chunk-size must be 4 when provided");
  return config;
}

std::vector<std::uint8_t> read_file(const std::string &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream)
    fail("cannot open " + path);
  const std::streamsize length = stream.tellg();
  if (length < 0)
    fail("cannot determine size of " + path);
  stream.seekg(0);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
  if (length != 0 &&
      !stream.read(reinterpret_cast<char *>(bytes.data()), length))
    fail("cannot read " + path);
  return bytes;
}

std::uint32_t read_le32(const std::vector<std::uint8_t> &bytes,
                        std::size_t offset) {
  return static_cast<std::uint32_t>(bytes[offset]) |
         (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
         (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
         (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::uint64_t read_le64(const std::vector<std::uint8_t> &bytes,
                        std::size_t offset) {
  std::uint64_t value = 0;
  for (std::size_t index = 0; index < 8; ++index)
    value |= static_cast<std::uint64_t>(bytes[offset + index]) << (index * 8);
  return value;
}

Corpus read_corpus(const std::string &path) {
  constexpr std::size_t header_bytes = 88;
  constexpr std::size_t record_bytes = 64;
  constexpr std::uint8_t magic[8] = {
      0x4d, 0x57, 0x4d, 0x54, 0x56, 0x31, 0x00, 0x00};
  const std::vector<std::uint8_t> raw = read_file(path);
  if (raw.size() < header_bytes)
    fail("MWMTV1 corpus is shorter than its header");
  if (!std::equal(std::begin(magic), std::end(magic), raw.begin()))
    fail("MWMTV1 corpus has the wrong magic");
  if (read_le32(raw, 8) != 1)
    fail("unsupported MWMTV1 version");
  const std::uint32_t count = read_le32(raw, 12);
  if (count == 0 ||
      raw.size() != header_bytes + static_cast<std::size_t>(count) * record_bytes)
    fail("MWMTV1 record count does not match file size");

  Corpus corpus;
  corpus.valid_count = count;
  corpus.fingerprint = read_le64(raw, 48);
  corpus.scalar.reserve(32);
  for (std::size_t index = 16; index < 48; ++index)
    corpus.scalar.push_back(raw[index]);
  corpus.points.reserve((static_cast<std::size_t>(count) + 1) * 32);
  corpus.expected.reserve(static_cast<std::size_t>(count) * 32);
  for (std::uint32_t record = 0; record < count; ++record) {
    const std::size_t offset =
        header_bytes + static_cast<std::size_t>(record) * record_bytes;
    for (std::size_t byte = 0; byte < 32; ++byte) {
      corpus.points.push_back(raw[offset + byte]);
      corpus.expected.push_back(raw[offset + 32 + byte]);
    }
  }
  for (std::size_t byte = 56; byte < 88; ++byte)
    corpus.points.push_back(raw[byte]);
  return corpus;
}

std::string version_string(std::uint32_t version) {
  return std::to_string(VK_VERSION_MAJOR(version)) + "." +
         std::to_string(VK_VERSION_MINOR(version)) + "." +
         std::to_string(VK_VERSION_PATCH(version));
}

std::uint32_t find_memory_type(VkPhysicalDevice physical_device,
                               std::uint32_t allowed,
                               VkMemoryPropertyFlags required) {
  VkPhysicalDeviceMemoryProperties properties{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &properties);
  for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
    if ((allowed & (1U << index)) != 0U &&
        (properties.memoryTypes[index].propertyFlags & required) == required)
      return index;
  }
  fail("no host-visible coherent Vulkan memory type is available");
}

Buffer make_buffer(VkPhysicalDevice physical_device, VkDevice device,
                   VkDeviceSize size) {
  Buffer buffer;
  buffer.size = size;
  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check(vkCreateBuffer(device, &buffer_info, nullptr, &buffer.handle),
        "vkCreateBuffer");

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device, buffer.handle, &requirements);
  VkMemoryAllocateInfo allocation_info{};
  allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocation_info.allocationSize = requirements.size;
  allocation_info.memoryTypeIndex = find_memory_type(
      physical_device, requirements.memoryTypeBits,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  check(vkAllocateMemory(device, &allocation_info, nullptr, &buffer.memory),
        "vkAllocateMemory");
  check(vkBindBufferMemory(device, buffer.handle, buffer.memory, 0),
        "vkBindBufferMemory");
  check(vkMapMemory(device, buffer.memory, 0, size, 0, &buffer.mapped),
        "vkMapMemory");
  std::memset(buffer.mapped, 0, static_cast<std::size_t>(size));
  return buffer;
}

void destroy_buffer(VkDevice device, Buffer &buffer) {
  if (buffer.mapped != nullptr) {
    std::memset(buffer.mapped, 0, static_cast<std::size_t>(buffer.size));
    vkUnmapMemory(device, buffer.memory);
  }
  if (buffer.handle != VK_NULL_HANDLE)
    vkDestroyBuffer(device, buffer.handle, nullptr);
  if (buffer.memory != VK_NULL_HANDLE)
    vkFreeMemory(device, buffer.memory, nullptr);
  buffer = {};
}

void validate(const Corpus &corpus, const Buffer &result_buffer,
              const Buffer &valid_buffer) {
  const auto *results =
      static_cast<const std::uint32_t *>(result_buffer.mapped);
  const auto *valid = static_cast<const std::uint32_t *>(valid_buffer.mapped);
  for (std::uint32_t record = 0; record < corpus.valid_count; ++record) {
    if (valid[record] != 1)
      fail("valid record " + std::to_string(record) + " was rejected");
    const std::size_t base = static_cast<std::size_t>(record) * 32;
    for (std::size_t byte = 0; byte < 32; ++byte) {
      if (results[base + byte] != corpus.expected[base + byte]) {
        std::ostringstream actual_hex;
        std::ostringstream expected_hex;
        actual_hex << std::hex << std::setfill('0');
        expected_hex << std::hex << std::setfill('0');
        for (std::size_t index = 0; index < 32; ++index) {
          actual_hex << std::setw(2) << results[base + index];
          expected_hex << std::setw(2)
                       << static_cast<unsigned>(corpus.expected[base + index]);
        }
        std::cerr << "actual_record_hex=" << actual_hex.str() << '\n';
        std::cerr << "expected_record_hex=" << expected_hex.str() << '\n';
        fail("Dalek mismatch at record " + std::to_string(record) +
             ", byte " + std::to_string(byte));
      }
    }
  }
  const std::uint32_t invalid_index = corpus.valid_count;
  if (valid[invalid_index] != 0)
    fail("Dalek-rejected point was accepted");
  const std::size_t invalid_base =
      static_cast<std::size_t>(invalid_index) * 32;
  for (std::size_t byte = 0; byte < 32; ++byte) {
    if (results[invalid_base + byte] != 0)
      fail("invalid point left nonzero output");
  }
}

} // namespace

int main(int argc, char **argv) {
  const Config config = parse_args(argc, argv);
  const Corpus corpus = read_corpus(config.vector_path);
  constexpr std::uint32_t stage_count = 10;
  const char *stage_names[stage_count] = {
      "scalar_digits",
      "decode_prepare",
      "decode_inverse",
      "decode_sqrt",
      "decode_finish",
      "multiply_init",
      "multiply_double",
      "multiply_add",
      "multiply_finish",
      "compress",
  };
  std::string shader_paths[stage_count];
  std::vector<std::uint8_t> shader_bytes[stage_count];
  for (std::uint32_t index = 0; index < stage_count; ++index) {
    shader_paths[index] =
        config.shader_prefix + "-" + stage_names[index] + ".spv";
    shader_bytes[index] = read_file(shader_paths[index]);
    if (shader_bytes[index].empty() ||
        (shader_bytes[index].size() % 4) != 0)
      fail("SPIR-V shader size is invalid: " + shader_paths[index]);
  }

  VkApplicationInfo application_info{};
  application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  application_info.pApplicationName = "wallet-vulkan-radix13-benchmark";
  application_info.applicationVersion = 1;
  application_info.pEngineName = "none";
  application_info.engineVersion = 1;
  application_info.apiVersion = VK_API_VERSION_1_1;
  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &application_info;
  VkInstance instance = VK_NULL_HANDLE;
  check(vkCreateInstance(&instance_info, nullptr, &instance),
        "vkCreateInstance");

  std::uint32_t physical_count = 0;
  check(vkEnumeratePhysicalDevices(instance, &physical_count, nullptr),
        "vkEnumeratePhysicalDevices(count)");
  if (physical_count == 0)
    fail("no Vulkan physical device");
  std::vector<VkPhysicalDevice> physical_devices(physical_count);
  check(vkEnumeratePhysicalDevices(instance, &physical_count,
                                   physical_devices.data()),
        "vkEnumeratePhysicalDevices(list)");
  const VkPhysicalDevice physical_device = physical_devices.front();
  VkPhysicalDeviceProperties physical_properties{};
  vkGetPhysicalDeviceProperties(physical_device, &physical_properties);

  std::uint32_t queue_family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                           &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_properties(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                           &queue_family_count,
                                           queue_properties.data());
  std::uint32_t queue_family = queue_family_count;
  for (std::uint32_t index = 0; index < queue_family_count; ++index) {
    if ((queue_properties[index].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U) {
      queue_family = index;
      break;
    }
  }
  if (queue_family == queue_family_count)
    fail("no Vulkan compute queue family");

  const float queue_priority = 1.0F;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = queue_family;
  queue_info.queueCount = 1;
  queue_info.pQueuePriorities = &queue_priority;
  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = 1;
  device_info.pQueueCreateInfos = &queue_info;
  VkDevice device = VK_NULL_HANDLE;
  check(vkCreateDevice(physical_device, &device_info, nullptr, &device),
        "vkCreateDevice");
  VkQueue queue = VK_NULL_HANDLE;
  vkGetDeviceQueue(device, queue_family, 0, &queue);

  const std::uint32_t dispatch_count = corpus.valid_count + 1;
  PushConstants push_constants{dispatch_count, 0, 0};
  Buffer scalar_buffer = make_buffer(
      physical_device, device, corpus.scalar.size() * sizeof(std::uint32_t));
  Buffer point_buffer = make_buffer(
      physical_device, device, corpus.points.size() * sizeof(std::uint32_t));
  Buffer scratch_buffer = make_buffer(
      physical_device, device,
      static_cast<VkDeviceSize>(dispatch_count) * 200 *
          sizeof(std::uint32_t));
  Buffer state_buffer = make_buffer(
      physical_device, device,
      static_cast<VkDeviceSize>(dispatch_count) * 80 *
          sizeof(std::uint32_t));
  Buffer result_buffer = make_buffer(
      physical_device, device,
      static_cast<VkDeviceSize>(dispatch_count) * 32 * sizeof(std::uint32_t));
  Buffer valid_buffer = make_buffer(
      physical_device, device,
      static_cast<VkDeviceSize>(dispatch_count) * sizeof(std::uint32_t));
  Buffer digit_buffer = make_buffer(
      physical_device, device, 64 * sizeof(std::uint32_t));
  Buffer table_buffer = make_buffer(
      physical_device, device,
      static_cast<VkDeviceSize>(dispatch_count) * 640 *
          sizeof(std::uint32_t));
  std::memcpy(scalar_buffer.mapped, corpus.scalar.data(),
              static_cast<std::size_t>(scalar_buffer.size));
  std::memcpy(point_buffer.mapped, corpus.points.data(),
              static_cast<std::size_t>(point_buffer.size));

  VkDescriptorSetLayoutBinding bindings[8]{};
  for (std::uint32_t index = 0; index < 8; ++index) {
    bindings[index].binding = index;
    bindings[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[index].descriptorCount = 1;
    bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }
  VkDescriptorSetLayoutCreateInfo descriptor_layout_info{};
  descriptor_layout_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor_layout_info.bindingCount = 8;
  descriptor_layout_info.pBindings = bindings;
  VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
  check(vkCreateDescriptorSetLayout(device, &descriptor_layout_info, nullptr,
                                    &descriptor_layout),
        "vkCreateDescriptorSetLayout");

  VkPushConstantRange push_range{};
  push_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  push_range.offset = 0;
  push_range.size = sizeof(PushConstants);
  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = 1;
  pipeline_layout_info.pSetLayouts = &descriptor_layout;
  pipeline_layout_info.pushConstantRangeCount = 1;
  pipeline_layout_info.pPushConstantRanges = &push_range;
  VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
  check(vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr,
                               &pipeline_layout),
        "vkCreatePipelineLayout");

  VkShaderModule shader_modules[stage_count]{};
  VkPipeline pipelines[stage_count]{};
  for (std::uint32_t index = 0; index < stage_count; ++index) {
    VkShaderModuleCreateInfo shader_info{};
    shader_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_info.codeSize = shader_bytes[index].size();
    shader_info.pCode = reinterpret_cast<const std::uint32_t *>(
        shader_bytes[index].data());
    check(vkCreateShaderModule(device, &shader_info, nullptr,
                               &shader_modules[index]),
          "vkCreateShaderModule");
    VkPipelineShaderStageCreateInfo stage_info{};
    stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_info.module = shader_modules[index];
    stage_info.pName = "main";
    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = stage_info;
    pipeline_info.layout = pipeline_layout;
    const VkResult pipeline_result =
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_info,
                                 nullptr, &pipelines[index]);
    if (pipeline_result != VK_SUCCESS) {
      fail(std::string("vkCreateComputePipelines(") + stage_names[index] +
           ") failed with VkResult " +
           std::to_string(static_cast<int>(pipeline_result)));
    }
  }

  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = 8;
  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = 1;
  pool_info.poolSizeCount = 1;
  pool_info.pPoolSizes = &pool_size;
  VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
  check(vkCreateDescriptorPool(device, &pool_info, nullptr, &descriptor_pool),
        "vkCreateDescriptorPool");
  VkDescriptorSetAllocateInfo descriptor_allocate{};
  descriptor_allocate.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptor_allocate.descriptorPool = descriptor_pool;
  descriptor_allocate.descriptorSetCount = 1;
  descriptor_allocate.pSetLayouts = &descriptor_layout;
  VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
  check(vkAllocateDescriptorSets(device, &descriptor_allocate,
                                 &descriptor_set),
        "vkAllocateDescriptorSets");

  Buffer *buffer_list[8] = {
      &scalar_buffer, &point_buffer, &scratch_buffer, &state_buffer,
      &result_buffer, &valid_buffer, &digit_buffer, &table_buffer};
  VkDescriptorBufferInfo buffer_infos[8]{};
  VkWriteDescriptorSet writes[8]{};
  for (std::uint32_t index = 0; index < 8; ++index) {
    buffer_infos[index].buffer = buffer_list[index]->handle;
    buffer_infos[index].offset = 0;
    buffer_infos[index].range = buffer_list[index]->size;
    writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[index].dstSet = descriptor_set;
    writes[index].dstBinding = index;
    writes[index].descriptorCount = 1;
    writes[index].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[index].pBufferInfo = &buffer_infos[index];
  }
  vkUpdateDescriptorSets(device, 8, writes, 0, nullptr);

  VkCommandPoolCreateInfo command_pool_info{};
  command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  command_pool_info.queueFamilyIndex = queue_family;
  VkCommandPool command_pool = VK_NULL_HANDLE;
  check(vkCreateCommandPool(device, &command_pool_info, nullptr,
                            &command_pool),
        "vkCreateCommandPool");
  VkCommandBufferAllocateInfo command_allocate{};
  command_allocate.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_allocate.commandPool = command_pool;
  command_allocate.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_allocate.commandBufferCount = 1;
  VkCommandBuffer command = VK_NULL_HANDLE;
  check(vkAllocateCommandBuffers(device, &command_allocate, &command),
        "vkAllocateCommandBuffers");
  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  VkFence fence = VK_NULL_HANDLE;
  check(vkCreateFence(device, &fence_info, nullptr, &fence),
        "vkCreateFence");
  VkQueryPoolCreateInfo query_info{};
  query_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  query_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  query_info.queryCount = 4;
  VkQueryPool query_pool = VK_NULL_HANDLE;
  check(vkCreateQueryPool(device, &query_info, nullptr, &query_pool),
        "vkCreateQueryPool");

  auto run_round = [&](bool collect_timing) -> Timing {
    check(vkResetCommandBuffer(command, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command, &begin_info), "vkBeginCommandBuffer");
    vkCmdResetQueryPool(command, query_pool, 0, 4);
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                        query_pool, 0);
    vkCmdBindDescriptorSets(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout, 0, 1, &descriptor_set, 0, nullptr);

    auto record_barrier = [&]() {
      VkMemoryBarrier memory_barrier{};
      memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      memory_barrier.dstAccessMask =
          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
      vkCmdPipelineBarrier(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                           1, &memory_barrier, 0, nullptr, 0, nullptr);
    };
    auto record_stage = [&](std::uint32_t stage, std::uint32_t groups,
                            std::uint32_t step_index, bool add_barrier) {
      push_constants.step_index = step_index;
      vkCmdBindPipeline(command, VK_PIPELINE_BIND_POINT_COMPUTE,
                        pipelines[stage]);
      vkCmdPushConstants(command, pipeline_layout,
                         VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(push_constants), &push_constants);
      vkCmdDispatch(command, groups, 1, 1);
      if (add_barrier)
        record_barrier();
    };

    const std::uint32_t full_groups =
        (dispatch_count + config.workgroup_size - 1U) /
        config.workgroup_size;
    record_stage(0, 1, 0, true);
    for (std::uint32_t stage = 1; stage <= 4; ++stage)
      record_stage(stage, full_groups, 0, true);
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        query_pool, 1);
    record_stage(5, full_groups, 0, true);
    if (config.multiply_chunk_size == 4) {
      for (int digit = 62; digit >= 0; digit -= 4)
        record_stage(7, full_groups, static_cast<std::uint32_t>(digit), true);
    } else {
      for (int digit = 62; digit >= 0; --digit) {
        for (std::uint32_t doubling = 0; doubling < 4;
             doubling += config.doublings_per_dispatch)
          record_stage(6, full_groups, 0, true);
        record_stage(7, full_groups, static_cast<std::uint32_t>(digit), true);
      }
    }
    record_stage(8, full_groups, 0, true);
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        query_pool, 2);
    record_stage(9, full_groups, 0, false);
    vkCmdWriteTimestamp(command, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        query_pool, 3);
    check(vkEndCommandBuffer(command), "vkEndCommandBuffer");

    check(vkResetFences(device, 1, &fence), "vkResetFences");
    const auto started = std::chrono::steady_clock::now();
    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command;
    check(vkQueueSubmit(queue, 1, &submit_info, fence), "vkQueueSubmit");
    check(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX),
          "vkWaitForFences");
    const auto stopped = std::chrono::steady_clock::now();
    validate(corpus, result_buffer, valid_buffer);

    std::uint64_t timestamps[4]{};
    check(vkGetQueryPoolResults(
              device, query_pool, 0, 4, sizeof(timestamps), timestamps,
              sizeof(std::uint64_t),
              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
          "vkGetQueryPoolResults");
    if (!collect_timing)
      return {};
    Timing timing;
    timing.host_seconds =
        std::chrono::duration<double>(stopped - started).count();
    const double timestamp_to_seconds =
        static_cast<double>(physical_properties.limits.timestampPeriod) /
        1'000'000'000.0;
    timing.gpu_seconds =
        static_cast<double>(timestamps[1] - timestamps[0]) *
        timestamp_to_seconds;
    timing.decode_seconds =
        static_cast<double>(timestamps[1] - timestamps[0]) *
        timestamp_to_seconds;
    timing.multiply_seconds =
        static_cast<double>(timestamps[2] - timestamps[1]) *
        timestamp_to_seconds;
    timing.compress_seconds =
        static_cast<double>(timestamps[3] - timestamps[2]) *
        timestamp_to_seconds;
    timing.gpu_seconds =
        timing.decode_seconds + timing.multiply_seconds +
        timing.compress_seconds;
    return timing;
  };

  for (std::uint32_t round = 0; round < config.warmup_rounds; ++round)
    run_round(false);
  double host_seconds = 0;
  double gpu_seconds = 0;
  double decode_seconds = 0;
  double multiply_seconds = 0;
  double compress_seconds = 0;
  for (std::uint32_t round = 0; round < config.rounds; ++round) {
    const auto timing = run_round(true);
    host_seconds += timing.host_seconds;
    gpu_seconds += timing.gpu_seconds;
    decode_seconds += timing.decode_seconds;
    multiply_seconds += timing.multiply_seconds;
    compress_seconds += timing.compress_seconds;
  }

  const std::uint64_t operations =
      static_cast<std::uint64_t>(corpus.valid_count) * config.rounds;
  if (config.implementation == "radix2625") {
    std::cout << "testbench=wallet_vulkan_mobile_derivation_v4\n";
    std::cout << "stage=Vulkan_M12_radix2625_chunked_d"
              << config.doublings_per_dispatch << '\n';
  } else {
    std::cout << "testbench=wallet_vulkan_radix13_derivation_v3\n";
    std::cout << "stage=Vulkan_M13_pure_u32_chunked_correctness_first\n";
  }
  std::cout << "algorithm=monero_generate_key_derivation_8_times_a_times_r\n";
  std::cout << "vulkan_device=" << physical_properties.deviceName << '\n';
  std::cout << "vulkan_api_version="
            << version_string(physical_properties.apiVersion) << '\n';
  if (config.implementation == "radix2625") {
    std::cout << "field_representation=radix_25_26_shader_int64\n";
    std::cout << "storage_layout=ten_limbs_per_field_element\n";
  } else {
    std::cout << "field_representation=radix_2_to_13_pure_u32\n";
    std::cout << "storage_layout=one_uint_per_logical_byte_correctness_first\n";
  }
  std::cout << "workgroup_size=" << config.workgroup_size << '\n';
  std::cout << "doublings_per_dispatch="
            << config.doublings_per_dispatch << '\n';
  std::cout << "multiply_chunk_size=" << config.multiply_chunk_size << '\n';
  std::cout << "compute_dispatches_per_round="
            << (config.multiply_chunk_size == 4
                    ? 24
                    : 8 + 63 *
                              (1 + 4 / config.doublings_per_dispatch))
            << '\n';
  std::cout << "points_per_round=" << corpus.valid_count << '\n';
  std::cout << "timed_rounds=" << config.rounds << '\n';
  std::cout << "warmup_rounds=" << config.warmup_rounds << '\n';
  std::cout << "operations=" << operations << '\n';
  std::cout.precision(9);
  std::cout << std::fixed;
  std::cout << "host_submit_wait_seconds=" << host_seconds << '\n';
  std::cout << "gpu_execution_seconds=" << gpu_seconds << '\n';
  std::cout << "gpu_decode_seconds=" << decode_seconds << '\n';
  std::cout << "gpu_multiply_seconds=" << multiply_seconds << '\n';
  std::cout << "gpu_compress_seconds=" << compress_seconds << '\n';
  std::cout.precision(3);
  std::cout << "derivations_per_second="
            << static_cast<double>(operations) / host_seconds << '\n';
  std::cout << std::hex;
  std::cout << "corpus_fingerprint_fnv1a64=0x" << corpus.fingerprint << '\n';
  std::cout << std::dec;
  std::cout << "invalid_point_contract=Dalek_rejected_returns_valid_0_and_zero_output\n";
  std::cout << "validation=pass\n";

  check(vkDeviceWaitIdle(device), "vkDeviceWaitIdle");
  destroy_buffer(device, scalar_buffer);
  destroy_buffer(device, point_buffer);
  destroy_buffer(device, scratch_buffer);
  destroy_buffer(device, state_buffer);
  destroy_buffer(device, result_buffer);
  destroy_buffer(device, valid_buffer);
  destroy_buffer(device, digit_buffer);
  destroy_buffer(device, table_buffer);
  vkDestroyQueryPool(device, query_pool, nullptr);
  vkDestroyFence(device, fence, nullptr);
  vkDestroyCommandPool(device, command_pool, nullptr);
  vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
  for (std::uint32_t index = 0; index < stage_count; ++index) {
    vkDestroyPipeline(device, pipelines[index], nullptr);
    vkDestroyShaderModule(device, shader_modules[index], nullptr);
  }
  vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
  vkDestroyDescriptorSetLayout(device, descriptor_layout, nullptr);
  vkDestroyDevice(device, nullptr);
  vkDestroyInstance(instance, nullptr);
  return 0;
}
