#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

[[noreturn]] void fail(const char *message, int code = 2) {
  std::cerr << "wallet Vulkan probe error: " << message << '\n';
  std::exit(code);
}

void check(VkResult result, const char *operation) {
  if (result != VK_SUCCESS) {
    std::cerr << "wallet Vulkan probe error: " << operation
              << " failed with VkResult " << static_cast<int>(result) << '\n';
    std::exit(2);
  }
}

std::string version_string(std::uint32_t version) {
  return std::to_string(VK_VERSION_MAJOR(version)) + "." +
      std::to_string(VK_VERSION_MINOR(version)) + "." +
      std::to_string(VK_VERSION_PATCH(version));
}

const char *yes_no(VkBool32 value) {
  return value == VK_TRUE ? "yes" : "no";
}

}  // namespace

int main() {
  std::uint32_t loader_version = VK_API_VERSION_1_0;
  const auto enumerate_instance_version =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion"));
  if (enumerate_instance_version != nullptr) {
    check(enumerate_instance_version(&loader_version),
          "vkEnumerateInstanceVersion");
  }

  VkApplicationInfo application_info{};
  application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  application_info.pApplicationName = "wallet-vulkan-capability-probe";
  application_info.applicationVersion = 1;
  application_info.pEngineName = "none";
  application_info.engineVersion = 1;
  application_info.apiVersion =
      loader_version >= VK_API_VERSION_1_1 ? VK_API_VERSION_1_1
                                          : VK_API_VERSION_1_0;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &application_info;

  VkInstance instance = VK_NULL_HANDLE;
  check(vkCreateInstance(&create_info, nullptr, &instance), "vkCreateInstance");

  std::uint32_t device_count = 0;
  check(vkEnumeratePhysicalDevices(instance, &device_count, nullptr),
        "vkEnumeratePhysicalDevices(count)");
  if (device_count == 0) {
    vkDestroyInstance(instance, nullptr);
    fail("no Vulkan physical device was reported", 3);
  }

  std::vector<VkPhysicalDevice> devices(device_count);
  check(vkEnumeratePhysicalDevices(instance, &device_count, devices.data()),
        "vkEnumeratePhysicalDevices(list)");

  std::cout << "schema=wallet_android_vulkan_capability_v1\n";
  std::cout << "vulkan_loader_version=" << version_string(loader_version)
            << '\n';
  std::cout << "physical_device_count=" << device_count << '\n';

  bool found_compute_queue = false;
  for (std::uint32_t device_index = 0; device_index < device_count;
       ++device_index) {
    const VkPhysicalDevice device = devices[device_index];
    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceFeatures features{};
    vkGetPhysicalDeviceProperties(device, &properties);
    vkGetPhysicalDeviceFeatures(device, &features);

    std::uint32_t queue_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count, nullptr);
    std::vector<VkQueueFamilyProperties> queues(queue_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_count,
                                             queues.data());

    std::uint32_t compute_queue_count = 0;
    std::uint32_t dedicated_compute_queue_count = 0;
    for (const VkQueueFamilyProperties &queue : queues) {
      if ((queue.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U) {
        compute_queue_count += queue.queueCount;
        found_compute_queue = true;
        if ((queue.queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0U) {
          dedicated_compute_queue_count += queue.queueCount;
        }
      }
    }

    std::uint32_t subgroup_size = 0;
    VkShaderStageFlags subgroup_stages = 0;
    VkSubgroupFeatureFlags subgroup_operations = 0;
    if (properties.apiVersion >= VK_API_VERSION_1_1) {
      const auto get_properties2 =
          reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
              vkGetInstanceProcAddr(instance,
                                    "vkGetPhysicalDeviceProperties2"));
      if (get_properties2 != nullptr) {
        VkPhysicalDeviceSubgroupProperties subgroup{};
        subgroup.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
        VkPhysicalDeviceProperties2 properties2{};
        properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        properties2.pNext = &subgroup;
        get_properties2(device, &properties2);
        subgroup_size = subgroup.subgroupSize;
        subgroup_stages = subgroup.supportedStages;
        subgroup_operations = subgroup.supportedOperations;
      }
    }

    const std::string prefix = "device_" + std::to_string(device_index) + "_";
    std::cout << prefix << "name=" << properties.deviceName << '\n';
    std::cout << prefix << "vendor_id=" << properties.vendorID << '\n';
    std::cout << prefix << "device_id=" << properties.deviceID << '\n';
    std::cout << prefix << "api_version="
              << version_string(properties.apiVersion) << '\n';
    std::cout << prefix << "driver_version=" << properties.driverVersion
              << '\n';
    std::cout << prefix << "shader_int64=" << yes_no(features.shaderInt64)
              << '\n';
    std::cout << prefix << "compute_queue_count=" << compute_queue_count
              << '\n';
    std::cout << prefix
              << "dedicated_compute_queue_count="
              << dedicated_compute_queue_count << '\n';
    std::cout << prefix << "subgroup_size=" << subgroup_size << '\n';
    std::cout << prefix << "subgroup_compute_supported="
              << (((subgroup_stages & VK_SHADER_STAGE_COMPUTE_BIT) != 0U)
                      ? "yes"
                      : "no")
              << '\n';
    std::cout << prefix << "subgroup_operations=" << subgroup_operations
              << '\n';
    std::cout << prefix << "max_compute_workgroup_invocations="
              << properties.limits.maxComputeWorkGroupInvocations << '\n';
    std::cout << prefix << "max_compute_workgroup_size_x="
              << properties.limits.maxComputeWorkGroupSize[0] << '\n';
    std::cout << prefix << "max_compute_workgroup_count_x="
              << properties.limits.maxComputeWorkGroupCount[0] << '\n';
    std::cout << prefix << "max_compute_shared_memory_bytes="
              << properties.limits.maxComputeSharedMemorySize << '\n';
    std::cout << prefix << "max_storage_buffer_range="
              << properties.limits.maxStorageBufferRange << '\n';
    std::cout << prefix << "timestamp_compute_and_graphics="
              << yes_no(properties.limits.timestampComputeAndGraphics)
              << '\n';
    std::cout << prefix << "timestamp_period_ns="
              << properties.limits.timestampPeriod << '\n';
  }

  vkDestroyInstance(instance, nullptr);
  std::cout << "compute_ready=" << (found_compute_queue ? "yes" : "no")
            << '\n';
  return found_compute_queue ? 0 : 4;
}
