#include "derivation_core.cuh"
#include "derivation_radix2625.cuh"
#include "vector_corpus.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using monero_cuda::u8;
using monero_cuda::u32;

enum class Variant {
  layout,
  ladder,
  radix16,
  radix2625_chunk,
  radix2625_direct,
  radix2625_radix8,
  radix2625_radix8_sqrt_ratio,
  radix2625_radix8_sqrt_ratio_batch,
  radix2625_radix8_sqrt_ratio_batch_lb5,
  radix2625_radix8_sqrt_ratio_batch_lb6,
  radix2625_radix4_sqrt_ratio_batch,
};

struct Config {
  std::string vector_path;
  Variant variant = Variant::radix16;
  int rounds = 3;
  int warmup_rounds = 1;
  int threads_per_block = 64;
  int inverse_threads_per_block = 64;
  int compress_threads_per_block = 128;
  int batch_inversion_chunk_size = 16;
  int device = 0;
};

void cuda_check(cudaError_t status, const char *expression,
                const char *file, int line) {
  if (status == cudaSuccess) return;
  throw std::runtime_error(std::string(file) + ":" +
                           std::to_string(line) + ": " + expression +
                           " failed: " + cudaGetErrorString(status));
}

#define CUDA_CHECK(expression) \
  cuda_check((expression), #expression, __FILE__, __LINE__)

template <typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(std::size_t count) : count_(count) {
    if (count == 0) return;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&pointer_),
                          count * sizeof(T)));
  }

  ~DeviceBuffer() {
    if (pointer_) cudaFree(pointer_);
  }

  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *get() { return pointer_; }
  const T *get() const { return pointer_; }
  std::size_t count() const { return count_; }

 private:
  T *pointer_ = nullptr;
  std::size_t count_ = 0;
};

class Event {
 public:
  Event() { CUDA_CHECK(cudaEventCreate(&event_)); }
  ~Event() { cudaEventDestroy(event_); }
  Event(const Event &) = delete;
  Event &operator=(const Event &) = delete;
  operator cudaEvent_t() const { return event_; }

 private:
  cudaEvent_t event_{};
};

[[noreturn]] void usage(const std::string &message = {}) {
  if (!message.empty()) std::cerr << message << '\n';
  std::cerr
      << "Usage: wallet-cuda-testbench --vectors PATH "
         "[--variant layout|ladder|radix16|radix2625-chunk|"
         "radix2625-direct|radix2625-radix8|"
         "radix2625-radix8-sqrt-ratio|"
         "radix2625-radix8-sqrt-ratio-batch|"
         "radix2625-radix8-sqrt-ratio-batch-lb5|"
         "radix2625-radix8-sqrt-ratio-batch-lb6|"
         "radix2625-radix4-sqrt-ratio-batch] [--rounds N] "
         "[--warmup-rounds N] [--threads-per-block N] "
         "[--inverse-threads-per-block N] [--compress-threads-per-block N] "
         "[--batch-inversion-chunk-size N] [--device N]\n";
  std::exit(2);
}

int positive(const char *text, const std::string &option) {
  if (!text) usage(option + " requires an integer");
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!end || *end != '\0' || value <= 0 ||
      value > std::numeric_limits<int>::max())
    usage(option + " requires a positive integer");
  return static_cast<int>(value);
}

int nonnegative(const char *text, const std::string &option) {
  if (!text) usage(option + " requires an integer");
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!end || *end != '\0' || value < 0 ||
      value > std::numeric_limits<int>::max())
    usage(option + " requires a nonnegative integer");
  return static_cast<int>(value);
}

Config parse_args(int argc, char **argv) {
  Config config;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--vectors") {
      if (++index >= argc) usage("--vectors requires a path");
      config.vector_path = argv[index];
    } else if (argument == "--variant") {
      if (++index >= argc) usage("--variant requires a value");
      const std::string value = argv[index];
      if (value == "layout")
        config.variant = Variant::layout;
      else if (value == "ladder")
        config.variant = Variant::ladder;
      else if (value == "radix16")
        config.variant = Variant::radix16;
      else if (value == "radix2625-chunk")
        config.variant = Variant::radix2625_chunk;
      else if (value == "radix2625-direct")
        config.variant = Variant::radix2625_direct;
      else if (value == "radix2625-radix8")
        config.variant = Variant::radix2625_radix8;
      else if (value == "radix2625-radix8-sqrt-ratio")
        config.variant = Variant::radix2625_radix8_sqrt_ratio;
      else if (value == "radix2625-radix8-sqrt-ratio-batch")
        config.variant =
            Variant::radix2625_radix8_sqrt_ratio_batch;
      else if (value ==
               "radix2625-radix8-sqrt-ratio-batch-lb5")
        config.variant =
            Variant::radix2625_radix8_sqrt_ratio_batch_lb5;
      else if (value ==
               "radix2625-radix8-sqrt-ratio-batch-lb6")
        config.variant =
            Variant::radix2625_radix8_sqrt_ratio_batch_lb6;
      else if (value ==
               "radix2625-radix4-sqrt-ratio-batch")
        config.variant =
            Variant::radix2625_radix4_sqrt_ratio_batch;
      else
        usage("unknown variant: " + value);
    } else if (argument == "--rounds") {
      if (++index >= argc) usage("--rounds requires a value");
      config.rounds = positive(argv[index], argument);
    } else if (argument == "--warmup-rounds") {
      if (++index >= argc) usage("--warmup-rounds requires a value");
      config.warmup_rounds = nonnegative(argv[index], argument);
    } else if (argument == "--threads-per-block") {
      if (++index >= argc)
        usage("--threads-per-block requires a value");
      config.threads_per_block = positive(argv[index], argument);
    } else if (argument == "--inverse-threads-per-block") {
      if (++index >= argc)
        usage("--inverse-threads-per-block requires a value");
      config.inverse_threads_per_block = positive(argv[index], argument);
    } else if (argument == "--compress-threads-per-block") {
      if (++index >= argc)
        usage("--compress-threads-per-block requires a value");
      config.compress_threads_per_block = positive(argv[index], argument);
    } else if (argument == "--batch-inversion-chunk-size") {
      if (++index >= argc)
        usage("--batch-inversion-chunk-size requires a value");
      config.batch_inversion_chunk_size = positive(argv[index], argument);
    } else if (argument == "--device") {
      if (++index >= argc) usage("--device requires a value");
      config.device = nonnegative(argv[index], argument);
    } else if (argument == "--help" || argument == "-h") {
      usage();
    } else {
      usage("unknown argument: " + argument);
    }
  }
  if (config.vector_path.empty()) usage("--vectors is required");
  return config;
}

__global__ void fold_scalar_kernel(u8 *folded, char *radix16_digits,
                                   char *radix8_digits,
                                   char *radix4_digits,
                                   const u8 *input) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    monero_cuda::fold_scalar(folded, input);
    monero_cuda2625::scalar_as_radix_16_group(radix16_digits, folded);
    monero_cuda2625::scalar_as_radix_8_group(radix8_digits, folded);
    monero_cuda2625::scalar_as_radix_4_group(radix4_digits, folded);
  }
}

__global__ void layout_kernel(const u8 *points, u8 *results, u8 *valid,
                              std::size_t count) {
  const std::size_t id =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= count) return;
  const std::size_t offset = id * 32;
  for (u32 byte = 0; byte < 32; ++byte)
    results[offset + byte] = points[offset + byte];
  valid[id] = 1;
}

__global__ void ladder_kernel(const u8 *folded_global, const u8 *points,
                              u8 *results, u8 *valid,
                              std::size_t count) {
  __shared__ u8 folded[32];
  for (u32 byte = threadIdx.x; byte < 32; byte += blockDim.x)
    folded[byte] = folded_global[byte];
  __syncthreads();
  const std::size_t id =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= count) return;
  const std::size_t offset = id * 32;
  u8 encoded[32], scalar[32], result[32];
  for (u32 byte = 0; byte < 32; ++byte) {
    encoded[byte] = points[offset + byte];
    scalar[byte] = folded[byte];
  }
  const bool accepted =
      monero_cuda::derive_ladder(result, scalar, encoded);
  for (u32 byte = 0; byte < 32; ++byte)
    results[offset + byte] = result[byte];
  valid[id] = accepted ? 1 : 0;
}

__global__ void radix16_kernel(const u8 *folded_global, const u8 *points,
                               u8 *results, u8 *valid,
                               std::size_t count) {
  __shared__ u8 folded[32];
  for (u32 byte = threadIdx.x; byte < 32; byte += blockDim.x)
    folded[byte] = folded_global[byte];
  __syncthreads();
  const std::size_t id =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= count) return;
  const std::size_t offset = id * 32;
  u8 encoded[32], scalar[32], result[32];
  for (u32 byte = 0; byte < 32; ++byte) {
    encoded[byte] = points[offset + byte];
    scalar[byte] = folded[byte];
  }
  const bool accepted =
      monero_cuda::derive_radix16(result, scalar, encoded);
  for (u32 byte = 0; byte < 32; ++byte)
    results[offset + byte] = result[byte];
  valid[id] = accepted ? 1 : 0;
}

__global__ void radix2625_projective_kernel(
    const char *radix16_digits, const u8 *points, u32 *projective,
    u8 *valid, std::size_t count) {
  const std::size_t id =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= count) return;
  const std::size_t byte_offset = id * 32;
  const std::size_t field_offset = id * 30;
  u8 encoded[32];
  for (u32 byte = 0; byte < 32; ++byte)
    encoded[byte] = points[byte_offset + byte];

  monero_cuda2625::Point point, derived;
  if (!monero_cuda2625::point_from_compressed_m5(point, encoded)) {
    for (u32 limb = 0; limb < 30; ++limb)
      projective[field_offset + limb] = 0;
    valid[id] = 0;
    return;
  }
  monero_cuda2625::point_scalar_multiply_radix16_group(
      derived, point, radix16_digits);
  monero_cuda2625::fe_store_device(projective + field_offset, derived.x);
  monero_cuda2625::fe_store_device(projective + field_offset + 10,
                                    derived.y);
  monero_cuda2625::fe_store_device(projective + field_offset + 20,
                                    derived.z);
  valid[id] = 1;
}

__global__ void radix2625_direct_kernel(
    const char *radix16_digits, const u8 *points, u8 *results,
    u8 *valid, std::size_t count) {
  const std::size_t id =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= count) return;
  const std::size_t offset = id * 32;
  u8 encoded[32];
  for (u32 byte = 0; byte < 32; ++byte)
    encoded[byte] = points[offset + byte];

  monero_cuda2625::Point point, derived;
  if (!monero_cuda2625::point_from_compressed_m5(point, encoded)) {
    for (u32 byte = 0; byte < 32; ++byte)
      results[offset + byte] = 0;
    valid[id] = 0;
    return;
  }
  monero_cuda2625::point_scalar_multiply_radix16_group(
      derived, point, radix16_digits);
  monero_cuda2625::point_to_compressed_m5(encoded, derived);
  for (u32 byte = 0; byte < 32; ++byte)
    results[offset + byte] = encoded[byte];
  valid[id] = 1;
}

__global__ void radix2625_radix8_kernel(
    const char *radix8_digits, const u8 *points, u8 *results,
    u8 *valid, std::size_t count) {
  const std::size_t id =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= count) return;
  const std::size_t offset = id * 32;
  u8 encoded[32];
  for (u32 byte = 0; byte < 32; ++byte)
    encoded[byte] = points[offset + byte];

  monero_cuda2625::Point point, derived;
  if (!monero_cuda2625::point_from_compressed_m5(point, encoded)) {
    for (u32 byte = 0; byte < 32; ++byte)
      results[offset + byte] = 0;
    valid[id] = 0;
    return;
  }
  monero_cuda2625::point_scalar_multiply_radix8_group(
      derived, point, radix8_digits);
  monero_cuda2625::point_to_compressed_m5(encoded, derived);
  for (u32 byte = 0; byte < 32; ++byte)
    results[offset + byte] = encoded[byte];
  valid[id] = 1;
}

__global__ void radix2625_radix8_sqrt_ratio_kernel(
    const char *radix8_digits, const u8 *points, u8 *results,
    u8 *valid, std::size_t count) {
  const std::size_t id =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= count) return;
  const std::size_t offset = id * 32;
  u8 encoded[32];
  for (u32 byte = 0; byte < 32; ++byte)
    encoded[byte] = points[offset + byte];

  monero_cuda2625::Point point, derived;
  if (!monero_cuda2625::point_from_compressed_m6(point, encoded)) {
    for (u32 byte = 0; byte < 32; ++byte)
      results[offset + byte] = 0;
    valid[id] = 0;
    return;
  }
  monero_cuda2625::point_scalar_multiply_radix8_group(
      derived, point, radix8_digits);
  monero_cuda2625::point_to_compressed_m5(encoded, derived);
  for (u32 byte = 0; byte < 32; ++byte)
    results[offset + byte] = encoded[byte];
  valid[id] = 1;
}

__device__ __forceinline__
void radix2625_radix8_sqrt_ratio_projective_body(
    const char *radix8_digits, const u8 *points, u32 *projective,
    u8 *valid, std::size_t count) {
  const std::size_t id =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= count) return;
  const std::size_t byte_offset = id * 32;
  const std::size_t field_offset = id * 30;
  u8 encoded[32];
  for (u32 byte = 0; byte < 32; ++byte)
    encoded[byte] = points[byte_offset + byte];

  monero_cuda2625::Point point, derived;
  if (!monero_cuda2625::point_from_compressed_m6(point, encoded)) {
    for (u32 limb = 0; limb < 30; ++limb)
      projective[field_offset + limb] = 0;
    valid[id] = 0;
    return;
  }
  monero_cuda2625::point_scalar_multiply_radix8_group(
      derived, point, radix8_digits);
  monero_cuda2625::fe_store_device(projective + field_offset,
                                   derived.x);
  monero_cuda2625::fe_store_device(projective + field_offset + 10,
                                   derived.y);
  monero_cuda2625::fe_store_device(projective + field_offset + 20,
                                   derived.z);
  valid[id] = 1;
}

__global__ void radix2625_radix8_sqrt_ratio_projective_kernel(
    const char *radix8_digits, const u8 *points, u32 *projective,
    u8 *valid, std::size_t count) {
  radix2625_radix8_sqrt_ratio_projective_body(
      radix8_digits, points, projective, valid, count);
}

__global__ __launch_bounds__(64, 5)
void radix2625_radix8_sqrt_ratio_projective_lb5_kernel(
    const char *radix8_digits, const u8 *points, u32 *projective,
    u8 *valid, std::size_t count) {
  radix2625_radix8_sqrt_ratio_projective_body(
      radix8_digits, points, projective, valid, count);
}

__global__ __launch_bounds__(64, 6)
void radix2625_radix8_sqrt_ratio_projective_lb6_kernel(
    const char *radix8_digits, const u8 *points, u32 *projective,
    u8 *valid, std::size_t count) {
  radix2625_radix8_sqrt_ratio_projective_body(
      radix8_digits, points, projective, valid, count);
}

__global__ void radix2625_radix4_sqrt_ratio_projective_kernel(
    const char *radix4_digits, const u8 *points, u32 *projective,
    u8 *valid, std::size_t count) {
  const std::size_t id =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= count) return;

  const std::size_t byte_offset = id * 32;
  const std::size_t field_offset = id * 30;
  u8 encoded[32];
  for (u32 byte = 0; byte < 32; ++byte)
    encoded[byte] = points[byte_offset + byte];

  monero_cuda2625::Point point, derived;
  if (!monero_cuda2625::point_from_compressed_m6(point, encoded)) {
    for (u32 limb = 0; limb < 30; ++limb)
      projective[field_offset + limb] = 0;
    valid[id] = 0;
    return;
  }

  monero_cuda2625::point_scalar_multiply_radix4_group(
      derived, point, radix4_digits);
  monero_cuda2625::fe_store_device(projective + field_offset,
                                   derived.x);
  monero_cuda2625::fe_store_device(projective + field_offset + 10,
                                   derived.y);
  monero_cuda2625::fe_store_device(projective + field_offset + 20,
                                   derived.z);
  valid[id] = 1;
}

__global__ void radix2625_inverse_kernel(
    const u32 *projective, u32 *inverse_z, const u8 *valid,
    std::size_t count, u32 chunk_size) {
  const std::size_t chunk =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t begin = chunk * chunk_size;
  if (begin >= count) return;
  const std::size_t end =
      begin + chunk_size < count ? begin + chunk_size : count;

  monero_cuda2625::Fe accumulator;
  monero_cuda2625::fe_one(accumulator);
  for (std::size_t record = begin; record < end; ++record) {
    const std::size_t field_offset = record * 30;
    const std::size_t inverse_offset = record * 10;
    monero_cuda2625::fe_store_device(inverse_z + inverse_offset,
                                     accumulator);
    monero_cuda2625::Fe z, next;
    if (valid[record] != 0)
      monero_cuda2625::fe_load_device(
          z, projective + field_offset + 20);
    else
      monero_cuda2625::fe_one(z);
    monero_cuda2625::fe_mul(next, accumulator, z);
    monero_cuda2625::fe_copy(accumulator, next);
  }

  monero_cuda2625::Fe inverse_product;
  monero_cuda2625::fe_inverse_m5(inverse_product, accumulator);
  std::size_t remaining = end;
  while (remaining != begin) {
    const std::size_t record = --remaining;
    const std::size_t field_offset = record * 30;
    const std::size_t inverse_offset = record * 10;
    monero_cuda2625::Fe prefix, z, inverse_record, next;
    monero_cuda2625::fe_load_device(prefix,
                                     inverse_z + inverse_offset);
    if (valid[record] != 0)
      monero_cuda2625::fe_load_device(
          z, projective + field_offset + 20);
    else
      monero_cuda2625::fe_one(z);
    monero_cuda2625::fe_mul(inverse_record, inverse_product, prefix);
    if (valid[record] != 0) {
      monero_cuda2625::fe_store_device(inverse_z + inverse_offset,
                                       inverse_record);
    } else {
      monero_cuda2625::Fe zero;
      monero_cuda2625::fe_zero(zero);
      monero_cuda2625::fe_store_device(inverse_z + inverse_offset,
                                       zero);
    }
    monero_cuda2625::fe_mul(next, inverse_product, z);
    monero_cuda2625::fe_copy(inverse_product, next);
  }
}

__global__ void radix2625_compress_kernel(
    const u32 *projective, const u32 *inverse_z, u8 *results,
    const u8 *valid, std::size_t count) {
  const std::size_t id =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id >= count) return;
  const std::size_t byte_offset = id * 32;
  if (valid[id] == 0) {
    for (u32 byte = 0; byte < 32; ++byte)
      results[byte_offset + byte] = 0;
    return;
  }

  const std::size_t field_offset = id * 30;
  const std::size_t inverse_offset = id * 10;
  monero_cuda2625::Fe projective_x, projective_y, inverse, x, y;
  monero_cuda2625::fe_load_device(projective_x,
                                  projective + field_offset);
  monero_cuda2625::fe_load_device(projective_y,
                                  projective + field_offset + 10);
  monero_cuda2625::fe_load_device(inverse,
                                  inverse_z + inverse_offset);
  monero_cuda2625::fe_mul(x, projective_x, inverse);
  monero_cuda2625::fe_mul(y, projective_y, inverse);
  u8 encoded[32];
  monero_cuda2625::fe_to_bytes(encoded, y);
  encoded[31] ^= monero_cuda2625::fe_is_negative(x) ? 128 : 0;
  for (u32 byte = 0; byte < 32; ++byte)
    results[byte_offset + byte] = encoded[byte];
}

const char *variant_name(Variant variant) {
  switch (variant) {
    case Variant::layout:
      return "cuda_c0_layout";
    case Variant::ladder:
      return "cuda_c1_ladder_addition_chain";
    case Variant::radix16:
      return "cuda_c2_radix16_niels_addition_chain";
    case Variant::radix2625_chunk:
      return "cuda_c3_radix2625_chunk_inversion";
    case Variant::radix2625_direct:
      return "cuda_c4_radix2625_direct";
    case Variant::radix2625_radix8:
      return "cuda_c5_radix2625_radix8_direct";
    case Variant::radix2625_radix8_sqrt_ratio:
      return "cuda_c6_radix2625_radix8_sqrt_ratio";
    case Variant::radix2625_radix8_sqrt_ratio_batch:
      return "cuda_c7_radix2625_radix8_sqrt_ratio_batch";
    case Variant::radix2625_radix8_sqrt_ratio_batch_lb5:
      return "cuda_c8_radix2625_radix8_sqrt_ratio_batch_lb5";
    case Variant::radix2625_radix8_sqrt_ratio_batch_lb6:
      return "cuda_c9_radix2625_radix8_sqrt_ratio_batch_lb6";
    case Variant::radix2625_radix4_sqrt_ratio_batch:
      return "cuda_c11_radix2625_radix4_sqrt_ratio_batch";
  }
  return "unknown";
}

cudaFuncAttributes function_attributes(Variant variant) {
  cudaFuncAttributes attributes{};
  if (variant == Variant::layout)
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, layout_kernel));
  else if (variant == Variant::ladder)
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, ladder_kernel));
  else if (variant == Variant::radix16)
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, radix16_kernel));
  else if (variant == Variant::radix2625_chunk)
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, radix2625_projective_kernel));
  else if (variant == Variant::radix2625_direct)
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     radix2625_direct_kernel));
  else if (variant == Variant::radix2625_radix8)
    CUDA_CHECK(cudaFuncGetAttributes(&attributes,
                                     radix2625_radix8_kernel));
  else if (variant ==
           Variant::radix2625_radix8_sqrt_ratio_batch)
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes,
        radix2625_radix8_sqrt_ratio_projective_kernel));
  else if (variant ==
           Variant::radix2625_radix8_sqrt_ratio_batch_lb5)
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes,
        radix2625_radix8_sqrt_ratio_projective_lb5_kernel));
  else if (variant ==
           Variant::radix2625_radix8_sqrt_ratio_batch_lb6)
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes,
        radix2625_radix8_sqrt_ratio_projective_lb6_kernel));
  else if (variant ==
           Variant::radix2625_radix4_sqrt_ratio_batch)
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes,
        radix2625_radix4_sqrt_ratio_projective_kernel));
  else
    CUDA_CHECK(cudaFuncGetAttributes(
        &attributes, radix2625_radix8_sqrt_ratio_kernel));
  return attributes;
}

void launch(const Config &config, int blocks, int inverse_blocks,
            int compress_blocks, const u8 *folded,
            const char *radix16_digits, const char *radix8_digits,
            const char *radix4_digits,
            const u8 *points, u8 *results, u8 *valid, u32 *projective,
            u32 *inverse_z, std::size_t count) {
  if (config.variant == Variant::layout) {
    layout_kernel<<<blocks, config.threads_per_block>>>(
        points, results, valid, count);
  } else if (config.variant == Variant::ladder) {
    ladder_kernel<<<blocks, config.threads_per_block>>>(
        folded, points, results, valid, count);
  } else if (config.variant == Variant::radix16) {
    radix16_kernel<<<blocks, config.threads_per_block>>>(
        folded, points, results, valid, count);
  } else if (config.variant == Variant::radix2625_chunk) {
    radix2625_projective_kernel<<<blocks, config.threads_per_block>>>(
        radix16_digits, points, projective, valid, count);
    radix2625_inverse_kernel<<<inverse_blocks,
                               config.inverse_threads_per_block>>>(
        projective, inverse_z, valid, count,
        static_cast<u32>(config.batch_inversion_chunk_size));
    radix2625_compress_kernel<<<compress_blocks,
                                config.compress_threads_per_block>>>(
        projective, inverse_z, results, valid, count);
  } else if (config.variant == Variant::radix2625_direct) {
    radix2625_direct_kernel<<<blocks, config.threads_per_block>>>(
        radix16_digits, points, results, valid, count);
  } else if (config.variant == Variant::radix2625_radix8) {
    radix2625_radix8_kernel<<<blocks, config.threads_per_block>>>(
        radix8_digits, points, results, valid, count);
  } else if (config.variant ==
             Variant::radix2625_radix8_sqrt_ratio_batch) {
    radix2625_radix8_sqrt_ratio_projective_kernel<<<
        blocks, config.threads_per_block>>>(
        radix8_digits, points, projective, valid, count);
    radix2625_inverse_kernel<<<inverse_blocks,
                               config.inverse_threads_per_block>>>(
        projective, inverse_z, valid, count,
        static_cast<u32>(config.batch_inversion_chunk_size));
    radix2625_compress_kernel<<<compress_blocks,
                                config.compress_threads_per_block>>>(
        projective, inverse_z, results, valid, count);
  } else if (config.variant ==
             Variant::radix2625_radix8_sqrt_ratio_batch_lb5) {
    radix2625_radix8_sqrt_ratio_projective_lb5_kernel<<<
        blocks, config.threads_per_block>>>(
        radix8_digits, points, projective, valid, count);
    radix2625_inverse_kernel<<<inverse_blocks,
                               config.inverse_threads_per_block>>>(
        projective, inverse_z, valid, count,
        static_cast<u32>(config.batch_inversion_chunk_size));
    radix2625_compress_kernel<<<compress_blocks,
                                config.compress_threads_per_block>>>(
        projective, inverse_z, results, valid, count);
  } else if (config.variant ==
             Variant::radix2625_radix8_sqrt_ratio_batch_lb6) {
    radix2625_radix8_sqrt_ratio_projective_lb6_kernel<<<
        blocks, config.threads_per_block>>>(
        radix8_digits, points, projective, valid, count);
    radix2625_inverse_kernel<<<inverse_blocks,
                               config.inverse_threads_per_block>>>(
        projective, inverse_z, valid, count,
        static_cast<u32>(config.batch_inversion_chunk_size));
    radix2625_compress_kernel<<<compress_blocks,
                                config.compress_threads_per_block>>>(
        projective, inverse_z, results, valid, count);
  } else if (config.variant ==
             Variant::radix2625_radix4_sqrt_ratio_batch) {
    radix2625_radix4_sqrt_ratio_projective_kernel<<<
        blocks, config.threads_per_block>>>(
        radix4_digits, points, projective, valid, count);
    radix2625_inverse_kernel<<<inverse_blocks,
                               config.inverse_threads_per_block>>>(
        projective, inverse_z, valid, count,
        static_cast<u32>(config.batch_inversion_chunk_size));
    radix2625_compress_kernel<<<compress_blocks,
                                config.compress_threads_per_block>>>(
        projective, inverse_z, results, valid, count);
  } else {
    radix2625_radix8_sqrt_ratio_kernel<<<
        blocks, config.threads_per_block>>>(
        radix8_digits, points, results, valid, count);
  }
  CUDA_CHECK(cudaGetLastError());
}

void validate_layout(const std::vector<u8> &points,
                     const std::vector<u8> &results,
                     const std::vector<u8> &valid) {
  if (points != results)
    throw std::runtime_error("layout kernel did not copy input bytes");
  for (std::size_t index = 0; index < valid.size(); ++index)
    if (valid[index] != 1)
      throw std::runtime_error("layout kernel left invalid status");
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Config config = parse_args(argc, argv);
    const auto corpus =
        monero_cuda_testbench::read_vector_corpus(config.vector_path);
    const std::size_t dispatch_count =
        static_cast<std::size_t>(corpus.count) + 1;
    std::vector<u8> points = corpus.points;
    points.insert(points.end(), corpus.invalid.begin(), corpus.invalid.end());
    std::vector<u8> results(dispatch_count * 32);
    std::vector<u8> valid(dispatch_count);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (config.device >= device_count)
      throw std::runtime_error("requested CUDA device does not exist");
    CUDA_CHECK(cudaSetDevice(config.device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, config.device));
    if (config.threads_per_block > properties.maxThreadsPerBlock)
      throw std::runtime_error("threads per block exceed device limit");
    const auto attributes = function_attributes(config.variant);
    if (config.threads_per_block > attributes.maxThreadsPerBlock)
      throw std::runtime_error("threads per block exceed kernel limit");
    cudaFuncAttributes inverse_attributes{};
    cudaFuncAttributes compress_attributes{};
    if (config.variant == Variant::radix2625_chunk ||
        config.variant ==
            Variant::radix2625_radix8_sqrt_ratio_batch ||
        config.variant ==
            Variant::radix2625_radix8_sqrt_ratio_batch_lb5 ||
        config.variant ==
            Variant::radix2625_radix8_sqrt_ratio_batch_lb6 ||
        config.variant ==
            Variant::radix2625_radix4_sqrt_ratio_batch) {
      CUDA_CHECK(cudaFuncGetAttributes(&inverse_attributes,
                                       radix2625_inverse_kernel));
      CUDA_CHECK(cudaFuncGetAttributes(&compress_attributes,
                                       radix2625_compress_kernel));
      if (config.inverse_threads_per_block >
          inverse_attributes.maxThreadsPerBlock)
        throw std::runtime_error(
            "inverse threads per block exceed kernel limit");
      if (config.compress_threads_per_block >
          compress_attributes.maxThreadsPerBlock)
        throw std::runtime_error(
            "compress threads per block exceed kernel limit");
    }

    DeviceBuffer<u8> scalar(32);
    DeviceBuffer<u8> folded(32);
    DeviceBuffer<char> radix16_digits(64);
    DeviceBuffer<char> radix8_digits(86);
    DeviceBuffer<char> radix4_digits(128);
    DeviceBuffer<u8> device_points(points.size());
    DeviceBuffer<u8> device_results(results.size());
    DeviceBuffer<u8> device_valid(valid.size());
    const bool uses_chunk_inversion =
        config.variant == Variant::radix2625_chunk ||
        config.variant ==
            Variant::radix2625_radix8_sqrt_ratio_batch ||
        config.variant ==
            Variant::radix2625_radix8_sqrt_ratio_batch_lb5 ||
        config.variant ==
            Variant::radix2625_radix8_sqrt_ratio_batch_lb6 ||
        config.variant ==
            Variant::radix2625_radix4_sqrt_ratio_batch;
    DeviceBuffer<u32> device_projective(
        uses_chunk_inversion ? dispatch_count * 30 : 0);
    DeviceBuffer<u32> device_inverse_z(
        uses_chunk_inversion ? dispatch_count * 10 : 0);
    CUDA_CHECK(cudaMemcpy(scalar.get(), corpus.scalar.data(), 32,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_points.get(), points.data(), points.size(),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_results.get(), 0, results.size()));
    CUDA_CHECK(cudaMemset(device_valid.get(), 0, valid.size()));

    Event scalar_start, scalar_stop;
    CUDA_CHECK(cudaEventRecord(scalar_start));
    fold_scalar_kernel<<<1, 1>>>(folded.get(), radix16_digits.get(),
                                 radix8_digits.get(),
                                 radix4_digits.get(), scalar.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(scalar_stop));
    CUDA_CHECK(cudaEventSynchronize(scalar_stop));
    float scalar_milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&scalar_milliseconds, scalar_start,
                                    scalar_stop));

    const std::size_t records_per_block =
        static_cast<std::size_t>(config.threads_per_block);
    if (dispatch_count >
        static_cast<std::size_t>(std::numeric_limits<int>::max()) *
            records_per_block)
      throw std::runtime_error("CUDA grid size overflows int");
    const int blocks = static_cast<int>(
        (dispatch_count + records_per_block - 1) /
        records_per_block);
    const std::size_t inverse_chunks =
        (dispatch_count + config.batch_inversion_chunk_size - 1) /
        config.batch_inversion_chunk_size;
    const int inverse_blocks = static_cast<int>(
        (inverse_chunks + config.inverse_threads_per_block - 1) /
        config.inverse_threads_per_block);
    const int compress_blocks = static_cast<int>(
        (dispatch_count + config.compress_threads_per_block - 1) /
        config.compress_threads_per_block);
    for (int round = 0; round < config.warmup_rounds; ++round)
      launch(config, blocks, inverse_blocks, compress_blocks, folded.get(),
             radix16_digits.get(), radix8_digits.get(),
             radix4_digits.get(), device_points.get(),
             device_results.get(), device_valid.get(),
             device_projective.get(), device_inverse_z.get(),
             dispatch_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    Event start, stop;
    const auto host_started = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaEventRecord(start));
    for (int round = 0; round < config.rounds; ++round)
      launch(config, blocks, inverse_blocks, compress_blocks, folded.get(),
             radix16_digits.get(), radix8_digits.get(),
             radix4_digits.get(), device_points.get(),
             device_results.get(), device_valid.get(),
             device_projective.get(), device_inverse_z.get(),
             dispatch_count);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    const auto host_elapsed =
        std::chrono::steady_clock::now() - host_started;
    float gpu_milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_milliseconds, start, stop));
    CUDA_CHECK(cudaMemcpy(results.data(), device_results.get(), results.size(),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(valid.data(), device_valid.get(), valid.size(),
                          cudaMemcpyDeviceToHost));

    if (config.variant == Variant::layout)
      validate_layout(points, results, valid);
    else
      monero_cuda_testbench::validate_results(corpus, results, valid);

    const double gpu_seconds =
        static_cast<double>(gpu_milliseconds) / 1000.0;
    const double host_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(
            host_elapsed)
            .count();
    const std::uint64_t operations =
        static_cast<std::uint64_t>(corpus.count) * config.rounds;
    const double rate = static_cast<double>(operations) / gpu_seconds;
    const double logical_mib =
        static_cast<double>(operations) * 64.0 / (1024.0 * 1024.0);

    std::cout << "result_begin\n";
    std::cout << "testbench=monero_wallet_cuda_testbench_v1\n";
    std::cout
        << "algorithm=monero_generate_key_derivation_8_times_a_times_r\n";
    std::cout << "variant=" << variant_name(config.variant) << '\n';
    std::cout << "cuda_device=" << properties.name << '\n';
    std::cout << "cuda_compute_capability=" << properties.major << '.'
              << properties.minor << '\n';
    std::cout << "cuda_multiprocessors=" << properties.multiProcessorCount
              << '\n';
    std::cout << "cuda_global_memory_bytes=" << properties.totalGlobalMem
              << '\n';
    std::cout << "kernel_registers_per_thread=" << attributes.numRegs << '\n';
    std::cout << "kernel_local_bytes_per_thread="
              << attributes.localSizeBytes << '\n';
    std::cout << "kernel_static_shared_bytes="
              << attributes.sharedSizeBytes << '\n';
    std::cout << "kernel_max_threads_per_block="
              << attributes.maxThreadsPerBlock << '\n';
    if (uses_chunk_inversion) {
      std::cout << "inverse_kernel_registers_per_thread="
                << inverse_attributes.numRegs << '\n';
      std::cout << "inverse_kernel_local_bytes_per_thread="
                << inverse_attributes.localSizeBytes << '\n';
      std::cout << "inverse_kernel_max_threads_per_block="
                << inverse_attributes.maxThreadsPerBlock << '\n';
      std::cout << "compress_kernel_registers_per_thread="
                << compress_attributes.numRegs << '\n';
      std::cout << "compress_kernel_local_bytes_per_thread="
                << compress_attributes.localSizeBytes << '\n';
      std::cout << "compress_kernel_max_threads_per_block="
                << compress_attributes.maxThreadsPerBlock << '\n';
    }
    std::cout << "corpus_fingerprint_fnv1a64=0x" << std::hex
              << corpus.fingerprint << std::dec << '\n';
    std::cout << "valid_points_per_round=" << corpus.count << '\n';
    std::cout << "dispatch_records_per_round=" << dispatch_count << '\n';
    std::cout << "timed_rounds=" << config.rounds << '\n';
    std::cout << "warmup_rounds=" << config.warmup_rounds << '\n';
    std::cout << "threads_per_block=" << config.threads_per_block << '\n';
    std::cout << "blocks=" << blocks << '\n';
    if (uses_chunk_inversion) {
      std::cout << "batch_inversion_chunk_size="
                << config.batch_inversion_chunk_size << '\n';
      std::cout << "inverse_threads_per_block="
                << config.inverse_threads_per_block << '\n';
      std::cout << "inverse_blocks=" << inverse_blocks << '\n';
      std::cout << "compress_threads_per_block="
                << config.compress_threads_per_block << '\n';
      std::cout << "compress_blocks=" << compress_blocks << '\n';
    }
    std::cout << "operations=" << operations << '\n';
    std::cout << std::fixed << std::setprecision(9);
    std::cout << "scalar_setup_seconds="
              << static_cast<double>(scalar_milliseconds) / 1000.0 << '\n';
    std::cout << "gpu_elapsed_seconds=" << gpu_seconds << '\n';
    std::cout << "host_submit_wait_seconds=" << host_seconds << '\n';
    std::cout << std::setprecision(3);
    if (config.variant == Variant::layout)
      std::cout << "layout_records_per_second=" << rate << '\n';
    else
      std::cout << "derivations_per_second=" << rate << '\n';
    std::cout << "logical_mib_per_second=" << logical_mib / gpu_seconds
              << '\n';
    std::cout << "result_checksum_fnv1a64=0x" << std::hex
              << monero_cuda_testbench::checksum_results(results) << std::dec
              << '\n';
    std::cout << "validation=pass\n";
    std::cout << "result_end\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "wallet-cuda-testbench error: " << error.what() << '\n';
    return 1;
  }
}
