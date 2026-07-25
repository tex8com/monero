#define __STDC_WANT_LIB_EXT1__ 1

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "monero_fast_metal.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace
{
  constexpr NSUInteger PROJECTIVE_THREADS = 256;
  constexpr NSUInteger INVERSE_THREADS = 16;
  constexpr NSUInteger COMPRESS_THREADS = 128;
  constexpr uint32_t BATCH_INVERSION_CHUNK = 16;

  struct Parameters
  {
    uint32_t count;
    uint32_t batch_chunk_size;
  };

  void secure_zero(void *memory, size_t bytes)
  {
    // macOS guarantees the C11 bounds-checked memset_s call is not removed by
    // the optimizer; unlike a volatile byte loop it still uses the platform's
    // optimized bulk-memory implementation.
    (void)memset_s(memory, bytes, 0, bytes);
  }

  std::string describe_error(NSString *prefix, NSError *error)
  {
    NSString *description = error == nil ? @"unknown error" : error.localizedDescription;
    NSString *combined = [NSString stringWithFormat:@"%@: %@", prefix, description];
    return std::string(combined.UTF8String == nullptr ? "Metal error" : combined.UTF8String);
  }

  std::vector<NSString *> library_candidates()
  {
    NSDictionary<NSString *, NSString *> *environment =
      NSProcessInfo.processInfo.environment;
    NSString *configured = environment[@"MONERO_METAL_LIBRARY_PATH"];
    if (configured.length != 0)
      return {configured};

    NSMutableArray<NSString *> *paths = [NSMutableArray array];
    NSString *resource = NSBundle.mainBundle.resourcePath;
    if (resource.length != 0)
    {
      [paths addObject:[resource stringByAppendingPathComponent:
        @"monero_wallet_derivation.metallib"]];
      [paths addObject:[resource stringByAppendingPathComponent:
        @"_up_/native-libs/monero_wallet_derivation.metallib"]];
      [paths addObject:[resource stringByAppendingPathComponent:
        @"native-libs/monero_wallet_derivation.metallib"]];
    }

    NSString *executable = NSBundle.mainBundle.executablePath;
    if (executable.length != 0)
    {
      NSString *directory = executable.stringByDeletingLastPathComponent;
      [paths addObject:[directory stringByAppendingPathComponent:
        @"monero_wallet_derivation.metallib"]];
    }

    std::vector<NSString *> result;
    result.reserve(paths.count);
    for (NSString *path in paths)
      result.push_back(path);
    return result;
  }

  class MetalDerivationContext
  {
  public:
    bool available()
    {
      std::lock_guard<std::mutex> lock(mutex_);
      return initialize_locked();
    }

    int64_t derive(
        uint8_t *results,
        const uint8_t *scalar,
        const uint8_t *points,
        uint8_t *valid,
        size_t count)
    {
      if (count == 0)
        return 0;
      if (results == nullptr || scalar == nullptr || points == nullptr || valid == nullptr)
        return MONERO_FAST_METAL_INVALID_ARGUMENT;
      if (count > std::numeric_limits<uint32_t>::max()
          || count > std::numeric_limits<size_t>::max() / 120
          || count > std::numeric_limits<size_t>::max() / 32)
        return MONERO_FAST_METAL_INVALID_ARGUMENT;

      @autoreleasepool
      {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!initialize_locked())
          return MONERO_FAST_METAL_UNAVAILABLE;

        const size_t point_bytes = count * 32;
        const size_t projective_bytes = count * 30 * sizeof(uint32_t);
        const size_t inverse_bytes = count * 10 * sizeof(uint32_t);
        Parameters parameters{
          static_cast<uint32_t>(count),
          BATCH_INVERSION_CHUNK
        };

        if (!ensure_buffers_locked(count))
        {
          last_error_ = "Metal shared-buffer allocation failed";
          zero_secret_buffers(point_bytes, projective_bytes, inverse_bytes);
          return MONERO_FAST_METAL_EXECUTION_ERROR;
        }
        std::memcpy(scalar_buffer_.contents, scalar, 32);
        std::memcpy(point_buffer_.contents, points, point_bytes);
        std::memcpy(parameter_buffer_.contents, &parameters, sizeof(parameters));

        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        if (command == nil)
        {
          last_error_ = "Metal command-buffer allocation failed";
          zero_secret_buffers(point_bytes, projective_bytes, inverse_bytes);
          return MONERO_FAST_METAL_EXECUTION_ERROR;
        }
        command.label = @"Monero wallet key derivation batch";

        id<MTLComputeCommandEncoder> projective_encoder =
          [command computeCommandEncoder];
        if (projective_encoder == nil)
        {
          last_error_ = "Metal projective encoder allocation failed";
          zero_secret_buffers(point_bytes, projective_bytes, inverse_bytes);
          return MONERO_FAST_METAL_EXECUTION_ERROR;
        }
        [projective_encoder setComputePipelineState:projective_pipeline_];
        [projective_encoder setBuffer:scalar_buffer_ offset:0 atIndex:0];
        [projective_encoder setBuffer:point_buffer_ offset:0 atIndex:1];
        [projective_encoder setBuffer:projective_buffer_ offset:0 atIndex:2];
        [projective_encoder setBuffer:valid_buffer_ offset:0 atIndex:3];
        [projective_encoder setBuffer:parameter_buffer_ offset:0 atIndex:4];
        [projective_encoder
          dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(
            std::min(PROJECTIVE_THREADS, projective_pipeline_.maxTotalThreadsPerThreadgroup),
            1,
            1)];
        [projective_encoder endEncoding];

        id<MTLComputeCommandEncoder> inverse_encoder =
          [command computeCommandEncoder];
        if (inverse_encoder == nil)
        {
          last_error_ = "Metal inverse encoder allocation failed";
          zero_secret_buffers(point_bytes, projective_bytes, inverse_bytes);
          return MONERO_FAST_METAL_EXECUTION_ERROR;
        }
        const NSUInteger inverse_count =
          (count + BATCH_INVERSION_CHUNK - 1) / BATCH_INVERSION_CHUNK;
        [inverse_encoder setComputePipelineState:inverse_pipeline_];
        [inverse_encoder setBuffer:projective_buffer_ offset:0 atIndex:0];
        [inverse_encoder setBuffer:inverse_buffer_ offset:0 atIndex:1];
        [inverse_encoder setBuffer:valid_buffer_ offset:0 atIndex:2];
        [inverse_encoder setBuffer:parameter_buffer_ offset:0 atIndex:3];
        [inverse_encoder
          dispatchThreads:MTLSizeMake(inverse_count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(
            std::min({
              INVERSE_THREADS,
              inverse_count,
              inverse_pipeline_.maxTotalThreadsPerThreadgroup
            }),
            1,
            1)];
        [inverse_encoder endEncoding];

        id<MTLComputeCommandEncoder> compress_encoder =
          [command computeCommandEncoder];
        if (compress_encoder == nil)
        {
          last_error_ = "Metal compression encoder allocation failed";
          zero_secret_buffers(point_bytes, projective_bytes, inverse_bytes);
          return MONERO_FAST_METAL_EXECUTION_ERROR;
        }
        [compress_encoder setComputePipelineState:compress_pipeline_];
        [compress_encoder setBuffer:projective_buffer_ offset:0 atIndex:0];
        [compress_encoder setBuffer:inverse_buffer_ offset:0 atIndex:1];
        [compress_encoder setBuffer:result_buffer_ offset:0 atIndex:2];
        [compress_encoder setBuffer:valid_buffer_ offset:0 atIndex:3];
        [compress_encoder setBuffer:parameter_buffer_ offset:0 atIndex:4];
        [compress_encoder
          dispatchThreads:MTLSizeMake(count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(
            std::min(COMPRESS_THREADS, compress_pipeline_.maxTotalThreadsPerThreadgroup),
            1,
            1)];
        [compress_encoder endEncoding];

        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted || command.error != nil)
        {
          last_error_ = describe_error(@"Metal key-derivation command failed", command.error);
          zero_secret_buffers(point_bytes, projective_bytes, inverse_bytes);
          return MONERO_FAST_METAL_EXECUTION_ERROR;
        }

        std::memcpy(results, result_buffer_.contents, point_bytes);
        std::memcpy(valid, valid_buffer_.contents, count);
        int64_t successes = 0;
        for (size_t index = 0; index < count; ++index)
          successes += valid[index] == 1 ? 1 : 0;

        zero_secret_buffers(point_bytes, projective_bytes, inverse_bytes);
        return successes;
      }
    }

    const char *last_error()
    {
      std::lock_guard<std::mutex> lock(mutex_);
      return last_error_.c_str();
    }

  private:
    bool initialize_locked()
    {
      if (initialized_)
        return ready_;
      initialized_ = true;

      device_ = MTLCreateSystemDefaultDevice();
      if (device_ == nil)
      {
        last_error_ = "No default Metal device";
        return false;
      }
      queue_ = [device_ newCommandQueue];
      if (queue_ == nil)
      {
        last_error_ = "Metal command-queue creation failed";
        return false;
      }

      NSFileManager *files = NSFileManager.defaultManager;
      NSString *library_path = nil;
      for (NSString *candidate : library_candidates())
      {
        if ([files isReadableFileAtPath:candidate])
        {
          library_path = candidate;
          break;
        }
      }
      if (library_path == nil)
      {
        last_error_ = "Packaged monero_wallet_derivation.metallib not found";
        return false;
      }

      NSError *error = nil;
      id<MTLLibrary> library =
        [device_ newLibraryWithURL:[NSURL fileURLWithPath:library_path] error:&error];
      if (library == nil)
      {
        last_error_ = describe_error(@"Cannot load wallet metallib", error);
        return false;
      }

      projective_pipeline_ = make_pipeline(
        library, @"derivation_m12_projective_chunkinvert");
      inverse_pipeline_ = make_pipeline(
        library, @"derivation_m12_chunk_inverse");
      compress_pipeline_ = make_pipeline(
        library, @"derivation_m12_compress_chunkinvert");
      ready_ = projective_pipeline_ != nil
        && inverse_pipeline_ != nil
        && compress_pipeline_ != nil;
      return ready_;
    }

    id<MTLComputePipelineState> make_pipeline(
        id<MTLLibrary> library,
        NSString *name)
    {
      id<MTLFunction> function = [library newFunctionWithName:name];
      if (function == nil)
      {
        last_error_ = std::string("Metal function missing: ") + name.UTF8String;
        return nil;
      }
      NSError *error = nil;
      id<MTLComputePipelineState> pipeline =
        [device_ newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil)
        last_error_ = describe_error(
          [NSString stringWithFormat:@"Cannot create %@ pipeline", name], error);
      return pipeline;
    }

    bool ensure_buffers_locked(size_t count)
    {
      if (buffer_capacity_ >= count)
        return true;

      const size_t point_bytes = count * 32;
      const size_t projective_bytes = count * 30 * sizeof(uint32_t);
      const size_t inverse_bytes = count * 10 * sizeof(uint32_t);
      scalar_buffer_ =
        [device_ newBufferWithLength:32 options:MTLResourceStorageModeShared];
      point_buffer_ =
        [device_ newBufferWithLength:point_bytes options:MTLResourceStorageModeShared];
      result_buffer_ =
        [device_ newBufferWithLength:point_bytes options:MTLResourceStorageModeShared];
      valid_buffer_ =
        [device_ newBufferWithLength:count options:MTLResourceStorageModeShared];
      projective_buffer_ =
        [device_ newBufferWithLength:projective_bytes options:MTLResourceStorageModeShared];
      inverse_buffer_ =
        [device_ newBufferWithLength:inverse_bytes options:MTLResourceStorageModeShared];
      parameter_buffer_ =
        [device_ newBufferWithLength:sizeof(Parameters)
                             options:MTLResourceStorageModeShared];

      if (scalar_buffer_ == nil || point_buffer_ == nil || result_buffer_ == nil
          || valid_buffer_ == nil || projective_buffer_ == nil
          || inverse_buffer_ == nil || parameter_buffer_ == nil)
        return false;

      scalar_buffer_.label = @"wallet_view_scalar";
      point_buffer_.label = @"wallet_transaction_public_keys";
      result_buffer_.label = @"wallet_key_derivations";
      valid_buffer_.label = @"wallet_key_derivation_validity";
      projective_buffer_.label = @"wallet_key_derivation_projective_xyz";
      inverse_buffer_.label = @"wallet_key_derivation_inverse_z";
      parameter_buffer_.label = @"wallet_key_derivation_parameters";
      buffer_capacity_ = count;
      return true;
    }

    static void zero_buffer(id<MTLBuffer> buffer, size_t bytes)
    {
      if (buffer != nil && buffer.contents != nullptr)
        secure_zero(buffer.contents, (std::min)(bytes, static_cast<size_t>(buffer.length)));
    }

    void zero_secret_buffers(
        size_t result_bytes,
        size_t projective_bytes,
        size_t inverse_bytes)
    {
      zero_buffer(scalar_buffer_, 32);
      zero_buffer(result_buffer_, result_bytes);
      zero_buffer(projective_buffer_, projective_bytes);
      zero_buffer(inverse_buffer_, inverse_bytes);
    }

    std::mutex mutex_;
    bool initialized_ = false;
    bool ready_ = false;
    std::string last_error_;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLComputePipelineState> projective_pipeline_ = nil;
    id<MTLComputePipelineState> inverse_pipeline_ = nil;
    id<MTLComputePipelineState> compress_pipeline_ = nil;
    size_t buffer_capacity_ = 0;
    id<MTLBuffer> scalar_buffer_ = nil;
    id<MTLBuffer> point_buffer_ = nil;
    id<MTLBuffer> result_buffer_ = nil;
    id<MTLBuffer> valid_buffer_ = nil;
    id<MTLBuffer> projective_buffer_ = nil;
    id<MTLBuffer> inverse_buffer_ = nil;
    id<MTLBuffer> parameter_buffer_ = nil;
  };

  MetalDerivationContext &context()
  {
    static MetalDerivationContext instance;
    return instance;
  }
}

extern "C" int fast_metal_derivation_available(void)
{
  @autoreleasepool
  {
    return context().available() ? 1 : 0;
  }
}

extern "C" int64_t fast_metal_generate_key_derivation_batch_same_scalar(
    uint8_t *results,
    const uint8_t *scalar,
    const uint8_t *points,
    uint8_t *valid,
    size_t count)
{
  return context().derive(results, scalar, points, valid, count);
}

extern "C" const char *fast_metal_derivation_last_error(void)
{
  return context().last_error();
}
