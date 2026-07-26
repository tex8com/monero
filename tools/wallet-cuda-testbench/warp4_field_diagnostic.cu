#include "derivation_radix2625.cuh"
#include "derivation_warp4.cuh"
#include "vector_corpus.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using monero_cuda2625::u8;
using monero_cuda2625::u32;

constexpr u32 kStages = 11u;
constexpr u32 kLimbs = 10u;

void cuda_check(cudaError_t status, const char *expression) {
  if (status == cudaSuccess) return;
  throw std::runtime_error(
      std::string(expression) + ": " + cudaGetErrorString(status));
}

#define CUDA_CHECK(expression) cuda_check((expression), #expression)

__global__ void scalar_field_diagnostic(
    const u8 *encoded, u32 *stages, u32 *accepted) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  monero_cuda2625::Fe y, y_squared, y_multiplied, curve_d, y_times_d;
  monero_cuda2625::Fe denominator;
  monero_cuda2625::Fe one, numerator, numerator_squared, product;
  monero_cuda2625::fe_from_bytes(y, encoded);
  monero_cuda2625::fe_square(y_squared, y);
  monero_cuda2625::fe_mul(y_multiplied, y, y);
  monero_cuda2625::fe_from_field_d(curve_d);
  monero_cuda2625::fe_mul(y_times_d, y, curve_d);
  monero_cuda2625::fe_mul(denominator, y_squared, curve_d);
  monero_cuda2625::fe_one(one);
  monero_cuda2625::fe_add(denominator, denominator, one);
  monero_cuda2625::fe_sub(numerator, y_squared, one);
  monero_cuda2625::fe_square(numerator_squared, numerator);
  monero_cuda2625::fe_mul(product, numerator, denominator);
  const monero_cuda2625::Fe values[kStages] = {
      y, y_squared, y_multiplied, curve_d, y_times_d, denominator,
      one, numerator, numerator_squared, product, y_squared};
  for (u32 stage = 0; stage < kStages; ++stage)
    monero_cuda2625::fe_store_device(
        stages + stage * kLimbs, values[stage]);
  *accepted = 1u;
}

__global__ void warp4_field_diagnostic(
    const u8 *encoded, u32 *stages, u32 *accepted) {
  if (blockIdx.x != 0 || threadIdx.x >= 4) return;
  monero_cuda_warp4::Fe y, y_squared, y_multiplied, curve_d, y_times_d;
  monero_cuda_warp4::Fe denominator;
  monero_cuda_warp4::Fe one, numerator, numerator_squared, product;
  monero_cuda_warp4::fe_from_bytes(y, encoded);
  monero_cuda_warp4::fe_square(y_squared, y);
  monero_cuda_warp4::fe_mul(y_multiplied, y, y);
  monero_cuda_warp4::fe_from_field_d(curve_d);
  monero_cuda_warp4::fe_mul(y_times_d, y, curve_d);
  monero_cuda_warp4::fe_mul(denominator, y_squared, curve_d);
  monero_cuda_warp4::fe_one(one);
  monero_cuda_warp4::fe_add(denominator, denominator, one);
  monero_cuda_warp4::fe_sub(numerator, y_squared, one);
  monero_cuda_warp4::fe_square(numerator_squared, numerator);
  monero_cuda_warp4::fe_mul(product, numerator, denominator);
  const monero_cuda_warp4::Fe values[kStages] = {
      y, y_squared, y_multiplied, curve_d, y_times_d, denominator,
      one, numerator, numerator_squared, product, y_squared};
  for (u32 stage = 0; stage < kStages; ++stage)
    monero_cuda_warp4::fe_store_device(
        stages + stage * kLimbs, values[stage]);
  if (threadIdx.x == 0) *accepted = 1u;
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2)
      throw std::runtime_error(
          "usage: warp4-field-diagnostic VECTORS.mwmtv1");
    const auto corpus =
        monero_cuda_testbench::read_vector_corpus(argv[1]);
    std::vector<u8> encoded(corpus.points.begin(),
                            corpus.points.begin() + 32);

    u8 *device_encoded = nullptr;
    u32 *scalar_stages = nullptr;
    u32 *warp_stages = nullptr;
    u32 *scalar_accepted = nullptr;
    u32 *warp_accepted = nullptr;
    CUDA_CHECK(cudaMalloc(&device_encoded, 32));
    CUDA_CHECK(cudaMalloc(&scalar_stages,
                          kStages * kLimbs * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&warp_stages,
                          kStages * kLimbs * sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&scalar_accepted, sizeof(u32)));
    CUDA_CHECK(cudaMalloc(&warp_accepted, sizeof(u32)));
    CUDA_CHECK(cudaMemcpy(device_encoded, encoded.data(), 32,
                          cudaMemcpyHostToDevice));

    scalar_field_diagnostic<<<1, 1>>>(
        device_encoded, scalar_stages, scalar_accepted);
    warp4_field_diagnostic<<<1, 4>>>(
        device_encoded, warp_stages, warp_accepted);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<u32> scalar(kStages * kLimbs);
    std::vector<u32> warp(kStages * kLimbs);
    u32 scalar_valid = 0;
    u32 warp_valid = 0;
    CUDA_CHECK(cudaMemcpy(scalar.data(), scalar_stages,
                          scalar.size() * sizeof(u32),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(warp.data(), warp_stages,
                          warp.size() * sizeof(u32),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&scalar_valid, scalar_accepted, sizeof(u32),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&warp_valid, warp_accepted, sizeof(u32),
                          cudaMemcpyDeviceToHost));

    bool pass = true;
    for (u32 stage = 0; stage < kStages; ++stage) {
      bool equal = true;
      for (u32 limb = 0; limb < kLimbs; ++limb)
        equal &= scalar[stage * kLimbs + limb] ==
                 warp[stage * kLimbs + limb];
      std::cout << "stage_" << stage << '='
                << (equal ? "pass" : "FAIL") << '\n';
      if (!equal) {
        pass = false;
        for (u32 limb = 0; limb < kLimbs; ++limb)
          std::cout << "  limb=" << limb
                    << " scalar=0x" << std::hex
                    << scalar[stage * kLimbs + limb]
                    << " warp=0x"
                    << warp[stage * kLimbs + limb]
                    << std::dec << '\n';
      }
    }
    std::cout << "scalar_decode=" << scalar_valid << '\n';
    std::cout << "warp4_decode=" << warp_valid << '\n';

    cudaFree(warp_accepted);
    cudaFree(scalar_accepted);
    cudaFree(warp_stages);
    cudaFree(scalar_stages);
    cudaFree(device_encoded);
    return pass && scalar_valid == warp_valid ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "warp4 diagnostic error: " << error.what() << '\n';
    return 2;
  }
}
