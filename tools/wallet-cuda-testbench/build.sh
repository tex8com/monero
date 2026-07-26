#!/usr/bin/env bash
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cuda_arch="${CUDA_ARCH:-sm_86}"
output="${1:-${tool_dir}/wallet-cuda-testbench}"
nvcc_binary="${NVCC:-nvcc}"

"${nvcc_binary}" \
  -O3 \
  -std=c++17 \
  -arch="${cuda_arch}" \
  -lineinfo \
  -Xptxas=-v \
  "${tool_dir}/main.cu" \
  -o "${output}"
