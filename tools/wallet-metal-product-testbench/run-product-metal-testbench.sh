#!/usr/bin/env bash
# Build and measure the exact native product host against public Dalek vectors.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${tool_dir}/../.." && pwd)"
vector_tool_dir="${source_root}/tools/wallet-crypto-testbench"
metal_source="${source_root}/tools/wallet-metal-testbench/derivation_radix2625_chunkinvert.metal"
metal_host="${source_root}/external/monero-fast-metal/src/monero_fast_metal.mm"
metal_include="${source_root}/external/monero-fast-metal/include"
results_root="${WALLET_METAL_PRODUCT_RESULTS_DIR:-${source_root}/build/wallet-metal-product-testbench}"
run_id="${WALLET_METAL_PRODUCT_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)}"
point_count="${WALLET_METAL_PRODUCT_POINTS:-8192}"
result_dir="${results_root}/${run_id}"

[[ "$(uname -s)" == "Darwin" ]] || { echo "The Metal product test requires macOS." >&2; exit 2; }
[[ "${point_count}" =~ ^[1-9][0-9]*$ ]] || { echo "WALLET_METAL_PRODUCT_POINTS must be positive." >&2; exit 2; }
[[ ! -e "${result_dir}" ]] || { echo "Result directory exists: ${result_dir}" >&2; exit 2; }
mkdir -p "${result_dir}"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

{
  echo "schema=wallet_metal_product_benchmark_v1"
  echo "run_id=${run_id}"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname)"
  echo "uname=$(uname -a)"
  echo "macos=$(sw_vers | tr '\n' ';')"
  echo "hardware=$(system_profiler SPHardwareDataType | tr '\n' ';')"
  echo "display=$(system_profiler SPDisplaysDataType | tr '\n' ';')"
  echo "xcode=$(xcodebuild -version | tr '\n' ';')"
  echo "metal_compiler=$(xcrun -sdk macosx metal -v 2>&1 | tr '\n' ';')"
  echo "kernel_sha256=$(sha256_file "${metal_source}")"
  echo "host_sha256=$(sha256_file "${metal_host}")"
  echo "testbench_sha256=$(sha256_file "${tool_dir}/main.cpp")"
  echo "points_per_round=${point_count}"
  echo "command=$0 $*"
} > "${result_dir}/metadata.env"

if [[ -n "${WALLET_METAL_PRODUCT_VECTOR_FILE:-}" ]]; then
  [[ -f "${WALLET_METAL_PRODUCT_VECTOR_FILE}" ]] || {
    echo "Vector file not found: ${WALLET_METAL_PRODUCT_VECTOR_FILE}" >&2
    exit 66
  }
  cp "${WALLET_METAL_PRODUCT_VECTOR_FILE}" "${result_dir}/vectors.mwmtv1"
  echo "source=${WALLET_METAL_PRODUCT_VECTOR_FILE}" > "${result_dir}/vector-export.log"
else
  (cd "${vector_tool_dir}" && cargo build --release --locked) \
    > "${result_dir}/vector-build.log" 2>&1
  "${vector_tool_dir}/target/release/monero-wallet-crypto-testbench" \
    --points "${point_count}" \
    --export-metal-vectors "${result_dir}/vectors.mwmtv1" \
    > "${result_dir}/vector-export.log" 2>&1
fi

xcrun -sdk macosx metal -c -mmacosx-version-min=12.0 \
  "${metal_source}" -o "${result_dir}/monero_wallet_derivation.air" \
  > "${result_dir}/metal-compile.log" 2>&1
xcrun -sdk macosx metallib \
  "${result_dir}/monero_wallet_derivation.air" \
  -o "${result_dir}/monero_wallet_derivation.metallib" \
  > "${result_dir}/metallib-link.log" 2>&1

xcrun clang++ -O3 -std=c++17 -mmacosx-version-min=12.0 \
  -I"${metal_include}" \
  -c "${tool_dir}/main.cpp" -o "${result_dir}/main.o" \
  > "${result_dir}/host-build.log" 2>&1
xcrun clang++ -O3 -std=c++17 -fobjc-arc -mmacosx-version-min=12.0 \
  -I"${metal_include}" \
  -c "${metal_host}" -o "${result_dir}/monero_fast_metal.o" \
  >> "${result_dir}/host-build.log" 2>&1
xcrun clang++ \
  "${result_dir}/main.o" "${result_dir}/monero_fast_metal.o" \
  -framework Foundation -framework Metal \
  -o "${result_dir}/wallet-metal-product-testbench" \
  >> "${result_dir}/host-build.log" 2>&1

export MONERO_METAL_LIBRARY_PATH="${result_dir}/monero_wallet_derivation.metallib"
/usr/bin/time -l "${result_dir}/wallet-metal-product-testbench" \
  --vectors "${result_dir}/vectors.mwmtv1" "$@" \
  > "${result_dir}/result.log" 2> "${result_dir}/time.log"

{
  echo "finished_utc=$(date -u +%FT%TZ)"
  echo "vectors_sha256=$(sha256_file "${result_dir}/vectors.mwmtv1")"
  echo "metallib_sha256=$(sha256_file "${result_dir}/monero_wallet_derivation.metallib")"
  echo "binary_sha256=$(sha256_file "${result_dir}/wallet-metal-product-testbench")"
} >> "${result_dir}/metadata.env"

echo "result_dir=${result_dir}"
cat "${result_dir}/result.log"
