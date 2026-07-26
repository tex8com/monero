#!/usr/bin/env bash
# Mac/Metal stage M1. It times only byte-checked D = 8 * a * R GPU kernels.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${tool_dir}/../.." && pwd)"
crypto_tool_dir="${source_root}/tools/wallet-crypto-testbench"
results_root="${WALLET_METAL_BENCH_RESULTS_DIR:-${source_root}/build/wallet-metal-testbench}"
run_id="${WALLET_METAL_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)}"
point_count="${WALLET_METAL_M1_POINTS:-64}"
result_dir="${results_root}/${run_id}"
kernel_source="${WALLET_METAL_KERNEL_SOURCE:-${tool_dir}/derivation.metal}"

[[ "$(uname -s)" == "Darwin" ]] || { echo "Metal M1 requires macOS." >&2; exit 2; }
[[ "${point_count}" =~ ^[1-9][0-9]*$ ]] || { echo "WALLET_METAL_M1_POINTS must be a positive integer." >&2; exit 2; }
[[ -f "${kernel_source}" ]] || { echo "Metal kernel source missing: ${kernel_source}" >&2; exit 2; }
[[ ! -e "${result_dir}" ]] || { echo "result directory already exists: ${result_dir}" >&2; exit 2; }
mkdir -p "${result_dir}"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

{
  echo "schema=wallet_metal_derivation_benchmark_v1"
  echo "run_id=${run_id}"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname)"
  echo "uname=$(uname -a)"
  echo "macos=$(sw_vers | tr '\n' ';')"
  echo "metal_device=$(system_profiler SPDisplaysDataType | tr '\n' ';')"
  echo "swiftc=$(xcrun swiftc --version | tr '\n' ';')"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "curve25519_dalek_version=4.1.3"
  echo "vector_generator_main_sha256=$(sha256_file "${crypto_tool_dir}/src/main.rs")"
  echo "wallet_fast_crypto_sha256=$(sha256_file "${source_root}/external/monero-fast-crypto/src/lib.rs")"
  echo "vector_generator_lock_sha256=$(sha256_file "${crypto_tool_dir}/Cargo.lock")"
  echo "metal_host_main_sha256=$(sha256_file "${tool_dir}/main.swift")"
  echo "metal_kernel_source=${kernel_source}"
  echo "metal_kernel_sha256=$(sha256_file "${kernel_source}")"
  echo "points_per_round=${point_count}"
  echo "command=$0 $*"
} >"${result_dir}/metadata.env"

(cd "${crypto_tool_dir}" && cargo build --release --locked) >"${result_dir}/vector-build.log" 2>&1
"${crypto_tool_dir}/target/release/monero-wallet-crypto-testbench" \
  --points "${point_count}" \
  --export-metal-vectors "${result_dir}/vectors.mwmtv1" \
  >"${result_dir}/vector-export.log" 2>&1

binary="${result_dir}/wallet-metal-derivation-testbench"
xcrun swiftc -O -framework Metal "${tool_dir}/main.swift" -o "${binary}" >"${result_dir}/build.log" 2>&1
/usr/bin/time -l "${binary}" \
  --vectors "${result_dir}/vectors.mwmtv1" \
  --kernel-source "${kernel_source}" \
  "$@" \
  >"${result_dir}/result.log" 2>"${result_dir}/time.log"

{
  echo "finished_utc=$(date -u +%FT%TZ)"
  echo "vector_sha256=$(sha256_file "${result_dir}/vectors.mwmtv1")"
  echo "binary_sha256=$(sha256_file "${binary}")"
} >>"${result_dir}/metadata.env"

echo "result_dir=${result_dir}"
cat "${result_dir}/result.log"
