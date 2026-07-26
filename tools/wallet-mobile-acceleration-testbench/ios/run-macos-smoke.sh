#!/usr/bin/env bash
# Run the mobile Metal runner on public vectors using the Mac GPU.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${tool_dir}/../../.." && pwd)"
rust_tool_dir="${source_root}/tools/wallet-crypto-testbench"
results_root="${WALLET_IOS_ACCEL_RESULTS_DIR:-${source_root}/build/wallet-mobile-acceleration-testbench}"
run_id="${WALLET_IOS_ACCEL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-ios-runner-mac-smoke}"
result_dir="${results_root}/${run_id}"
point_count="${WALLET_IOS_ACCEL_SMOKE_POINTS:-8}"
kernel_source="${source_root}/tools/wallet-metal-testbench/derivation_radix2625_chunkinvert.metal"

[[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "WALLET_IOS_ACCEL_RUN_ID contains unsupported characters." >&2
  exit 2
}
[[ "${point_count}" =~ ^[1-9][0-9]*$ ]] || {
  echo "WALLET_IOS_ACCEL_SMOKE_POINTS must be a positive integer." >&2
  exit 2
}
[[ ! -e "${result_dir}" ]] || {
  echo "result directory already exists: ${result_dir}" >&2
  exit 2
}
mkdir -p "${result_dir}"
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

(cd "${rust_tool_dir}" && cargo build --release --locked) \
  >"${result_dir}/vector-build.log" 2>&1
"${rust_tool_dir}/target/release/monero-wallet-crypto-testbench" \
  --points "${point_count}" \
  --export-metal-vectors "${result_dir}/vectors.mwmtv1" \
  >"${result_dir}/vector-export.log" 2>&1

binary="${result_dir}/wallet-metal-mobile-runner-mac-smoke"
xcrun swiftc -O -framework Metal \
  "${tool_dir}/WalletMetalMobileTestbench.swift" \
  "${tool_dir}/MobileRunnerMacSmoke.swift" \
  -o "${binary}" >"${result_dir}/build.log" 2>&1

"${binary}" "${result_dir}/vectors.mwmtv1" "${kernel_source}" \
  >"${result_dir}/result.log" 2>"${result_dir}/error.log"
grep -qx "validation=pass" "${result_dir}/result.log"

{
  echo "schema=wallet_ios_metal_runner_mac_smoke_v1"
  echo "run_id=${run_id}"
  echo "started_and_finished_utc=$(date -u +%FT%TZ)"
  echo "host_uname=$(uname -a)"
  echo "points_per_round=${point_count}"
  echo "runner_sha256=$(sha256_file "${tool_dir}/WalletMetalMobileTestbench.swift")"
  echo "smoke_host_sha256=$(sha256_file "${tool_dir}/MobileRunnerMacSmoke.swift")"
  echo "kernel_sha256=$(sha256_file "${kernel_source}")"
  echo "vector_sha256=$(sha256_file "${result_dir}/vectors.mwmtv1")"
  echo "binary_sha256=$(sha256_file "${binary}")"
  echo "physical_iphone_validation=not_run_no_device"
} >"${result_dir}/metadata.env"

echo "result_dir=${result_dir}"
cat "${result_dir}/result.log"
