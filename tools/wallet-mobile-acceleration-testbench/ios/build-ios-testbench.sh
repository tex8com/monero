#!/usr/bin/env bash
# Compile the isolated Metal runner for real-iPhone and iPhone-Simulator SDKs.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${tool_dir}/../../.." && pwd)"
results_root="${WALLET_IOS_ACCEL_RESULTS_DIR:-${source_root}/build/wallet-mobile-acceleration-testbench}"
run_id="${WALLET_IOS_ACCEL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-ios-metal-build}"
result_dir="${results_root}/${run_id}"
source_file="${tool_dir}/WalletMetalMobileTestbench.swift"

[[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "WALLET_IOS_ACCEL_RUN_ID may contain only letters, numbers, dot, underscore and dash." >&2
  exit 2
}
[[ ! -e "${result_dir}" ]] || {
  echo "result directory already exists: ${result_dir}" >&2
  exit 2
}
[[ -f "${source_file}" ]] || { echo "missing Swift runner: ${source_file}" >&2; exit 2; }
command -v xcrun >/dev/null || { echo "Xcode command-line tools are required." >&2; exit 2; }

mkdir -p "${result_dir}/iphoneos" "${result_dir}/iphonesimulator"
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
iphoneos_sdk="$(xcrun --sdk iphoneos --show-sdk-path)"
simulator_sdk="$(xcrun --sdk iphonesimulator --show-sdk-path)"

{
  echo "schema=wallet_ios_metal_testbench_build_v1"
  echo "run_id=${run_id}"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "host_uname=$(uname -a)"
  echo "xcode=$(xcodebuild -version | tr '\n' ';')"
  echo "swift=$(xcrun swiftc --version | tr '\n' ';')"
  echo "iphoneos_sdk=${iphoneos_sdk}"
  echo "iphonesimulator_sdk=${simulator_sdk}"
  echo "deployment_target=17.0"
  echo "runner_sha256=$(sha256_file "${source_file}")"
  echo "metal_kernel_sha256=$(sha256_file "${source_root}/tools/wallet-metal-testbench/derivation_radix2625_chunkinvert.metal")"
} >"${result_dir}/metadata.env"

xcrun --sdk iphoneos swiftc \
  -parse-as-library \
  -target arm64-apple-ios17.0 \
  -sdk "${iphoneos_sdk}" \
  -module-name WalletMetalMobileTestbench \
  -emit-module \
  -emit-module-path "${result_dir}/iphoneos/WalletMetalMobileTestbench.swiftmodule" \
  "${source_file}" >"${result_dir}/iphoneos-build.log" 2>&1

xcrun --sdk iphonesimulator swiftc \
  -parse-as-library \
  -target arm64-apple-ios17.0-simulator \
  -sdk "${simulator_sdk}" \
  -module-name WalletMetalMobileTestbench \
  -emit-module \
  -emit-module-path "${result_dir}/iphonesimulator/WalletMetalMobileTestbench.swiftmodule" \
  "${source_file}" >"${result_dir}/iphonesimulator-build.log" 2>&1

{
  echo "iphoneos_module_sha256=$(sha256_file "${result_dir}/iphoneos/WalletMetalMobileTestbench.swiftmodule")"
  echo "iphonesimulator_module_sha256=$(sha256_file "${result_dir}/iphonesimulator/WalletMetalMobileTestbench.swiftmodule")"
  echo "physical_iphone_execution=not_run_no_device"
  echo "simulator_performance_status=not_valid_for_a_series_measurement"
  echo "finished_utc=$(date -u +%FT%TZ)"
} >>"${result_dir}/metadata.env"

{
  echo "iphoneos_compile=pass"
  echo "iphonesimulator_compile=pass"
  echo "physical_iphone_validation=pending"
} | tee "${result_dir}/result.log"
echo "result_dir=${result_dir}"
