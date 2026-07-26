#!/usr/bin/env bash
# Build an arm64 Android Vulkan capability probe and optionally run it via ADB.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${tool_dir}/../../.." && pwd)"
results_root="${WALLET_ANDROID_ACCEL_RESULTS_DIR:-${source_root}/build/wallet-mobile-acceleration-testbench}"
run_id="${WALLET_ANDROID_ACCEL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-android-vulkan-probe}"
result_dir="${results_root}/${run_id}"
build_dir="${result_dir}/build"
build_only=0

usage() {
  echo "Usage: $0 [--build-only]"
}

while (($# > 0)); do
  case "$1" in
    --build-only) build_only=1 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

[[ "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]] || {
  echo "WALLET_ANDROID_ACCEL_RUN_ID may contain only letters, numbers, dot, underscore and dash." >&2
  exit 2
}
[[ ! -e "${result_dir}" ]] || {
  echo "result directory already exists: ${result_dir}" >&2
  exit 2
}

android_sdk_root="${ANDROID_SDK_ROOT:-/Users/rolandkohlhuber/Library/Android/sdk}"
probe_ndk_root="${ANDROID_NDK_ROOT:-${ANDROID_NDK_HOME:-}}"
if [[ -z "${probe_ndk_root}" ]]; then
  for candidate in "${android_sdk_root}"/ndk/*; do
    [[ -d "${candidate}" ]] && probe_ndk_root="${candidate}"
  done
fi
[[ -f "${probe_ndk_root}/build/cmake/android.toolchain.cmake" ]] || {
  echo "Android NDK CMake toolchain not found. Set ANDROID_NDK_ROOT." >&2
  exit 2
}
command -v cmake >/dev/null || { echo "cmake is required." >&2; exit 2; }
command -v ninja >/dev/null || { echo "ninja is required." >&2; exit 2; }

mkdir -p "${build_dir}"
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

{
  echo "schema=wallet_android_vulkan_probe_run_v1"
  echo "run_id=${run_id}"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "host_uname=$(uname -a)"
  echo "android_ndk=${probe_ndk_root}"
  echo "cmake=$(cmake --version | head -n 1)"
  echo "probe_source_sha256=$(sha256_file "${tool_dir}/vulkan_probe.cpp")"
  echo "cmake_source_sha256=$(sha256_file "${tool_dir}/CMakeLists.txt")"
  echo "build_only=${build_only}"
  echo "command=$0 $*"
} >"${result_dir}/metadata.env"

cmake -S "${tool_dir}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${probe_ndk_root}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  >"${result_dir}/configure.log" 2>&1
cmake --build "${build_dir}" --parallel >"${result_dir}/build.log" 2>&1

binary="${build_dir}/wallet-vulkan-probe"
[[ -x "${binary}" ]] || { echo "probe binary was not produced." >&2; exit 2; }
{
  echo "binary_sha256=$(sha256_file "${binary}")"
  echo "binary_file=$(file "${binary}")"
} >>"${result_dir}/metadata.env"

if ((build_only == 1)); then
  echo "status=build_only_pass" | tee "${result_dir}/result.log"
  echo "finished_utc=$(date -u +%FT%TZ)" >>"${result_dir}/metadata.env"
  echo "result_dir=${result_dir}"
  exit 0
fi

adb_binary="${ADB:-${android_sdk_root}/platform-tools/adb}"
[[ -x "${adb_binary}" ]] || { echo "adb not found: ${adb_binary}" >&2; exit 2; }

probe_serial="${ANDROID_SERIAL:-}"
if [[ -z "${probe_serial}" ]]; then
  connected_serials=($("${adb_binary}" devices | awk '$2 == "device" {print $1}'))
  ((${#connected_serials[@]} == 1)) || {
    echo "exactly one authorized Android device is required; found ${#connected_serials[@]}." >&2
    exit 2
  }
  probe_serial="${connected_serials[0]}"
fi
adb_device=("${adb_binary}" -s "${probe_serial}")
"${adb_device[@]}" get-state | grep -qx device || {
  echo "Android device is not authorized and online: ${probe_serial}" >&2
  exit 2
}

remote_binary="/data/local/tmp/wallet-vulkan-probe-${run_id}"
cleanup_remote() {
  case "${remote_binary}" in
    /data/local/tmp/wallet-vulkan-probe-*)
      "${adb_device[@]}" shell rm -f "${remote_binary}" >/dev/null 2>&1 || true
      ;;
  esac
}
trap cleanup_remote EXIT

{
  echo "android_serial=${probe_serial}"
  echo "android_model=$("${adb_device[@]}" shell getprop ro.product.model | tr -d '\r')"
  echo "android_device=$("${adb_device[@]}" shell getprop ro.product.device | tr -d '\r')"
  echo "android_soc_model=$("${adb_device[@]}" shell getprop ro.soc.model | tr -d '\r')"
  echo "android_release=$("${adb_device[@]}" shell getprop ro.build.version.release | tr -d '\r')"
  echo "android_sdk=$("${adb_device[@]}" shell getprop ro.build.version.sdk | tr -d '\r')"
  echo "android_abi=$("${adb_device[@]}" shell getprop ro.product.cpu.abi | tr -d '\r')"
  echo "device_cpu_count=$("${adb_device[@]}" shell getconf _NPROCESSORS_ONLN | tr -d '\r')"
} >>"${result_dir}/metadata.env"

"${adb_device[@]}" push "${binary}" "${remote_binary}" >"${result_dir}/adb-push.log"
"${adb_device[@]}" shell chmod 700 "${remote_binary}"
"${adb_device[@]}" shell "${remote_binary}" | tr -d '\r' >"${result_dir}/result.log"
"${adb_device[@]}" shell dumpsys thermalservice >"${result_dir}/thermal-after.txt" 2>&1 || true

grep -qx "compute_ready=yes" "${result_dir}/result.log" || {
  echo "Vulkan probe did not report a usable compute queue." >&2
  cat "${result_dir}/result.log"
  exit 3
}
echo "finished_utc=$(date -u +%FT%TZ)" >>"${result_dir}/metadata.env"
echo "result_dir=${result_dir}"
cat "${result_dir}/result.log"
