#!/usr/bin/env bash
# Build and optionally run workload-identical C/Ref10 and Rust Android baselines.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${tool_dir}/../../.." && pwd)"
rust_tool_dir="${source_root}/tools/wallet-crypto-testbench"
ref10_tool_dir="${source_root}/tools/wallet-original-crypto-testbench"
results_root="${WALLET_ANDROID_ACCEL_RESULTS_DIR:-${source_root}/build/wallet-mobile-acceleration-testbench}"
run_id="${WALLET_ANDROID_ACCEL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-android-cpu-reference}"
result_dir="${results_root}/${run_id}"
point_count="${WALLET_ANDROID_CPU_POINTS:-8192}"
rounds="${WALLET_ANDROID_CPU_ROUNDS:-20}"
warmup_rounds="${WALLET_ANDROID_CPU_WARMUP_ROUNDS:-3}"
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
for value in "${point_count}" "${rounds}" "${warmup_rounds}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "all point and round settings must be positive integers." >&2
    exit 2
  }
done
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
android_toolchain="${probe_ndk_root}/toolchains/llvm/prebuilt/darwin-x86_64/bin"
android_cc="${android_toolchain}/aarch64-linux-android24-clang"
[[ -x "${android_cc}" ]] || {
  echo "Android arm64 compiler not found. Set ANDROID_NDK_ROOT." >&2
  exit 2
}
rustup target list --installed | grep -qx aarch64-linux-android || {
  echo "Rust target aarch64-linux-android is not installed." >&2
  exit 2
}
boost_prefix="$(brew --prefix boost 2>/dev/null || true)"
[[ -f "${boost_prefix}/include/boost/preprocessor/stringize.hpp" ]] || {
  echo "Boost headers are required to compile Monero crypto-ops.c." >&2
  exit 2
}

mkdir -p "${result_dir}"
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

{
  echo "schema=wallet_android_cpu_reference_run_v1"
  echo "run_id=${run_id}"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "host_uname=$(uname -a)"
  echo "android_ndk=${probe_ndk_root}"
  echo "android_cc=$("${android_cc}" --version | head -n 1)"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "points_per_round=${point_count}"
  echo "timed_rounds=${rounds}"
  echo "warmup_rounds=${warmup_rounds}"
  echo "build_only=${build_only}"
  echo "rust_main_sha256=$(sha256_file "${rust_tool_dir}/src/main.rs")"
  echo "wallet_adapter_sha256=$(sha256_file "${source_root}/external/monero-fast-crypto/src/lib.rs")"
  echo "ref10_main_sha256=$(sha256_file "${ref10_tool_dir}/main.c")"
  echo "crypto_ops_sha256=$(sha256_file "${source_root}/src/crypto/crypto-ops.c")"
} >"${result_dir}/metadata.env"

(cd "${rust_tool_dir}" && cargo build --release --locked) \
  >"${result_dir}/host-vector-build.log" 2>&1
host_vector_generator="${rust_tool_dir}/target/release/monero-wallet-crypto-testbench"
"${host_vector_generator}" \
  --points "${point_count}" \
  --export-metal-vectors "${result_dir}/vectors.mwmtv1" \
  >"${result_dir}/vector-export.log" 2>&1

(cd "${rust_tool_dir}" && \
  CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="${android_cc}" \
  cargo build --release --locked --target aarch64-linux-android) \
  >"${result_dir}/rust-android-build.log" 2>&1
rust_binary="${rust_tool_dir}/target/aarch64-linux-android/release/monero-wallet-crypto-testbench"

ref10_binary="${result_dir}/monero-original-ref10-android-arm64"
"${android_cc}" -O3 -std=c11 -Wall -Wextra -Werror -pedantic -pthread \
  -I"${source_root}/src" \
  -I"${source_root}/src/crypto" \
  -I"${source_root}/contrib/epee/include" \
  -I"${boost_prefix}/include" \
  "${ref10_tool_dir}/main.c" \
  "${source_root}/src/crypto/crypto-ops.c" \
  "${source_root}/src/crypto/crypto-ops-data.c" \
  -o "${ref10_binary}" >"${result_dir}/ref10-android-build.log" 2>&1

{
  echo "vector_sha256=$(sha256_file "${result_dir}/vectors.mwmtv1")"
  echo "rust_binary_sha256=$(sha256_file "${rust_binary}")"
  echo "ref10_binary_sha256=$(sha256_file "${ref10_binary}")"
  echo "rust_binary_file=$(file "${rust_binary}")"
  echo "ref10_binary_file=$(file "${ref10_binary}")"
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

device_workers="${WALLET_ANDROID_CPU_WORKERS:-$("${adb_device[@]}" shell getconf _NPROCESSORS_ONLN | tr -d '\r')}"
[[ "${device_workers}" =~ ^[1-9][0-9]*$ ]] || {
  echo "device worker count is not a positive integer: ${device_workers}" >&2
  exit 2
}

remote_prefix="/data/local/tmp/wallet-accel-${run_id}"
remote_rust="${remote_prefix}-rust"
remote_ref10="${remote_prefix}-ref10"
remote_vectors="${remote_prefix}-vectors.mwmtv1"
cleanup_remote() {
  case "${remote_prefix}" in
    /data/local/tmp/wallet-accel-*)
      "${adb_device[@]}" shell rm -f \
        "${remote_rust}" "${remote_ref10}" "${remote_vectors}" \
        >/dev/null 2>&1 || true
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
  echo "device_workers=${device_workers}"
} >>"${result_dir}/metadata.env"

"${adb_device[@]}" push "${rust_binary}" "${remote_rust}" >"${result_dir}/adb-push.log"
"${adb_device[@]}" push "${ref10_binary}" "${remote_ref10}" >>"${result_dir}/adb-push.log"
"${adb_device[@]}" push "${result_dir}/vectors.mwmtv1" "${remote_vectors}" >>"${result_dir}/adb-push.log"
"${adb_device[@]}" shell chmod 700 "${remote_rust}" "${remote_ref10}"
"${adb_device[@]}" shell dumpsys thermalservice >"${result_dir}/thermal-before.txt" 2>&1 || true

run_remote_logged() {
  local output_file="$1"
  shift
  local attempt
  for attempt in 1 2; do
    if "${adb_device[@]}" shell "$@" | tr -d '\r' >"${output_file}.partial"; then
      mv "${output_file}.partial" "${output_file}"
      return 0
    fi
    echo "ADB interrupted benchmark attempt ${attempt}; waiting once and retrying." >&2
    "${adb_device[@]}" wait-for-device
  done
  return 1
}

run_remote_logged "${result_dir}/ref10-one-worker.log" "${remote_ref10}" \
  --vectors "${remote_vectors}" \
  --rounds "${rounds}" \
  --warmup-rounds "${warmup_rounds}" \
  --workers 1

run_remote_logged "${result_dir}/rust-one-worker.log" "${remote_rust}" \
  --points "${point_count}" \
  --rounds "${rounds}" \
  --warmup-rounds "${warmup_rounds}" \
  --workers 1

run_remote_logged "${result_dir}/rust-device-workers.log" "${remote_rust}" \
  --points "${point_count}" \
  --rounds "${rounds}" \
  --warmup-rounds "${warmup_rounds}" \
  --workers "${device_workers}"

"${adb_device[@]}" shell dumpsys thermalservice >"${result_dir}/thermal-after.txt" 2>&1 || true

{
  echo "section=original_c_ref10_one_worker"
  cat "${result_dir}/ref10-one-worker.log"
  echo "section=rust_one_worker"
  cat "${result_dir}/rust-one-worker.log"
  echo "section=rust_device_workers"
  cat "${result_dir}/rust-device-workers.log"
} >"${result_dir}/result.log"

grep -qx "preflight=pass" "${result_dir}/ref10-one-worker.log"
[[ "$(grep -c '^preflight=pass$' "${result_dir}/rust-one-worker.log")" -eq 1 ]]
[[ "$(grep -c '^preflight=pass$' "${result_dir}/rust-device-workers.log")" -eq 1 ]]
echo "finished_utc=$(date -u +%FT%TZ)" >>"${result_dir}/metadata.env"

echo "result_dir=${result_dir}"
cat "${result_dir}/result.log"
