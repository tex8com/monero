#!/usr/bin/env bash
# Build and run the byte-checked three-stage pure-u32 Vulkan derivation.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${tool_dir}/../../.." && pwd)"
rust_tool_dir="${source_root}/tools/wallet-crypto-testbench"
metal_source="${source_root}/tools/wallet-metal-testbench/derivation_radix13_u32.metal"
results_root="${WALLET_ANDROID_ACCEL_RESULTS_DIR:-${source_root}/build/wallet-mobile-acceleration-testbench}"
run_id="${WALLET_ANDROID_ACCEL_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-android-vulkan-radix13}"
result_dir="${results_root}/${run_id}"
build_dir="${result_dir}/build"
point_count="${WALLET_ANDROID_GPU_POINTS:-8192}"
rounds="${WALLET_ANDROID_GPU_ROUNDS:-5}"
warmup_rounds="${WALLET_ANDROID_GPU_WARMUP_ROUNDS:-2}"
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
  echo "WALLET_ANDROID_ACCEL_RUN_ID has unsupported characters." >&2
  exit 2
}
for value in "${point_count}" "${rounds}" "${warmup_rounds}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || {
    echo "point and round settings must be positive integers." >&2
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
[[ -f "${probe_ndk_root}/build/cmake/android.toolchain.cmake" ]] || {
  echo "Android NDK CMake toolchain not found. Set ANDROID_NDK_ROOT." >&2
  exit 2
}
command -v cmake >/dev/null || { echo "cmake is required." >&2; exit 2; }
command -v ninja >/dev/null || { echo "ninja is required." >&2; exit 2; }

glslang_source="${GLSLANG_VALIDATOR:-}"
if [[ -z "${glslang_source}" ]]; then
  glslang_source="$(command -v glslangValidator 2>/dev/null || true)"
fi
if [[ -z "${glslang_source}" ]]; then
  glslang_source="${android_sdk_root}/emulator/lib64/vulkan/glslangValidator"
fi
spirv_opt_source="${SPIRV_OPT:-${probe_ndk_root}/shader-tools/darwin-x86_64/spirv-opt}"
[[ -f "${glslang_source}" ]] || {
  echo "glslangValidator not found. Set GLSLANG_VALIDATOR." >&2
  exit 2
}
[[ -f "${spirv_opt_source}" ]] || {
  echo "spirv-opt not found. Set SPIRV_OPT." >&2
  exit 2
}

mkdir -p "${build_dir}" "${result_dir}/shaders"
cp "${glslang_source}" "${result_dir}/glslangValidator"
cp "${spirv_opt_source}" "${result_dir}/spirv-opt"
chmod 700 "${result_dir}/glslangValidator" "${result_dir}/spirv-opt"
sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

{
  echo "schema=wallet_android_vulkan_radix13_run_v2"
  echo "run_id=${run_id}"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "host_uname=$(uname -a)"
  echo "android_ndk=${probe_ndk_root}"
  echo "points_per_round=${point_count}"
  echo "timed_rounds=${rounds}"
  echo "warmup_rounds=${warmup_rounds}"
  echo "build_only=${build_only}"
  echo "generator_sha256=$(sha256_file "${tool_dir}/generate_radix13_glsl.py")"
  echo "host_sha256=$(sha256_file "${tool_dir}/vulkan_radix13_benchmark.cpp")"
  echo "metal_m13_sha256=$(sha256_file "${metal_source}")"
} >"${result_dir}/metadata.env"

(cd "${rust_tool_dir}" && cargo build --release --locked) \
  >"${result_dir}/host-vector-build.log" 2>&1
host_vector_generator="${rust_tool_dir}/target/release/monero-wallet-crypto-testbench"
"${host_vector_generator}" \
  --points "${point_count}" \
  --export-metal-vectors "${result_dir}/vectors.mwmtv1" \
  >"${result_dir}/vector-export.log" 2>&1

for stage in decode multiply compress; do
  source_file="${result_dir}/shaders/derivation_radix13_${stage}.comp"
  raw_file="${result_dir}/shaders/derivation_radix13_${stage}_raw.spv"
  shader_file="${result_dir}/shaders/derivation_radix13_${stage}.spv"
  python3 "${tool_dir}/generate_radix13_glsl.py" \
    --metal-source "${metal_source}" \
    --stage "${stage}" \
    --output "${source_file}"
  "${result_dir}/glslangValidator" -V --target-env vulkan1.1 \
    -S comp "${source_file}" -o "${raw_file}" \
    >"${result_dir}/shader-${stage}.log" 2>&1
  "${result_dir}/spirv-opt" --eliminate-dead-functions \
    "${raw_file}" -o "${shader_file}"
  echo "shader_${stage}_sha256=$(sha256_file "${shader_file}")" \
    >>"${result_dir}/metadata.env"
done

cmake -S "${tool_dir}" -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="${probe_ndk_root}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-24 \
  >"${result_dir}/configure.log" 2>&1
cmake --build "${build_dir}" --target wallet-vulkan-radix13-benchmark \
  >"${result_dir}/build.log" 2>&1
binary="${build_dir}/wallet-vulkan-radix13-benchmark"
[[ -x "${binary}" ]] || { echo "Vulkan benchmark was not produced." >&2; exit 2; }
{
  echo "vector_sha256=$(sha256_file "${result_dir}/vectors.mwmtv1")"
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

remote_prefix="/data/local/tmp/wallet-vk-r13-${run_id}"
remote_binary="${remote_prefix}-benchmark"
remote_vectors="${remote_prefix}-vectors.mwmtv1"
remote_decode="${remote_prefix}-decode.spv"
remote_multiply="${remote_prefix}-multiply.spv"
remote_compress="${remote_prefix}-compress.spv"
remote_output="${remote_prefix}-output.txt"
remote_status="${remote_prefix}-status.txt"
cleanup_remote() {
  case "${remote_prefix}" in
    /data/local/tmp/wallet-vk-r13-*)
      "${adb_device[@]}" shell rm -f \
        "${remote_binary}" "${remote_vectors}" "${remote_decode}" \
        "${remote_multiply}" "${remote_compress}" \
        "${remote_output}" "${remote_status}" >/dev/null 2>&1 || true
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
} >>"${result_dir}/metadata.env"

"${adb_device[@]}" push "${binary}" "${remote_binary}" >"${result_dir}/adb-push.log"
"${adb_device[@]}" push "${result_dir}/vectors.mwmtv1" "${remote_vectors}" >>"${result_dir}/adb-push.log"
"${adb_device[@]}" push "${result_dir}/shaders/derivation_radix13_decode.spv" "${remote_decode}" >>"${result_dir}/adb-push.log"
"${adb_device[@]}" push "${result_dir}/shaders/derivation_radix13_multiply.spv" "${remote_multiply}" >>"${result_dir}/adb-push.log"
"${adb_device[@]}" push "${result_dir}/shaders/derivation_radix13_compress.spv" "${remote_compress}" >>"${result_dir}/adb-push.log"
"${adb_device[@]}" shell chmod 700 "${remote_binary}"
"${adb_device[@]}" shell dumpsys thermalservice >"${result_dir}/thermal-before.txt" 2>&1 || true

remote_command="${remote_binary} --vectors ${remote_vectors} --decode-shader ${remote_decode} --multiply-shader ${remote_multiply} --compress-shader ${remote_compress} --rounds ${rounds} --warmup-rounds ${warmup_rounds}"
"${adb_device[@]}" shell \
  "rm -f '${remote_output}' '${remote_status}'; nohup sh -c '${remote_command} > ${remote_output} 2>&1; echo \$? > ${remote_status}' </dev/null >/dev/null 2>&1 &"

complete=0
for attempt in $(seq 1 180); do
  if "${adb_device[@]}" shell test -f "${remote_status}" >/dev/null 2>&1; then
    complete=1
    break
  fi
  sleep 1
done
((complete == 1)) || {
  echo "remote Vulkan benchmark did not finish within 180 seconds." >&2
  exit 3
}
"${adb_device[@]}" shell cat "${remote_output}" | tr -d '\r' >"${result_dir}/result.log"
remote_exit="$("${adb_device[@]}" shell cat "${remote_status}" | tr -d '\r')"
"${adb_device[@]}" shell dumpsys thermalservice >"${result_dir}/thermal-after.txt" 2>&1 || true
[[ "${remote_exit}" == "0" ]] || {
  echo "remote Vulkan benchmark failed with exit ${remote_exit}." >&2
  cat "${result_dir}/result.log" >&2
  exit 3
}
grep -qx "validation=pass" "${result_dir}/result.log"
echo "finished_utc=$(date -u +%FT%TZ)" >>"${result_dir}/metadata.env"

echo "result_dir=${result_dir}"
cat "${result_dir}/result.log"
