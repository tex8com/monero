#!/usr/bin/env bash
# Mac/Metal stage M0. It validates the planned derivation buffer ABI only.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${tool_dir}/../.." && pwd)"
results_root="${WALLET_METAL_BENCH_RESULTS_DIR:-${source_root}/build/wallet-metal-testbench}"
run_id="${WALLET_METAL_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)}"
result_dir="${results_root}/${run_id}"

[[ "$(uname -s)" == "Darwin" ]] || { echo "Metal M0 requires macOS." >&2; exit 2; }
[[ ! -e "${result_dir}" ]] || { echo "result directory already exists: ${result_dir}" >&2; exit 2; }
mkdir -p "${result_dir}"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }

{
  echo "schema=wallet_metal_layout_benchmark_v1"
  echo "run_id=${run_id}"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname)"
  echo "uname=$(uname -a)"
  echo "macos=$(sw_vers | tr '\n' ';')"
  echo "metal_device=$(system_profiler SPDisplaysDataType | tr '\n' ';')"
  echo "swiftc=$(xcrun swiftc --version | tr '\n' ';')"
  echo "testbench_main_sha256=$(sha256_file "${tool_dir}/main.swift")"
  echo "command=$0 $*"
} >"${result_dir}/metadata.env"

binary="${result_dir}/wallet-metal-layout-testbench"
xcrun swiftc -O -framework Metal "${tool_dir}/main.swift" -o "${binary}" >"${result_dir}/build.log" 2>&1
/usr/bin/time -l "${binary}" "$@" >"${result_dir}/result.log" 2>"${result_dir}/time.log"

{
  echo "finished_utc=$(date -u +%FT%TZ)"
  echo "binary_sha256=$(sha256_file "${binary}")"
} >>"${result_dir}/metadata.env"

echo "result_dir=${result_dir}"
cat "${result_dir}/result.log"
