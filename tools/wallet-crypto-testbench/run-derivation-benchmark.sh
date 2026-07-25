#!/usr/bin/env bash
# Reproducible local CPU benchmark for the exact Monero 8*a*R derivation.
# No network, blockchain, wallet file, key material, or CI service is used.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${tool_dir}/../.." && pwd)"
results_root="${WALLET_CRYPTO_BENCH_RESULTS_DIR:-${source_root}/build/wallet-crypto-testbench}"
run_id="${WALLET_CRYPTO_BENCH_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)}"
result_dir="${results_root}/${run_id}"

if [[ -e "${result_dir}" ]]; then
  echo "result directory already exists: ${result_dir}" >&2
  exit 2
fi
mkdir -p "${result_dir}"

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

cpu_description() {
  if [[ "$(uname -s)" == "Darwin" ]]; then
    sysctl -n machdep.cpu.brand_string 2>/dev/null || sysctl -n hw.model
  else
    awk -F: '/model name/ {gsub(/^ /, "", $2); print $2; exit}' /proc/cpuinfo
  fi
}

{
  echo "schema=wallet_crypto_benchmark_v1"
  echo "run_id=${run_id}"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname)"
  echo "uname=$(uname -a)"
  echo "cpu=$(cpu_description)"
  echo "logical_cpus=$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
  echo "rustc=$(rustc -V 2>/dev/null || true)"
  echo "cargo=$(cargo -V 2>/dev/null || true)"
  echo "monero_commit=$(git -C "${source_root}" rev-parse HEAD 2>/dev/null || true)"
  echo "monero_status=$(git -C "${source_root}" status --porcelain 2>/dev/null | wc -l | tr -d ' ')"
  echo "source_revision=${WALLET_CRYPTO_BENCH_SOURCE_REVISION:-$(git -C "${source_root}" rev-parse HEAD 2>/dev/null || true)}"
  echo "testbench_main_sha256=$(sha256_file "${tool_dir}/src/main.rs")"
  echo "testbench_cargo_toml_sha256=$(sha256_file "${tool_dir}/Cargo.toml")"
  echo "testbench_lock_sha256=$(sha256_file "${tool_dir}/Cargo.lock")"
  echo "fast_crypto_lib_sha256=$(sha256_file "${source_root}/external/monero-fast-crypto/src/lib.rs")"
  echo "fast_crypto_cargo_toml_sha256=$(sha256_file "${source_root}/external/monero-fast-crypto/Cargo.toml")"
  echo "fast_crypto_lock_sha256=$(sha256_file "${source_root}/external/monero-fast-crypto/Cargo.lock")"
  echo "command=$0 $*"
} >"${result_dir}/metadata.env"

(cd "${tool_dir}" && cargo build --release --locked) >"${result_dir}/build.log" 2>&1
binary="${tool_dir}/target/release/monero-wallet-crypto-testbench"
[[ -x "${binary}" ]] || { echo "benchmark binary missing: ${binary}" >&2; exit 1; }

run_linux_with_proc_sampling() {
  local result_log="$1" time_log="$2"
  shift 2
  local ticks start_ns end_ns pid status=0
  local last_user_ticks=0 last_system_ticks=0 max_rss_bytes=0
  local sampled_user sampled_system sampled_rss sampled_hwm

  ticks="$(getconf CLK_TCK)"
  start_ns="$(date +%s%N)"
  "${binary}" "$@" >"${result_log}" 2>"${time_log}" &
  pid=$!
  while kill -0 "${pid}" 2>/dev/null; do
    if [[ -r "/proc/${pid}/stat" ]]; then
      read -r sampled_user sampled_system sampled_rss < <(awk '{print $14, $15, $24}' "/proc/${pid}/stat")
      if [[ "${sampled_user:-}" =~ ^[0-9]+$ ]]; then
        last_user_ticks="${sampled_user}"
      fi
      if [[ "${sampled_system:-}" =~ ^[0-9]+$ ]]; then
        last_system_ticks="${sampled_system}"
      fi
      if [[ "${sampled_rss:-}" =~ ^[0-9]+$ ]]; then
        local current_rss_bytes=$(( sampled_rss * $(getconf PAGESIZE) ))
        if (( current_rss_bytes > max_rss_bytes )); then
          max_rss_bytes="${current_rss_bytes}"
        fi
      fi
    fi
    if [[ -r "/proc/${pid}/status" ]]; then
      sampled_hwm="$(awk '/^VmHWM:/ {print $2; exit}' "/proc/${pid}/status")"
      if [[ "${sampled_hwm:-}" =~ ^[0-9]+$ ]]; then
        local hwm_bytes=$(( sampled_hwm * 1024 ))
        if (( hwm_bytes > max_rss_bytes )); then
          max_rss_bytes="${hwm_bytes}"
        fi
      fi
    fi
    sleep 0.1
  done
  if wait "${pid}"; then
    :
  else
    status=$?
  fi
  end_ns="$(date +%s%N)"
  {
    echo "resource_method=linux_proc_sampling_100ms"
    echo "elapsed_ns=$(( end_ns - start_ns ))"
    echo "cpu_user_seconds=$(awk -v ticks="${last_user_ticks}" -v hz="${ticks}" 'BEGIN { printf "%.6f", ticks / hz }')"
    echo "cpu_system_seconds=$(awk -v ticks="${last_system_ticks}" -v hz="${ticks}" 'BEGIN { printf "%.6f", ticks / hz }')"
    echo "max_rss_bytes_sampled=${max_rss_bytes}"
  } >>"${time_log}"
  return "${status}"
}

if [[ "$(uname -s)" == "Darwin" && -x /usr/bin/time ]]; then
  /usr/bin/time -l "${binary}" "$@" >"${result_dir}/result.log" 2>"${result_dir}/time.log"
elif [[ -x /usr/bin/time ]]; then
  /usr/bin/time -v "${binary}" "$@" >"${result_dir}/result.log" 2>"${result_dir}/time.log"
else
  run_linux_with_proc_sampling "${result_dir}/result.log" "${result_dir}/time.log" "$@"
fi

{
  echo "finished_utc=$(date -u +%FT%TZ)"
  echo "binary_sha256=$(sha256_file "${binary}")"
} >>"${result_dir}/metadata.env"

echo "result_dir=${result_dir}"
cat "${result_dir}/result.log"
