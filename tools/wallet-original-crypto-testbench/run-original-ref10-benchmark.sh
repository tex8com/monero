#!/usr/bin/env bash
# Build and run the isolated historical Monero Ref10 derivation benchmark.
set -euo pipefail

tool_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "${tool_dir}/../.." && pwd)"
crypto_tool_dir="${source_root}/tools/wallet-crypto-testbench"
results_root="${WALLET_ORIGINAL_REF10_RESULTS_DIR:-${source_root}/build/wallet-original-ref10-testbench}"
run_id="${WALLET_ORIGINAL_REF10_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname -s)}"
point_count="${WALLET_ORIGINAL_REF10_POINTS:-131072}"
rounds="${WALLET_ORIGINAL_REF10_ROUNDS:-100}"
warmup_rounds="${WALLET_ORIGINAL_REF10_WARMUP_ROUNDS:-2}"
workers="${WALLET_ORIGINAL_REF10_WORKERS:-1}"
result_dir="${results_root}/${run_id}"

[[ "$(uname -s)" == "Darwin" ]] || { echo "This benchmark script currently records macOS /usr/bin/time -l metrics." >&2; exit 2; }
for value in "${point_count}" "${rounds}" "${warmup_rounds}" "${workers}"; do
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || { echo "all count settings must be positive integers" >&2; exit 2; }
done
[[ ! -e "${result_dir}" ]] || { echo "result directory already exists: ${result_dir}" >&2; exit 2; }
mkdir -p "${result_dir}"

sha256_file() { shasum -a 256 "$1" | awk '{print $1}'; }
boost_prefix="$(brew --prefix boost 2>/dev/null || true)"
[[ -n "${boost_prefix}" && -f "${boost_prefix}/include/boost/preprocessor/stringize.hpp" ]] || {
  echo "Boost headers are required to compile Monero crypto-ops.c." >&2
  exit 2
}

{
  echo "schema=monero_original_ref10_derivation_benchmark_v1"
  echo "run_id=${run_id}"
  echo "started_utc=$(date -u +%FT%TZ)"
  echo "hostname=$(hostname)"
  echo "uname=$(uname -a)"
  echo "macos=$(sw_vers | tr '\n' ';')"
  echo "cc=$(xcrun clang --version | tr '\n' ';')"
  echo "boost_prefix=${boost_prefix}"
  echo "rustc=$(rustc --version)"
  echo "cargo=$(cargo --version)"
  echo "historical_source_commit=e0a84c6e8^"
  echo "historical_function=crypto_ops::generate_key_derivation"
  echo "historical_crypto_cpp_sha256=$(git -C "${source_root}" show e0a84c6e8^:src/crypto/crypto.cpp | shasum -a 256 | awk '{print $1}')"
  echo "crypto_ops_c_sha256=$(sha256_file "${source_root}/src/crypto/crypto-ops.c")"
  echo "crypto_ops_data_c_sha256=$(sha256_file "${source_root}/src/crypto/crypto-ops-data.c")"
  echo "testbench_main_sha256=$(sha256_file "${tool_dir}/main.c")"
  echo "vector_generator_main_sha256=$(sha256_file "${crypto_tool_dir}/src/main.rs")"
  echo "points_per_round=${point_count}"
  echo "timed_rounds=${rounds}"
  echo "warmup_rounds=${warmup_rounds}"
  echo "workers=${workers}"
  echo "command=$0 $*"
} >"${result_dir}/metadata.env"

(cd "${crypto_tool_dir}" && cargo build --release --locked) >"${result_dir}/vector-build.log" 2>&1
"${crypto_tool_dir}/target/release/monero-wallet-crypto-testbench" \
  --points "${point_count}" \
  --export-metal-vectors "${result_dir}/vectors.mwmtv1" \
  >"${result_dir}/vector-export.log" 2>&1

binary="${result_dir}/monero-original-ref10-testbench"
xcrun clang -O3 -std=c11 -Wall -Wextra -Werror -pedantic -pthread \
  -I"${source_root}/src" -I"${source_root}/src/crypto" -I"${source_root}/contrib/epee/include" -I"${boost_prefix}/include" \
  "${tool_dir}/main.c" "${source_root}/src/crypto/crypto-ops.c" "${source_root}/src/crypto/crypto-ops-data.c" \
  -o "${binary}" >"${result_dir}/build.log" 2>&1

/usr/bin/time -l "${binary}" \
  --vectors "${result_dir}/vectors.mwmtv1" \
  --rounds "${rounds}" --warmup-rounds "${warmup_rounds}" --workers "${workers}" \
  "$@" >"${result_dir}/result.log" 2>"${result_dir}/time.log"

{
  echo "finished_utc=$(date -u +%FT%TZ)"
  echo "vector_sha256=$(sha256_file "${result_dir}/vectors.mwmtv1")"
  echo "binary_sha256=$(sha256_file "${binary}")"
} >>"${result_dir}/metadata.env"

echo "result_dir=${result_dir}"
cat "${result_dir}/result.log"
