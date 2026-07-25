#include "monero_fast_metal.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
  constexpr uint8_t MAGIC[8] = {
    0x4d, 0x57, 0x4d, 0x54, 0x56, 0x31, 0x00, 0x00
  };
  constexpr size_t HEADER_BYTES = 88;

  struct Corpus
  {
    uint8_t scalar[32];
    std::vector<uint8_t> points;
    std::vector<uint8_t> expected;
    size_t valid_count;
  };

  uint32_t read_le32(const uint8_t *bytes)
  {
    return static_cast<uint32_t>(bytes[0])
      | static_cast<uint32_t>(bytes[1]) << 8
      | static_cast<uint32_t>(bytes[2]) << 16
      | static_cast<uint32_t>(bytes[3]) << 24;
  }

  std::vector<uint8_t> read_file(const std::string &path)
  {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input)
      throw std::runtime_error("cannot open vector file");
    const std::streamsize length = input.tellg();
    if (length < 0)
      throw std::runtime_error("cannot determine vector-file size");
    input.seekg(0);
    std::vector<uint8_t> bytes(static_cast<size_t>(length));
    if (!input.read(reinterpret_cast<char *>(bytes.data()), length))
      throw std::runtime_error("cannot read vector file");
    return bytes;
  }

  Corpus read_corpus(const std::string &path)
  {
    const std::vector<uint8_t> bytes = read_file(path);
    if (bytes.size() < HEADER_BYTES
        || std::memcmp(bytes.data(), MAGIC, sizeof(MAGIC)) != 0)
      throw std::runtime_error("invalid MWMTV1 header");
    if (read_le32(bytes.data() + 8) != 1)
      throw std::runtime_error("unsupported MWMTV1 version");
    const size_t count = read_le32(bytes.data() + 12);
    if (count == 0
        || count > (std::numeric_limits<size_t>::max() - HEADER_BYTES) / 64
        || bytes.size() != HEADER_BYTES + count * 64)
      throw std::runtime_error("invalid MWMTV1 record count");

    Corpus corpus;
    std::memcpy(corpus.scalar, bytes.data() + 16, 32);
    corpus.valid_count = count;
    corpus.points.reserve((count + 1) * 32);
    corpus.expected.reserve((count + 1) * 32);
    for (size_t index = 0; index < count; ++index)
    {
      const size_t offset = HEADER_BYTES + index * 64;
      corpus.points.insert(
        corpus.points.end(), bytes.begin() + offset, bytes.begin() + offset + 32);
      corpus.expected.insert(
        corpus.expected.end(),
        bytes.begin() + offset + 32,
        bytes.begin() + offset + 64);
    }
    corpus.points.insert(
      corpus.points.end(), bytes.begin() + 56, bytes.begin() + 88);
    corpus.expected.resize((count + 1) * 32, 0);
    return corpus;
  }

  size_t positive(const char *text, const char *name)
  {
    char *end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (end == text || end == nullptr || *end != '\0' || parsed == 0
        || parsed > std::numeric_limits<size_t>::max())
      throw std::runtime_error(std::string(name) + " requires a positive integer");
    return static_cast<size_t>(parsed);
  }

  void validate(
      const Corpus &corpus,
      const std::vector<uint8_t> &results,
      const std::vector<uint8_t> &valid,
      int64_t successes)
  {
    if (successes != static_cast<int64_t>(corpus.valid_count))
      throw std::runtime_error("unexpected Metal success count");
    for (size_t index = 0; index < corpus.valid_count; ++index)
    {
      if (valid[index] != 1)
        throw std::runtime_error("valid point was rejected");
    }
    if (valid.back() != 0)
      throw std::runtime_error("invalid point was accepted");
    if (results != corpus.expected)
      throw std::runtime_error("Metal result differs from Dalek vector");
  }
}

int main(int argc, char **argv)
{
  try
  {
    std::string vector_path;
    size_t rounds = 1;
    size_t warmup_rounds = 1;
    for (int index = 1; index < argc; ++index)
    {
      const std::string argument = argv[index];
      if (argument == "--vectors" && index + 1 < argc)
        vector_path = argv[++index];
      else if (argument == "--rounds" && index + 1 < argc)
        rounds = positive(argv[++index], "--rounds");
      else if (argument == "--warmup-rounds" && index + 1 < argc)
        warmup_rounds = positive(argv[++index], "--warmup-rounds");
      else
        throw std::runtime_error("usage: --vectors PATH [--rounds N] [--warmup-rounds N]");
    }
    if (vector_path.empty())
      throw std::runtime_error("--vectors is required");

    const Corpus corpus = read_corpus(vector_path);
    const size_t dispatch_count = corpus.valid_count + 1;
    std::vector<uint8_t> results(dispatch_count * 32);
    std::vector<uint8_t> valid(dispatch_count);

    if (fast_metal_derivation_available() != 1)
      throw std::runtime_error(
        std::string("Metal backend unavailable: ")
        + fast_metal_derivation_last_error());

    auto run = [&](size_t run_count, bool timed) {
      std::chrono::nanoseconds elapsed{0};
      for (size_t round = 0; round < run_count; ++round)
      {
        const auto start = std::chrono::steady_clock::now();
        const int64_t successes =
          fast_metal_generate_key_derivation_batch_same_scalar(
            results.data(),
            corpus.scalar,
            corpus.points.data(),
            valid.data(),
            dispatch_count);
        const auto end = std::chrono::steady_clock::now();
        if (successes < 0)
          throw std::runtime_error(
            std::string("Metal dispatch failed: ")
            + fast_metal_derivation_last_error());
        validate(corpus, results, valid, successes);
        if (timed)
          elapsed += end - start;
      }
      return elapsed;
    };

    run(warmup_rounds, false);
    const std::chrono::nanoseconds elapsed = run(rounds, true);
    const double seconds = static_cast<double>(elapsed.count()) / 1.0e9;
    const size_t operations = corpus.valid_count * rounds;
    const size_t logical_bytes =
      (32 + corpus.valid_count * 32 + corpus.valid_count * 32
        + corpus.valid_count) * rounds;

    std::cout << "testbench=wallet_metal_product_backend_v1\n";
    std::cout << "backend=native_objcxx_precompiled_metallib\n";
    std::cout << "algorithm=monero_generate_key_derivation_8_times_a_times_r\n";
    std::cout << "points_per_round=" << corpus.valid_count << '\n';
    std::cout << "dispatch_records_per_round=" << dispatch_count << '\n';
    std::cout << "timed_rounds=" << rounds << '\n';
    std::cout << "warmup_rounds=" << warmup_rounds << '\n';
    std::cout << "operations=" << operations << '\n';
    std::cout << "host_submit_wait_ns=" << elapsed.count() << '\n';
    std::cout << std::fixed << std::setprecision(9)
      << "host_submit_wait_seconds=" << seconds << '\n';
    std::cout << std::setprecision(3)
      << "derivations_per_second=" << static_cast<double>(operations) / seconds
      << '\n';
    std::cout << "logical_payload_bytes=" << logical_bytes << '\n';
    std::cout << std::setprecision(6)
      << "logical_payload_mib_per_second="
      << static_cast<double>(logical_bytes) / seconds / 1048576.0 << '\n';
    std::cout << "dalek_byte_validation=pass\n";
    std::cout << "invalid_point_validation=pass\n";
    std::cout << "validation=pass\n";
    return 0;
  }
  catch (const std::exception &error)
  {
    std::cerr << "wallet Metal product testbench error: "
      << error.what() << '\n';
    return 1;
  }
}
