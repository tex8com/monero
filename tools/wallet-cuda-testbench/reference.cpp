#include "derivation_core.cuh"
#include "derivation_radix2625.cuh"
#include "vector_corpus.hpp"

#include <chrono>
#include <array>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

enum class Variant {
  ladder,
  radix16,
  radix2625,
  radix2625_radix8,
  radix2625_radix8_sqrt_ratio,
  radix2625_radix4_sqrt_ratio,
  all,
};

struct Config {
  std::string vector_path;
  Variant variant = Variant::all;
};

std::uint64_t splitmix64(std::uint64_t &state) {
  state += 0x9e3779b97f4a7c15ull;
  std::uint64_t value = state;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

void validate_radix8_recode_one(const std::uint8_t *input) {
  std::uint8_t folded[32];
  char digits[86];
  monero_cuda::fold_scalar(folded, input);
  monero_cuda2625::scalar_as_radix_8_group(digits, folded);

  std::array<std::uint8_t, 33> reconstructed{};
  for (int index = 85; index >= 0; --index) {
    std::uint32_t carry = 0;
    for (auto &byte : reconstructed) {
      const std::uint32_t value =
          (static_cast<std::uint32_t>(byte) << 3) | carry;
      byte = static_cast<std::uint8_t>(value);
      carry = value >> 8;
    }
    if (carry != 0)
      throw std::runtime_error("radix-8 reconstruction overflow");

    const int digit = static_cast<int>(digits[index]);
    if (digit < -4 || digit > 4)
      throw std::runtime_error("radix-8 digit outside [-4, 4]");
    if (digit >= 0) {
      std::uint32_t add = static_cast<std::uint32_t>(digit);
      for (auto &byte : reconstructed) {
        const std::uint32_t value =
            static_cast<std::uint32_t>(byte) + add;
        byte = static_cast<std::uint8_t>(value);
        add = value >> 8;
      }
      if (add != 0)
        throw std::runtime_error("radix-8 addition overflow");
    } else {
      std::uint32_t subtract = static_cast<std::uint32_t>(-digit);
      for (auto &byte : reconstructed) {
        const std::uint32_t value = static_cast<std::uint32_t>(byte);
        byte = static_cast<std::uint8_t>(value - subtract);
        subtract = value < subtract ? 1 : 0;
      }
      if (subtract != 0)
        throw std::runtime_error("radix-8 reconstruction underflow");
    }
  }

  for (std::size_t byte = 0; byte < 32; ++byte)
    if (reconstructed[byte] != folded[byte])
      throw std::runtime_error("radix-8 reconstruction mismatch");
  if (reconstructed[32] != 0)
    throw std::runtime_error("radix-8 reconstruction high byte set");
}

void validate_radix8_recode() {
  std::array<std::uint8_t, 32> input{};
  validate_radix8_recode_one(input.data());
  input.fill(0xff);
  validate_radix8_recode_one(input.data());
  input.fill(0);
  input[0] = 1;
  validate_radix8_recode_one(input.data());
  for (std::size_t byte = 0; byte < 32; ++byte)
    input[byte] = monero_cuda::SCALAR_L[byte];
  validate_radix8_recode_one(input.data());

  std::uint64_t state = 0x435544415f52385full;
  for (std::size_t sample = 0; sample < 10000; ++sample) {
    for (std::size_t word = 0; word < 4; ++word) {
      const std::uint64_t value = splitmix64(state);
      for (std::size_t byte = 0; byte < 8; ++byte)
        input[word * 8 + byte] =
            static_cast<std::uint8_t>(value >> (byte * 8));
    }
    validate_radix8_recode_one(input.data());
  }
}

void validate_radix4_recode_one(const std::uint8_t *input) {
  std::uint8_t folded[32];
  char digits[128];
  monero_cuda::fold_scalar(folded, input);
  monero_cuda2625::scalar_as_radix_4_group(digits, folded);

  std::array<std::uint8_t, 33> reconstructed{};
  for (int index = 127; index >= 0; --index) {
    std::uint32_t carry = 0;
    for (auto &byte : reconstructed) {
      const std::uint32_t value =
          (static_cast<std::uint32_t>(byte) << 2) | carry;
      byte = static_cast<std::uint8_t>(value);
      carry = value >> 8;
    }
    if (carry != 0)
      throw std::runtime_error("radix-4 reconstruction overflow");

    const int digit = static_cast<int>(digits[index]);
    if (digit < -2 || digit > 2)
      throw std::runtime_error("radix-4 digit outside [-2, 2]");
    if (digit >= 0) {
      std::uint32_t add = static_cast<std::uint32_t>(digit);
      for (auto &byte : reconstructed) {
        const std::uint32_t value =
            static_cast<std::uint32_t>(byte) + add;
        byte = static_cast<std::uint8_t>(value);
        add = value >> 8;
      }
      if (add != 0)
        throw std::runtime_error("radix-4 addition overflow");
    } else {
      std::uint32_t subtract = static_cast<std::uint32_t>(-digit);
      for (auto &byte : reconstructed) {
        const std::uint32_t value = static_cast<std::uint32_t>(byte);
        byte = static_cast<std::uint8_t>(value - subtract);
        subtract = value < subtract ? 1 : 0;
      }
      if (subtract != 0)
        throw std::runtime_error("radix-4 reconstruction underflow");
    }
  }

  for (std::size_t byte = 0; byte < 32; ++byte)
    if (reconstructed[byte] != folded[byte])
      throw std::runtime_error("radix-4 reconstruction mismatch");
  if (reconstructed[32] != 0)
    throw std::runtime_error("radix-4 reconstruction high byte set");
}

void validate_radix4_recode() {
  std::array<std::uint8_t, 32> input{};
  validate_radix4_recode_one(input.data());
  input.fill(0xff);
  validate_radix4_recode_one(input.data());
  input.fill(0);
  input[0] = 1;
  validate_radix4_recode_one(input.data());
  for (std::size_t byte = 0; byte < 32; ++byte)
    input[byte] = monero_cuda::SCALAR_L[byte];
  validate_radix4_recode_one(input.data());

  std::uint64_t state = 0x435544415f52345full;
  for (std::size_t sample = 0; sample < 10000; ++sample) {
    for (std::size_t word = 0; word < 4; ++word) {
      const std::uint64_t value = splitmix64(state);
      for (std::size_t byte = 0; byte < 8; ++byte)
        input[word * 8 + byte] =
            static_cast<std::uint8_t>(value >> (byte * 8));
    }
    validate_radix4_recode_one(input.data());
  }
}

[[noreturn]] void usage(const std::string &message = {}) {
  if (!message.empty()) std::cerr << message << '\n';
  std::cerr
      << "Usage: wallet-cuda-reference --vectors PATH "
         "[--variant ladder|radix16|radix2625|radix2625-radix8|"
         "radix2625-radix8-sqrt-ratio|"
         "radix2625-radix4-sqrt-ratio|all]\n";
  std::exit(2);
}

Config parse_args(int argc, char **argv) {
  Config config;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--vectors") {
      if (++index >= argc) usage("--vectors requires a path");
      config.vector_path = argv[index];
    } else if (argument == "--variant") {
      if (++index >= argc) usage("--variant requires a value");
      const std::string value = argv[index];
      if (value == "ladder")
        config.variant = Variant::ladder;
      else if (value == "radix16")
        config.variant = Variant::radix16;
      else if (value == "radix2625")
        config.variant = Variant::radix2625;
      else if (value == "radix2625-radix8")
        config.variant = Variant::radix2625_radix8;
      else if (value == "radix2625-radix8-sqrt-ratio")
        config.variant = Variant::radix2625_radix8_sqrt_ratio;
      else if (value == "radix2625-radix4-sqrt-ratio")
        config.variant = Variant::radix2625_radix4_sqrt_ratio;
      else if (value == "all")
        config.variant = Variant::all;
      else
        usage("unknown variant: " + value);
    } else if (argument == "--help" || argument == "-h") {
      usage();
    } else {
      usage("unknown argument: " + argument);
    }
  }
  if (config.vector_path.empty()) usage("--vectors is required");
  return config;
}

void run_variant(const monero_cuda_testbench::VectorCorpus &corpus,
                 Variant variant) {
  std::uint8_t folded_scalar[32];
  monero_cuda::fold_scalar(folded_scalar, corpus.scalar.data());
  char radix16_digits[64];
  monero_cuda2625::scalar_as_radix_16_group(radix16_digits,
                                             folded_scalar);
  char radix8_digits[86];
  monero_cuda2625::scalar_as_radix_8_group(radix8_digits,
                                            folded_scalar);
  char radix4_digits[128];
  monero_cuda2625::scalar_as_radix_4_group(radix4_digits,
                                            folded_scalar);
  const std::size_t dispatch_count =
      static_cast<std::size_t>(corpus.count) + 1;
  std::vector<std::uint8_t> points = corpus.points;
  points.insert(points.end(), corpus.invalid.begin(), corpus.invalid.end());
  std::vector<std::uint8_t> results(dispatch_count * 32);
  std::vector<std::uint8_t> valid(dispatch_count);

  const auto started = std::chrono::steady_clock::now();
  for (std::size_t record = 0; record < dispatch_count; ++record) {
    bool accepted = false;
    if (variant == Variant::ladder) {
      accepted = monero_cuda::derive_ladder(
          results.data() + record * 32, folded_scalar,
          points.data() + record * 32);
    } else if (variant == Variant::radix16) {
      accepted = monero_cuda::derive_radix16(
          results.data() + record * 32, folded_scalar,
          points.data() + record * 32);
    } else {
      monero_cuda2625::Point point, derived;
      if (variant == Variant::radix2625_radix8_sqrt_ratio ||
          variant == Variant::radix2625_radix4_sqrt_ratio)
        accepted = monero_cuda2625::point_from_compressed_m6(
            point, points.data() + record * 32);
      else
        accepted = monero_cuda2625::point_from_compressed_m5(
            point, points.data() + record * 32);
      if (accepted) {
        if (variant == Variant::radix2625)
          monero_cuda2625::point_scalar_multiply_radix16_group(
              derived, point, radix16_digits);
        else if (variant == Variant::radix2625_radix4_sqrt_ratio)
          monero_cuda2625::point_scalar_multiply_radix4_group(
              derived, point, radix4_digits);
        else
          monero_cuda2625::point_scalar_multiply_radix8_group(
              derived, point, radix8_digits);
        monero_cuda2625::point_to_compressed_m5(
            results.data() + record * 32, derived);
      } else {
        for (std::size_t byte = 0; byte < 32; ++byte)
          results[record * 32 + byte] = 0;
      }
    }
    valid[record] = accepted ? 1 : 0;
  }
  const auto elapsed = std::chrono::steady_clock::now() - started;
  monero_cuda_testbench::validate_results(corpus, results, valid);
  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(elapsed)
          .count();
  const char *name = variant == Variant::ladder
                         ? "portable_ladder"
                         : (variant == Variant::radix16
                                ? "portable_radix16"
                                : (variant == Variant::radix2625
                                       ? "portable_radix2625"
                                       : (variant ==
                                                  Variant::radix2625_radix8
                                              ? "portable_radix2625_radix8"
                                              : (variant ==
                                                         Variant::radix2625_radix8_sqrt_ratio
                                                     ? "portable_radix2625_radix8_sqrt_ratio"
                                                     : "portable_radix2625_radix4_sqrt_ratio"))));
  std::cout << "result_begin\n";
  std::cout << "backend=portable_cpp\n";
  std::cout << "variant=" << name << '\n';
  std::cout << "records=" << corpus.count << '\n';
  std::cout << "elapsed_seconds=" << std::fixed << std::setprecision(9)
            << seconds << '\n';
  std::cout << "derivations_per_second=" << std::setprecision(3)
            << static_cast<double>(corpus.count) / seconds << '\n';
  std::cout << "result_checksum_fnv1a64=0x" << std::hex
            << monero_cuda_testbench::checksum_results(results) << std::dec
            << '\n';
  std::cout << "validation=pass\n";
  std::cout << "result_end\n";
}

}  // namespace

int main(int argc, char **argv) {
  try {
    const Config config = parse_args(argc, argv);
    validate_radix8_recode();
    validate_radix4_recode();
    const auto corpus =
        monero_cuda_testbench::read_vector_corpus(config.vector_path);
    std::cout << "testbench=monero_wallet_cuda_portable_reference_v1\n";
    std::cout << "radix8_recode_samples=10004\n";
    std::cout << "radix8_recode_validation=pass\n";
    std::cout << "radix4_recode_samples=10004\n";
    std::cout << "radix4_recode_validation=pass\n";
    std::cout << "corpus_fingerprint_fnv1a64=0x" << std::hex
              << corpus.fingerprint << std::dec << '\n';
    if (config.variant == Variant::ladder ||
        config.variant == Variant::all)
      run_variant(corpus, Variant::ladder);
    if (config.variant == Variant::radix16 ||
        config.variant == Variant::all)
      run_variant(corpus, Variant::radix16);
    if (config.variant == Variant::radix2625 ||
        config.variant == Variant::all)
      run_variant(corpus, Variant::radix2625);
    if (config.variant == Variant::radix2625_radix8 ||
        config.variant == Variant::all)
      run_variant(corpus, Variant::radix2625_radix8);
    if (config.variant == Variant::radix2625_radix8_sqrt_ratio ||
        config.variant == Variant::all)
      run_variant(corpus, Variant::radix2625_radix8_sqrt_ratio);
    if (config.variant == Variant::radix2625_radix4_sqrt_ratio ||
        config.variant == Variant::all)
      run_variant(corpus, Variant::radix2625_radix4_sqrt_ratio);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "wallet-cuda-reference error: " << error.what() << '\n';
    return 1;
  }
}
