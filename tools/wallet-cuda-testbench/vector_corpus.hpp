#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace monero_cuda_testbench {

struct VectorCorpus {
  std::array<std::uint8_t, 32> scalar{};
  std::array<std::uint8_t, 32> invalid{};
  std::vector<std::uint8_t> points;
  std::vector<std::uint8_t> expected;
  std::uint64_t fingerprint = 0;
  std::uint32_t count = 0;
};

inline std::uint32_t load_le32(const std::uint8_t *bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         (static_cast<std::uint32_t>(bytes[1]) << 8) |
         (static_cast<std::uint32_t>(bytes[2]) << 16) |
         (static_cast<std::uint32_t>(bytes[3]) << 24);
}

inline std::uint64_t load_le64(const std::uint8_t *bytes) {
  std::uint64_t value = 0;
  for (unsigned i = 0; i < 8; ++i)
    value |= static_cast<std::uint64_t>(bytes[i]) << (8u * i);
  return value;
}

inline VectorCorpus read_vector_corpus(const std::string &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open vector file: " + path);
  const std::streamoff end = input.tellg();
  if (end < 0) throw std::runtime_error("cannot stat vector file: " + path);
  if (static_cast<std::uint64_t>(end) >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    throw std::runtime_error("vector file is too large");
  std::vector<std::uint8_t> raw(static_cast<std::size_t>(end));
  input.seekg(0);
  if (!raw.empty())
    input.read(reinterpret_cast<char *>(raw.data()),
               static_cast<std::streamsize>(raw.size()));
  if (!input) throw std::runtime_error("cannot read vector file: " + path);

  constexpr std::array<std::uint8_t, 8> magic = {
      0x4d, 0x57, 0x4d, 0x54, 0x56, 0x31, 0x00, 0x00};
  constexpr std::size_t header_bytes = 88;
  if (raw.size() < header_bytes)
    throw std::runtime_error("vector file is shorter than MWMTV1 header");
  for (std::size_t i = 0; i < magic.size(); ++i)
    if (raw[i] != magic[i])
      throw std::runtime_error("vector file magic is not MWMTV1");
  if (load_le32(raw.data() + 8) != 1)
    throw std::runtime_error("unsupported MWMTV1 version");

  VectorCorpus corpus;
  corpus.count = load_le32(raw.data() + 12);
  if (corpus.count == 0)
    throw std::runtime_error("vector corpus contains zero records");
  const std::size_t count = corpus.count;
  if (count > (std::numeric_limits<std::size_t>::max() - header_bytes) / 64)
    throw std::runtime_error("vector record count overflows size_t");
  if (raw.size() != header_bytes + count * 64)
    throw std::runtime_error("vector file size does not match record count");
  for (std::size_t i = 0; i < 32; ++i) {
    corpus.scalar[i] = raw[16 + i];
    corpus.invalid[i] = raw[56 + i];
  }
  corpus.fingerprint = load_le64(raw.data() + 48);
  corpus.points.resize(count * 32);
  corpus.expected.resize(count * 32);
  for (std::size_t record = 0; record < count; ++record) {
    const std::size_t source = header_bytes + record * 64;
    const std::size_t destination = record * 32;
    for (std::size_t byte = 0; byte < 32; ++byte) {
      corpus.points[destination + byte] = raw[source + byte];
      corpus.expected[destination + byte] = raw[source + 32 + byte];
    }
  }
  return corpus;
}

inline std::uint64_t checksum_results(const std::vector<std::uint8_t> &bytes) {
  std::uint64_t fingerprint = 0xcbf29ce484222325ull;
  for (std::uint8_t byte : bytes) {
    fingerprint ^= byte;
    fingerprint *= 0x100000001b3ull;
  }
  return fingerprint;
}

inline void validate_results(const VectorCorpus &corpus,
                             const std::vector<std::uint8_t> &results,
                             const std::vector<std::uint8_t> &valid) {
  const std::size_t dispatch_count =
      static_cast<std::size_t>(corpus.count) + 1;
  if (results.size() != dispatch_count * 32 ||
      valid.size() != dispatch_count)
    throw std::runtime_error("result buffer shape is invalid");
  for (std::size_t record = 0; record < corpus.count; ++record) {
    if (valid[record] != 1)
      throw std::runtime_error("valid point was rejected at record " +
                               std::to_string(record));
    for (std::size_t byte = 0; byte < 32; ++byte) {
      if (results[record * 32 + byte] !=
          corpus.expected[record * 32 + byte])
        throw std::runtime_error(
            "Dalek mismatch at record " + std::to_string(record) +
            ", byte " + std::to_string(byte));
    }
  }
  if (valid[corpus.count] != 0)
    throw std::runtime_error("invalid compressed point was accepted");
  for (std::size_t byte = 0; byte < 32; ++byte)
    if (results[static_cast<std::size_t>(corpus.count) * 32 + byte] != 0)
      throw std::runtime_error(
          "invalid point left a nonzero output byte");
}

}  // namespace monero_cuda_testbench
