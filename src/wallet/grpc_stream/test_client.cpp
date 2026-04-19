// Smoke test for cuprate_grpc_stream_client against a live cuprated gRPC server.
//
// Not linked into the wallet — built separately to validate the client
// class end-to-end before wiring it into wallet2.cpp.
//
// Build:
//   cmake -DMONERO_ENABLE_GRPC_STREAM=ON -DBUILD_GRPC_STREAM_TESTS=ON ../..
//   make cuprate_grpc_stream_test
// Run:
//   ./src/wallet/grpc_stream/cuprate_grpc_stream_test <host:port> <start> <stop> [chunk_hint=1000]

#include "grpc_block_stream_client.h"

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <host:port> <start_height> <stop_height> [chunk_hint=1000]\n",
            argv[0]);
        return 2;
    }
    const std::string target = argv[1];
    const uint64_t start_h = std::strtoull(argv[2], nullptr, 10);
    const uint64_t stop_h  = std::strtoull(argv[3], nullptr, 10);
    const uint32_t chunk_hint = (argc >= 5)
        ? static_cast<uint32_t>(std::strtoul(argv[4], nullptr, 10))
        : 1000U;

    cuprate_grpc_stream::cuprate_grpc_stream_client client;
    if (!client.connect(target)) {
        std::fprintf(stderr, "connect failed: %s\n", client.last_error_message().c_str());
        return 1;
    }
    if (!client.open_stream(start_h, stop_h, /*prune=*/true, chunk_hint, "wallet-smoke-1")) {
        std::fprintf(stderr, "open_stream failed: %s\n", client.last_error_message().c_str());
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::string payload;
    uint64_t chunks = 0, blocks = 0, bytes = 0;
    while (client.next_chunk_payload(payload, 60000)) {
        chunks++;
        blocks += client.last_chunk_n_blocks();
        bytes  += client.last_chunk_payload_bytes();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double avg_mbs = (total_ms > 0.0)
        ? (static_cast<double>(bytes) / 1024.0 / 1024.0) / (total_ms / 1000.0)
        : 0.0;
    std::printf(
        "\n[TEST RESULT] ok=%d chunks=%llu blocks=%llu bytes=%llu total_ms=%.1f avg_mbs=%.2f err_code=%d err_msg='%s'\n",
        client.stream_ended_ok() ? 1 : 0,
        (unsigned long long)chunks,
        (unsigned long long)blocks,
        (unsigned long long)bytes,
        total_ms,
        avg_mbs,
        client.last_error_code(),
        client.last_error_message().c_str());
    client.close();
    return client.stream_ended_ok() ? 0 : 1;
}
