// Smoke/diagnostic test for cuprate_grpc_block_stream_source against a live
// cuprated gRPC server.
//
// Not linked into the wallet — built separately to validate the client
// class end-to-end before wiring it into wallet2.cpp.
//
// Build:
//   cmake -DMONERO_ENABLE_GRPC_STREAM=ON -DBUILD_GRPC_STREAM_TESTS=ON ../..
//   make cuprate_grpc_stream_test
// Run:
//   ./src/wallet/grpc_stream/cuprate_grpc_stream_test <host:port> <start> <stop> [chunk_hint=1000]
//
// Set CUPRATE_GRPC_TEST_RANGE_POOL=1 together with the explicitly opt-in
// range-pool variables to drain real block payload through the parallel
// transport. CUPRATE_GRPC_TEST_FANOUT=1 is a stricter transport ceiling test:
// it snapshots the tip, opens disjoint direct streams, and drains each on its
// own thread without an ordered spool. Both deliberately do not parse or
// wallet-scan payload; neither is an E2E wallet benchmark result.

#include "grpc_block_stream_client.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

bool enabled_exactly_one(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1' && value[1] == '\0';
}

uint64_t bounded_env_u64(const char* name, uint64_t fallback,
                         uint64_t minimum, uint64_t maximum)
{
    const char* value = std::getenv(name);
    if (!value || !*value)
        return fallback;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (!end || *end != '\0' || parsed < minimum || parsed > maximum)
        return fallback;
    return static_cast<uint64_t>(parsed);
}

// Transport-only ceiling diagnostic. The bootstrap gives every worker the
// same finite chain tip; workers then drain disjoint height ranges directly.
// It intentionally has no shared ordering, spool, parser or scanner.
int run_direct_fanout(const std::string& target, uint64_t start_h,
                      uint64_t stop_h, uint32_t chunk_hint)
{
    const uint64_t requested_lanes = bounded_env_u64(
        "CUPRATE_GRPC_RANGE_CONNECTIONS", 4, 2, 16);
    const uint64_t bootstrap_blocks = bounded_env_u64(
        "CUPRATE_GRPC_RANGE_BOOTSTRAP_BLOCKS", 1000, 1000, 20000);
    const uint64_t bootstrap_stop = stop_h == 0
        ? start_h + bootstrap_blocks - 1
        : std::min(stop_h, start_h + bootstrap_blocks - 1);

    std::fprintf(stderr,
        "[TEST MODE] transport_only=1 source=direct_fanout requested_lanes=%llu bootstrap_blocks=%llu\n",
        static_cast<unsigned long long>(requested_lanes),
        static_cast<unsigned long long>(bootstrap_blocks));

    const auto t0 = std::chrono::steady_clock::now();
    cuprate_grpc_stream::cuprate_grpc_stream_client bootstrap;
    if (!bootstrap.connect(target) || !bootstrap.open_stream(start_h,
            bootstrap_stop, /*prune=*/true, chunk_hint,
            "wallet-fanout-bootstrap")) {
        std::fprintf(stderr, "bootstrap open failed: %s\n",
            bootstrap.last_error_message().c_str());
        return 1;
    }

    uint64_t chunks = 0, blocks = 0, bytes = 0;
    uint64_t next_start = start_h;
    uint64_t snapshot_tip = 0;
    std::string payload;
    while (bootstrap.next_chunk_payload(payload, 60000)) {
        ++chunks;
        blocks += bootstrap.last_chunk_n_blocks();
        bytes += bootstrap.last_chunk_payload_bytes();
        next_start = bootstrap.last_chunk_start_height()
            + bootstrap.last_chunk_n_blocks();
        if (snapshot_tip == 0)
            snapshot_tip = bootstrap.last_chunk_chain_tip();
        payload.clear();
    }
    const bool bootstrap_ok = bootstrap.stream_ended_ok();
    const int bootstrap_error = bootstrap.last_error_code();
    const std::string bootstrap_message = bootstrap.last_error_message();
    bootstrap.close();
    if (!bootstrap_ok || snapshot_tip < start_h) {
        std::fprintf(stderr,
            "bootstrap terminated ok=%d tip=%llu err_code=%d err_msg='%s'\n",
            bootstrap_ok ? 1 : 0, static_cast<unsigned long long>(snapshot_tip),
            bootstrap_error, bootstrap_message.c_str());
        return 1;
    }
    if (stop_h != 0)
        snapshot_tip = std::min(snapshot_tip, stop_h);

    std::atomic<uint64_t> worker_chunks{0};
    std::atomic<uint64_t> worker_blocks{0};
    std::atomic<uint64_t> worker_bytes{0};
    std::atomic<bool> worker_failed{false};
    std::mutex error_mu;
    int worker_error_code = 0;
    std::string worker_error_message;
    const auto set_error = [&](int code, const std::string& message) {
        bool expected = false;
        if (worker_failed.compare_exchange_strong(expected, true)) {
            std::lock_guard<std::mutex> lock(error_mu);
            worker_error_code = code;
            worker_error_message = message;
        }
    };

    if (next_start <= snapshot_tip) {
        const uint64_t remaining = snapshot_tip - next_start + 1;
        const uint64_t lanes = std::min(requested_lanes, remaining);
        std::vector<std::thread> workers;
        workers.reserve(static_cast<size_t>(lanes));
        uint64_t lane_start = next_start;
        for (uint64_t lane = 0; lane < lanes; ++lane) {
            const uint64_t lane_blocks = remaining / lanes
                + (lane < (remaining % lanes) ? 1 : 0);
            const uint64_t lane_stop = lane_start + lane_blocks - 1;
            workers.emplace_back([&, lane, lane_start, lane_stop]() {
                cuprate_grpc_stream::cuprate_grpc_stream_client client;
                if (!client.connect(target) || !client.open_stream(lane_start,
                        lane_stop, /*prune=*/true, chunk_hint,
                        "wallet-fanout-l" + std::to_string(lane))) {
                    set_error(client.last_error_code(),
                        "fanout lane open: " + client.last_error_message());
                    return;
                }
                std::string lane_payload;
                uint64_t local_chunks = 0, local_blocks = 0, local_bytes = 0;
                while (client.next_chunk_payload(lane_payload, 60000)) {
                    ++local_chunks;
                    local_blocks += client.last_chunk_n_blocks();
                    local_bytes += client.last_chunk_payload_bytes();
                    lane_payload.clear();
                }
                const bool ended_ok = client.stream_ended_ok();
                const int error_code = client.last_error_code();
                const std::string error_message = client.last_error_message();
                client.close();
                if (!ended_ok) {
                    set_error(error_code, "fanout lane terminated: " + error_message);
                    return;
                }
                worker_chunks.fetch_add(local_chunks);
                worker_blocks.fetch_add(local_blocks);
                worker_bytes.fetch_add(local_bytes);
            });
            lane_start = lane_stop + 1;
        }
        for (std::thread& worker : workers)
            worker.join();
    }

    chunks += worker_chunks.load();
    blocks += worker_blocks.load();
    bytes += worker_bytes.load();
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double avg_mbs = total_ms > 0.0
        ? (static_cast<double>(bytes) / 1024.0 / 1024.0) / (total_ms / 1000.0)
        : 0.0;
    const bool complete = !worker_failed.load()
        && blocks == snapshot_tip - start_h + 1;
    std::printf(
        "\n[TEST RESULT] ok=%d chunks=%llu blocks=%llu bytes=%llu total_ms=%.1f avg_mbs=%.2f err_code=%d err_msg='%s'\n",
        complete ? 1 : 0, static_cast<unsigned long long>(chunks),
        static_cast<unsigned long long>(blocks), static_cast<unsigned long long>(bytes),
        total_ms, avg_mbs, worker_error_code, worker_error_message.c_str());
    return complete ? 0 : 1;
}

} // namespace

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

    if (enabled_exactly_one("CUPRATE_GRPC_TEST_FANOUT"))
        return run_direct_fanout(target, start_h, stop_h, chunk_hint);

    const bool use_range_pool = enabled_exactly_one("CUPRATE_GRPC_TEST_RANGE_POOL");
    std::unique_ptr<cuprate_grpc_stream::cuprate_grpc_block_stream_source> client;
    if (use_range_pool)
        client = std::make_unique<cuprate_grpc_stream::cuprate_grpc_range_pool>();
    else
        client = std::make_unique<cuprate_grpc_stream::cuprate_grpc_stream_client>();

    std::fprintf(stderr, "[TEST MODE] transport_only=1 source=%s\n",
        use_range_pool ? "range_pool" : "single_stream");
    if (!client->connect(target)) {
        std::fprintf(stderr, "connect failed: %s\n", client->last_error_message().c_str());
        return 1;
    }
    if (!client->open_stream(start_h, stop_h, /*prune=*/true, chunk_hint, "wallet-smoke-1")) {
        std::fprintf(stderr, "open_stream failed: %s\n", client->last_error_message().c_str());
        return 1;
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::string payload;
    uint64_t chunks = 0, blocks = 0, bytes = 0;
    while (client->next_chunk_payload(payload, 60000)) {
        chunks++;
        blocks += client->last_chunk_n_blocks();
        bytes  += client->last_chunk_payload_bytes();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double avg_mbs = (total_ms > 0.0)
        ? (static_cast<double>(bytes) / 1024.0 / 1024.0) / (total_ms / 1000.0)
        : 0.0;
    std::printf(
        "\n[TEST RESULT] ok=%d chunks=%llu blocks=%llu bytes=%llu total_ms=%.1f avg_mbs=%.2f err_code=%d err_msg='%s'\n",
        client->stream_ended_ok() ? 1 : 0,
        (unsigned long long)chunks,
        (unsigned long long)blocks,
        (unsigned long long)bytes,
        total_ms,
        avg_mbs,
        client->last_error_code(),
        client->last_error_message().c_str());
    const bool ended_ok = client->stream_ended_ok();
    client->close();
    return ended_ok ? 0 : 1;
}
