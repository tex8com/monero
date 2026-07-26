// Cuprate gRPC streaming sync client.
//
// Pimpl-wrapped C++ client for the cuprate.stream.v1.BlockStream service.
// Hides grpc++ headers from wallet2.cpp so the wallet can be built without
// pulling abseil/grpc into every translation unit.
//
// Usage pattern:
//   cuprate_grpc_stream_client client;
//   if (!client.connect("host:18091")) return false;
//   if (!client.open_stream(start_h, 0, /*prune=*/true, /*chunk_hint=*/1000)) return false;
//   std::string epee_payload;
//   while (client.next_chunk_payload(epee_payload, /*timeout_ms=*/30000)) {
//     // epee_payload is exactly the wire format the bin RPC's GetBlocks
//     // returns -- decode with the existing
//     // COMMAND_RPC_GET_BLOCKS_FAST::response deserializer.
//     ...
//   }
//   client.close();
//
// Thread model: a single background recv thread (started by open_stream)
// drains the gRPC stream into a bounded queue. next_chunk_payload pops from
// the queue with a timeout. close() shuts the stream down deterministically.
//
// Errors and end-of-stream are both reported as `next_chunk_payload returns
// false`. Use last_error_code()/last_error_message() to distinguish.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cuprate_grpc_stream {

// Common source interface used by the wallet.  The ordinary client is one
// HTTP/2 connection; the range pool below is an opt-in desktop transport that
// keeps several *disjoint*, finite height ranges in flight but exposes only
// the next contiguous chunk to wallet2.  Keeping this boundary here makes the
// wallet's consensus/scan path independent of the transport fan-out.
class cuprate_grpc_block_stream_source
{
public:
    virtual ~cuprate_grpc_block_stream_source() = default;

    virtual bool connect(const std::string& target) = 0;
    virtual bool open_stream(uint64_t start_height,
                             uint64_t stop_height,
                             bool prune,
                             uint32_t chunk_blocks_hint,
                             const std::string& client_request_id,
                             const std::vector<std::string>& chain_locator = {}) = 0;
    virtual bool next_chunk_payload(std::string& out_epee_bytes, uint32_t timeout_ms) = 0;
    virtual bool stream_ended_ok() const = 0;
    virtual void close() = 0;

    virtual uint64_t    last_chunk_start_height() const = 0;
    virtual uint64_t    last_chunk_seq() const = 0;
    virtual uint64_t    last_chunk_chain_tip() const = 0;
    virtual uint32_t    last_chunk_n_blocks() const = 0;
    virtual uint32_t    last_chunk_payload_bytes() const = 0;
    virtual std::string last_chunk_server_request_id() const = 0;
    virtual double      last_chunk_read_call_ms() const = 0;
    virtual double      last_chunk_queue_dwell_ms() const = 0;
    virtual size_t      queue_depth() const = 0;
    virtual size_t      queue_payload_bytes() const = 0;
    virtual size_t      max_queue_depth() const = 0;
    virtual size_t      max_queue_payload_bytes() const = 0;
    virtual double      total_queue_full_wait_ms() const = 0;
    virtual double      total_read_wait_ms() const = 0;
    virtual int         last_error_code() const = 0;
    virtual std::string last_error_message() const = 0;
};

class cuprate_grpc_stream_client final : public cuprate_grpc_block_stream_source
{
public:
    cuprate_grpc_stream_client();
    ~cuprate_grpc_stream_client();

    cuprate_grpc_stream_client(const cuprate_grpc_stream_client&) = delete;
    cuprate_grpc_stream_client& operator=(const cuprate_grpc_stream_client&) = delete;

    // Establish a gRPC channel to `target` (e.g. "host:18091"). Returns true
    // on success. Note: gRPC channel creation is lazy — actual TCP connection
    // happens on the first RPC call. This function only validates the target
    // string and constructs the channel object.
    bool connect(const std::string& target);

    // Open a server-streaming StreamBlocks call. Spawns a background thread
    // that drains the stream into the internal queue. Returns true on
    // successful stream open. Pass stop_height = 0 to stream until tip.
    //
    // `chunk_blocks_hint` follows the proto contract — server clamps to
    // [16, 10000]. For high-throughput tests use CUPRATE_GRPC_CHUNK_HINT=10000.
    //
    // `client_request_id` is echoed in the cuprated [GRPC StreamBlocks] log
    // line of every chunk — pass a per-sync-session id so wallet logs and
    // server logs line up without a shared clock.
    bool open_stream(uint64_t start_height,
                     uint64_t stop_height,
                     bool prune,
                     uint32_t chunk_blocks_hint,
                     const std::string& client_request_id,
                     const std::vector<std::string>& chain_locator = {});

    // Pop the next chunk's epee payload (blocking up to `timeout_ms`).
    // Returns false on timeout, end-of-stream, or error.
    //
    // On success, `out_epee_bytes` is exactly the wire format the bin RPC
    // GetBlocks returns. Decode with the wallet's existing
    // COMMAND_RPC_GET_BLOCKS_FAST::response deserializer (load_t_from_binary).
    //
    // Diagnostics about the chunk (server_request_id, chunk_seq, n_blocks,
    // chain_tip, start_height) are exposed via last_chunk_*() accessors.
    bool next_chunk_payload(std::string& out_epee_bytes, uint32_t timeout_ms);

    // True iff the stream has terminated cleanly (server reached tip /
    // stop_height) — distinguishes from "still streaming, just timed out
    // waiting for a chunk".
    bool stream_ended_ok() const;

    // Close the stream and join the recv thread. Idempotent.
    void close();

    // Last-chunk diagnostics (set by next_chunk_payload on success).
    uint64_t    last_chunk_start_height() const;
    uint64_t    last_chunk_seq()         const;
    uint64_t    last_chunk_chain_tip()   const;
    uint32_t    last_chunk_n_blocks()    const;
    uint32_t    last_chunk_payload_bytes() const;
    std::string last_chunk_server_request_id() const;
    // Per-chunk client-local timings. `read_call_ms` includes gRPC/HTTP2
    // delivery and protobuf materialization; `queue_dwell_ms` is only the
    // bounded hand-off queue between receive thread and wallet pull.
    double      last_chunk_read_call_ms() const;
    double      last_chunk_queue_dwell_ms() const;

    // Live bounded-buffer and receive-loop telemetry. Queue byte counts are
    // deliberately reported alongside chunk counts: block payloads vary
    // substantially, so a count-only capacity cannot explain RAM pressure.
    size_t      queue_depth() const;
    size_t      queue_payload_bytes() const;
    size_t      max_queue_depth() const;
    size_t      max_queue_payload_bytes() const;
    double      total_queue_full_wait_ms() const;
    double      total_read_wait_ms() const;

    // Last-error info (set on any failure).
    int         last_error_code()    const;
    std::string last_error_message() const;

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

// Experimental, explicitly opt-in parallel range transport.  It first gets a
// bounded bootstrap range to establish a fixed tip snapshot, then opens up to
// sixteen non-overlapping [start, stop] streams.  A lane may receive ahead, but
// next_chunk_payload() releases chunks only at the exact next height.  The
// benchmark may additionally request a local gRPC subchannel pool per lane so
// those logical streams cannot be coalesced onto one TCP connection.  This is
// deliberately a transport optimiser, not a parallel wallet scanner.
//
// It is selected only by wallet2 when CUPRATE_GRPC_RANGE_CONNECTIONS is set to
// 2..16.  The normal one-stream implementation remains the product default and
// the mobile-safe behaviour.  A reorg is still handled by wallet2's existing
// chain validation; close() cancels every active lane immediately.
class cuprate_grpc_range_pool final : public cuprate_grpc_block_stream_source
{
public:
    cuprate_grpc_range_pool();
    ~cuprate_grpc_range_pool();

    cuprate_grpc_range_pool(const cuprate_grpc_range_pool&) = delete;
    cuprate_grpc_range_pool& operator=(const cuprate_grpc_range_pool&) = delete;

    bool connect(const std::string& target) override;
    bool open_stream(uint64_t start_height,
                     uint64_t stop_height,
                     bool prune,
                     uint32_t chunk_blocks_hint,
                     const std::string& client_request_id,
                     const std::vector<std::string>& chain_locator = {}) override;
    bool next_chunk_payload(std::string& out_epee_bytes, uint32_t timeout_ms) override;
    bool stream_ended_ok() const override;
    void close() override;

    uint64_t    last_chunk_start_height() const override;
    uint64_t    last_chunk_seq() const override;
    uint64_t    last_chunk_chain_tip() const override;
    uint32_t    last_chunk_n_blocks() const override;
    uint32_t    last_chunk_payload_bytes() const override;
    std::string last_chunk_server_request_id() const override;
    double      last_chunk_read_call_ms() const override;
    double      last_chunk_queue_dwell_ms() const override;
    size_t      queue_depth() const override;
    size_t      queue_payload_bytes() const override;
    size_t      max_queue_depth() const override;
    size_t      max_queue_payload_bytes() const override;
    double      total_queue_full_wait_ms() const override;
    double      total_read_wait_ms() const override;
    int         last_error_code() const override;
    std::string last_error_message() const override;

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

} // namespace cuprate_grpc_stream
