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

#include <cstdint>
#include <memory>
#include <string>

namespace cuprate_grpc_stream {

class cuprate_grpc_stream_client
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
    // [16, 2000]. Sweet spot per smoke tests: 1000.
    //
    // `client_request_id` is echoed in the cuprated [GRPC StreamBlocks] log
    // line of every chunk — pass a per-sync-session id so wallet logs and
    // server logs line up without a shared clock.
    bool open_stream(uint64_t start_height,
                     uint64_t stop_height,
                     bool prune,
                     uint32_t chunk_blocks_hint,
                     const std::string& client_request_id);

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

    // Last-error info (set on any failure).
    int         last_error_code()    const;
    std::string last_error_message() const;

private:
    struct impl;
    std::unique_ptr<impl> p_;
};

} // namespace cuprate_grpc_stream
