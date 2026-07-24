#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace cuprate_grpc_stream {

// A bounded, ordered scheduler over independent gRPC channels. Ranges never
// overlap; a completed range is committed by wallet2 before its slot is reused.
// This preserves chain order while allowing network transfer to run ahead.
class cuprate_grpc_block_range_pool
{
public:
    struct impl;
    cuprate_grpc_block_range_pool();
    ~cuprate_grpc_block_range_pool();
    cuprate_grpc_block_range_pool(const cuprate_grpc_block_range_pool&) = delete;
    cuprate_grpc_block_range_pool& operator=(const cuprate_grpc_block_range_pool&) = delete;

    bool open(const std::string& target, uint64_t start_height, uint64_t stop_height,
              uint32_t range_blocks, size_t max_channels,
              const std::string& client_request_id,
              const std::vector<std::string>& chain_locator);
    bool next_chunk_payload(std::string& out_epee_bytes, uint32_t timeout_ms);
    void close();
    bool stream_ended_ok() const;
    int last_error_code() const;
    std::string last_error_message() const;
    uint64_t last_chunk_start_height() const;
    uint64_t last_chunk_seq() const;
    uint64_t last_chunk_chain_tip() const;
    uint32_t last_chunk_n_blocks() const;
    uint32_t last_chunk_payload_bytes() const;
    std::string last_chunk_server_request_id() const;

private:
    std::unique_ptr<impl> p_;
};

} // namespace cuprate_grpc_stream
