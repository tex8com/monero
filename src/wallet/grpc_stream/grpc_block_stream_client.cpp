// Cuprate gRPC streaming sync client — skeleton implementation.
//
// This file provides only the Pimpl scaffolding so the build wires up
// cleanly. The real client (channel + recv thread + queue + logging)
// lands in the next commit.

#include "grpc_block_stream_client.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <grpcpp/grpcpp.h>
#include "cuprate_stream.grpc.pb.h"

namespace cuprate_grpc_stream {

struct cuprate_grpc_stream_client::impl {
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<cuprate::stream::v1::BlockStream::Stub> stub;

    int         last_error_code = 0;
    std::string last_error_message;
    bool        stream_ended_ok = false;

    uint64_t    last_chunk_start_height = 0;
    uint64_t    last_chunk_seq = 0;
    uint64_t    last_chunk_chain_tip = 0;
    uint32_t    last_chunk_n_blocks = 0;
    uint32_t    last_chunk_payload_bytes = 0;
    std::string last_chunk_server_request_id;
};

cuprate_grpc_stream_client::cuprate_grpc_stream_client()
  : p_(std::make_unique<impl>())
{}

cuprate_grpc_stream_client::~cuprate_grpc_stream_client() = default;

bool cuprate_grpc_stream_client::connect(const std::string& target)
{
    if (target.empty()) {
        p_->last_error_code = -1;
        p_->last_error_message = "empty target";
        return false;
    }
    grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(64 * 1024 * 1024);
    ch_args.SetMaxSendMessageSize(64 * 1024 * 1024);
    p_->channel = grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), ch_args);
    p_->stub = cuprate::stream::v1::BlockStream::NewStub(p_->channel);
    std::fprintf(stderr, "[GRPC client] connect target=%s (channel created — connect is lazy)\n", target.c_str());
    return true;
}

bool cuprate_grpc_stream_client::open_stream(uint64_t /*start_height*/,
                                             uint64_t /*stop_height*/,
                                             bool /*prune*/,
                                             uint32_t /*chunk_blocks_hint*/,
                                             const std::string& /*client_request_id*/)
{
    p_->last_error_code = -2;
    p_->last_error_message = "open_stream: not implemented in skeleton (commit 2)";
    return false;
}

bool cuprate_grpc_stream_client::next_chunk_payload(std::string& /*out_epee_bytes*/, uint32_t /*timeout_ms*/)
{
    p_->last_error_code = -2;
    p_->last_error_message = "next_chunk_payload: not implemented in skeleton (commit 2)";
    return false;
}

bool cuprate_grpc_stream_client::stream_ended_ok() const { return p_->stream_ended_ok; }
void cuprate_grpc_stream_client::close() {}

uint64_t    cuprate_grpc_stream_client::last_chunk_start_height()      const { return p_->last_chunk_start_height; }
uint64_t    cuprate_grpc_stream_client::last_chunk_seq()               const { return p_->last_chunk_seq; }
uint64_t    cuprate_grpc_stream_client::last_chunk_chain_tip()         const { return p_->last_chunk_chain_tip; }
uint32_t    cuprate_grpc_stream_client::last_chunk_n_blocks()          const { return p_->last_chunk_n_blocks; }
uint32_t    cuprate_grpc_stream_client::last_chunk_payload_bytes()     const { return p_->last_chunk_payload_bytes; }
std::string cuprate_grpc_stream_client::last_chunk_server_request_id() const { return p_->last_chunk_server_request_id; }

int         cuprate_grpc_stream_client::last_error_code()    const { return p_->last_error_code; }
std::string cuprate_grpc_stream_client::last_error_message() const { return p_->last_error_message; }

} // namespace cuprate_grpc_stream
