// Cuprate gRPC streaming sync client — full implementation.
//
// Recv-thread model:
//   open_stream() spawns one background thread that drains the gRPC stream
//   (reader->Read) into a bounded std::deque under mutex+condvar.
//   next_chunk_payload() pops one chunk with timeout. close() cancels the
//   ClientContext (forces Read to return false) and joins the thread.
//
// Bounded queue: a full queue means the wallet (consumer) is slower than
// the cuprate server. The recv thread blocks on the queue's "not full"
// condition, which propagates back through gRPC flow control to the server
// — same backpressure model as the cuprated side, just on the receiving
// end. Capacity is small on purpose so the bp shows up in PERF logs
// instead of bloating memory.

#include "grpc_block_stream_client.h"
#include "grpc_stream_status.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

#include <grpc/impl/channel_arg_names.h>
#include <grpcpp/grpcpp.h>
#include "cuprate_stream.grpc.pb.h"

namespace cuprate_grpc_stream {

std::atomic<int64_t>& last_successful_chunk_unix_ms()
{
    static std::atomic<int64_t> s_last{0};
    return s_last;
}

namespace {
    constexpr size_t DEFAULT_QUEUE_CAPACITY = 32;
    constexpr size_t MAX_QUEUE_CAPACITY = 128;
    constexpr int MAX_GRPC_MESSAGE_BYTES = 1024 * 1024 * 1024;
    constexpr int GRPC_HTTP2_BUFFER_BYTES = 64 * 1024 * 1024;
    constexpr int GRPC_HTTP2_MAX_FRAME_BYTES = 16 * 1024 * 1024 - 1;

    size_t queue_capacity()
    {
        static const size_t cap = []() -> size_t {
            const char *env = std::getenv("CUPRATE_GRPC_QUEUE_CAPACITY");
            if (!env || !*env)
                return DEFAULT_QUEUE_CAPACITY;

            errno = 0;
            char *endptr = nullptr;
            const unsigned long parsed = std::strtoul(env, &endptr, 10);
            if (errno != 0 || !endptr || *endptr != '\0' || parsed == 0)
                return DEFAULT_QUEUE_CAPACITY;

            return static_cast<size_t>(std::max(1UL, std::min(parsed, static_cast<unsigned long>(MAX_QUEUE_CAPACITY))));
        }();
        return cap;
    }
}

struct chunk_record {
    std::string  payload;
    uint64_t     start_height = 0;
    uint64_t     chunk_seq = 0;
    uint64_t     chain_tip = 0;
    uint32_t     n_blocks = 0;
    uint32_t     payload_bytes = 0;
    std::string  server_request_id;
};

struct cuprate_grpc_stream_client::impl {
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<cuprate::stream::v1::BlockStream::Stub> stub;

    std::unique_ptr<grpc::ClientContext> ctx;
    std::unique_ptr<grpc::ClientReader<cuprate::stream::v1::BlockChunk>> reader;
    std::thread recv_thread;

    std::mutex                mu;
    std::condition_variable   cv_not_empty;
    std::condition_variable   cv_not_full;
    std::deque<chunk_record>  queue;
    bool                      done = false;
    bool                      cancelled = false;

    int         last_error_code = 0;
    std::string last_error_message;
    bool        stream_ended_ok = false;

    uint64_t    last_chunk_start_height = 0;
    uint64_t    last_chunk_seq = 0;
    uint64_t    last_chunk_chain_tip = 0;
    uint32_t    last_chunk_n_blocks = 0;
    uint32_t    last_chunk_payload_bytes = 0;
    std::string last_chunk_server_request_id;

    std::string client_request_id;
    std::chrono::steady_clock::time_point t_open;
    std::chrono::steady_clock::time_point t_first_chunk;
    bool        first_chunk_seen = false;
    uint64_t    total_chunks = 0;
    uint64_t    total_blocks = 0;
    uint64_t    total_bytes = 0;
    uint64_t    backpressure_waits = 0;
};

cuprate_grpc_stream_client::cuprate_grpc_stream_client()
  : p_(std::make_unique<impl>())
{}

cuprate_grpc_stream_client::~cuprate_grpc_stream_client()
{
    close();
}

bool cuprate_grpc_stream_client::connect(const std::string& target)
{
    if (target.empty()) {
        p_->last_error_code = -1;
        p_->last_error_message = "empty target";
        return false;
    }
    grpc::ChannelArguments ch_args;
    ch_args.SetMaxReceiveMessageSize(MAX_GRPC_MESSAGE_BYTES);
    ch_args.SetMaxSendMessageSize(MAX_GRPC_MESSAGE_BYTES);
    ch_args.SetInt(GRPC_ARG_HTTP2_STREAM_LOOKAHEAD_BYTES, GRPC_HTTP2_BUFFER_BYTES);
    ch_args.SetInt(GRPC_ARG_HTTP2_WRITE_BUFFER_SIZE, GRPC_HTTP2_BUFFER_BYTES);
    ch_args.SetInt(GRPC_ARG_HTTP2_MAX_FRAME_SIZE, GRPC_HTTP2_MAX_FRAME_BYTES);
    ch_args.SetInt(GRPC_ARG_HTTP2_BDP_PROBE, 1);
    p_->channel = grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), ch_args);
    p_->stub = cuprate::stream::v1::BlockStream::NewStub(p_->channel);
    std::fprintf(stderr,
        "[GRPC client] CONNECT target=%s queue_capacity=%zu max_msg=%d http2_buffer=%d max_frame=%d (channel created -- gRPC connect is lazy, first RPC opens TCP)\n",
        target.c_str(),
        queue_capacity(),
        MAX_GRPC_MESSAGE_BYTES,
        GRPC_HTTP2_BUFFER_BYTES,
        GRPC_HTTP2_MAX_FRAME_BYTES);
    return true;
}

bool cuprate_grpc_stream_client::open_stream(uint64_t start_height,
                                             uint64_t stop_height,
                                             bool prune,
                                             uint32_t chunk_blocks_hint,
                                             const std::string& client_request_id,
                                             const std::vector<std::string>& chain_locator)
{
    if (!p_->stub) {
        p_->last_error_code = -1;
        p_->last_error_message = "open_stream: connect() not called";
        return false;
    }
    if (p_->reader) {
        p_->last_error_code = -1;
        p_->last_error_message = "open_stream: stream already open (close first)";
        return false;
    }

    p_->stream_ended_ok = false;
    p_->done = false;
    p_->cancelled = false;
    p_->last_error_code = 0;
    p_->last_error_message.clear();
    p_->client_request_id = client_request_id;
    p_->t_open = std::chrono::steady_clock::now();
    p_->first_chunk_seen = false;
    p_->total_chunks = 0;
    p_->total_blocks = 0;
    p_->total_bytes = 0;
    p_->backpressure_waits = 0;

    cuprate::stream::v1::StreamBlocksRequest req;
    req.set_start_height(start_height);
    req.set_stop_height(stop_height);
    req.set_prune(prune);
    req.set_chunk_blocks_hint(chunk_blocks_hint);
    req.set_no_miner_tx(false);
    req.set_client_request_id(client_request_id);
    for (const std::string& hash : chain_locator)
        req.add_chain_locator(hash);

    p_->ctx = std::make_unique<grpc::ClientContext>();
    p_->reader = p_->stub->StreamBlocks(p_->ctx.get(), req);

    std::fprintf(stderr,
        "[GRPC client] OPEN client_req_id=%s start=%llu stop=%llu prune=%d chunk_hint=%u locator_hashes=%zu\n",
        client_request_id.c_str(),
        (unsigned long long)start_height,
        (unsigned long long)stop_height,
        prune ? 1 : 0,
        chunk_blocks_hint,
        chain_locator.size());

    p_->recv_thread = std::thread([this]() {
        cuprate::stream::v1::BlockChunk chunk;
        while (p_->reader->Read(&chunk)) {
            const auto t_recv = std::chrono::steady_clock::now();
            if (!p_->first_chunk_seen) {
                p_->t_first_chunk = t_recv;
                p_->first_chunk_seen = true;
                const double first_ms = std::chrono::duration<double, std::milli>(
                    t_recv - p_->t_open).count();
                std::fprintf(stderr,
                    "[GRPC client] FIRST_CHUNK_LATENCY_MS=%.1f client_req_id=%s server_req_id=%s\n",
                    first_ms,
                    p_->client_request_id.c_str(),
                    chunk.server_request_id().c_str());
            }

            chunk_record rec;
            rec.payload           = std::move(*chunk.mutable_payload());
            rec.start_height      = chunk.start_height();
            rec.chunk_seq         = chunk.chunk_seq();
            rec.chain_tip         = chunk.chain_tip();
            rec.n_blocks          = chunk.n_blocks();
            rec.payload_bytes     = chunk.payload_bytes();
            rec.server_request_id = chunk.server_request_id();

            std::unique_lock<std::mutex> lk(p_->mu);
            const auto t_enq0 = std::chrono::steady_clock::now();
            p_->cv_not_full.wait(lk, [this]() {
                return p_->queue.size() < queue_capacity() || p_->cancelled;
            });
            const double enq_wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_enq0).count();
            if (p_->cancelled) break;
            if (enq_wait_ms > 5.0) {
                ++p_->backpressure_waits;
                std::fprintf(stderr,
                    "[GRPC client] BACKPRESSURE seq=%llu enq_wait_ms=%.1f queue_full -- wallet slower than server\n",
                    (unsigned long long)rec.chunk_seq, enq_wait_ms);
            }

            const uint64_t total_chunks_now = p_->total_chunks + 1;
            const uint64_t total_blocks_now = p_->total_blocks + rec.n_blocks;
            const uint64_t total_bytes_now  = p_->total_bytes  + rec.payload_bytes;
            const double cum_ms = std::chrono::duration<double, std::milli>(
                t_recv - p_->t_open).count();
            const double cum_mbs = (cum_ms > 0.0)
                ? (static_cast<double>(total_bytes_now) / 1024.0 / 1024.0) / (cum_ms / 1000.0)
                : 0.0;
            std::fprintf(stderr,
                "[GRPC client] CHUNK seq=%llu start=%llu n_blocks=%u payload_bytes=%u cum_chunks=%llu cum_blocks=%llu cum_bytes=%llu cum_ms=%.1f cum_mbs=%.2f tip=%llu enq_wait_ms=%.1f client_req_id=%s server_req_id=%s\n",
                (unsigned long long)rec.chunk_seq,
                (unsigned long long)rec.start_height,
                rec.n_blocks,
                rec.payload_bytes,
                (unsigned long long)total_chunks_now,
                (unsigned long long)total_blocks_now,
                (unsigned long long)total_bytes_now,
                cum_ms,
                cum_mbs,
                (unsigned long long)rec.chain_tip,
                enq_wait_ms,
                p_->client_request_id.c_str(),
                rec.server_request_id.c_str());

            p_->total_chunks = total_chunks_now;
            p_->total_blocks = total_blocks_now;
            p_->total_bytes  = total_bytes_now;

            p_->queue.push_back(std::move(rec));
            lk.unlock();
            // Publish liveness timestamp read by the Qt GUI badge.
            last_successful_chunk_unix_ms().store(
                std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count(),
                std::memory_order_relaxed);
            p_->cv_not_empty.notify_one();
        }

        // Stream done — call Finish() to get final status.
        const grpc::Status status = p_->reader->Finish();
        std::lock_guard<std::mutex> lk(p_->mu);
        p_->done = true;
        if (status.ok()) {
            p_->stream_ended_ok = true;
        } else {
            p_->last_error_code = static_cast<int>(status.error_code());
            p_->last_error_message = status.error_message();
        }
        const double total_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - p_->t_open).count();
        const double avg_mbs = (total_ms > 0.0)
            ? (static_cast<double>(p_->total_bytes) / 1024.0 / 1024.0) / (total_ms / 1000.0)
            : 0.0;
        std::fprintf(stderr,
            "[GRPC client] CLOSE client_req_id=%s reason=%s chunks=%llu blocks=%llu bytes=%llu total_ms=%.1f avg_mbs=%.2f bp_waits=%llu grpc_code=%d msg='%s'\n",
            p_->client_request_id.c_str(),
            p_->stream_ended_ok ? "stream_end_ok"
              : (p_->cancelled ? "client_cancel" : "stream_error"),
            (unsigned long long)p_->total_chunks,
            (unsigned long long)p_->total_blocks,
            (unsigned long long)p_->total_bytes,
            total_ms,
            avg_mbs,
            (unsigned long long)p_->backpressure_waits,
            p_->last_error_code,
            p_->last_error_message.c_str());
        p_->cv_not_empty.notify_all();
    });

    return true;
}

bool cuprate_grpc_stream_client::next_chunk_payload(std::string& out_epee_bytes, uint32_t timeout_ms)
{
    std::unique_lock<std::mutex> lk(p_->mu);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    if (!p_->cv_not_empty.wait_until(lk, deadline, [this]() {
        return !p_->queue.empty() || p_->done || p_->cancelled;
    })) {
        // Timeout. Don't set last_error — caller may try again.
        return false;
    }
    if (p_->queue.empty()) {
        // done or cancelled with nothing left.
        return false;
    }
    chunk_record rec = std::move(p_->queue.front());
    p_->queue.pop_front();
    p_->last_chunk_start_height       = rec.start_height;
    p_->last_chunk_seq                = rec.chunk_seq;
    p_->last_chunk_chain_tip          = rec.chain_tip;
    p_->last_chunk_n_blocks           = rec.n_blocks;
    p_->last_chunk_payload_bytes      = rec.payload_bytes;
    p_->last_chunk_server_request_id  = rec.server_request_id;
    out_epee_bytes = std::move(rec.payload);
    lk.unlock();
    p_->cv_not_full.notify_one();
    return true;
}

bool cuprate_grpc_stream_client::stream_ended_ok() const
{
    std::lock_guard<std::mutex> lk(p_->mu);
    return p_->stream_ended_ok;
}

void cuprate_grpc_stream_client::close()
{
    {
        std::lock_guard<std::mutex> lk(p_->mu);
        if (!p_->reader && !p_->recv_thread.joinable()) return;
        p_->cancelled = true;
        if (p_->ctx) p_->ctx->TryCancel();
    }
    p_->cv_not_empty.notify_all();
    p_->cv_not_full.notify_all();
    if (p_->recv_thread.joinable()) p_->recv_thread.join();
    p_->reader.reset();
    p_->ctx.reset();
    {
        std::lock_guard<std::mutex> lk(p_->mu);
        p_->queue.clear();
    }
}

uint64_t    cuprate_grpc_stream_client::last_chunk_start_height()      const { return p_->last_chunk_start_height; }
uint64_t    cuprate_grpc_stream_client::last_chunk_seq()               const { return p_->last_chunk_seq; }
uint64_t    cuprate_grpc_stream_client::last_chunk_chain_tip()         const { return p_->last_chunk_chain_tip; }
uint32_t    cuprate_grpc_stream_client::last_chunk_n_blocks()          const { return p_->last_chunk_n_blocks; }
uint32_t    cuprate_grpc_stream_client::last_chunk_payload_bytes()     const { return p_->last_chunk_payload_bytes; }
std::string cuprate_grpc_stream_client::last_chunk_server_request_id() const { return p_->last_chunk_server_request_id; }

int         cuprate_grpc_stream_client::last_error_code()    const { return p_->last_error_code; }
std::string cuprate_grpc_stream_client::last_error_message() const { return p_->last_error_message; }

} // namespace cuprate_grpc_stream
