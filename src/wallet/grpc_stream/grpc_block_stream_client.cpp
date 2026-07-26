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
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <memory>
#include <mutex>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

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
    constexpr int DEFAULT_GRPC_HTTP2_BUFFER_BYTES = 64 * 1024 * 1024;
    constexpr int MIN_GRPC_HTTP2_BUFFER_BYTES = 16 * 1024 * 1024;
    constexpr int MAX_GRPC_HTTP2_BUFFER_BYTES = 64 * 1024 * 1024;
    constexpr int DEFAULT_GRPC_HTTP2_MAX_FRAME_BYTES = 16 * 1024 * 1024 - 1;
    constexpr int MIN_GRPC_HTTP2_MAX_FRAME_BYTES = 16 * 1024;
    constexpr int MAX_GRPC_HTTP2_MAX_FRAME_BYTES = 16 * 1024 * 1024 - 1;
    // Keep the production default under the operating system's automatic TCP
    // buffer policy.  Benchmarks may override one wallet gRPC channel without
    // changing a host-wide TCP sysctl (important on phones and shared hosts).
    constexpr int MAX_TCP_RECEIVE_BUFFER_BYTES = 32 * 1024 * 1024;

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

    int tcp_receive_buffer_bytes()
    {
        static const int bytes = []() -> int {
            const char *env = std::getenv("CUPRATE_GRPC_TCP_RECEIVE_BUFFER_BYTES");
            if (!env || !*env)
                return -1; // Omit the arg: gRPC leaves SO_RCVBUF to the OS.

            errno = 0;
            char *endptr = nullptr;
            const long parsed = std::strtol(env, &endptr, 10);
            if (errno != 0 || !endptr || *endptr != '\0' || parsed <= 0)
                return -1;

            return static_cast<int>(std::min(
                parsed, static_cast<long>(MAX_TCP_RECEIVE_BUFFER_BYTES)));
        }();
        return bytes;
    }

    bool bdp_probe_enabled()
    {
        static const bool enabled = []() -> bool {
            const char *env = std::getenv("CUPRATE_GRPC_BDP_PROBE");
            if (!env || !*env)
                return true;

            // Keep automatic BDP tuning as the production default.  A
            // benchmark can opt out on one process when it needs the
            // explicitly configured high-latency receive window from the
            // first HTTP/2 SETTINGS frame.
            return std::strcmp(env, "0") != 0;
        }();
        return enabled;
    }

    bool local_subchannel_pool_enabled()
    {
        static const bool enabled = []() -> bool {
            const char *env = std::getenv("CUPRATE_GRPC_LOCAL_SUBCHANNEL_POOL");
            // gRPC normally pools subchannels globally.  That is desirable
            // for the normal one-stream wallet, but it would silently turn a
            // range-pool's several Channel objects back into one TCP/HTTP2
            // connection.  Keep the override explicit and benchmark-only.
            return env && std::strcmp(env, "1") == 0;
        }();
        return enabled;
    }

    int http2_buffer_bytes()
    {
        static const int bytes = []() -> int {
            const char *env = std::getenv("CUPRATE_GRPC_HTTP2_LOOKAHEAD_BYTES");
            if (!env || !*env)
                return DEFAULT_GRPC_HTTP2_BUFFER_BYTES;

            errno = 0;
            char *endptr = nullptr;
            const long parsed = std::strtol(env, &endptr, 10);
            if (errno != 0 || !endptr || *endptr != '\0')
                return DEFAULT_GRPC_HTTP2_BUFFER_BYTES;

            // Only benchmark profiles may choose 16, 32 or 64 MiB.  The
            // bounded range prevents an environment setting from turning one
            // mobile stream into an unbounded receive-credit allocation.
            return static_cast<int>(std::max(
                static_cast<long>(MIN_GRPC_HTTP2_BUFFER_BYTES),
                std::min(parsed, static_cast<long>(MAX_GRPC_HTTP2_BUFFER_BYTES))));
        }();
        return bytes;
    }

    int http2_max_frame_bytes()
    {
        static const int bytes = []() -> int {
            const char *env = std::getenv("CUPRATE_GRPC_HTTP2_MAX_FRAME_BYTES");
            if (!env || !*env)
                return DEFAULT_GRPC_HTTP2_MAX_FRAME_BYTES;

            errno = 0;
            char *endptr = nullptr;
            const long parsed = std::strtol(env, &endptr, 10);
            if (errno != 0 || !endptr || *endptr != '\0')
                return DEFAULT_GRPC_HTTP2_MAX_FRAME_BYTES;

            // HTTP/2 allows 16,384 through 16,777,215 bytes.  This remains a
            // client-local benchmark dial; the product default is unchanged.
            return static_cast<int>(std::max(
                static_cast<long>(MIN_GRPC_HTTP2_MAX_FRAME_BYTES),
                std::min(parsed, static_cast<long>(MAX_GRPC_HTTP2_MAX_FRAME_BYTES))));
        }();
        return bytes;
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
    // Monotonic timestamps remain local to the client: they identify queue
    // residence precisely without attempting to compare clocks with TEX8.
    std::chrono::steady_clock::time_point enqueued_at;
    double       read_call_ms = 0.0;
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
    size_t                    queue_payload_bytes = 0;
    size_t                    max_queue_depth = 0;
    size_t                    max_queue_payload_bytes = 0;
    double                    total_queue_full_wait_ms = 0.0;
    double                    total_read_wait_ms = 0.0;
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
    double      last_chunk_read_call_ms = 0.0;
    double      last_chunk_queue_dwell_ms = 0.0;

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
    const int http2_buffer = http2_buffer_bytes();
    ch_args.SetInt(GRPC_ARG_HTTP2_STREAM_LOOKAHEAD_BYTES, http2_buffer);
    ch_args.SetInt(GRPC_ARG_HTTP2_WRITE_BUFFER_SIZE, http2_buffer);
    const int http2_max_frame = http2_max_frame_bytes();
    ch_args.SetInt(GRPC_ARG_HTTP2_MAX_FRAME_SIZE, http2_max_frame);
    const bool bdp_probe = bdp_probe_enabled();
    ch_args.SetInt(GRPC_ARG_HTTP2_BDP_PROBE, bdp_probe ? 1 : 0);
    const bool local_subchannel_pool = local_subchannel_pool_enabled();
    if (local_subchannel_pool)
        ch_args.SetInt(GRPC_ARG_USE_LOCAL_SUBCHANNEL_POOL, 1);
    const int tcp_receive_buffer = tcp_receive_buffer_bytes();
    if (tcp_receive_buffer > 0)
        ch_args.SetInt(GRPC_ARG_TCP_RECEIVE_BUFFER_SIZE, tcp_receive_buffer);
    p_->channel = grpc::CreateCustomChannel(target, grpc::InsecureChannelCredentials(), ch_args);
    p_->stub = cuprate::stream::v1::BlockStream::NewStub(p_->channel);
    std::fprintf(stderr,
        "[GRPC client] CONNECT target=%s queue_capacity=%zu max_msg=%d http2_buffer=%d max_frame=%d bdp_probe=%d local_subchannel_pool=%d tcp_rcvbuf=%d (channel created -- gRPC connect is lazy, first RPC opens TCP)\n",
        target.c_str(),
        queue_capacity(),
        MAX_GRPC_MESSAGE_BYTES,
        http2_buffer,
        http2_max_frame,
        bdp_probe ? 1 : 0,
        local_subchannel_pool ? 1 : 0,
        tcp_receive_buffer);
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
    {
        std::lock_guard<std::mutex> lk(p_->mu);
        p_->queue.clear();
        p_->queue_payload_bytes = 0;
        p_->max_queue_depth = 0;
        p_->max_queue_payload_bytes = 0;
        p_->total_queue_full_wait_ms = 0.0;
        p_->total_read_wait_ms = 0.0;
    }

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
        for (;;) {
            const auto t_read0 = std::chrono::steady_clock::now();
            const bool read_ok = p_->reader->Read(&chunk);
            const auto t_recv = std::chrono::steady_clock::now();
            const double read_wait_ms = std::chrono::duration<double, std::milli>(
                t_recv - t_read0).count();
            {
                std::lock_guard<std::mutex> lk(p_->mu);
                p_->total_read_wait_ms += read_wait_ms;
            }
            if (!read_ok)
                break;
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
            rec.read_call_ms      = read_wait_ms;

            std::unique_lock<std::mutex> lk(p_->mu);
            const auto t_enq0 = std::chrono::steady_clock::now();
            p_->cv_not_full.wait(lk, [this]() {
                return p_->queue.size() < queue_capacity() || p_->cancelled;
            });
            const double enq_wait_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_enq0).count();
            p_->total_queue_full_wait_ms += enq_wait_ms;
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

            rec.enqueued_at = std::chrono::steady_clock::now();
            p_->queue_payload_bytes += rec.payload.size();
            p_->queue.push_back(std::move(rec));
            p_->max_queue_depth = std::max(p_->max_queue_depth, p_->queue.size());
            p_->max_queue_payload_bytes = std::max(
                p_->max_queue_payload_bytes, p_->queue_payload_bytes);
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
            "[GRPC client] CLOSE client_req_id=%s reason=%s chunks=%llu blocks=%llu bytes=%llu total_ms=%.1f avg_mbs=%.2f bp_waits=%llu queue_full_wait_ms=%.1f read_wait_ms=%.1f queue_highwater_chunks=%zu queue_highwater_bytes=%zu grpc_code=%d msg='%s'\n",
            p_->client_request_id.c_str(),
            p_->stream_ended_ok ? "stream_end_ok"
              : (p_->cancelled ? "client_cancel" : "stream_error"),
            (unsigned long long)p_->total_chunks,
            (unsigned long long)p_->total_blocks,
            (unsigned long long)p_->total_bytes,
            total_ms,
            avg_mbs,
            (unsigned long long)p_->backpressure_waits,
            p_->total_queue_full_wait_ms,
            p_->total_read_wait_ms,
            p_->max_queue_depth,
            p_->max_queue_payload_bytes,
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
    const double queue_dwell_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - rec.enqueued_at).count();
    p_->queue_payload_bytes -= rec.payload.size();
    p_->last_chunk_start_height       = rec.start_height;
    p_->last_chunk_seq                = rec.chunk_seq;
    p_->last_chunk_chain_tip          = rec.chain_tip;
    p_->last_chunk_n_blocks           = rec.n_blocks;
    p_->last_chunk_payload_bytes      = rec.payload_bytes;
    p_->last_chunk_server_request_id  = rec.server_request_id;
    p_->last_chunk_read_call_ms       = rec.read_call_ms;
    p_->last_chunk_queue_dwell_ms     = queue_dwell_ms;
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
        p_->queue_payload_bytes = 0;
    }
}

uint64_t cuprate_grpc_stream_client::last_chunk_start_height() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->last_chunk_start_height; }
uint64_t cuprate_grpc_stream_client::last_chunk_seq() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->last_chunk_seq; }
uint64_t cuprate_grpc_stream_client::last_chunk_chain_tip() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->last_chunk_chain_tip; }
uint32_t cuprate_grpc_stream_client::last_chunk_n_blocks() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->last_chunk_n_blocks; }
uint32_t cuprate_grpc_stream_client::last_chunk_payload_bytes() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->last_chunk_payload_bytes; }
std::string cuprate_grpc_stream_client::last_chunk_server_request_id() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->last_chunk_server_request_id; }
double cuprate_grpc_stream_client::last_chunk_read_call_ms() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->last_chunk_read_call_ms; }
double cuprate_grpc_stream_client::last_chunk_queue_dwell_ms() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->last_chunk_queue_dwell_ms; }

size_t cuprate_grpc_stream_client::queue_depth() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->queue.size(); }
size_t cuprate_grpc_stream_client::queue_payload_bytes() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->queue_payload_bytes; }
size_t cuprate_grpc_stream_client::max_queue_depth() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->max_queue_depth; }
size_t cuprate_grpc_stream_client::max_queue_payload_bytes() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->max_queue_payload_bytes; }
double cuprate_grpc_stream_client::total_queue_full_wait_ms() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->total_queue_full_wait_ms; }
double cuprate_grpc_stream_client::total_read_wait_ms() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->total_read_wait_ms; }

int cuprate_grpc_stream_client::last_error_code() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->last_error_code; }
std::string cuprate_grpc_stream_client::last_error_message() const { std::lock_guard<std::mutex> lk(p_->mu); return p_->last_error_message; }

namespace {

size_t range_pool_connections()
{
    const char *env = std::getenv("CUPRATE_GRPC_RANGE_CONNECTIONS");
    if (!env || !*env)
        return 2;

    errno = 0;
    char *endptr = nullptr;
    const unsigned long parsed = std::strtoul(env, &endptr, 10);
    if (errno != 0 || !endptr || *endptr != '\0')
        return 2;
    // The range pool is an explicit desktop benchmark mode.  Sixteen lanes
    // matches the proven physical-connection fan-out used by the legacy fast
    // wallet tests, while the normal product path remains one stream.
    return static_cast<size_t>(std::max(2UL, std::min(parsed, 16UL)));
}

uint64_t range_pool_blocks(const char *env_name, uint64_t default_value,
                           uint64_t min_value, uint64_t max_value)
{
    const char *env = std::getenv(env_name);
    if (!env || !*env)
        return default_value;

    errno = 0;
    char *endptr = nullptr;
    const unsigned long long parsed = std::strtoull(env, &endptr, 10);
    if (errno != 0 || !endptr || *endptr != '\0' || parsed == 0)
        return default_value;
    return std::max(min_value, std::min(static_cast<uint64_t>(parsed), max_value));
}

uint64_t range_pool_bytes(const char *env_name, uint64_t default_value,
                          uint64_t min_value, uint64_t max_value)
{
    return range_pool_blocks(env_name, default_value, min_value, max_value);
}

bool range_pool_spool_enabled()
{
    const char *env = std::getenv("CUPRATE_GRPC_RANGE_SPOOL");
    return env && std::strcmp(env, "1") == 0;
}

uint64_t saturating_add(uint64_t value, uint64_t increment)
{
    return increment > std::numeric_limits<uint64_t>::max() - value
        ? std::numeric_limits<uint64_t>::max()
        : value + increment;
}

} // anonymous namespace

struct cuprate_grpc_range_pool::impl {
    enum class phase { inactive, bootstrap, lanes, ended, failed };

    struct lane {
        struct spooled_record {
            uint64_t start_height = 0;
            uint64_t source_seq = 0;
            uint64_t chain_tip = 0;
            uint32_t n_blocks = 0;
            uint32_t payload_bytes = 0;
            std::string server_request_id;
            double read_call_ms = 0.0;
            double queue_dwell_ms = 0.0;
            uint64_t file_offset = 0;
            uint64_t file_bytes = 0;
        };

        std::unique_ptr<cuprate_grpc_stream_client> client;
        uint64_t start = 0;
        uint64_t stop = 0;
        std::thread spool_thread;
        std::mutex spool_mu;
        std::condition_variable spool_cv;
        std::deque<spooled_record> spool_records;
        bool spool_done = false;
        bool spool_ended_ok = false;
        int spool_error_code = 0;
        std::string spool_error_message;
        std::string spool_path;
        std::FILE *spool_writer = nullptr;
        std::FILE *spool_reader = nullptr;
        uint64_t spool_write_offset = 0;
    };

    std::string target;
    std::string client_request_id;
    uint32_t chunk_blocks_hint = 0;
    bool prune = true;
    uint64_t requested_stop = 0;
    size_t connections = 2;
    uint64_t bootstrap_blocks = 10000;
    uint64_t range_blocks = 40000;
    bool spool_enabled = false;
    std::string spool_directory;
    uint64_t max_spool_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    std::mutex spool_budget_mu;
    std::condition_variable spool_budget_cv;
    uint64_t spool_bytes = 0;
    uint64_t max_spool_bytes_seen = 0;
    bool shutting_down = false;
    phase state = phase::inactive;

    std::unique_ptr<cuprate_grpc_stream_client> bootstrap;
    // A deque keeps lane addresses stable after a spooling thread captures
    // one.  A vector would need to move mutexes/condition variables when it
    // grows, which is both invalid and unsafe for active receiver threads.
    std::deque<lane> lanes;
    size_t consume_lane = 0;
    uint64_t next_expected_height = 0;
    uint64_t next_wave_start = 0;
    uint64_t snapshot_tip = 0; // inclusive top height captured by bootstrap
    uint64_t wave_number = 0;
    uint64_t delivered_seq = 0;
    bool stream_ended_ok = false;
    int last_error_code = 0;
    std::string last_error_message;

    uint64_t last_chunk_start_height = 0;
    uint64_t last_chunk_seq = 0;
    uint64_t last_chunk_chain_tip = 0;
    uint32_t last_chunk_n_blocks = 0;
    uint32_t last_chunk_payload_bytes = 0;
    std::string last_chunk_server_request_id;
    double last_chunk_read_call_ms = 0.0;
    double last_chunk_queue_dwell_ms = 0.0;

    void copy_last(const cuprate_grpc_stream_client &source)
    {
        last_chunk_start_height = source.last_chunk_start_height();
        last_chunk_seq = delivered_seq++;
        last_chunk_chain_tip = source.last_chunk_chain_tip();
        last_chunk_n_blocks = source.last_chunk_n_blocks();
        last_chunk_payload_bytes = source.last_chunk_payload_bytes();
        last_chunk_server_request_id = source.last_chunk_server_request_id();
        last_chunk_read_call_ms = source.last_chunk_read_call_ms();
        last_chunk_queue_dwell_ms = source.last_chunk_queue_dwell_ms();
    }

    bool fail(int code, const std::string &message)
    {
        last_error_code = code;
        last_error_message = message;
        state = phase::failed;
        stream_ended_ok = false;
        {
            std::lock_guard<std::mutex> lock(spool_budget_mu);
            shutting_down = true;
        }
        spool_budget_cv.notify_all();
        std::fprintf(stderr, "[GRPC range_pool] FAIL code=%d msg=%s -- cancelling all lanes\n",
            code, message.c_str());
        if (bootstrap)
            bootstrap->close();
        for (lane &entry : lanes)
            entry.client->close();
        return false;
    }

    static bool write_all(std::FILE *file, const char *data, size_t bytes)
    {
        while (bytes != 0) {
            const size_t written = std::fwrite(data, 1, bytes, file);
            if (written == 0)
                return false;
            data += written;
            bytes -= written;
        }
        return std::fflush(file) == 0;
    }

    static bool read_all(std::FILE *file, char *data, size_t bytes)
    {
        while (bytes != 0) {
            const size_t read = std::fread(data, 1, bytes, file);
            if (read == 0)
                return false;
            data += read;
            bytes -= read;
        }
        return true;
    }

    void release_spool_bytes(uint64_t bytes)
    {
        {
            std::lock_guard<std::mutex> lock(spool_budget_mu);
            spool_bytes = bytes > spool_bytes ? 0 : spool_bytes - bytes;
        }
        spool_budget_cv.notify_all();
    }

    void start_spooler(lane &entry)
    {
        entry.spool_writer = std::fopen(entry.spool_path.c_str(), "w+b");
        entry.spool_reader = std::fopen(entry.spool_path.c_str(), "rb");
        if (!entry.spool_writer || !entry.spool_reader) {
            if (entry.spool_writer) std::fclose(entry.spool_writer);
            if (entry.spool_reader) std::fclose(entry.spool_reader);
            entry.spool_writer = nullptr;
            entry.spool_reader = nullptr;
            fail(-3, "cannot create range spool file: " + entry.spool_path);
            return;
        }

        entry.spool_thread = std::thread([this, &entry]() {
            for (;;) {
                std::string payload;
                if (!entry.client->next_chunk_payload(payload, 60000)) {
                    {
                        std::lock_guard<std::mutex> lock(entry.spool_mu);
                        entry.spool_done = true;
                        entry.spool_ended_ok = entry.client->stream_ended_ok();
                        entry.spool_error_code = entry.client->last_error_code();
                        entry.spool_error_message = entry.client->last_error_message();
                    }
                    entry.spool_cv.notify_all();
                    return;
                }

                lane::spooled_record record;
                record.start_height = entry.client->last_chunk_start_height();
                record.source_seq = entry.client->last_chunk_seq();
                record.chain_tip = entry.client->last_chunk_chain_tip();
                record.n_blocks = entry.client->last_chunk_n_blocks();
                record.payload_bytes = entry.client->last_chunk_payload_bytes();
                record.server_request_id = entry.client->last_chunk_server_request_id();
                record.read_call_ms = entry.client->last_chunk_read_call_ms();
                record.queue_dwell_ms = entry.client->last_chunk_queue_dwell_ms();
                record.file_bytes = payload.size();

                {
                    std::unique_lock<std::mutex> lock(spool_budget_mu);
                    spool_budget_cv.wait(lock, [this, &payload]() {
                        return shutting_down
                            || payload.size() <= max_spool_bytes - spool_bytes;
                    });
                    if (shutting_down)
                        return;
                    spool_bytes += payload.size();
                    max_spool_bytes_seen = std::max(max_spool_bytes_seen, spool_bytes);
                }

                record.file_offset = entry.spool_write_offset;
                const bool written = write_all(entry.spool_writer,
                    payload.data(), payload.size());
                if (!written) {
                    release_spool_bytes(record.file_bytes);
                    {
                        std::lock_guard<std::mutex> lock(entry.spool_mu);
                        entry.spool_done = true;
                        entry.spool_ended_ok = false;
                        entry.spool_error_code = -3;
                        entry.spool_error_message = "range spool write failed";
                    }
                    // A spooling failure cannot safely leave future ranges
                    // running: cancel all transports and let the ordered
                    // consumer fail over through the existing wallet path.
                    for (lane &other : lanes)
                        other.client->close();
                    entry.spool_cv.notify_all();
                    return;
                }
                entry.spool_write_offset += record.file_bytes;
                {
                    std::lock_guard<std::mutex> lock(entry.spool_mu);
                    entry.spool_records.push_back(std::move(record));
                }
                entry.spool_cv.notify_one();
            }
        });
    }

    bool pop_spooled(lane &entry, std::string &payload, uint32_t timeout_ms,
                     bool &stream_done)
    {
        stream_done = false;
        lane::spooled_record record;
        {
            std::unique_lock<std::mutex> lock(entry.spool_mu);
            const auto deadline = std::chrono::steady_clock::now()
                + std::chrono::milliseconds(timeout_ms);
            if (!entry.spool_cv.wait_until(lock, deadline, [&entry]() {
                    return !entry.spool_records.empty() || entry.spool_done;
                }))
                return false;
            if (entry.spool_records.empty()) {
                stream_done = entry.spool_done;
                return false;
            }
            record = std::move(entry.spool_records.front());
            entry.spool_records.pop_front();
        }

        payload.resize(static_cast<size_t>(record.file_bytes));
        const bool readable = std::fseek(entry.spool_reader,
                static_cast<long>(record.file_offset), SEEK_SET) == 0
            && read_all(entry.spool_reader, payload.data(), payload.size());
        release_spool_bytes(record.file_bytes);
        if (!readable) {
            payload.clear();
            std::lock_guard<std::mutex> lock(entry.spool_mu);
            entry.spool_done = true;
            entry.spool_ended_ok = false;
            entry.spool_error_code = -3;
            entry.spool_error_message = "range spool read failed";
            return false;
        }

        last_chunk_start_height = record.start_height;
        last_chunk_seq = delivered_seq++;
        last_chunk_chain_tip = record.chain_tip;
        last_chunk_n_blocks = record.n_blocks;
        last_chunk_payload_bytes = record.payload_bytes;
        last_chunk_server_request_id = record.server_request_id;
        last_chunk_read_call_ms = record.read_call_ms;
        last_chunk_queue_dwell_ms = record.queue_dwell_ms;
        return true;
    }

    bool lane_spool_ended_ok(lane &entry, int &code, std::string &message)
    {
        std::lock_guard<std::mutex> lock(entry.spool_mu);
        code = entry.spool_error_code;
        message = entry.spool_error_message;
        return entry.spool_done && entry.spool_ended_ok;
    }

    void close_lanes()
    {
        for (lane &entry : lanes)
            entry.client->close();
        for (lane &entry : lanes) {
            if (entry.spool_thread.joinable())
                entry.spool_thread.join();
            if (entry.spool_writer) std::fclose(entry.spool_writer);
            if (entry.spool_reader) std::fclose(entry.spool_reader);
            entry.spool_writer = nullptr;
            entry.spool_reader = nullptr;
            if (!entry.spool_path.empty())
                std::remove(entry.spool_path.c_str());
        }
        lanes.clear();
    }

    bool open_next_wave()
    {
        if (next_wave_start > snapshot_tip) {
            state = phase::ended;
            stream_ended_ok = true;
            return true;
        }

        close_lanes();
        consume_lane = 0;
        uint64_t range_start = next_wave_start;
        for (size_t lane_index = 0;
             lane_index < connections && range_start <= snapshot_tip;
             ++lane_index)
        {
            const uint64_t range_end = std::min(
                snapshot_tip, saturating_add(range_start, range_blocks - 1));
            lanes.emplace_back();
            lane &entry = lanes.back();
            entry.start = range_start;
            entry.stop = range_end;
            entry.client = std::make_unique<cuprate_grpc_stream_client>();
            if (!entry.client->connect(target))
                return fail(entry.client->last_error_code(),
                    "range lane connect failed: " + entry.client->last_error_message());

            std::ostringstream lane_id;
            lane_id << client_request_id << "-w" << wave_number << "-l" << lane_index
                    << "-h" << range_start;
            if (!entry.client->open_stream(range_start, range_end, prune,
                    chunk_blocks_hint, lane_id.str()))
                return fail(entry.client->last_error_code(),
                    "range lane open failed: " + entry.client->last_error_message());
            range_start = saturating_add(range_end, 1);
        }
        next_wave_start = range_start;
        ++wave_number;
        if (spool_enabled) {
            for (size_t lane_index = 0; lane_index < lanes.size(); ++lane_index) {
                lanes[lane_index].spool_path = spool_directory + "/lane-"
                    + std::to_string(wave_number - 1) + "-"
                    + std::to_string(lane_index) + ".bin";
                start_spooler(lanes[lane_index]);
                if (state == phase::failed)
                    return false;
            }
        }
        state = phase::lanes;
        std::fprintf(stderr,
            "[GRPC range_pool] WAVE open=%llu lanes=%zu first=%llu next_wave=%llu snapshot_tip=%llu range_blocks=%llu spool=%d spool_limit_bytes=%llu\n",
            (unsigned long long)(wave_number - 1), lanes.size(),
            (unsigned long long)lanes.front().start,
            (unsigned long long)next_wave_start,
            (unsigned long long)snapshot_tip,
            (unsigned long long)range_blocks,
            spool_enabled ? 1 : 0,
            (unsigned long long)max_spool_bytes);
        return true;
    }
};

cuprate_grpc_range_pool::cuprate_grpc_range_pool()
  : p_(std::make_unique<impl>())
{}

cuprate_grpc_range_pool::~cuprate_grpc_range_pool()
{
    close();
}

bool cuprate_grpc_range_pool::connect(const std::string& target)
{
    if (target.empty()) {
        p_->last_error_code = -1;
        p_->last_error_message = "empty target";
        return false;
    }
    p_->target = target;
    return true;
}

bool cuprate_grpc_range_pool::open_stream(uint64_t start_height,
                                          uint64_t stop_height,
                                          bool prune,
                                          uint32_t chunk_blocks_hint,
                                          const std::string& client_request_id,
                                          const std::vector<std::string>& chain_locator)
{
    if (p_->target.empty()) {
        p_->last_error_code = -1;
        p_->last_error_message = "open_stream: connect() not called";
        return false;
    }

    close();
    p_->connections = range_pool_connections();
    p_->bootstrap_blocks = range_pool_blocks(
        "CUPRATE_GRPC_RANGE_BOOTSTRAP_BLOCKS", 10000, 1000, 20000);
    p_->range_blocks = range_pool_blocks(
        "CUPRATE_GRPC_RANGE_BLOCKS", 40000, 10000, 100000);
    p_->spool_enabled = range_pool_spool_enabled();
    p_->max_spool_bytes = range_pool_bytes(
        "CUPRATE_GRPC_RANGE_MAX_SPOOL_BYTES",
        4ULL * 1024ULL * 1024ULL * 1024ULL,
        512ULL * 1024ULL * 1024ULL,
        8ULL * 1024ULL * 1024ULL * 1024ULL);
    {
        std::lock_guard<std::mutex> lock(p_->spool_budget_mu);
        p_->spool_bytes = 0;
        p_->max_spool_bytes_seen = 0;
        p_->shutting_down = false;
    }
    p_->spool_directory.clear();
    if (p_->spool_enabled) {
        const char *root = std::getenv("CUPRATE_GRPC_RANGE_SPOOL_DIR");
        const std::string spool_root = (root && *root) ? root : "/tmp";
        p_->spool_directory = spool_root + "/cuprate-grpc-range-pool-"
            + std::to_string(static_cast<unsigned long long>(::getpid()));
        if (::mkdir(p_->spool_directory.c_str(), 0700) != 0) {
            p_->last_error_code = -3;
            p_->last_error_message = "cannot create range spool directory: "
                + p_->spool_directory;
            return false;
        }
    }
    p_->client_request_id = client_request_id;
    p_->chunk_blocks_hint = chunk_blocks_hint;
    p_->prune = prune;
    p_->requested_stop = stop_height;
    p_->next_expected_height = start_height;
    p_->next_wave_start = start_height;
    p_->snapshot_tip = 0;
    p_->wave_number = 0;
    p_->delivered_seq = 0;
    p_->stream_ended_ok = false;
    p_->last_error_code = 0;
    p_->last_error_message.clear();

    const uint64_t bootstrap_stop_unbounded = saturating_add(
        start_height, p_->bootstrap_blocks - 1);
    const uint64_t bootstrap_stop = stop_height == 0
        ? bootstrap_stop_unbounded
        : std::min(stop_height, bootstrap_stop_unbounded);
    p_->bootstrap = std::make_unique<cuprate_grpc_stream_client>();
    if (!p_->bootstrap->connect(p_->target))
        return p_->fail(p_->bootstrap->last_error_code(),
            "bootstrap connect failed: " + p_->bootstrap->last_error_message());
    if (!p_->bootstrap->open_stream(start_height, bootstrap_stop, prune,
            chunk_blocks_hint, client_request_id + "-bootstrap", chain_locator))
        return p_->fail(p_->bootstrap->last_error_code(),
            "bootstrap open failed: " + p_->bootstrap->last_error_message());
    p_->state = impl::phase::bootstrap;
    std::fprintf(stderr,
        "[GRPC range_pool] OPEN id=%s start=%llu bootstrap_stop=%llu connections=%zu range_blocks=%llu spool=%d spool_dir=%s spool_limit_bytes=%llu queue_capacity_is_per_lane\n",
        client_request_id.c_str(), (unsigned long long)start_height,
        (unsigned long long)bootstrap_stop, p_->connections,
        (unsigned long long)p_->range_blocks, p_->spool_enabled ? 1 : 0,
        p_->spool_enabled ? p_->spool_directory.c_str() : "<disabled>",
        (unsigned long long)p_->max_spool_bytes);
    return true;
}

bool cuprate_grpc_range_pool::next_chunk_payload(std::string& out_epee_bytes,
                                                  uint32_t timeout_ms)
{
    for (;;) {
        if (p_->state == impl::phase::bootstrap) {
            if (p_->bootstrap->next_chunk_payload(out_epee_bytes, timeout_ms)) {
                p_->copy_last(*p_->bootstrap);
                const uint64_t chunk_end = saturating_add(
                    p_->last_chunk_start_height, p_->last_chunk_n_blocks);
                if (p_->last_chunk_start_height != p_->next_expected_height
                    || p_->last_chunk_n_blocks == 0)
                    return p_->fail(-2, "bootstrap contiguity violation");
                p_->next_expected_height = chunk_end;
                p_->next_wave_start = chunk_end;
                p_->snapshot_tip = p_->last_chunk_chain_tip;
                if (p_->requested_stop != 0)
                    p_->snapshot_tip = std::min(p_->snapshot_tip, p_->requested_stop);
                return true;
            }
            if (!p_->bootstrap->stream_ended_ok())
                return p_->fail(p_->bootstrap->last_error_code(),
                    "bootstrap terminated: " + p_->bootstrap->last_error_message());
            p_->bootstrap->close();
            p_->bootstrap.reset();
            if (p_->next_wave_start > p_->snapshot_tip) {
                p_->state = impl::phase::ended;
                p_->stream_ended_ok = true;
                return false;
            }
            if (!p_->open_next_wave())
                return false;
            continue;
        }

        if (p_->state == impl::phase::lanes) {
            if (p_->consume_lane >= p_->lanes.size()) {
                if (p_->next_wave_start > p_->snapshot_tip) {
                    p_->state = impl::phase::ended;
                    p_->stream_ended_ok = true;
                    return false;
                }
                if (!p_->open_next_wave())
                    return false;
                continue;
            }

            impl::lane &lane = p_->lanes[p_->consume_lane];
            bool lane_done = false;
            const bool lane_payload = p_->spool_enabled
                ? p_->pop_spooled(lane, out_epee_bytes, timeout_ms, lane_done)
                : lane.client->next_chunk_payload(out_epee_bytes, timeout_ms);
            if (lane_payload) {
                if (!p_->spool_enabled)
                    p_->copy_last(*lane.client);
                const uint64_t chunk_end = saturating_add(
                    p_->last_chunk_start_height, p_->last_chunk_n_blocks);
                if (p_->last_chunk_start_height != p_->next_expected_height
                    || p_->last_chunk_n_blocks == 0
                    || chunk_end > saturating_add(lane.stop, 1))
                    return p_->fail(-2, "range lane contiguity or boundary violation");
                p_->next_expected_height = chunk_end;
                std::fprintf(stderr,
                    "[GRPC range_pool] CHUNK global_seq=%llu lane=%zu start=%llu blocks=%u expected_next=%llu\n",
                    (unsigned long long)p_->last_chunk_seq, p_->consume_lane,
                    (unsigned long long)p_->last_chunk_start_height,
                    p_->last_chunk_n_blocks,
                    (unsigned long long)p_->next_expected_height);
                return true;
            }
            int lane_error_code = 0;
            std::string lane_error_message;
            const bool lane_ended_ok = p_->spool_enabled
                ? p_->lane_spool_ended_ok(lane, lane_error_code, lane_error_message)
                : lane.client->stream_ended_ok();
            if (!lane_ended_ok) {
                if (!p_->spool_enabled) {
                    lane_error_code = lane.client->last_error_code();
                    lane_error_message = lane.client->last_error_message();
                }
                return p_->fail(lane_error_code,
                    "range lane terminated: " + lane_error_message);
            }
            if (p_->next_expected_height != saturating_add(lane.stop, 1))
                return p_->fail(-2, "range lane ended before its fixed boundary");
            lane.client->close();
            ++p_->consume_lane;
            continue;
        }

        return false;
    }
}

bool cuprate_grpc_range_pool::stream_ended_ok() const { return p_->stream_ended_ok; }

void cuprate_grpc_range_pool::close()
{
    const bool had_spool = p_->spool_enabled;
    const uint64_t spool_highwater = p_->max_spool_bytes_seen;
    {
        std::lock_guard<std::mutex> lock(p_->spool_budget_mu);
        p_->shutting_down = true;
    }
    p_->spool_budget_cv.notify_all();
    if (p_->bootstrap)
        p_->bootstrap->close();
    p_->close_lanes();
    p_->bootstrap.reset();
    if (!p_->spool_directory.empty())
        ::rmdir(p_->spool_directory.c_str());
    p_->spool_directory.clear();
    if (had_spool) {
        std::fprintf(stderr,
            "[GRPC range_pool] SPOOL_CLOSE highwater_bytes=%llu retained_bytes=%llu files_removed=1\n",
            (unsigned long long)spool_highwater,
            (unsigned long long)p_->spool_bytes);
    }
    // close() is intentionally idempotent and is reached both from wallet2
    // and the range-pool destructor.  The files above are already gone, so do
    // not emit a second, misleading cleanup record on that destructor path.
    p_->spool_enabled = false;
    if (p_->state != impl::phase::failed)
        p_->state = impl::phase::inactive;
}

uint64_t cuprate_grpc_range_pool::last_chunk_start_height() const { return p_->last_chunk_start_height; }
uint64_t cuprate_grpc_range_pool::last_chunk_seq() const { return p_->last_chunk_seq; }
uint64_t cuprate_grpc_range_pool::last_chunk_chain_tip() const { return p_->last_chunk_chain_tip; }
uint32_t cuprate_grpc_range_pool::last_chunk_n_blocks() const { return p_->last_chunk_n_blocks; }
uint32_t cuprate_grpc_range_pool::last_chunk_payload_bytes() const { return p_->last_chunk_payload_bytes; }
std::string cuprate_grpc_range_pool::last_chunk_server_request_id() const { return p_->last_chunk_server_request_id; }
double cuprate_grpc_range_pool::last_chunk_read_call_ms() const { return p_->last_chunk_read_call_ms; }
double cuprate_grpc_range_pool::last_chunk_queue_dwell_ms() const { return p_->last_chunk_queue_dwell_ms; }

size_t cuprate_grpc_range_pool::queue_depth() const
{
    size_t value = p_->bootstrap ? p_->bootstrap->queue_depth() : 0;
    for (const impl::lane &lane : p_->lanes) value += lane.client->queue_depth();
    return value;
}
size_t cuprate_grpc_range_pool::queue_payload_bytes() const
{
    size_t value = p_->bootstrap ? p_->bootstrap->queue_payload_bytes() : 0;
    for (const impl::lane &lane : p_->lanes) value += lane.client->queue_payload_bytes();
    return value;
}
size_t cuprate_grpc_range_pool::max_queue_depth() const
{
    size_t value = p_->bootstrap ? p_->bootstrap->max_queue_depth() : 0;
    for (const impl::lane &lane : p_->lanes) value += lane.client->max_queue_depth();
    return value;
}
size_t cuprate_grpc_range_pool::max_queue_payload_bytes() const
{
    size_t value = p_->bootstrap ? p_->bootstrap->max_queue_payload_bytes() : 0;
    for (const impl::lane &lane : p_->lanes) value += lane.client->max_queue_payload_bytes();
    return value;
}
double cuprate_grpc_range_pool::total_queue_full_wait_ms() const
{
    double value = p_->bootstrap ? p_->bootstrap->total_queue_full_wait_ms() : 0.0;
    for (const impl::lane &lane : p_->lanes) value += lane.client->total_queue_full_wait_ms();
    return value;
}
double cuprate_grpc_range_pool::total_read_wait_ms() const
{
    double value = p_->bootstrap ? p_->bootstrap->total_read_wait_ms() : 0.0;
    for (const impl::lane &lane : p_->lanes) value += lane.client->total_read_wait_ms();
    return value;
}
int cuprate_grpc_range_pool::last_error_code() const { return p_->last_error_code; }
std::string cuprate_grpc_range_pool::last_error_message() const { return p_->last_error_message; }

} // namespace cuprate_grpc_stream
