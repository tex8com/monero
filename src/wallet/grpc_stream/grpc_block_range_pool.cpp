#include "grpc_block_range_pool.h"
#include "grpc_block_stream_client.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cuprate_grpc_stream {
namespace {
constexpr size_t kMaxChannels = 8;
constexpr uint32_t kMinRangeBlocks = 16;
constexpr uint32_t kMaxRangeBlocks = 512;
}

struct cuprate_grpc_block_range_pool::impl {
    struct slot {
        std::unique_ptr<cuprate_grpc_stream_client> client;
        uint64_t start = 0;
        uint64_t stop = 0;
        bool active = false;
    };
    std::string target, request_id, error;
    std::vector<std::string> locator;
    std::vector<slot> slots;
    uint64_t next_expected = 0, next_assign = 0, stop_height = 0;
    uint32_t range_blocks = 0;
    size_t target_channels = 1;
    bool opened = false, warmup_complete = false, ended_ok = false;
    int error_code = 0;
    uint64_t last_start = 0, last_seq = 0, last_tip = 0;
    uint32_t last_blocks = 0, last_payload_bytes = 0;
    std::string last_server_id;
};

cuprate_grpc_block_range_pool::cuprate_grpc_block_range_pool() : p_(std::make_unique<impl>()) {}
cuprate_grpc_block_range_pool::~cuprate_grpc_block_range_pool() { close(); }

namespace {
bool open_slot(cuprate_grpc_block_range_pool::impl& p, size_t index, bool use_locator)
{
    if (p.next_assign > p.stop_height) return true;
    if (index >= p.slots.size()) p.slots.resize(index + 1);
    auto& slot = p.slots[index];
    slot.client = std::make_unique<cuprate_grpc_stream_client>();
    slot.start = p.next_assign;
    slot.stop = std::min(p.stop_height, slot.start + static_cast<uint64_t>(p.range_blocks) - 1);
    p.next_assign = slot.stop + 1;
    if (!slot.client->connect(p.target) || !slot.client->open_stream(
            slot.start, slot.stop, true, p.range_blocks,
            p.request_id + "-c" + std::to_string(index),
            use_locator ? p.locator : std::vector<std::string>{}, 1)) {
        p.error_code = slot.client->last_error_code();
        p.error = slot.client->last_error_message();
        return false;
    }
    slot.active = true;
    std::fprintf(stderr, "[GRPC pool] OPEN slot=%zu range=[%llu..%llu] locator=%d\n", index,
        (unsigned long long)slot.start, (unsigned long long)slot.stop, use_locator ? 1 : 0);
    return true;
}

bool fill_slots(cuprate_grpc_block_range_pool::impl& p)
{
    for (size_t i = 0; i < p.target_channels && p.next_assign <= p.stop_height; ++i) {
        if (i >= p.slots.size() || !p.slots[i].active) {
            if (!open_slot(p, i, false)) return false;
        }
    }
    return true;
}
}

bool cuprate_grpc_block_range_pool::open(const std::string& target, uint64_t start_height,
    uint64_t stop_height, uint32_t range_blocks, size_t max_channels,
    const std::string& client_request_id, const std::vector<std::string>& chain_locator)
{
    close();
    if (target.empty() || stop_height < start_height) {
        p_->error_code = -1; p_->error = "invalid finite range pool request"; return false;
    }
    p_->target = target; p_->request_id = client_request_id; p_->locator = chain_locator;
    p_->stop_height = stop_height;
    p_->range_blocks = std::max(kMinRangeBlocks, std::min(kMaxRangeBlocks, range_blocks));
    p_->target_channels = std::max<size_t>(1, std::min(kMaxChannels, max_channels));
    p_->next_assign = start_height;
    p_->next_expected = 0;
    p_->error.clear(); p_->error_code = 0; p_->ended_ok = false;
    p_->last_start = p_->last_seq = p_->last_tip = 0;
    p_->last_blocks = p_->last_payload_bytes = 0;
    p_->last_server_id.clear();
    p_->opened = true;
    // The locator is used only by this warm-up range. Once its range ends, all
    // later requests are explicit, non-overlapping height ranges.
    if (!open_slot(*p_, 0, true)) { close(); return false; }
    return true;
}

bool cuprate_grpc_block_range_pool::next_chunk_payload(std::string& out, uint32_t timeout_ms)
{
    if (!p_->opened) { p_->error_code = -1; p_->error = "pool not open"; return false; }
    if (p_->warmup_complete && !fill_slots(*p_)) { close(); return false; }
    impl::slot* wanted = nullptr;
    for (auto& slot : p_->slots) {
        if (slot.active && (p_->next_expected == 0 || (slot.start <= p_->next_expected && p_->next_expected <= slot.stop))) {
            wanted = &slot; break;
        }
    }
    if (!wanted) { p_->ended_ok = p_->next_assign > p_->stop_height; return false; }
    if (!wanted->client->next_chunk_payload(out, timeout_ms)) {
        p_->error_code = wanted->client->last_error_code();
        p_->error = wanted->client->last_error_message();
        return false;
    }
    p_->last_start = wanted->client->last_chunk_start_height();
    p_->last_seq = wanted->client->last_chunk_seq();
    p_->last_tip = wanted->client->last_chunk_chain_tip();
    p_->last_blocks = wanted->client->last_chunk_n_blocks();
    p_->last_payload_bytes = wanted->client->last_chunk_payload_bytes();
    p_->last_server_id = wanted->client->last_chunk_server_request_id();
    if (p_->next_expected == 0) p_->next_expected = p_->last_start;
    const uint64_t end = p_->last_start + p_->last_blocks;
    if (p_->last_start != p_->next_expected || p_->last_blocks == 0 || end > wanted->stop + 1) {
        p_->error_code = -1; p_->error = "range gap, overlap, or invalid chunk"; return false;
    }
    p_->next_expected = end;
    if (end == wanted->stop + 1) {
        wanted->client->close(); wanted->active = false;
        p_->warmup_complete = true;
    }
    std::fprintf(stderr, "[GRPC pool] CHUNK start=%llu blocks=%u expected_next=%llu slots=%zu\n",
        (unsigned long long)p_->last_start, p_->last_blocks,
        (unsigned long long)p_->next_expected, p_->target_channels);
    return true;
}

void cuprate_grpc_block_range_pool::close() { for (auto& s : p_->slots) if (s.client) s.client->close(); p_->slots.clear(); p_->opened = false; }
bool cuprate_grpc_block_range_pool::stream_ended_ok() const { return p_->ended_ok; }
int cuprate_grpc_block_range_pool::last_error_code() const { return p_->error_code; }
std::string cuprate_grpc_block_range_pool::last_error_message() const { return p_->error; }
uint64_t cuprate_grpc_block_range_pool::last_chunk_start_height() const { return p_->last_start; }
uint64_t cuprate_grpc_block_range_pool::last_chunk_seq() const { return p_->last_seq; }
uint64_t cuprate_grpc_block_range_pool::last_chunk_chain_tip() const { return p_->last_tip; }
uint32_t cuprate_grpc_block_range_pool::last_chunk_n_blocks() const { return p_->last_blocks; }
uint32_t cuprate_grpc_block_range_pool::last_chunk_payload_bytes() const { return p_->last_payload_bytes; }
std::string cuprate_grpc_block_range_pool::last_chunk_server_request_id() const { return p_->last_server_id; }
} // namespace cuprate_grpc_stream
