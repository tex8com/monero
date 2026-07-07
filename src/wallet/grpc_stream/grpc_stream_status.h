// Tiny header exposing the gRPC-stream liveness timestamp without pulling in
// the full wallet2.h (which would drag in all of Boost and epee into every
// including translation unit). Both wallet2.cpp (which publishes) and the
// Qt Wallet wrapper (which reads for the GUI badge) include only this.

#pragma once

#include <atomic>
#include <cstdint>

namespace cuprate_grpc_stream {

// Unix ms timestamp of the last successful gRPC-stream chunk pop. 0 means
// no chunk has ever arrived. Readers compare against "now - 10 000 ms" to
// decide "stream live right now".
std::atomic<int64_t>& last_successful_chunk_unix_ms();

} // namespace cuprate_grpc_stream
