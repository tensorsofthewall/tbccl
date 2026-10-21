#pragma once

// Test/diagnostic accessors into a Communicator's runtime, private to libtbccl (NOT installed, not public API).

#include <tbccl/async_transfer.hpp>
#include <tbccl/communicator.hpp>
#include <tbccl/transport.hpp>

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace tbccl::detail
{

// Ranks this communicator currently has an established DATA connection to (world_size 2: its one peer from bootstrap; above that, only what lazy
// establishment has connected so far). The control plane is always a full mesh and is not reported here.
std::vector<std::size_t> debug_connected_data_peers(const Communicator &comm);

// Test hook: the next `count` P2P admissions on this communicator fail as if allocation failed (ErrorCode::ResourceExhausted).
void debug_fail_next_admissions(const Communicator &comm, int count);

// Test hook: while paused, no peer lane starts a queued P2P transfer (the progress threads idle). Submission must keep working; resuming lets everything run.
void debug_set_progress_paused(const Communicator &comm, bool paused);

// The N=2 all_gather exchange (send and receive in flight together on the worker's two duplex lanes), exposed so a test can drive it over a controlled
// transport. `fatal` is the communicator's non-blocking abort request; it is called as soon as a child fails or the second admission throws. If it throws, or the reason cannot be built, the legs are interrupted directly. The exchange never returns or throws while an admitted leg is still running.
void run_n2_exchange(
    TensorCommWorker &worker, Transport &transport, AsyncMemoryBackend &send_backend, AsyncMemoryBackend &recv_backend, std::size_t bytes, const char *what,
    const std::function<void(const std::string &)> &fatal);

// One rank of the chunk-pipelined two-rank all_reduce (see communicator.cpp), exposed so a test can drive it over a controlled transport and fake backends. Root: receives into
// `recv_base`, calls `reduce_chunk` once per chunk, in order, only after that chunk was received, then posts the chunk's result from `send_base`. Non-root: sends `send_base` and
// receives the result into `recv_base` (the same backend, in place) range by range, each only after the send of that range finished. Same failure contract as run_n2_exchange.
void run_n2_all_reduce_pipeline(
    TensorCommWorker &worker, Transport &transport, bool root, AsyncMemoryBackend &recv_base, AsyncMemoryBackend &send_base, std::size_t total_bytes, std::size_t chunk_bytes,
    std::size_t element_size, const std::function<void(std::size_t byte_offset, std::size_t elements)> &reduce_chunk, const char *what,
    const std::function<void(const std::string &)> &fatal);
// The chunk size the pipeline would use (0 = the sequential schedule applies); `setting` is TBCCL_ALLREDUCE_PIPELINE_BYTES.
std::size_t n2_pipeline_chunk_bytes(std::size_t setting, std::size_t total_bytes, std::size_t count, DataType datatype);

// Test hook: the next `count` exchanges fail to admit their second (receive) transfer, as if allocation failed, after the send has been admitted.
inline std::atomic<int> g_n2_exchange_fail_second_enqueue{0};
// Test hooks: the next `count` exchanges fail (bad_alloc) while building the abort reason, and right after both legs were admitted (before the first wait).
inline std::atomic<int> g_n2_exchange_fail_abort_message{0};
inline std::atomic<int> g_n2_exchange_fail_after_admission{0};
// Test hook: the n-th (1-based) chunk admission of the pipelined all_reduce root fails as if allocation failed (0 = off).
inline std::atomic<int> g_n2_pipeline_fail_admission{0};

} // namespace tbccl::detail
