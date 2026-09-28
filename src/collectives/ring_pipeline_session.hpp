#pragma once

// Internal (non-installed) chunk arithmetic and chunk-level progress
// tracking shared by every pipelined ring collective
// (all_gather_pipelined.cpp / reduce_scatter_pipelined.cpp /
// all_reduce_pipelined.cpp, the last via composing the first two).
// Not part of the public tbccl:: surface.
//
// This is deliberately separate from RingSession (ring_executor.hpp),
// which tracks completion at ring-*step* granularity — one flag per
// step. Pipelining needs finer-grained readiness: a sender forwarding
// step s's data chunk-by-chunk must be able to wait for just the
// chunk it is about to send, not the whole step, so neighboring ranks
// can overlap chunk transfers within a step instead of waiting for an
// entire segment to arrive before forwarding any of it.

#include <algorithm>
#include <cstddef>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace tbccl::detail
{

// Overflow-safe ceiling division: the number of chunk_bytes-sized
// pieces needed to cover total_bytes, with the last piece possibly
// shorter. Returns 0 for total_bytes == 0. Requires chunk_bytes > 0
// (validate_chunk_bytes() below enforces this before any chunk math
// runs). If chunk_bytes >= total_bytes, this correctly returns 1 —
// "treat the operation as one chunk" falls out of the formula, no
// separate clamping needed.
inline std::size_t compute_chunk_count(
    std::size_t total_bytes,
    std::size_t chunk_bytes)
{
    if (total_bytes == 0)
    {
        return 0;
    }

    return 1 + (total_bytes - 1) / chunk_bytes;
}

// Byte length of the chunk_index'th (0-based) chunk of a total_bytes
// region split into compute_chunk_count(total_bytes, chunk_bytes)
// chunks of chunk_bytes each (the last one possibly shorter).
inline std::size_t chunk_length(
    std::size_t total_bytes,
    std::size_t chunk_bytes,
    std::size_t chunk_index)
{
    const std::size_t offset = chunk_index * chunk_bytes;
    return std::min(chunk_bytes, total_bytes - offset);
}

// Rejects an invalid requested chunk size before any communication,
// rather than silently rounding it. `element_size` is the datatype
// size for typed (reduction) collectives, so a chunk always contains
// a whole number of elements; pass 1 for AllGather, which has no such
// requirement (any positive byte count is acceptable).
inline void validate_chunk_bytes(
    const char *context,
    std::size_t chunk_bytes,
    std::size_t element_size)
{
    if (chunk_bytes == 0)
    {
        throw std::runtime_error(
            std::string(context) +
            ": chunk_bytes must be greater than 0 for a nonzero payload");
    }

    if (element_size > 1 && chunk_bytes % element_size != 0)
    {
        throw std::runtime_error(
            std::string(context) + ": chunk_bytes (" +
            std::to_string(chunk_bytes) +
            ") must be a multiple of the datatype size (" +
            std::to_string(element_size) + ")");
    }
}

// Chunk-level progress tracking for one pipelined ring invocation.
// Scoped to exactly one call, constructed fresh on the stack by each
// pipelined algorithm — never reused across invocations, so a
// previous operation's progress or failure state can never spuriously
// satisfy or block a later, unrelated one (same invocation-scoping
// discipline as RingSession).
//
// completed_chunks_[step] is a monotonically increasing count of how
// many chunks of `step` the receiver has finished (written for
// AllGather, reduced-and-folded for ReduceScatter) — not a per-chunk
// flag array — because the receiver always processes a step's chunks
// strictly in index order (0, 1, 2, ...), so "at least K chunks of
// step done" is equivalent to "chunk K-1 of step done" and is cheaper
// to track and wait on.
class RingPipelineSession
{
public:
    explicit RingPipelineSession(std::size_t steps)
        : completed_chunks_(steps, 0)
    {
    }

    // Called by the receiver once chunk (completed_chunks_[step])
    // of `step` — the next one in order — has been fully written
    // (AllGather) or reduced into the working buffer (ReduceScatter).
    // Wakes any sender waiting on this step's chunk progress.
    void mark_chunk_complete(std::size_t step)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++completed_chunks_[step];
        cv_.notify_all();
    }

    // Called by the sender before forwarding the `required_chunks`'th
    // chunk (1-based count) of `step`. Blocks until at least that many
    // chunks of `step` have completed. Returns false — without having
    // sent anything for this chunk — once report_failure() has been
    // called by either side; the sender must stop immediately in that
    // case.
    bool wait_for_chunks(std::size_t step, std::size_t required_chunks)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(
            lock,
            [&]()
            {
                return completed_chunks_[step] >= required_chunks ||
                       failed_;
            });

        return !failed_;
    }

    // Marks the session failed and wakes every waiter, so a sender
    // blocked on a chunk that will now never arrive (because the
    // other side just threw) does not hang forever. Safe to call from
    // either side, including after the other side already called it.
    void report_failure()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        failed_ = true;
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::size_t> completed_chunks_;
    bool failed_ = false;
};

} // namespace tbccl::detail
