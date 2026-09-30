#pragma once

// Phase 32 Part I-M: the portable async tensor-transfer substrate.
// Everything in this header is transport- and memory-backend-agnostic
// -- it references only tbccl::Transport (transport.hpp) and the small
// AsyncMemoryBackend interface below, never CUDA/Metal types (those
// stay in benchmarks/tensor/, matching this library's existing
// boundary).
//
// Design summary (see docs/phase32_report.md for the full rationale):
//   ChunkPlan       -- pure function: (total bytes, chunk hint,
//                       alignment) -> chunk offsets/sizes. No I/O.
//   StagingPool     -- fixed-depth, preallocated, reusable host-visible
//                       buffers with race-safe acquire/release.
//   AsyncMemoryBackend -- what a memory backend must provide to move
//                       one chunk into/out of a staging buffer. Called
//                       only from TensorCommWorker's own threads, never
//                       the enqueuing caller's thread -- this is what
//                       makes the backend's work genuinely not block
//                       the caller.
//   TransferRequest -- describes one transfer: direction, backend
//                       hooks, transport, byte count, chunk hint.
//   TransferWork    -- lightweight completion handle. Constructing one
//                       never touches a device or blocks.
//   TensorCommWorker -- one persistent worker per communication
//                       context, processing a bounded FIFO queue of
//                       TransferRequests. Requests are processed one at
//                       a time and in submission order (Part AU) --
//                       chunks within ONE request may pipeline across
//                       two persistent internal threads (a "staging"
//                       thread and a "network" thread) for depth > 1,
//                       but distinct requests are never interleaved on
//                       the wire, since Phase 32 reuses one persistent
//                       TCP connection (Part AF item 118) and
//                       interleaving would corrupt framing.

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace tbccl
{

class Transport;

namespace detail
{
class TransferWorkAccess;
} // namespace detail

// ---------------------------------------------------------------------
// ChunkPlan
// ---------------------------------------------------------------------

struct Chunk
{
    std::size_t offset = 0;
    std::size_t size = 0;
};

// Splits `total_bytes` into chunks of (up to) `chunk_hint` bytes each,
// aligned to `alignment` where possible. Part O item 62's required
// cases are all handled explicitly:
//   - chunk_hint == 0 or chunk_hint >= total_bytes: one chunk, the
//     whole payload (also the tiny-payload / "full buffer" case).
//   - total_bytes == 0: returns an empty plan (zero chunks).
//   - non-divisible total_bytes: every chunk is chunk_hint bytes except
//     a smaller final partial chunk.
//   - alignment only affects where a chunk boundary falls when it can
//     do so without leaving a gap or exceeding total_bytes -- it never
//     causes a byte to be skipped or duplicated.
std::vector<Chunk> plan_chunks(
    std::size_t total_bytes,
    std::size_t chunk_hint,
    std::size_t alignment = 1);

// ---------------------------------------------------------------------
// StagingPool
// ---------------------------------------------------------------------

enum class StagingSlotState
{
    Free,
    SourcePreparing,
    Ready,
    TransportActive,
    DestPreparing,
    Complete,
};

// A fixed-depth set of preallocated, reusable host-visible buffers,
// each `slot_bytes` bytes. Race-safe acquire()/release() (Part M item
// 54) -- acquire() blocks (via condition_variable, never spins,
// Part AG item 121) until a slot is Free, and marks it in use;
// release() returns it to Free and wakes one waiter.
class StagingPool
{
public:
    StagingPool(std::size_t slot_bytes, std::size_t depth);
    ~StagingPool();

    StagingPool(const StagingPool &) = delete;
    StagingPool &operator=(const StagingPool &) = delete;

    std::size_t slot_bytes() const noexcept { return slot_bytes_; }
    std::size_t depth() const noexcept { return depth_; }

    // Returns the index of a now-owned slot; blocks until one is free.
    std::size_t acquire();

    // `data()`'s validity is tied to the pool's own lifetime, not to
    // any particular acquire()/release() cycle -- buffers are
    // preallocated once at construction and never reallocated (Part M
    // item 52).
    void *data(std::size_t slot_index) noexcept;
    const void *data(std::size_t slot_index) const noexcept;

    void release(std::size_t slot_index);

    // Diagnostic-only: lets tests and benchmarks observe slot state
    // transitions without adding production-path overhead elsewhere.
    void set_state(std::size_t slot_index, StagingSlotState state);
    StagingSlotState state(std::size_t slot_index) const;

private:
    std::size_t slot_bytes_;
    std::size_t depth_;
    std::vector<std::unique_ptr<std::byte[]>> buffers_;
    std::vector<StagingSlotState> states_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
};

// ---------------------------------------------------------------------
// AsyncMemoryBackend
// ---------------------------------------------------------------------

// What a memory backend (host, CUDA, Metal -- implementations live
// outside this library, see benchmarks/tensor/) must provide to move
// one chunk into or out of a staging buffer it does not own. Called
// only from a TensorCommWorker's internal threads (Part H item 35-37):
// a backend whose real work is asynchronous on-device (a CUDA event, a
// Metal shared event) is free to block *this* call until that
// on-device work completes, since blocking here never blocks the
// caller that enqueued the transfer.
class AsyncMemoryBackend
{
public:
    virtual ~AsyncMemoryBackend() = default;

    // Copies `chunk.size` bytes, starting at `chunk.offset` of the
    // already-ready source tensor, into `staging` (a buffer at least
    // `chunk.size` bytes). Must be safe to call for chunks in
    // increasing offset order without any other synchronization from
    // the caller.
    virtual void stage_source_chunk(
        const Chunk &chunk,
        void *staging) = 0;

    // Copies `chunk.size` bytes from `staging` (already filled by a
    // completed network recv) into the destination tensor at
    // `chunk.offset`, completing any device-side commit (e.g. an H2D
    // copy) before returning.
    virtual void commit_destination_chunk(
        const Chunk &chunk,
        const void *staging) = 0;
};

// ---------------------------------------------------------------------
// TransferWork
// ---------------------------------------------------------------------

enum class TransferDirection
{
    Send,
    Recv,
};

// Lightweight completion handle (Part J). Constructing/copying/holding
// one never touches a device, calls cudaStreamSynchronize, enumerates
// devices, or allocates meaningfully more than the handle itself.
// "Completed" means the destination side's commit_destination_chunk
// has returned for every chunk (Part J item 43) -- not merely that
// bytes left a socket.
class TransferWork
{
public:
    void wait();
    bool is_completed() const;
    bool has_error() const;
    std::string error() const;

private:
    friend class TensorCommWorker;
    friend class detail::TransferWorkAccess;

    struct State;
    std::shared_ptr<State> state_;

    TransferWork(); // only TensorCommWorker constructs a live one
};

namespace detail
{

// Grants TensorCommWorker's implementation (a private, non-member
// PIMPL type defined entirely in tensor_comm_worker.cpp -- not itself a
// friend of TransferWork) access to TransferWork::State, the same
// pattern World.hpp uses for detail::RingExecutorAccess -- a small
// public accessor class kept out of TransferWork's own public surface.
class TransferWorkAccess
{
public:
    static std::shared_ptr<TransferWork::State> state_of(TransferWork &work)
    {
        return work.state_;
    }

    static void complete_ok(const std::shared_ptr<TransferWork::State> &state);

    static void complete_error(
        const std::shared_ptr<TransferWork::State> &state,
        const std::string &message);
};

} // namespace detail

// ---------------------------------------------------------------------
// TransferRequest
// ---------------------------------------------------------------------

struct TransferRequest
{
    std::uint64_t transfer_id = 0;
    TransferDirection direction = TransferDirection::Send;

    // Non-owning; must outlive the TransferWork this request produces.
    AsyncMemoryBackend *backend = nullptr;
    Transport *transport = nullptr;

    std::size_t total_bytes = 0;
    // 0 means "one chunk, the whole payload" (see plan_chunks()).
    std::size_t chunk_hint = 0;
    std::size_t alignment = 1;
};

// ---------------------------------------------------------------------
// TensorCommWorker
// ---------------------------------------------------------------------

// One persistent communication context. Owns a bounded FIFO request
// queue and exactly two persistent threads (a staging thread and a
// network thread, Part K item 44's "not thread per transfer/chunk")
// used to pipeline chunks *within* one request when `pipeline_depth` >
// 1; distinct requests are always fully processed in submission order,
// never interleaved on the wire (Part AU).
class TensorCommWorker
{
public:
    // `pipeline_depth` sizes the internal StagingPool used for every
    // enqueued request's chunks (created lazily per request at the
    // request's own chunk size, since different requests may use
    // different chunk_hint values -- see the .cpp for why a single
    // fixed-size pool across heterogeneous requests would either waste
    // memory or reject valid requests). `queue_depth` bounds how many
    // TransferRequests may be waiting; enqueue() blocks (Part AG) once
    // full rather than growing unbounded.
    explicit TensorCommWorker(
        std::size_t pipeline_depth = 2,
        std::size_t queue_depth = 8);

    ~TensorCommWorker();

    TensorCommWorker(const TensorCommWorker &) = delete;
    TensorCommWorker &operator=(const TensorCommWorker &) = delete;

    // Never blocks on backend/transport/device work -- only on queue
    // capacity (Part AG). Returns immediately with a live TransferWork
    // once the request is queued.
    TransferWork enqueue(TransferRequest request);

    struct Stats
    {
        std::size_t submitted = 0;
        std::size_t completed = 0;
        std::size_t failed = 0;
    };

    Stats stats() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tbccl
