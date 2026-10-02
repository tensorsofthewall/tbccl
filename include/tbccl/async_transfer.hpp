#pragma once

#include <functional>

// The portable async tensor-transfer substrate.
// Everything in this header is transport- and memory-backend-agnostic
// -- it references only tbccl::Transport (transport.hpp) and the small
// AsyncMemoryBackend interface below, never CUDA/Metal types (those
// stay in benchmarks/tensor/, matching this library's existing
// boundary).
//
// Design summary:
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
//                       a time and in submission order --
//                       chunks within ONE request may pipeline across
//                       two persistent internal threads (a "staging"
//                       thread and a "network" thread) for depth > 1,
//                       but distinct requests are never interleaved on
//                       the wire, since async tensor-transfer reuses one persistent
//                       TCP connection and
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
// aligned to `alignment` where possible. the required
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
// each `slot_bytes` bytes. Race-safe acquire()/release() --
// acquire() blocks (via condition_variable, never spins,
//) until a slot is Free, and marks it in use;
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
    // preallocated once at construction and never reallocated.
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
// only from a TensorCommWorker's internal threads: a backend whose
// real work is asynchronous on-device (a CUDA event, a Metal shared
// event) is free to block *this* call until that on-device work
// completes, since blocking here never blocks the caller that enqueued
// the transfer.
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

    // Capability, not type check. A backend whose tensor memory is
    // ALREADY directly readable/writable by a Transport (plain host CPU
    // memory over TCP, currently the only such case) can report true
    // here to let TensorCommWorker skip the StagingPool entirely for
    // this transfer -- no TBCCL-owned memcpy, no staging thread. A
    // backend that must stage through host-visible memory to move data
    // at all (CUDA, Metal-private-staged) leaves this false (the
    // default) and keeps using stage_source_chunk()/
    // commit_destination_chunk() as before. This does NOT mean
    // kernel-level zero-copy (TCP still copies through the kernel
    // normally) -- it means zero *additional* TBCCL-owned copies on top
    // of that.
    virtual bool supports_direct_transport_access() const noexcept { return false; }

    // Valid only when supports_direct_transport_access() is true.
    // Returns a pointer to the WHOLE already-ready source tensor
    // (`total_bytes` from the owning TransferRequest), for a single
    // direct Transport::send() -- no chunking, no staging thread.
    virtual const void *direct_source_data() const noexcept { return nullptr; }

    // Valid only when supports_direct_transport_access() is true.
    // Returns a pointer to the WHOLE destination tensor, ready for a
    // single direct Transport::recv() to write into.
    virtual void *direct_destination_data() noexcept { return nullptr; }
};

// ---------------------------------------------------------------------
// TransferWork
// ---------------------------------------------------------------------

enum class TransferDirection
{
    Send,
    Recv,
};

// Lightweight completion handle. Constructing/copying/holding one
// never touches a device, calls cudaStreamSynchronize, enumerates
// devices, or allocates meaningfully more than the handle itself.
// "Completed" means the destination side's commit_destination_chunk
// has returned for every chunk -- not merely that bytes left a socket.
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

    // Constructs a fresh, live TransferWork whose state is not tied to
    // any TensorCommWorker request -- used by the Communicator's
    // collective executor (which drives n2_all_reduce_tensor() on its
    // own background thread, not via TensorCommWorker::enqueue) to
    // return a standalone Work handle for an AllReduce. The returned
    // TransferWork and the shared_ptr<State> obtained via state_of() on
    // it refer to the same state, so complete_ok()/complete_error() on
    // that state correctly completes the Work the caller is holding.
    static TransferWork make();
};

} // namespace detail

// ---------------------------------------------------------------------
// TransferRequest
// ---------------------------------------------------------------------

// Buffer lifetime contract: whatever memory `backend` reads from or
// writes to on this request's behalf -- whether via
// stage_source_chunk()/commit_destination_chunk() (staged path) or
// direct_source_data()/direct_destination_data() (direct path,
// backend->supports_direct_transport_access()==true) -- must remain
// valid for as long as this request's TransferWork has not yet
// completed (i.e. until wait()/is_completed()==true). This is true of
// the staged path too, not a new restriction the direct path
// introduces; the direct path just makes it more consequential, since
// there the CALLER's own source/destination buffer is what the network
// reads/writes directly, with no TBCCL-owned copy in between to fall
// back on. TBCCL never makes a hidden defensive copy to relax this --
// doing so would silently reintroduce the staging cost the direct path
// exists to avoid.
struct TransferRequest
{
    std::uint64_t transfer_id = 0;
    TransferDirection direction = TransferDirection::Send;

    // Non-owning; must outlive the TransferWork this request produces.
    AsyncMemoryBackend *backend = nullptr;
    Transport *transport = nullptr;

    std::size_t total_bytes = 0;
    // 0 means "one chunk, the whole payload" (see plan_chunks()).
    // Ignored on the direct path: a direct-capable backend always
    // transfers the whole buffer in one Transport call, regardless of
    // chunk_hint, since there is no staging slot size to bound.
    std::size_t chunk_hint = 0;
    std::size_t alignment = 1;
};

// ---------------------------------------------------------------------
// TensorCommWorker
// ---------------------------------------------------------------------

// One persistent communication context. Owns a bounded FIFO request
// queue and exactly two persistent threads (a staging thread and a
// network thread, the "not thread per transfer/chunk") used to
// pipeline chunks *within* one request when `pipeline_depth` > 1;
// distinct requests are always fully processed in submission order,
// never interleaved on the wire.
class TensorCommWorker
{
public:
    // `pipeline_depth` sizes the internal StagingPool used for every
    // enqueued request's chunks. The pool is cached and reused across
    // requests whose (chunk_capacity, depth) match the previous request
    // -- measurement showed that a fresh allocation per request
    // costs an order of magnitude more than a reused one for large
    // buffers (first-touch page faults, not memcpy bandwidth). A
    // request with a different chunk size does still pay a fresh
    // allocation (the pool's buffers are sized to a specific
    // chunk_capacity; a single fixed-size pool across heterogeneous
    // requests would either waste memory or reject valid requests) --
    // see the .cpp for the caching logic. `queue_depth` bounds how many
    // TransferRequests may be waiting; enqueue() blocks once
    // full rather than growing unbounded.
    explicit TensorCommWorker(
        std::size_t pipeline_depth = 2,
        std::size_t queue_depth = 8);

    ~TensorCommWorker();

    TensorCommWorker(const TensorCommWorker &) = delete;
    TensorCommWorker &operator=(const TensorCommWorker &) = delete;

    // Terminal, idempotent, non-blocking. Rejects new enqueue() calls, fails every queued request without executing
    // it, and makes a not-yet-started dequeue fail. The active request is unwound by interrupting its Transport (done
    // by the owner); it becomes terminal only after the staging/network threads have stopped touching its buffers.
    void abort(const std::string &reason);

    // True while a request is queued or being processed.
    bool busy() const;

    // Blocks (condition variable, no polling) until nothing is queued or active.
    void wait_idle();

    // Called (from the network thread, before the Work is failed) when a started transfer fails. Set once, before use.
    void set_fatal_handler(std::function<void(const std::string &)> handler);

    // Never blocks on backend/transport/device work -- only on queue
    // capacity. Returns immediately with a live TransferWork once the
    // request is queued.
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
