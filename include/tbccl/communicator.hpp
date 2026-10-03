#pragma once

// The public, framework-independent communication runtime
// entry point. See docs/framework_integration_architecture.md for the
// full design and docs/public_api.md for usage documentation.
//
// This header has zero CUDA/Objective-C/Metal dependency, matching the
// existing invariant already stated at the top of this project's
// CMakeLists.txt ("the tbccl library ... has no CUDA/Metal awareness").
// CUDA buffer support is added by an OPTIONAL component (built only
// when TBCCL_ENABLE_CUDA is on) that calls register_memory_provider_factory()
// once at startup -- core tbccl never depends on that component; it is
// an explicit, testable extension point rather than a weak-symbol trick.
// MemoryKind::MetalShared needs no such extension: it is handled by the
// SAME built-in host-pointer provider as MemoryKind::Host (see
// docs/framework_integration_architecture.md Section 5).

#include <tbccl/async_transfer.hpp>
#include <tbccl/buffer.hpp>
#include <tbccl/hetero_allreduce.hpp>
#include <tbccl/peer_capabilities.hpp>
#include <tbccl/rank_directory.hpp>
#include <tbccl/reduction.hpp>
#include <tbccl/types.hpp>
#include <tbccl/work.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace tbccl
{

// ---------------------------------------------------------------------
// Memory provider extension point
// ---------------------------------------------------------------------

// What an external-buffer provider for one MemoryKind must supply. A
// provider wraps ONE caller-owned BufferView (never allocates/frees it)
// and additionally owns whatever TBCCL-private scratch/staging resources
// its kind needs (Part 25: staging buffers, events, streams are fine to
// own; the framework tensor itself is never owned).
class ExternalMemoryProvider
{
public:
    virtual ~ExternalMemoryProvider() = default;

    // Wraps the caller's external buffer directly (non-owning). Used as
    // both recv_backend and send_backend for a non-root AllReduce leg or
    // a plain P2P transfer (n2_all_reduce_tensor allows the same object
    // in both roles -- see hetero_allreduce.hpp).
    virtual AsyncMemoryBackend &primary_backend() = 0;

    // A provider-owned scratch AsyncMemoryBackend of the same byte
    // capacity as the wrapped buffer, allocated lazily on first use and
    // reused thereafter (no payload-sized allocation per collective).
    // Used only as the AllReduce root's landing zone for the peer's
    // contribution, so the caller's own buffer (holding the local
    // input) is never overwritten before the local reduce runs.
    virtual AsyncMemoryBackend &scratch_backend() = 0;

    // Local SUM reduce: scratch_backend's contents are added into
    // primary_backend's contents, in place. Called only on the AllReduce
    // root, strictly after the reduce-to-root leg completes and strictly
    // before the broadcast-back leg is enqueued (matching
    // LocalReduceBackend's existing contract in hetero_allreduce.hpp).
    virtual LocalReduceBackend &reduce_backend() = 0;
};

// Constructs a provider for one call's worth of work, wrapping `buffer`
// (already-validated, non-owning) and respecting `context` (
// readiness semantics -- e.g. a CUDA provider waits on the supplied
// stream before D2H).
using MemoryProviderFactory = std::function<
    std::unique_ptr<ExternalMemoryProvider>(const BufferView &buffer, const ExecutionContext &context)>;

// Opaque, communicator-scoped storage for one MemoryKind's provider resources (e.g. persistent pinned
// staging). Empty until a slot-aware factory fills it; owned by the Communicator and released after its worker
// threads have stopped, so resources outlive every operation that can use them. Per-communicator, never
// process-global.
using ProviderResourceSlot = std::shared_ptr<void>;

// Slot-aware variant of MemoryProviderFactory: the factory may lazily create resources in `communicator_slot`
// and share them with the providers it returns. Calls for one communicator and kind are serialized.
using MemoryProviderFactoryEx = std::function<std::unique_ptr<ExternalMemoryProvider>(
    const BufferView &buffer, const ExecutionContext &context, ProviderResourceSlot &communicator_slot)>;

void register_memory_provider_factory_ex(MemoryKind kind, MemoryProviderFactoryEx factory);

// Registers (or replaces) the provider factory for `kind`. MemoryKind::Host
// and MemoryKind::MetalShared already have a built-in factory and do not
// need to be registered -- calling this for them overrides the built-in
// one, which is supported but not expected. MemoryKind::Cuda has NO
// built-in factory: a Communicator asked to move a Cuda-kind buffer
// before one is registered returns ErrorCode::Unsupported with a clear
// message, rather than silently falling back (Part 39).
void register_memory_provider_factory(MemoryKind kind, MemoryProviderFactory factory);

// True if a factory (built-in or registered) exists for `kind`.
bool memory_kind_registered(MemoryKind kind);

// ---------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------

// The legacy name of Endpoint (rank_directory.hpp), kept so existing callers compile unchanged.
using CommunicatorPeerEndpoint = Endpoint;

// Framework-neutral bootstrap configuration -- modeled
// directly on the already-clean TcpWorldOptions shape. No c10d::Store,
// Python dict, or exo topology object anywhere near this type; an
// out-of-tree framework adapter translates ITS bootstrap mechanism into
// this struct.
//
// Describe the world with `world_size`, one shared `communicator_id` and a `rank_directory` holding every
// rank's explicit control and data endpoint. The directory is assembled by whoever bootstraps the ranks (an adapter reading
// its own store, an application, a launcher); libtbccl never discovers peers. Connections follow one rule: for every pair
// the lower rank connects and the higher rank accepts, so a rank listens only if a higher rank exists
// (rank_accepts_connections()).
//
// `peers` is the pre-N-rank-runtime convenience for world_size 1 or 2 and is mutually exclusive with `rank_directory`: it is
// resolved into explicit endpoints (control = peers[r], data = peers[r].host : peers[r].port + 1000) with the nil
// communicator id. New code and every N>2 world must use `rank_directory`.
struct CommunicatorOptions
{
    std::size_t rank = 0;

    // 0 means "derive": rank_directory.entries.size(), or peers.size() for the legacy form. If set it must agree.
    std::size_t world_size = 0;

    // Shared by every rank of one communicator; a rank presenting a different id is rejected during the handshake.
    // The nil id (the default) is valid and only distinguishes nothing.
    CommunicatorId communicator_id;

    RankDirectory rank_directory;

    // Legacy world_size 1 or 2 form; see above.
    std::vector<CommunicatorPeerEndpoint> peers;

    // Bounds the whole bootstrap (connect, accept, handshakes, capability exchange).
    std::chrono::milliseconds bootstrap_timeout{10000};
};

// ---------------------------------------------------------------------
// Capabilities
// ---------------------------------------------------------------------

// Read-only view of this communicator's negotiated capabilities (Part
// AO/AP: translates the existing PeerCapabilities/NegotiationResult
// protocol, run exactly once at construction and cached here -- never
// re-negotiated per collective).
class Capabilities
{
public:
    bool supports_memory_kind(MemoryKind kind) const noexcept;
    bool supports_collective_all_reduce(MemoryKind kind, DataType datatype, ReduceOp op) const noexcept;
    // byte-generic collectives (no datatype/op: nothing is interpreted or reduced).
    bool supports_collective_broadcast(MemoryKind kind) const noexcept;
    bool supports_collective_all_gather(MemoryKind kind) const noexcept;
    std::size_t effective_max_chunk() const noexcept { return negotiation_.effective_max_chunk; }
    std::size_t effective_alignment() const noexcept { return negotiation_.effective_alignment; }
    const NegotiationResult &negotiation() const noexcept { return negotiation_; }
    const PeerCapabilities &local() const noexcept { return local_; }
    const PeerCapabilities &remote() const noexcept { return remote_; }

private:
    friend class Communicator;
    NegotiationResult negotiation_;
    PeerCapabilities local_;
    PeerCapabilities remote_;
};

// ---------------------------------------------------------------------
// Communicator
// ---------------------------------------------------------------------

// Owns the communication runtime state required across operations:
// negotiated capabilities, the data-plane Transport, a persistent
// TensorCommWorker (P2P progress), and a persistent single-collective
// executor thread (AllReduce progress, promoted from the bucketed
// all-reduce overlap work's BucketAllReduceWorker design with per-job
// failure isolation instead of whole-worker abort -- see
// communicator.cpp). Construction is expensive (real network bootstrap +
// capability negotiation); operation submission is cheap (Part 104/105).
// Never touches externally-owned caller buffers beyond reading/writing
// the bytes the caller asked to move (Part 148).
class Communicator
{
public:
    // Throws std::runtime_error (bounded, no infinite connect/accept
    // hang; respects options.bootstrap_timeout) on bootstrap
    // failure, wrong/unsupported world size, or capability
    // negotiation failure.
    static std::unique_ptr<Communicator> create(const CommunicatorOptions &options);

    ~Communicator();

    Communicator(const Communicator &) = delete;
    Communicator &operator=(const Communicator &) = delete;

    std::size_t rank() const noexcept;
    std::size_t world_size() const noexcept;
    const Capabilities &capabilities() const noexcept;

    // True once a prior operation's transport/protocol failure has
    // poisoned this communicator -- subsequent operations fail
    // immediately with ErrorCode::PeerFailure rather than hanging. No
    // automatic reconnection is attempted.
    bool failed() const noexcept;

    // communicator-wide, destructive, idempotent, thread-safe abort. NOT the same as Work::wait(timeout), which only stops
    // the caller waiting and leaves the operation (and the communicator) intact. After abort(): new operations throw
    // immediately; queued Works fail without running; the active operation is interrupted (its socket is shut down; no peer
    // cooperation needed) and its Work fails only after TBCCL has stopped touching its buffers; abort() returns once that
    // has happened. The communicator cannot be reused. The first reason (from abort() or from a fatal
    // transport/protocol/device error) is kept. Limit: finite local device work already running (e.g. a user CUDA kernel
    // TBCCL is synchronizing on) is not preempted. A caller-side Work::wait(timeout) expiring never aborts; failures
    // detected before protocol start (invalid arguments) do not poison the communicator.
    void abort(const std::string &reason = "");
    bool aborted() const noexcept;
    std::string abort_reason() const;

    // This communicator's resource slot for `kind` (null if its factory never used one). Diagnostics only.
    ProviderResourceSlot provider_resources(MemoryKind kind) const;

    // Async P2P. `peer` must be the communicator's single other rank (0
    // or 1). Returns a live Work; never blocks on transport/device work,
    // only on TensorCommWorker's bounded queue (same contract as
    // TensorCommWorker::enqueue). Throws std::runtime_error with a
    // message tagged by ErrorCode (see error_code_name()) for validation
    // failures caught before any network activity (null/undersized
    // buffer, unregistered memory kind, wrong peer, communicator
    // already failed).
    Work send(
        const BufferView &buffer,
        std::size_t count,
        DataType datatype,
        std::size_t peer,
        const ExecutionContext &context = {});

    Work recv(
        const BufferView &buffer,
        std::size_t count,
        DataType datatype,
        std::size_t peer,
        const ExecutionContext &context = {});

    // Async N=2 SUM AllReduce, routed to the existing, unmodified
    // n2_all_reduce_tensor(). Root is always rank 0; there is no
    // caller-visible root parameter. Returns ErrorCode::Unsupported (as
    // a thrown std::runtime_error whose message starts with
    // "unsupported: ") if world_size() != 2, op != ReduceOp::Sum, or
    // either buffer's memory kind has no registered provider. send_buf
    // and recv_buf may be the same BufferView (in-place, the common
    // case) or different (TBCCL copies send_buf's content into
    // recv_buf's location first when they differ).
    Work all_reduce(
        const BufferView &send_buf,
        const BufferView &recv_buf,
        std::size_t count,
        DataType datatype,
        ReduceOp op,
        const ExecutionContext &context = {});

    // byte-generic N=2 Broadcast. In place from the caller's view: on `root` the buffer is the
    // source, on the other rank it is the destination; `buffer.bytes` is authoritative and both
    // ranks must pass the same size. Runs in the same FIFO ordering domain as all_reduce(). Throws
    // "unsupported: ..." for world_size != 2 or an unregistered memory kind, "invalid_argument: ..."
    // for root >= world_size or a null non-empty buffer.
    Work broadcast(const BufferView &buffer, std::size_t root, const ExecutionContext &context = {});

    // byte-generic N=2 AllGather. `outputs` must hold exactly world_size() buffers, each of
    // `input.bytes` bytes; on completion outputs[r] holds rank r's input on BOTH ranks (including
    // outputs[rank()] <- input, performed inside the runtime through the memory providers; skipped if
    // they alias). Deadlock-safe ordering: rank 0 sends then receives, rank 1 receives then sends.
    // Not performance-tuned (intended for small metadata).
    Work all_gather(
        const BufferView &input,
        const std::vector<BufferView> &outputs,
        const ExecutionContext &context = {});

private:
    Communicator();

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace tbccl
