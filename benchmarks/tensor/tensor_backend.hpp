#pragma once

// Phase 17 tensor-transfer backend abstraction. This is deliberately
// small and benchmark-specific (not a general tensor runtime): each
// backend owns exactly one deterministically-generated source buffer
// and one destination buffer of the same capacity, and exposes the
// individual pipeline stages (device data generation, device-to-host
// staging, network send/recv, host-to-device staging) as separate
// calls so a caller can time each stage independently rather than
// only ever measuring an opaque end-to-end transfer.
//
// This header, and every backend built against it, lives entirely
// under benchmarks/ -- the core tbccl library (include/tbccl,
// src/core, src/transport, src/collectives) has no dependency on it
// and no CUDA/Metal awareness. World::send()/World::recv() remain the
// only network primitives a backend uses; no new public TBCCL API is
// introduced.
//
// Buffer lifetime (Part H): a backend's source/destination buffers
// must not be reallocated or released while any stage that reads or
// writes them (device kernel, staging copy, World::send/recv) is
// still in flight. Every stage method below is synchronous/blocking
// with respect to its own effects -- when a stage method returns, its
// effect is guaranteed complete and it is always safe to proceed to
// the next stage or to reuse/reallocate the buffer. Overlapping
// stages (double-buffering) is explicitly out of scope for Phase 17.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace tbccl
{
class World;
} // namespace tbccl

namespace tbccl_bench::tensor
{

// One shared cross-platform deterministic byte pattern. Every backend
// (CPU, CUDA kernel, Metal compute shader) must produce exactly this
// value for a given (i, seed) pair, so that a destination buffer's
// expected contents never depend on which backend produced the
// source. Uses explicit fixed-width unsigned arithmetic throughout so
// the result cannot depend on `int` width or signedness.
inline std::uint8_t pattern_byte(std::size_t i, std::uint32_t seed) noexcept
{
    const std::uint64_t index = static_cast<std::uint64_t>(i);
    const std::uint64_t value =
        index * 131u + (index >> 8) * 17u + static_cast<std::uint64_t>(seed);
    return static_cast<std::uint8_t>(value & 0xffu);
}

enum class BackendKind
{
    Host,
    CudaPageable,
    CudaPinned,
    MetalShared,
    MetalPrivateStaged,
};

// Human-readable name used for CLI parsing and CSV output (e.g.
// "cuda-pinned"). Throws std::runtime_error for an unrecognized name.
std::string backend_kind_name(BackendKind kind);
BackendKind parse_backend_kind(const std::string &name);

// True for backends compiled into this build (host is always true;
// cuda-*/metal-* are true only when TBCCL_ENABLE_CUDA/TBCCL_ENABLE_METAL
// were on at build time). Checked before make_backend() so an
// unavailable backend produces a clear, distinct error rather than a
// generic communication failure (Part C, requirement 10).
bool backend_kind_available(BackendKind kind);

struct AllocationStats
{
    std::size_t capacity_bytes = 0;
    std::size_t allocation_count = 0;
    std::size_t reuse_count = 0;
};

// One backend instance owns one source buffer and one destination
// buffer, each `capacity()` bytes once allocate() has been called
// (Part 8: "do not assume the same buffer can safely act as source,
// network receive destination and GPU output simultaneously" -- kept
// as two independent allocations rather than one).
class TensorBackend
{
public:
    virtual ~TensorBackend() = default;

    TensorBackend(const TensorBackend &) = delete;
    TensorBackend &operator=(const TensorBackend &) = delete;

    virtual BackendKind kind() const noexcept = 0;

    // (Re)allocates the source/destination buffers for `bytes` bytes
    // each. A call with the same `bytes` as the current allocation
    // reuses it (stats().reuse_count increments, not
    // allocation_count) rather than reallocating -- this is what lets
    // a benchmark's warmup + N-iteration loop be a true
    // allocate-once/reuse-many-times measurement (Part 30).
    virtual void allocate(std::size_t bytes) = 0;

    virtual std::size_t capacity() const noexcept = 0;

    virtual AllocationStats stats() const noexcept = 0;

    // Fills the source buffer with pattern_byte(i, seed) for every
    // i in [0, capacity()), genuinely executed on this backend's
    // compute device (a GPU kernel for CUDA/Metal backends, never a
    // host-generated buffer merely copied onto the device).
    virtual void initialize_source(std::uint32_t seed) = 0;

    // Blocks until initialize_source()'s device-side work has
    // completed. Split out from initialize_source() so producer
    // completion cost can be measured/excluded from steady-state
    // timing separately (Part 27, Part 36).
    virtual void prepare_source() = 0;

    // Copies the (already-prepared) source buffer into a host-visible
    // staging area. Once this returns, host_send_data() may safely
    // read it (Part 8's completion guarantee).
    virtual void stage_device_to_host() = 0;

    // A pointer to the host-visible source staging area, exactly
    // capacity() bytes, valid once stage_device_to_host() has
    // returned. Exposed (in addition to host_send_data()) so a
    // no-network "staging-only" benchmark mode can measure staging
    // cost by copying directly between two backend instances'
    // staging areas, without involving a World at all.
    virtual const void *source_staging_data() const noexcept = 0;

    // A pointer to the host-visible destination staging area, exactly
    // capacity() bytes, that must be filled (by host_recv_data() or,
    // for staging-only measurement, a direct memcpy) before
    // stage_host_to_device() is called.
    virtual void *destination_staging_data() noexcept = 0;

    // Sends exactly capacity() bytes of the host-visible source
    // staging area to `peer` over `world`. Equivalent to
    // `world.send(peer, source_staging_data(), capacity())`.
    virtual void host_send_data(tbccl::World &world, std::size_t peer) = 0;

    // Receives exactly capacity() bytes from `peer` over `world` into
    // the host-visible destination staging area. Equivalent to
    // `world.recv(peer, destination_staging_data(), capacity())`.
    virtual void host_recv_data(tbccl::World &world, std::size_t peer) = 0;

    // Copies the (received) host-visible destination staging area
    // into the destination device buffer and completes any
    // device-side synchronization (Part 8's "destination accelerator
    // operation is complete" guarantee) -- after this returns,
    // verify_destination() reflects the transfer.
    virtual void stage_host_to_device() = 0;

    // Blocks until every asynchronous operation this backend has
    // issued (producer, staging copies, consumer) is complete. A
    // general synchronization point a caller can use between
    // iterations; individual stage methods above already guarantee
    // their own specific effects are complete when they return.
    virtual void synchronize() = 0;

    // True byte-for-byte verification (never a checksum) of the
    // source buffer against pattern_byte(i, seed), read back from the
    // real backing store (a GPU readback for CUDA/Metal backends).
    virtual bool verify_source(std::uint32_t seed) const = 0;

    // Same, but against the destination buffer.
    virtual bool verify_destination(std::uint32_t seed) const = 0;

protected:
    TensorBackend() = default;
};

// Throws std::runtime_error if `kind` was not compiled into this
// build (see backend_kind_available()) or if device initialization
// fails (e.g. no CUDA device present at runtime).
std::unique_ptr<TensorBackend> make_backend(BackendKind kind);

} // namespace tbccl_bench::tensor
