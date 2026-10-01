#pragma once

// Phase 38: an experimental N=2 SUM AllReduce built directly on the Phase
// 32 async tensor-transfer substrate (async_transfer.hpp), rather than on
// World::send()/recv() like the existing reduce()/broadcast()/all_reduce()
// in collectives.hpp. See docs/phase38_collective_design.md for the full
// rationale.
//
// This is additive and explicit-opt-in only: tbccl::all_reduce() and its
// Auto/Reference/Ring algorithm selection are completely unmodified by
// this file. No existing collective is touched.
//
// This header has no CUDA/Metal awareness, matching the project's
// existing boundary (include/tbccl, src/core, src/collectives stay
// device-free) -- concrete LocalReduceBackend implementations for
// CUDA/Metal live under benchmarks/tensor/.

#include <tbccl/async_transfer.hpp>
#include <tbccl/reduction.hpp>
#include <tbccl/transport.hpp>

#include <cstddef>

namespace tbccl
{

// Performs this rank's local SUM reduction of its own input against a
// just-received peer contribution. Implementations are backend-specific
// (host, CUDA, Metal-shared) and know, out of band, which raw
// source/destination pointers to operate on and where to leave the
// result -- this interface itself never sees a pointer, matching
// AsyncMemoryBackend's own "generic TBCCL never sees device types"
// boundary (Part N/BO of the Phase 38 plan).
class LocalReduceBackend
{
public:
    virtual ~LocalReduceBackend() = default;

    // Computes the element-wise SUM of this rank's local input and the
    // just-received peer contribution, leaving the result wherever the
    // caller's own `send_backend` (see n2_all_reduce_tensor below) will
    // read it from for the broadcast-back send. Called on the root rank
    // only, exactly once per AllReduce, strictly after the reduce-to-root
    // TransferWork has completed and strictly before the broadcast-back
    // TransferRequest is enqueued.
    virtual void reduce_sum(std::size_t count, DataType datatype) = 0;
};

// Executes one N=2 reduce(root)+broadcast(root) SUM AllReduce using the
// async tensor-transfer substrate. `transport`/`worker` must already be
// connected to the single peer and otherwise idle (no other transfer in
// flight) for the duration of this call -- this function enqueues exactly
// two sequential TransferRequests (never both outstanding at once) and
// waits for each in turn, so at most one AllReduce-related transfer is
// ever in flight (Part BC).
//
// Two separate AsyncMemoryBackend references are taken rather than one,
// because a receive leg and a send leg naturally want different roles on
// some backends (e.g. a Metal-shared TensorBackend's source buffer is
// read-only by contract once prepared; see docs/phase38_collective_design.md
// Part 5 for why this is NOT just "two pointers into one buffer" for every
// backend). For a backend where source and destination really are two
// independently-addressable, fully-mutable buffers under the caller's
// control (e.g. CudaChunkedAsyncBackend), `recv_backend` and `send_backend`
// may be the SAME object -- the function does not require them to differ.
//
//   ROOT (rank == root):     recv_backend receives the peer's contribution
//                            into its destination location; reduce_backend
//                            combines it with the local input and leaves
//                            the result wherever send_backend's source
//                            location is; send_backend then sends it back.
//   NON-ROOT (rank != root): send_backend sends the local input (already
//                            staged at its source location by the caller,
//                            before this call) to root; recv_backend then
//                            receives the final result into its
//                            destination location -- the caller already
//                            knows where that is, since it constructed
//                            recv_backend around it.
//
// `reduce_backend` must be non-null if and only if `rank == root`.
//
// Throws std::runtime_error if `root >= 2`, or if either TransferWork
// completes with an error (the error message is propagated, not
// swallowed) -- this function never blocks forever on a failed transfer
// (Part AN of the Phase 38 plan).
void n2_all_reduce_tensor(
    Transport &transport,
    TensorCommWorker &worker,
    AsyncMemoryBackend &recv_backend,
    AsyncMemoryBackend &send_backend,
    LocalReduceBackend *reduce_backend,
    std::size_t rank,
    std::size_t root,
    std::size_t total_bytes,
    std::size_t chunk_hint,
    std::size_t count,
    DataType datatype);

} // namespace tbccl
