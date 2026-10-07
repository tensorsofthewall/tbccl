#pragma once

// Internal (non-installed) implementation API for reduce_scatter()
// algorithm variants. Not part of the public tbccl:: surface — kept
// under src/ and exposed to tests only via a private include path, not
// the public include/ tree.

#include <cstddef>

#include <tbccl/reduction.hpp>
#include <tbccl/world.hpp>

namespace tbccl::detail
{

// Shared argument validation for every reduce_scatter() algorithm
// variant: DataType/ReduceOp validation, the three overflow checks
// (element count, segment bytes, total bytes), and the buffer null
// checks. Throws before any communication on invalid arguments.
// Returns the overflow-checked total_count = world.size() *
// recv_count. Callers still need their own recv_count == 0 fast-path
// handling (kept in each algorithm, not here).
std::size_t validate_reduce_scatter_args(
    const World &world,
    const void *send_buffer,
    const void *recv_buffer,
    std::size_t recv_count,
    DataType datatype,
    ReduceOp op);

// The existing centralized (reduce-to-rank-0-then-scatter)
// implementation, unchanged in behavior. The public reduce_scatter()
// routes through the algorithm selector (see algorithm_selector.hpp)
// and may call this or reduce_scatter_ring() depending on world size,
// segment size, and any TBCCL_ALGORITHM/TBCCL_REDUCE_SCATTER_ALGORITHM
// override; this remains the correctness oracle other algorithms are
// compared against.
void reduce_scatter_reference(
    World &world,
    const void *send_buffer,
    void *recv_buffer,
    std::size_t recv_count,
    DataType datatype,
    ReduceOp op);

// Ring reduce-scatter: N-1 steps around a logical ring, each rank
// folding the chunk it most recently received into its own working
// copy and forwarding the newly-reduced chunk to its next neighbor.
// See reduce_scatter_ring.cpp for the exact send/receive chunk
// formulas and concurrency design (shifted by one position relative to
// all_gather_ring's formulas, since a chunk here must be reduced
// in place before being forwarded).
void reduce_scatter_ring(
    World &world,
    const void *send_buffer,
    void *recv_buffer,
    std::size_t recv_count,
    DataType datatype,
    ReduceOp op);

// Experimental: chunked/pipelined ring reduce-scatter. Same N-1 step
// ring topology and send_chunk(s)/recv_chunk(s) formulas as
// reduce_scatter_ring() — each step's recv_count-element segment is
// itself split into chunk_bytes-sized byte chunks (a whole number of
// elements each; the final chunk possibly shorter), received and
// reduced into the typed working buffer one chunk at a time instead of
// one recv()+apply_reduction() call per whole segment. See
// reduce_scatter_pipelined.cpp for the exact schedule. Every rank in
// the World must be called with the identical chunk_bytes value for
// one invocation. Not connected to the public Auto selector or any
// public API; internal/benchmark-only, and kept
// independent from reduce_scatter_ring() (not a drop-in replacement —
// both remain separately callable baselines). Requires chunk_bytes > 0
// and chunk_bytes % datatype_size(datatype) == 0 for a nonzero
// recv_count; throws before any communication otherwise.
void reduce_scatter_pipelined(
    World &world,
    const void *send_buffer,
    void *recv_buffer,
    std::size_t recv_count,
    DataType datatype,
    ReduceOp op,
    std::size_t chunk_bytes);

} // namespace tbccl::detail
