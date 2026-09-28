#pragma once

// Internal (non-installed) implementation API for all_gather()
// algorithm variants. Not part of the public tbccl:: surface — kept
// under src/ and exposed to tests only via a private include path, not
// the public include/ tree.

#include <cstddef>

#include <tbccl/world.hpp>

namespace tbccl::detail
{

// Shared argument validation for every all_gather() algorithm variant
// — buffer null checks and the world.size() * bytes_per_rank overflow
// check. Throws before any communication on invalid arguments; callers
// still need their own bytes_per_rank == 0 and world.size() == 1
// fast-path handling (kept in each algorithm, not here, since the
// single-rank fast path differs slightly in shape between reference
// and ring).
void validate_all_gather_args(
    const World &world,
    const void *send_buffer,
    const void *recv_buffer,
    std::size_t bytes_per_rank);

// The existing centralized/gather-then-broadcast implementation,
// unchanged in behavior. The public all_gather() routes through the
// algorithm selector (see algorithm_selector.hpp) and may call this or
// all_gather_ring() depending on world size, message size, and any
// TBCCL_ALGORITHM/TBCCL_ALL_GATHER_ALGORITHM override; this remains
// the correctness oracle other algorithms are compared against.
void all_gather_reference(
    World &world,
    const void *send_buffer,
    void *recv_buffer,
    std::size_t bytes_per_rank);

// Ring all-gather: N-1 steps around a logical ring, each rank
// forwarding the chunk it most recently received to its next neighbor
// while simultaneously receiving the next chunk from its previous
// neighbor. See all_gather_ring.cpp for the exact send/receive chunk
// formulas and concurrency design.
void all_gather_ring(
    World &world,
    const void *send_buffer,
    void *recv_buffer,
    std::size_t bytes_per_rank);

// Experimental: chunked/pipelined ring all-gather. Same N-1 step ring
// topology and send_chunk()/recv_chunk() formulas as all_gather_ring()
// — the only difference is that each step's bytes_per_rank-sized
// segment is itself split into chunk_bytes-sized pieces (the last one
// possibly shorter), forwarded chunk-by-chunk instead of as one
// send()/recv() call per step, so a chunk can begin forwarding to the
// next rank as soon as it individually arrives rather than waiting for
// its whole segment. Every rank in the World must be called with the
// identical chunk_bytes value for one invocation — see
// all_gather_pipelined.cpp. Not connected to the public Auto selector
// or any public API; internal/benchmark-only, and kept
// independent from all_gather_ring() (not a drop-in replacement — both
// remain separately callable baselines). Requires chunk_bytes > 0 for
// a nonzero bytes_per_rank; throws before any communication otherwise.
void all_gather_pipelined(
    World &world,
    const void *send_buffer,
    void *recv_buffer,
    std::size_t bytes_per_rank,
    std::size_t chunk_bytes);

} // namespace tbccl::detail
