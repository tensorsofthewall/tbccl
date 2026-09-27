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

} // namespace tbccl::detail
