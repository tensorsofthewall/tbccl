#pragma once

// Internal (non-installed) implementation API for all_reduce()
// algorithm variants. Not part of the public tbccl:: surface — kept
// under src/ and exposed to tests only via a private include path, not
// the public include/ tree.

#include <cstddef>

#include <tbccl/reduction.hpp>
#include <tbccl/world.hpp>

namespace tbccl::detail
{

// Shared argument validation for every all_reduce() algorithm variant:
// DataType/ReduceOp validation, the count*datatype_size overflow
// check, and buffer null checks (every rank needs both buffers
// non-null when count > 0, since every rank both contributes and
// receives). Throws before any communication. Does NOT check
// all_reduce_ring()'s extra `count % world.size() == 0` requirement —
// that is specific to the ring composition (it needs equal segments)
// and checked separately, since the reference implementation has no
// such restriction and must keep accepting arbitrary counts.
void validate_all_reduce_args(
    const void *send_buffer,
    const void *recv_buffer,
    std::size_t count,
    DataType datatype,
    ReduceOp op);

// The existing reduce(root=0)+broadcast(root=0) implementation,
// unchanged in behavior — this is what the public all_reduce() calls,
// and it remains the correctness oracle other algorithms are compared
// against.
void all_reduce_reference(
    World &world,
    const void *send_buffer,
    void *recv_buffer,
    std::size_t count,
    DataType datatype,
    ReduceOp op);

// Ring all-reduce: ring reduce-scatter over the full input followed by
// ring all-gather of each rank's reduced segment — composes the two
// already-validated ring primitives rather than implementing a third
// ring transport engine. Requires `count % world.size() == 0` (equal
// segments); rejected clearly before any communication otherwise. See
// all_reduce_ring.cpp for why no barrier is needed between the two
// phases and for its exact-aliasing behavior.
void all_reduce_ring(
    World &world,
    const void *send_buffer,
    void *recv_buffer,
    std::size_t count,
    DataType datatype,
    ReduceOp op);

} // namespace tbccl::detail
