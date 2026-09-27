#pragma once

#include <cstddef>

#include <tbccl/world.hpp>

namespace tbccl
{

// Collectives are synchronous: all ranks in a World must invoke them
// in matching order, and each collective has exclusive use of every
// World peer connection for the duration of its call. World's
// connections are an unframed byte stream per peer, so concurrent
// unrelated send()/recv() traffic over the same World while a
// collective is executing — from another thread, or messages left in
// flight from a previous phase — is not supported and will corrupt the
// control/payload protocol. Sequence application communication and
// collective calls strictly one after another (see
// tests/barrier_test.cpp and tests/broadcast_test.cpp for the intended
// before/collective/after pattern). No operation IDs, framing, or
// multiplexing exist yet to relax this.

// Blocks the calling rank until every rank in `world` has called
// barrier().
void barrier(World &world);

// `root` already holds the source contents in `buffer`; every other
// rank receives exactly `bytes` bytes from `root` into its own
// `buffer`. On successful return, every rank's buffer holds the bytes
// root's buffer held at entry. `root` must be a valid rank
// (`root < world.size()`); `buffer` may be nullptr only when `bytes`
// is 0. Byte-oriented only — no typed overloads or datatypes yet.
void broadcast(
    World &world,
    void *buffer,
    std::size_t bytes,
    std::size_t root);

} // namespace tbccl
