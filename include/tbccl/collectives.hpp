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

// Every rank contributes exactly `bytes_per_rank` bytes from
// `send_buffer`. On successful return, every rank's `recv_buffer`
// holds all contributions concatenated in rank order: bytes
// `[0, bytes_per_rank)` are rank 0's contribution, bytes
// `[bytes_per_rank, 2*bytes_per_rank)` are rank 1's, and so on, so
// `recv_buffer` must be at least `world.size() * bytes_per_rank`
// bytes. `send_buffer`/`recv_buffer` may be nullptr only when
// `bytes_per_rank` is 0. Not in-place: for non-zero payloads,
// `send_buffer` and `recv_buffer` must refer to separate storage —
// overlap is not detected and produces undefined results.
void all_gather(
    World &world,
    const void *send_buffer,
    void *recv_buffer,
    std::size_t bytes_per_rank);

} // namespace tbccl
