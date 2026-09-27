#pragma once

#include <cstddef>

#include <tbccl/reduction.hpp>
#include <tbccl/world.hpp>

namespace tbccl
{

// Collectives are synchronous: all ranks in a World must invoke them
// in matching order, and each collective has exclusive use of every
// World peer connection for the duration of its call. World's
// connections are an unframed byte stream per peer, so concurrent
// unrelated send()/recv() traffic over the same World while a
// collective is executing — from another thread, or messages left in
// flight from an earlier operation — is not supported and will corrupt the
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

// Every rank contributes `count` elements of `datatype` from
// `send_buffer`; on successful return, `root`'s `recv_buffer` holds the
// element-wise reduction (via `op`) of every rank's contribution, root's
// own included. All ranks must call with the same `count`, `datatype`,
// `op`, and `root` (`root < world.size()`). `send_buffer` must be
// non-null on every rank when `count > 0`; `recv_buffer` must be
// non-null on `root` when `count > 0`, but is ignored (may be nullptr)
// on non-root ranks, which do not receive a result. `count == 0` is
// valid and performs no communication. `send_buffer` and root's
// `recv_buffer` must be suitably aligned for `datatype`; if they are
// the same pointer, that aliasing is handled, but no other overlap
// between them is supported. Root reduces peer contributions in
// ascending rank order (its own contribution first), so results are
// reproducible for a given input across runs, but floating-point Sum/
// Product results are not associativity-independent — a different
// reduction order (e.g. a future tree/ring implementation) may produce
// a different Float32/Float64 result for the same inputs. Integer
// Sum/Product wrap using two's-complement modular arithmetic rather
// than invoking signed-overflow undefined behavior.
void reduce(
    World &world,
    const void *send_buffer,
    void *recv_buffer,
    std::size_t count,
    DataType datatype,
    ReduceOp op,
    std::size_t root);

// Every rank contributes `count` elements of `datatype` from
// `send_buffer`; on successful return, every rank's `recv_buffer` holds
// the identical element-wise reduction (via `op`) of every rank's
// contribution. All ranks must call with the same `count`, `datatype`,
// and `op`. There is no root parameter: internally this reduces to
// rank 0 and broadcasts the result back out, but that is an
// implementation detail, not part of the contract. Unlike reduce(),
// `send_buffer` and `recv_buffer` must both be non-null on *every*
// rank when `count > 0`, since every rank receives the result; this is
// validated before any communication is attempted, so no rank can
// strand its peers by failing after they have already started
// sending. `count == 0` is valid and performs no communication.
// `send_buffer` and `recv_buffer` may be the same pointer (the
// underlying reduce()+broadcast() composition supports this exactly:
// non-root ranks never read recv_buffer during reduce(), and every
// rank's send is already complete before broadcast() overwrites the
// same memory with the final result); no other overlap is supported.
void all_reduce(
    World &world,
    const void *send_buffer,
    void *recv_buffer,
    std::size_t count,
    DataType datatype,
    ReduceOp op);

} // namespace tbccl
