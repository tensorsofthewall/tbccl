# Communicators

A `Communicator` is one rank of an N-rank world. It owns the connections to every peer, the worker threads that move data, and the negotiated capabilities of every rank.

## Bootstrap

TBCCL never discovers peers. Whoever starts the ranks (an adapter reading its own key-value store, an application, a launcher) gives every rank:

- the same `CommunicatorId` (16 random bytes; it keeps independent communicators between the same hosts from cross-connecting and is not a secret), and
- a `RankDirectory` with each rank's explicit **control** and **data** endpoint.

The usual sequence is:

1. Each rank binds its listeners first (a port of 0 means any free port) so the actual endpoints can be published.
2. The application collects every rank's endpoint and shares one communicator id.
3. Each rank calls `Communicator::create()` with its rank, the world size, the id and the directory.

The C ABI exposes the same model through caller-exchanged 256-byte opaque endpoint blobs ([C API bootstrap](../reference/c-abi-bootstrap.md)). `Communicator::create()` is expensive (real network setup and capability negotiation): create one communicator per logical group and reuse it.

## Connections

For every pair of ranks the **lower rank dials and the higher rank accepts**, so the last rank binds nothing. Each pair has:

- one **control** connection: handshake, abort and goodbye frames, and the collective descriptors and verdicts;
- two **data** connections: `Data` for point-to-point payloads and `CollectiveData` for collective payloads (see {doc}`ordering-domains`).

At world size 2 all connections are made during bootstrap. Above 2, the control plane is a full mesh and data connections are opened lazily by the first transfer over an edge, so a collective only opens the edges its algorithm uses. A world of size 1 opens no socket. World sizes 1 to 4 are validated; at most 8 are accepted (`kMaxFullMeshWorldSize`).

## Handshake

Every connection starts with a fixed-size handshake that carries the wire protocol version, the communicator id, the rank, the world size and the connection role. A wrong communicator id, a duplicate rank, a rank outside the world, a world-size or protocol-version mismatch, or an unexpected role is rejected on **both** sides with a `protocol_mismatch:` error, never a hang. The whole bootstrap is bounded by `bootstrap_timeout`.

## Queries

`rank()`, `world_size()`, `capabilities()` (a per-rank table), `failed()` and `aborted()` are cheap and never block.

## Lifetime

Destroying a communicator stops its workers, joins its threads and releases its sockets; it never touches caller-owned buffers. Destroying a `Work` handle does not cancel the operation, and it does not make the buffer safe to free. See {doc}`failure-handling` for abort and destruction with work in flight.
