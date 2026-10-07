# Ordering domains: point-to-point and collectives

Point-to-point (P2P) transfers and collectives are **independent ordering domains**. They may be in flight at the same time on one communicator, to the same peer, from any threads, and their relative submission order is allowed to differ between ranks.

## What is guaranteed

| Property | Guarantee |
|---|---|
| P2P order | FIFO per (peer, direction) |
| Collective order | every rank issues the same collectives in the same order |
| Simultaneous send and receive on one peer pair | supported; a ring (every rank sends to its successor while receiving from its predecessor) completes |
| P2P and collectives together | safe, in any relative order across ranks |
| Failure | a failure in either domain fails the communicator |

Legal examples: rank 0 calls `all_reduce` then `send(1)` while rank 1 calls `recv(0)` then `all_reduce`; one rank calls `send` then `all_reduce` while the other calls `all_reduce` then `recv`; one application thread issues P2P calls while another issues collectives.

Still required of the application: matching P2P calls per peer pair (there are no tags and no `ANY_SOURCE`), the same collective sequence on every rank, and buffers that stay valid until the `Work` is terminal.

## How it works

Each peer pair has two data connections, each with its own duplex worker:

- `Data` carries P2P messages, each with a 16-byte header (magic, reserved field, 64-bit length) followed by the payload.
- `CollectiveData` carries collective payloads, unframed.

A byte on the `Data` connection can only be read by a P2P receive, and a byte on the `CollectiveData` connection only by a collective. Because the two kinds of traffic never meet on a byte stream, their relative order cannot matter. The property is structural, not timing dependent. The same advertised data endpoint accepts both connections; the connection role in the handshake, not arrival order, decides which one a socket is. See {doc}`../adr/0005-independent-p2p-and-collective-ordering` and {doc}`../adr/0006-separate-connections-per-traffic-domain`.

## Costs and limits

- Each peer pair holds two data sockets instead of one, and up to four network threads instead of two. Above world size 2 the collective connection is opened lazily by the first collective transfer to that peer.
- A peer that connects only one of the two data roles fails the bootstrap with a bounded timeout that names the missing role.
- Wire protocol 4 introduced this design. Peers speaking wire protocol 3 do not interoperate; see the [wire protocol reference](../reference/wire-protocol.md).
