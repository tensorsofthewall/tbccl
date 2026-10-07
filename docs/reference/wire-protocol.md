# Wire protocol (version 4)

The wire protocol is private to libtbccl: it is how ranks talk to each other, and it is never visible through the C ABI. It is documented here so operators can reason about compatibility, ports and failures. Do not implement it independently.

## Version

`kWireProtocolVersion` is **4**. Version 3 and version 4 do not interoperate ([Versioning](versioning.md)). The package version, the C ABI version and the wire version are independent.

## Connections per rank pair

| Connection | Role value | Carries |
|---|---|---|
| control | `Control` = 1 | handshake, abort and goodbye frames, capability exchange, collective descriptors and verdicts |
| data | `Data` = 2 | point-to-point payloads, each framed |
| collective data | `CollectiveData` = 3 | collective payloads, unframed |

Version 3 had only the control and one data connection. The lower rank of a pair dials and the higher rank accepts. A rank advertises one control and one data endpoint; **the same data listener accepts both data roles**, and the role in the handshake (never the arrival order) decides which connection a socket is.

Establishment: at world size 2 all connections are made during bootstrap (the lower rank accepts both data roles in either order). At world size above 2 data connections are opened lazily, one per traffic domain, on first use. A peer that connects only one data role ends in a bounded `timeout:` error that names the missing role; a duplicate of one role is a `protocol_mismatch: duplicate rank`.

## Handshake

The first message on every connection is a fixed 128-byte `Hello` from the dialing side, answered with a 128-byte `HelloAck` from the accepting side. The `Hello` carries the wire version, the communicator id, the rank, the world size and the connection role. The `HelloAck` carries `Ok` or a rejection reason (bad version, wrong communicator, duplicate rank, rank out of range, world-size mismatch, unexpected role, unexpected rank). A rejection surfaces as `protocol_mismatch:` on **both** sides. The accepting side checks the version first.

All integers are serialized big-endian into fixed-size buffers; no structure is ever sent directly.

## Frames

| Message | Size | Where |
|---|---|---|
| `Hello` / `HelloAck` | 128 bytes | first message on every connection |
| control frame (abort, goodbye) | 272 bytes | control connection after the handshake |
| point-to-point header | 16 bytes: magic `TBMP`, a reserved field, a 64-bit payload length | `Data` connection, before each payload |
| collective descriptor and verdict | fixed size | control connection, before each collective at world size other than 2, and for `barrier` at any size |

A P2P receive validates the header as soon as it arrives and never reads beyond the posted size. Collective payloads on `CollectiveData` carry no header; their order is fixed by the collective sequence.

## Security

The communicator id in the handshake is an identity token that keeps independent communicators apart; it is not a secret and has no cryptographic property. The protocol and the library contain no encryption or authentication code. Use it only between peers you trust, on networks you trust.
