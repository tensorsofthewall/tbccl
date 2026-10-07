# ADR 0004: Explicit wire protocol versioning with strict rejection

- Status: accepted
- Date: 2026-10-03

## Context

Ranks of one communicator may be built and started separately, possibly from different TBCCL versions. A silent mismatch in what ranks send corrupts data or hangs.

## Decision

Every connection begins with a fixed-size handshake that carries `kWireProtocolVersion`, the communicator id, the rank, the world size and the connection role. The accepting side checks the version first and answers any mismatch with a rejection, so the failure surfaces as `protocol_mismatch:` on both sides. Peers with different wire versions are never accepted: there is no negotiation across versions, and the library does not infer a peer's version from connection arrival order. The wire version is independent of the package version and of the C ABI version, and it is bumped whenever what peers exchange changes incompatibly (3 to 4 when the second data connection was added).

## Consequences

- Mixed-version worlds are impossible by design and fail promptly and cleanly, in both dial directions, with nothing but the handshake exchanged.
- Adapters declare the wire protocol versions they were validated against (torch-tbccl also warns when it is linked against a version it has not been tested with).
- A wire change is a release-significant event (a minor version at least); see [Versioning](../reference/versioning.md).
