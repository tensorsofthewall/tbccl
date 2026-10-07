# Transports

TBCCL separates the control plane and the data plane so that TCP is one implementation of a `Transport` interface, not an architectural assumption.

## What exists

| Component | Status |
|---|---|
| `Transport` interface (`include/tbccl/transport.hpp`) | implemented |
| `TcpTransport` | implemented; the only transport TBCCL has ever carried traffic over |
| `PeerCapabilities` exchange and `negotiate()` | implemented |

TBCCL is **not** an RDMA implementation. No RDMA, native Thunderbolt or Windows transport exists in this code base, and no claim is made about whether such a transport could be built for a given piece of hardware. The Thunderbolt 4 support described in these documents is ordinary TCP over a Thunderbolt-bridged network interface ([Thunderbolt link guide](../guides/thunderbolt-link.md)).

## The Transport seam

`Transport` offers `send` and `recv` with an exact-byte-count contract and a `capabilities()` description (reliable, ordered, zero-copy, registered-memory, direct-device-memory, preferred alignment, maximum in-flight). Code above the interface branches on capabilities, never on a transport's concrete type, so there is no `if (transport == TCP)` in transfer-scheduling code. The public signatures avoid POSIX-specific types, which is a design intention for portability; only Linux and macOS are built and tested.

## Capability negotiation

When a connection is set up, ranks exchange a `PeerCapabilities` record once over the control connection: wire protocol version, operating system, advertised transports, advertised memory backends, advertised asynchronous capabilities, and chunk and alignment hints. `negotiate()` picks a transport from what this build actually implements (never a claimed-but-unimplemented capability) and reports the common memory backends as information.

## Platforms

Linux (x86-64) and macOS (arm64 and x86-64) are the supported platforms for the C ABI layout checks; the validated hardware pairing is a Linux/NVIDIA host and an Apple-silicon Mac (see [Platform capabilities](../reference/platform-capabilities.md)). Only IPv4 endpoints are supported.
