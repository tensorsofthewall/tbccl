# TBCCL transport architecture

Phase 32 introduced a control-plane/data-plane split so that TCP is one
implementation of a `Transport` interface, not a permanent architectural
assumption. This document records the current state and the intended
extension points for transports and platforms this project does not yet
implement.

## Layering

```
Async Tensor Transfer Layer (TransferRequest, TransferWork, ChunkPlan,
StagingPool, TensorCommWorker)          -- backend/transport-agnostic
        |                         |
        v                         v
Memory Backend                Transport
(Host/CUDA/Metal,             (Transport interface, transport.hpp)
benchmarks/tensor/)                 |
                                     v
                              TcpTransport (implemented)
                              RdmaTransport (not implemented)
                              NativeTbTransport (not implemented)
```

`Transport` (`include/tbccl/transport.hpp`) is the seam: `send`/`recv`
with the same exact-byte-count contract as `Connection`, plus
`capabilities()` returning a `TransportCapabilities` struct (reliable,
ordered, zero-copy, registered-memory, direct-device-memory,
preferred alignment, max in-flight). Code above this layer must branch
on capabilities, never on a transport's concrete type (Phase 32 Part F
item 28's explicit rule) -- there should never be an `if (transport ==
TCP)` in transfer-scheduling code.

`PeerCapabilities` (`include/tbccl/peer_capabilities.hpp`) is the
control-plane payload exchanged once at connection setup
(`exchange_capabilities()`), independent of any particular transfer:
protocol version, OS, advertised transports, advertised memory
backends, advertised async capabilities, and chunk/alignment hints.
`negotiate()` picks the transport (intersected against what this build
actually implements, never a claimed-but-unimplemented capability) and
reports the common memory backends as information for the caller, not
as a forced selection.

## Implemented this phase

| Component | Status |
|---|---|
| `Transport` interface | Implemented |
| `TcpTransport` (adapts existing `Connection`) | Implemented |
| `PeerCapabilities` / `exchange_capabilities()` / `negotiate()` | Implemented |
| Async substrate (`TransferRequest`/`TransferWork`/`TensorCommWorker`/`StagingPool`/`ChunkPlan`) | Implemented, host path validated |
| CUDA/Metal async backend adapters | Implemented as a worker-thread-driven wrapper over the existing synchronous `TensorBackend` stage calls (see docs/phase32_report.md item 41-45) -- not the full event/stream-overlap design Part S describes |

## Future transport backends (planned/unknown, not implemented)

### TCP

**Platforms**: macOS, Linux, and (unverified, unbuilt) Windows via a
future Winsock-backed `Connection`/`Transport` implementation behind
the same interface.

**Role**: universal fallback/baseline. This is the only transport this
project has ever run traffic over.

### Linux RDMA

**Status: unknown / future investigation required.** A plausible
provider is `ibverbs` (possibly via a Thunderbolt-specific provider, if
one exists for this hardware) or `rdma_cm`. No code has been written,
no library has been linked, and no claim is made here about whether
this machine's Thunderbolt/NHI hardware path can actually carry RDMA
traffic at all -- that is exactly the kind of claim Phase 32 was
explicitly told not to manufacture (Part AR item 156-158).

### macOS native transport

**Status: unknown / future investigation required.** No native
(non-TCP) Thunderbolt transport API has been identified or evaluated on
macOS for this project. Nothing beyond the TCP path is claimed.

### Windows native/RDMA

**Status: unknown / future investigation required.** Phase 32 does not
compile or run on Windows. The interfaces above (`Transport`,
`PeerCapabilities`) deliberately avoid POSIX-specific types
(`pollfd`, `sockaddr`, file descriptors) in their public signatures so
a future Windows `TcpTransport` built on Winsock could implement the
same `Transport` interface without changing anything above it -- this
is a design intention, not a tested claim, since it has never been
built or compiled on Windows.

## Why RDMA is not implemented yet

RDMA (via `ibverbs`, GPUDirect, or any other registered-memory
transport) is architecturally anticipated -- `TransportCapabilities`
already has `supports_zero_copy`/`supports_registered_memory`/
`supports_direct_device_memory` fields an RDMA transport would set true
-- but Phase 32's explicit scope (Part AR) excludes implementing it.
The rationale, per the plan: build and validate the substrate
(`TransferRequest`/`TransferWork`/`TensorCommWorker`/`StagingPool`/
`ChunkPlan`) against the one transport this project can actually test
today (TCP over the Thunderbolt-bridged link), so that a future RDMA
backend is a `Transport` implementation swap, not a rewrite of transfer
scheduling, staging, or the memory-backend abstraction.
