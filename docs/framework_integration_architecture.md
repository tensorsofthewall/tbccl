# Framework Integration Architecture (Phase 41)

## 1. Purpose and non-goals

This document defines the framework-independent public runtime boundary
for TBCCL. It is written before any Phase 41 implementation, per the
phase plan's own ordering requirement.

`libtbccl` **never** depends on PyTorch, Python, vLLM, exo, or MLX.
Those projects depend on `libtbccl`, via out-of-tree adapters built in
later phases. Forbidden dependency edges:

```
libtbccl -> torch     FORBIDDEN
libtbccl -> vllm      FORBIDDEN
libtbccl -> mlx       FORBIDDEN
libtbccl -> exo       FORBIDDEN
```

Desired edges (future phases, not built here):

```
torch-tbccl -> libtbccl
vllm-tbccl  -> libtbccl
mlx-tbccl   -> libtbccl
```

## 2. Layering

```
          APPLICATION
              |
       FRAMEWORK ADAPTER        (future, out-of-tree)
              |
              v
     +------------------+
     | TBCCL public API |       <- Phase 41 builds this
     +------------------+
              |
         Communicator
              |
    +---------+----------+
    |                     |
Collectives              P2P
    |                     |
    +---------+----------+
              |
       async execution     (TensorCommWorker, promoted)
              |
       memory providers    (AsyncMemoryBackend implementations)
              |
          Transport        (TcpTransport, existing)
              |
             TCP
```

## 3. Ownership audit (existing internal types)

| Current type | Owns what | Public-ready as-is? | Phase 41 role |
|---|---|---|---|
| `StagingPool` | Preallocated host slots | Yes, internal | Stays fully internal |
| `AsyncMemoryBackend` | Nothing (pure interface) | Yes | Stays as the SPI a `BufferView`-backed provider implements, unchanged |
| `TransferWork` | Shared completion state | Yes | Promoted directly to the public `Work` type (type alias, no rewrite) |
| `TransferRequest` | Non-owning raw pointers | Mostly | Stays internal; `Communicator` builds these from `BufferView` |
| `TensorCommWorker` | FIFO queue + 2 threads + cached `StagingPool` | Yes | Becomes `Communicator`'s P2P progress engine (owned, not exposed) |
| `Transport`/`TcpTransport` | `TcpTransport` owns a `Connection` | Yes | Stays internal to `Communicator` construction |
| `World`/`TcpWorldOptions`/`create_tcp_world` | `World` owns a `RingExecutor`; options describe static rank/peer list | Partially | `TcpWorldOptions`' shape becomes the model for `CommunicatorOptions`; `World` itself stays internal (used only for the readiness barrier, as established in Phase 38/39) |
| `PeerCapabilities`/`negotiate()` | Nothing (pure functions) | Yes, but uncached | `Communicator` runs negotiation once at construction and caches/exposes the result via `capabilities()` |
| `n2_all_reduce_tensor()` | Nothing (free function over refs) | No (N=2/SUM-only, root is a parameter) | Reused **unchanged** as the engine behind `Communicator::all_reduce`; root is fixed to rank 0 internally by the public API (Part X), SUM/N=2-only is surfaced honestly via `Unsupported` for anything else |
| `collectives.hpp` sync ops | Nothing (free functions over `World&`) | Yes, but a different concurrency model | Not exposed in the new public API this phase; remains available as the old internal/benchmark path |
| `HostAsyncBackend` | Nothing (wraps external `void*`) | **Yes, as-is** | Direct backing for `MemoryKind::Host` (and, per Section 5, `MemoryKind::MetalShared`) `BufferView`s |
| `CudaChunkedAsyncBackend` | Device buffers + pinned scratch (self-allocated via `allocate()`) | No | A new class, `CudaExternalAsyncBackend`, wraps an externally-owned device pointer while still owning its own persistent pinned staging scratch + copy stream + readiness event |
| `TensorBackendAsyncAdapter`/`MetalSharedDirectAsyncBackend` | Wrap a `TensorBackend&`, which owns memory | No (indirectly owning) | Not reused for the external-buffer path; see Section 5 |
| `BucketAllReduceWorker` | Thread + FIFO of non-owning job structs | No (benchmark-only, abort-on-first-failure) | Architecturally promoted (not reused verbatim) into `Communicator`'s internal collective executor, with per-job failure isolation instead of whole-worker abort |
| `tbccl`/`tbccl_tensor_backend` CMake targets | N/A | No `install()`/`export()` exists anywhere | Added from scratch this phase |

## 4. Public concepts

- `tbccl::BufferView` -- non-owning external buffer descriptor.
- `tbccl::DataType`, `tbccl::ReduceOp`, `tbccl::MemoryKind` -- plain enums.
- `tbccl::Work` -- promoted `TransferWork` (type alias, zero behavior change).
- `tbccl::Communicator` -- owns bootstrap, transport, workers, capabilities.
- `tbccl::CommunicatorOptions` -- framework-neutral bootstrap config (rank,
  world_size via peer list, timeouts, session id) -- directly modeled on
  the already-clean `TcpWorldOptions` shape, not reinvented.
- `tbccl::Capabilities` -- read-only view of negotiated
  `PeerCapabilities`/`NegotiationResult`.
- `tbccl::ErrorCode` -- structured error enum (`Success`, `InvalidArgument`,
  `Unsupported`, `TransportError`, `PeerFailure`, `Timeout`, `DeviceError`,
  `InternalError`).

Not exposed publicly: `TensorCommWorker`, `StagingPool`, `ChunkPlan`,
`TcpConnection`, `World`, `RingExecutor`, any benchmark `TensorBackend`
class.

## 5. BufferView and the Metal simplification

```cpp
enum class MemoryKind { Host, Cuda, MetalShared };

struct BufferView {
    MemoryKind memory_kind = MemoryKind::Host;
    void *data = nullptr;
    std::size_t bytes = 0;
    int device_ordinal = 0; // meaningful only for Cuda
};
```

`BufferView` never owns memory (Part 23/24). `device_ordinal` is present
for `Cuda` only; `MetalShared`'s `data` is required to already be the
buffer's CPU-visible pointer (the same pointer `MTLBuffer.contents`
yields) -- this is not a TBCCL-internal detail the caller must discover;
it is the external contract for this memory kind.

**Key simplification, derived directly from the Phase 40 audit**: for
the transport layer, a `MemoryKind::MetalShared` buffer is *already*
just a CPU-visible pointer + byte count -- indistinguishable from
`MemoryKind::Host` once `.contents` has been taken. `HostAsyncBackend`
(audit: "already exactly a wrap-an-external-Host-BufferView backend
with zero changes needed") is therefore reused, unchanged, as the
provider for both `Host` and `MetalShared` buffers. `MemoryKind` exists
to (a) document provenance/GPU-visibility expectations to the caller and
(b) leave room for a future, genuinely different `MetalPrivate` kind
(out of scope, explicitly unsupported) -- it does not imply a different
code path today. No new Objective-C/Metal code is required for Phase
41's external Metal-shared buffer support as a result.

CUDA is the one memory kind that needs new code: an external device
pointer cannot be treated as host-visible, so a dedicated
`CudaExternalAsyncBackend` (Section 7) is required.

## 6. Work

`tbccl::Work` is `tbccl::TransferWork` by alias -- the audit confirmed
its existing shape (`wait()`/`is_completed()`/`has_error()`/`error()`,
shared-state, repeated-wait-safe, friend-gated construction) is already
public-API-grade. No second async state machine is introduced. The
internal `detail::TransferWorkAccess` friend class gains one addition --
a factory usable by the collective executor (Section 8), not just
`TensorCommWorker` -- rather than inventing a parallel `Work` type for
collectives (avoiding Part BG's "second redundant worker layer"
anti-pattern).

Destruction semantics (Part Q item 96): **operation state survives Work
wrapper destruction** -- `TransferWork` already holds a `shared_ptr`, so
a destroyed `Work` handle whose underlying operation is still in flight
does not cancel it; the operation completes and its state is simply
unobservable. This matches the plan's preferred long-term behavior and
required no new code.

## 7. External CUDA provider

`CudaExternalAsyncBackend` (new): constructed from a caller-owned device
pointer + byte count (no `allocate()`), owns only:

- a persistent pinned host staging buffer (reused across transfers, like
  `CudaChunkedAsyncBackend`'s existing scratch),
- a persistent copy stream,
- one persistent, reused `cudaEvent_t` for producer-stream readiness
  (Part L) -- not allocated per chunk or per call.

Source readiness: the caller supplies an optional execution context
(Section 9); if a CUDA stream is given, TBCCL records an event on it and
makes its own copy stream wait on that event before starting D2H,
instead of requiring `cudaDeviceSynchronize()`. If no stream is given,
the buffer is assumed already ready (matching every prior phase's
benchmark-owned-tensor convention).

Destination completion: `Work::wait()` conservatively host-blocks until
H2D completes (Part M item 72) -- sufficient for Phase 41; a future
phase can expose a completion event through the same `ExecutionContext`
abstraction without redesigning this class.

TBCCL never calls `cudaFree()` on an externally-supplied pointer (Part
J); ownership tests confirm the caller can free it exactly once after
communicator/provider destruction.

## 8. Communicator and the collective executor

`Communicator` owns: rank, world_size (from peer list, not separately
configurable, matching `TcpWorldOptions`), one `Transport` connection,
one `TensorCommWorker` (P2P progress), one small collective-executor
thread (promoted from `BucketAllReduceWorker`'s design, generalized
beyond "bucket" naming and given **per-job failure isolation** instead
of whole-worker abort-forever -- the audit flagged this as the one
change needed before promotion), and the cached `NegotiationResult`.

`Communicator::all_reduce(send, recv, count, dtype, op)` routes to the
existing, **unchanged** `n2_all_reduce_tensor()`. World size is not
hardcoded to 2 at the `Communicator` level (Part T) -- a `Communicator`
with `world_size != 2` returns `ErrorCode::Unsupported` from
`all_reduce()` rather than pretending to support it. Root selection
(Part X) is fixed internally to rank 0; the old benchmark-only root
override remains available on the pre-existing `n2_all_reduce_tensor()`
entry point for diagnostics, not through the new public API.

## 9. Execution context

```cpp
enum class ExecutionContextKind { Host, CudaStream };
struct ExecutionContext {
    ExecutionContextKind kind = ExecutionContextKind::Host;
    void *native_handle = nullptr; // cudaStream_t, opaque at this layer
};
```

Generic public headers (`include/tbccl/*.hpp`) never include
`<cuda_runtime.h>`; `native_handle` is cast to `cudaStream_t` only inside
the CUDA-specific translation unit implementing
`CudaExternalAsyncBackend`.

## 10. Bootstrap

`CommunicatorOptions` is modeled directly on `TcpWorldOptions` (already
framework-neutral): rank, peer endpoint list, timeout. No `c10d::Store`,
Python dict, or exo topology object anywhere in `libtbccl` -- those
translate into `CommunicatorOptions` in out-of-tree adapters (Phase 42+).
Session identity reuses the existing capability-exchange handshake; no
duplicate handshake is added.

## 11. C ABI

v0, explicitly unstable, opaque handles only (`tbcclComm_t`,
`tbcclWork_t`), no STL types crossing the boundary. The C++ API is
canonical; the C ABI is a thin wrapper over it (Part AI direction A --
lower-risk given the existing codebase is entirely C++).

## 12. What Phase 41 deliberately does not change

- `n2_all_reduce_tensor()`'s algorithm: unchanged.
- `TensorCommWorker`'s internals: unchanged.
- `collectives.hpp`'s synchronous `World`-based API: unchanged, still
  present, not deprecated.
- Metal direct-vs-adapter default from Phase 40: unchanged (adapter
  remains default; direct remains opt-in).
- Selector, TCP transport, AER/thermal safety practice: unchanged.
