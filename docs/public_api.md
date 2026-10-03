# TBCCL Public API

This documents the framework-independent public C++ API introduced in
Phase 41. See `docs/framework_integration_architecture.md` for the
design rationale and `examples/async_allreduce.cpp` for a complete,
minimal, public-headers-only usage example.

## Install / link

```sh
cmake -S . -B build
cmake --build build
cmake --install build --prefix /path/to/install
```

```cmake
find_package(TBCCL CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE TBCCL::tbccl)
```

This installs the core library and `include/tbccl/*.hpp` only. It
requires nothing beyond a C++17 compiler and Threads to build and link
against for `MemoryKind::Host`/`MemoryKind::MetalShared` buffers.

**CUDA support is an optional installable component.** Configure with
`-DTBCCL_ENABLE_CUDA=ON` and the install additionally contains
`libtbccl_cuda.a` and `TBCCLCudaTargets.cmake`; a host-only configure
installs neither and stays fully valid. Core `tbccl` never depends on CUDA.

```cmake
find_package(TBCCL CONFIG REQUIRED)
if(TBCCL_cuda_FOUND)
    target_link_libraries(your_app PRIVATE TBCCL::tbccl_cuda)  # pulls CUDA::cudart
endif()
```

then call `tbccl::register_cuda_support()` (declared in the public
`<tbccl/cuda_support.hpp>`, idempotent) once before constructing any
`Communicator` that will move CUDA buffers. The consuming project does not
need to enable the CUDA language. Both `tbccl` and `tbccl_cuda` are built
position-independent so they can be linked into shared objects.

Metal-shared support needs **no separate step at all** -- see "Memory
kinds" below.

## Lifetime and ownership

- `Communicator::create()` is expensive (real network bootstrap +
  capability negotiation). Create one per logical group and reuse it;
  `send()`/`recv()`/`all_reduce()` are cheap to call repeatedly.
- `BufferView` **never** owns the memory it points to. The caller
  guarantees it remains valid until the returned `Work` completes.
  TBCCL never frees, reallocates, or takes ownership of it.
- `Communicator`'s destructor stops its internal workers, joins
  threads, and releases the transport -- it never touches any
  caller-owned buffer.
- A `Work` handle's destruction does not cancel the operation: the
  underlying shared state outlives the handle, and the Communicator's
  own worker(s) complete the operation regardless of whether anyone is
  still holding a `Work` for it.

## Communicator

A `Communicator` is one rank of an N-rank world (`world_size` 1 to 4 validated; at most `kMaxFullMeshWorldSize` = 8 accepted). libtbccl never discovers
peers: whoever bootstraps the ranks (an adapter reading its own store, an application, a launcher) gives every rank the same `CommunicatorId` and a
`RankDirectory` with each rank's explicit **control** and **data** endpoint.

```cpp
// 1. Bind this rank's listeners first (port 0 = any free port) so the ACTUAL endpoints can be published.
auto listeners = tbccl::CommunicatorListeners::bind("10.0.0.5");       // only a rank with a higher rank above it listens
tbccl::RankEndpoint mine{rank, listeners->control(), listeners->data()};
// 2. ... publish `mine` however you like, collect every rank's endpoint, share one CommunicatorId (CommunicatorId::generate() on one rank) ...
tbccl::CommunicatorOptions opts;
opts.rank = rank;
opts.world_size = n;
opts.communicator_id = id;
opts.rank_directory.entries = {rank0_endpoint, rank1_endpoint, ...};   // ordered by rank; validated before any network work
opts.listeners = listeners;                                              // optional: otherwise the directory entry is bound
auto comm = tbccl::Communicator::create(opts);
```

The legacy two-rank form still works and resolves into the same explicit endpoints (control = `peers[r]`, data = `peers[0].port + 1000`, nil communicator id):

```cpp
opts.rank = 0; // or 1
opts.peers = {{"host0", port0}, {"host1", port1}};
```

Connection rule: for every pair the **lower rank dials and the higher rank accepts**, so the last rank binds nothing and exactly two sockets (control, data) exist per pair.
Every connection starts with a handshake that carries the wire protocol version (`kWireProtocolVersion`, independent of the package version), the communicator id,
the rank, the world size and the connection role. A wrong communicator id, a duplicate rank, a rank outside the world, a world-size or protocol-version mismatch is
rejected on both sides with a `protocol_mismatch:` error, never a hang; the whole bootstrap is bounded by `bootstrap_timeout`. `world_size == 1` opens no socket.

`comm->rank()`, `comm->world_size()`, `comm->capabilities()` (a per-rank table: `capabilities().for_rank(r)`), `comm->failed()`, `comm->aborted()` are cheap, non-blocking queries.

## Memory kinds

```cpp
enum class tbccl::MemoryKind { Host, Cuda, MetalShared };
```

- **Host**: an ordinary `malloc`/`new`/`std::vector`-backed CPU pointer.
  Works out of the box.
- **MetalShared**: the CPU-visible `.contents` pointer of an
  `MTLBuffer` allocated with `MTLResourceStorageModeShared`. Works out
  of the box, with **zero Metal-specific code anywhere in
  Communicator** -- once you have taken `.contents`, it is ordinary
  CPU-visible memory as far as TBCCL's transport is concerned (see the
  architecture doc's Section 5). `MTLResourceStorageModePrivate`
  buffers are not supported (not CPU-addressable).
- **Cuda**: a `cudaMalloc`'d device pointer. Requires
  `tbccl::register_cuda_support()` to have been called once (see
  "Install / link" above); otherwise `Communicator` returns an
  `Unsupported` error for it, same as any other unregistered kind.

## BufferView

```cpp
tbccl::BufferView view{tbccl::MemoryKind::Host, ptr, bytes, /*device_ordinal=*/0};
```

`bytes` is explicit, not inferred from a collective's `count`/`datatype`
-- every call validates `count * datatype_size(datatype) <= bytes`
with overflow-safe arithmetic and throws on mismatch. A zero-count call
is always valid regardless of `bytes`. `device_ordinal` is meaningful
only for `MemoryKind::Cuda`.

## Execution context (CUDA stream readiness)

```cpp
tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, my_cuda_stream};
auto work = comm->send(view, count, datatype, peer, ctx);
```

If you write to a CUDA buffer asynchronously on your own stream and
then call `send()`/`all_reduce()` without a host synchronize, pass that
stream as an `ExecutionContext`: TBCCL records an event on it and makes
its own copy stream wait on that event before reading the buffer, so
you never need `cudaDeviceSynchronize()` as a correctness crutch. See
`tests/communicator_cuda_test.cpp`'s `test_delayed_producer_stream_dependency`
for a verified example (a kernel that deliberately delays its write,
proving the dependency is genuinely enforced, not a lucky race).

Omitting `context` (the default) means "already synchronized, safe to
read now" -- the convention every prior phase's benchmark-owned tensor
already used.

## Datatypes: reduction types vs byte transport (Phase 49)

`tbccl::DataType` names the element type of an **arithmetic** collective only: `Int32, Int64, Float32, Float64` (original values 0-3, never changed) and, appended in
Phase 49, `Int8 = 4, UInt8 = 5, Float16 = 6, BFloat16 = 7`. `datatype_size()` is the single size table; `reduction_supported(datatype, op)` is the single support
predicate; `validate_reduction()` throws `unsupported: reduction dtype=... op=... (supported ops for this dtype: ...)` before any communication.
`Communicator::capabilities().supports_collective_all_reduce(kind, datatype, op)` answers per memory kind. Float16/BFloat16 SUM widens to float32, adds once and rounds once
(ties-to-even); Int8/UInt8 SUM is addition modulo 256; neither provides Product/Min/Max. FP8, packed INT4/FP4 and any other quantized payload are **not** `DataType`s: they travel as
opaque bytes through `send`/`recv`, `broadcast` and `all_gather`. See `docs/quantized_payloads.md`.

For P2P, `count` and `datatype` are only a size check (`count * datatype_size(datatype) <= view.bytes`); the transfer moves `view.bytes`. Describe an opaque payload as
`DataType::UInt8` with `count = bytes`.

## P2P

```cpp
auto work = comm->send(view, count, datatype, peer_rank);
// ... or ...
auto work = comm->recv(view, count, datatype, peer_rank);
work.wait();
if (work.has_error()) { /* work.error() is a human-readable message */ }
```

`peer_rank` is any other rank of the world (a self send/recv, or a rank outside the world, throws `invalid_argument:` and poisons nothing). There is no `ANY_SOURCE` and no
message tag: P2P is FIFO per peer and direction. Operations to different peers are independent (a receive that cannot complete yet never delays traffic with another peer), and
send and receive on one peer pair do not block each other, so two ranks may send to each other simultaneously and a ring (every rank sends to its successor while receiving from its
predecessor) completes. P2P messages carry a length header: a receive posted for a different byte count than the sender sent fails with `protocol_mismatch: ... size mismatch`, poisons the
communicator (the stream position is unknown) and never reads beyond the posted size. A P2P `Work` shares a peer's lane with that peer's collective traffic, so do not interleave a collective with
P2P to the same peer in an order that differs between ranks.

## Collectives

All collectives are asynchronous (`Work`), run in call order on every rank (one collective executes at a time per communicator), and share the ordering contract: every rank issues the same
collectives in the same order. Phase 51 selects the N>2 algorithm internally (rank 0 decides per collective and every rank runs the same plan; N=2 keeps its specialised path and is never rerouted). Phase 50's root-based algorithms remain as the `reference` fallback (debug override only). Full table and thresholds: `docs/collective_algorithms.md`; floating-point reduction semantics: `docs/numerical_reduction_semantics.md`.

| call | N=1 | N=2 | N>2 default algorithm |
|---|---|---|---|
| `barrier()` | completes locally | descriptor exchange | reference (control-plane gather/release) below N=9; dissemination from N=9 |
| `broadcast(buffer, root)` | no-op | specialised single transfer | binomial tree |
| `all_gather(input, outputs)` | local copy | specialised pairwise exchange | ring |
| `all_reduce(send, recv, count, dtype, op)` | local | specialised heterogeneous engine (unchanged) | recursive doubling (power-of-two N <= 4, small), binomial tree (small), ring reduce-scatter + all-gather from ~96 KiB x (N-2) |

Data connections for N>2 are created lazily by the first transfer over an edge; the control plane stays a full mesh. `ExternalMemoryProvider` gained one **optional** virtual, `reduce_backend_range` (reduce a received sub-range into a buffer; default: unsupported, which makes the ring all-reduce unavailable for that provider and the planner fall back). Host and CUDA providers implement it. Package version 0.4.0 at Phase 51 (0.5.0 from Phase 52), wire protocol version 3.

`broadcast` and `all_gather` are byte-generic (any dtype, FP8, packed INT4, ...). `all_reduce` accepts `ReduceOp::Sum` of Float32, Float64, Int32, Int64 everywhere, Int8/UInt8 (modulo-256 sum, associative
so it extends to N>2), and Float16/BFloat16 **only for N=2**: for N>2 they are rejected up front with `unsupported: ... N>2 reduction semantics are not defined`. The N=2 path keeps its
descriptor-free wire format and cost.

**Collective sequence and descriptors (N != 2, and `barrier` at any N).** Each collective starts with every rank sending a small descriptor (sequence number, kind, root, count, dtype, op, bytes,
local memory kind, forced algorithm, and whether this rank can run it) to rank 0 **over the control plane**, which answers every rank with a verdict carrying the chosen algorithm before any payload moves (the dissemination barrier validates descriptors at rank 0 asynchronously instead of waiting for a verdict):
a rank-local capability problem (an unregistered memory kind, an Int8 reduction on a `MemoryKind::MetalShared` buffer) fails the collective on **every** rank with `unsupported:` naming the rank,
without poisoning the communicator; a disagreement on sequence, kind, root, element count, dtype, op or byte count fails every rank with `protocol_mismatch:` and aborts the communicator. Set `TBCCL_TRACE=1` for a
per-rank log of sequence, kind, bytes, dtype, peer and verdict.

## Abort and failure

`abort(reason)` is communicator-wide (Phase 45, generalized in Phase 50): the aborting rank tells every peer over the control channel before it interrupts its own work, each peer aborts and relays,
every blocked operation on every healthy rank fails, and the communicator becomes terminal. A peer that dies abruptly (its control connection closes without a `Goodbye`) aborts the others; a clean
destruction sends `Goodbye`, so an idle rank finishing early does not abort anyone. A rank that is alive but silent cannot be detected by the library: any rank may abort (a timeout in the caller),
and that reaches the blocked ranks. No recovery, shrinking or renumbering exists: everyone fails. A `Work` becomes terminal only after no TBCCL thread can touch its buffer.

## Errors

The C++ API throws `std::runtime_error`. Messages are prefixed with a
lowercase tag matching an `ErrorCode` (`invalid_argument:`,
`unsupported:`, `transport_error:`, `peer_failure:`, ...) so a caller
can branch on the prefix without needing a separate C-ABI-style error
code this phase. Once a transport/protocol failure occurs,
`Communicator::failed()` becomes `true` and all subsequent operations
fail immediately rather than hanging -- no automatic reconnection is
attempted.

## Thread safety

A single `Communicator` may be called from multiple application
threads concurrently for *submission* -- `TensorCommWorker`'s queue and
the collective executor are internally synchronized. Collective
**ordering** across ranks remains the caller's responsibility (same
contract every collective library has): if two threads on the same
rank submit `all_reduce()` calls concurrently, the peer rank must issue
its matching calls in the same relative order.

## Known limitations (honest, not silently dropped -- see docs/phase41_report.md)

- World size 1 to 4 is validated (full mesh, two sockets per rank pair; up to 8 accepted). N>2 collectives are unoptimized reference algorithms; N>2 Float16/BFloat16 reduction is rejected.
- Only `ReduceOp::Sum` is supported for AllReduce.
- CUDA support requires building from source and linking the optional
  device component directly; it is not yet part of the installable
  package.
- No C ABI yet (recommended as the first piece of Phase 42 -- the C++
  API design was deliberately kept simple enough, with no STL types in
  the Communicator/Work/BufferView surface beyond `std::string` error
  messages, that wrapping it is expected to be straightforward).
- Out-of-place AllReduce (`send_buf.data != recv_buf.data`) is not
  supported for `MemoryKind::Cuda`.


## Phase 52: nonblocking submission, structured errors and the C ABI

**Submission never waits.** `Communicator::send` / `recv` (and collectives, as before) return a `Work` without waiting for socket progress, a matching operation on the peer, staging or lane capacity. Each peer lane admits into a growing queue of lightweight descriptors (no payload copy, no staging, no device scratch at admission); the persistent per-peer TX/RX threads move a request to the bounded active/staging resources when its turn comes. FIFO per (peer, direction) is preserved; a stalled peer never delays posting to or progress with another peer. A call either accepts immediately or throws immediately (`ErrorCode::ResourceExhausted` for an allocation failure at admission, `Aborted` after an abort). With no tags the receiver must still post matching P2P operations in a compatible logical order per peer pair. `BufferView` does not own memory: it must stay valid and unmodified until the operation is terminal, and destroying a `Work` neither cancels the operation nor releases that obligation. See `docs/phase52_submission_audit.md` and `docs/grouped_operations_audit.md` (individual nonblocking posting is sufficient; no groups).

**Structured errors.** `tbccl::Error` (`<tbccl/error.hpp>`, derives from `std::runtime_error`) carries an `ErrorCode` chosen where the failure happens; `Work::error_code()` is the structured terminal result (independent of `Work::error()`'s text, which keeps its historical `tag: detail` shape for humans and the torch-tbccl mapper). `ErrorCode::ResourceExhausted` was added; an untyped exception is `InternalError`, `std::bad_alloc` is `ResourceExhausted`. `Work::wait_for(timeout)` is a timed wait that neither cancels nor changes the operation. `TensorCommWorker(..., queue_depth == TensorCommWorker::kUnboundedAdmission)` is the unbounded-admission mode the Communicator uses; a non-zero depth keeps the old blocking backpressure for standalone users.

**C ABI v1.** `include/tbccl/tbccl.h` and `TBCCL::tbccl_c` are the stable C interface (`docs/c_abi_v1.md` is authoritative; quickstart `docs/c_api_quickstart.md`, bootstrap `docs/c_api_bootstrap.md`, threading `docs/c_api_thread_safety.md`). The C++ API above is not an ABI promise. The shim wraps this runtime and adds nothing to it: fixed-width constants (no C enums), `struct_size`-prefixed structs, opaque handles, caller-exchanged endpoint blobs for bootstrap, byte-based P2P, barrier/broadcast/all_gather/all_reduce, `Work` queries whose API status is separate from the operation result, and a CUDA-free header. Install: `include/tbccl/tbccl.h` and `TBCCL::tbccl_c` (the same symbol set in host-only and CUDA builds).
