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

```cpp
tbccl::CommunicatorOptions opts;
opts.rank = 0; // or 1
opts.peers = {{"host0", port0}, {"host1", port1}}; // world_size = peers.size()
auto comm = tbccl::Communicator::create(opts);
```

This phase's async P2P/AllReduce data path supports exactly
`world_size == 2` (matching every proven async mechanism in this
codebase); `create()` throws for any other size rather than pretending
to support it.

`comm->rank()`, `comm->world_size()`, `comm->capabilities()`,
`comm->failed()` are cheap, non-blocking queries.

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

## P2P

```cpp
auto work = comm->send(view, count, datatype, peer_rank);
// ... or ...
auto work = comm->recv(view, count, datatype, peer_rank);
work.wait();
if (work.has_error()) { /* work.error() is a human-readable message */ }
```

`peer_rank` must be this communicator's single other rank (0 or 1).
Multiple outstanding `Work` objects from the same `Communicator` are
supported (`TensorCommWorker` processes them FIFO, one at a time, in
submission order).

## AllReduce

```cpp
auto work = comm->all_reduce(send_view, recv_view, count, datatype, tbccl::ReduceOp::Sum);
```

- `send_view`/`recv_view` may be the **same** `BufferView` (in-place,
  the common case, as in `examples/async_allreduce.cpp`) or different
  (TBCCL copies `send_view`'s content into `recv_view`'s location
  first -- Host/MetalShared only; out-of-place is not supported for
  `MemoryKind::Cuda` buffers this phase, pass the same view for both).
- Only `ReduceOp::Sum` is supported this phase; anything else returns
  `Unsupported`.
- The root is always rank 0 internally -- there is no caller-visible
  root parameter. (The older, internal `tbccl::n2_all_reduce_tensor()`
  entry point still exposes a `root` parameter for benchmark
  diagnostics; the public `Communicator::all_reduce()` does not.)
- All ranks must call `all_reduce()` in the same order (collective
  ordering is the caller's responsibility, same as every collective
  library). Only one collective may be in flight on a given
  `Communicator` at a time (single-active-collective, matching Phase
  39's proven design).

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

- World size is fixed at exactly 2 for the full async data path.
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
