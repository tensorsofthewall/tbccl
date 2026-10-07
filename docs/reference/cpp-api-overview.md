# C++ API overview

The public C++ API lives in `include/tbccl/` and is generated in full in the [C++ API reference](cpp-api.md). This page explains how the pieces fit. The C++ API is a source-level API and is **not** an ABI promise; use the [C ABI](c-abi.md) for a stable binary interface. A complete, minimal, public-headers-only example is `examples/async_allreduce.cpp`.

## Install and link

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build --prefix /path/to/install
```

```cmake
find_package(TBCCL CONFIG REQUIRED)
target_link_libraries(your_app PRIVATE TBCCL::tbccl)
```

This installs the core library and `include/tbccl/*.hpp`. It needs only a C++17 compiler and Threads for `Host` and `MetalShared` buffers. CUDA support is an optional installable component: configure with `-DTBCCL_ENABLE_CUDA=ON` and the install also contains `libtbccl_cuda.a` and `TBCCLCudaTargets.cmake`.

```cmake
find_package(TBCCL CONFIG REQUIRED)
if(TBCCL_cuda_FOUND)
    target_link_libraries(your_app PRIVATE TBCCL::tbccl_cuda)  # pulls CUDA::cudart
endif()
```

Then call `tbccl::register_cuda_support()` (declared in `<tbccl/cuda_support.hpp>`, idempotent) once before constructing a `Communicator` that moves CUDA buffers. The consuming project does not need to enable the CUDA language. Both libraries are built position-independent.

## Creating a communicator

```cpp
// 1. Bind this rank's listeners first (port 0 = any free port) so the ACTUAL endpoints can be published.
auto listeners = tbccl::CommunicatorListeners::bind("10.0.0.5");
tbccl::RankEndpoint mine{rank, listeners->control(), listeners->data()};
// 2. Publish `mine`, collect every rank's endpoint, and share one CommunicatorId (CommunicatorId::generate() on one rank).
tbccl::CommunicatorOptions opts;
opts.rank = rank;
opts.world_size = n;
opts.communicator_id = id;
opts.rank_directory.entries = {rank0_endpoint, rank1_endpoint, /* ... ordered by rank */};
opts.listeners = listeners;   // optional: otherwise the directory entry is bound
auto comm = tbccl::Communicator::create(opts);
```

The older two-rank form still works: set `opts.rank` and `opts.peers = {{"host0", port0}, {"host1", port1}}`; each rank's control endpoint is `peers[r]`, its data endpoint is the same host with the port plus 1000, and the communicator id is nil. See [Communicators](../concepts/communicators.md) for the connection and handshake rules.

## Buffers and execution context

```cpp
tbccl::BufferView view{tbccl::MemoryKind::Host, ptr, bytes, /*device_ordinal=*/0};
tbccl::ExecutionContext ctx{tbccl::ExecutionContextKind::CudaStream, my_cuda_stream};
auto work = comm->send(view, count, datatype, peer, ctx);
```

`bytes` is explicit. Every call validates `count * datatype_size(datatype) <= bytes` with overflow-safe arithmetic and throws on mismatch; a zero-count call is always valid. For P2P, `count` and `datatype` are only a size check and the transfer moves `view.bytes`; describe an opaque payload as `DataType::UInt8` with `count = bytes`. Pass an `ExecutionContext` when a CUDA buffer is written asynchronously on your own stream; omit it for already-synchronized buffers. See [Memory providers](../concepts/memory-providers.md).

## Point-to-point

```cpp
auto work = comm->send(view, count, datatype, peer_rank);   // or comm->recv(...)
work.wait();
if (work.has_error()) { /* work.error_code(), work.error() */ }
```

`peer_rank` is any other rank of the world; a self send, or a rank outside the world, throws `invalid_argument:` and poisons nothing. There is no `ANY_SOURCE` and no message tag: P2P is FIFO per peer and direction. Operations to different peers are independent, and send and receive on one pair do not block each other. P2P messages carry a length header; a receive posted for a different byte count than the sender sent fails with `protocol_mismatch: ... size mismatch` and poisons the communicator. See [Ordering domains](../concepts/ordering-domains.md).

## Collectives

| call | N=1 | N=2 | N>2 default algorithm |
|---|---|---|---|
| `barrier()` | completes locally | descriptor exchange | reference below N=9; dissemination from N=9 |
| `broadcast(buffer, root)` | no-op | specialised single transfer | binomial tree |
| `all_gather(input, outputs)` | local copy | specialised pairwise exchange | ring |
| `all_reduce(send, recv, count, dtype, op)` | local | specialised heterogeneous engine | recursive doubling, binomial tree or ring by size |

`all_reduce` accepts `ReduceOp::Sum` of Float32, Float64, Int32, Int64 everywhere, Int8 and UInt8, and Float16 and BFloat16 only at world size 2. `broadcast` and `all_gather` are byte-generic. See [Collectives](../concepts/collectives.md), [Collective algorithms](collective-algorithms.md) and [Numerical semantics](numerical-semantics.md).

## Work

Every post returns a `Work`. `wait()`, `wait_for(timeout)`, `is_completed()`, `has_error()`, `error()` and `error_code()` are safe from any thread. Destroying a `Work` handle does not cancel the operation; the shared state outlives the handle and the communicator's workers complete the operation regardless.

## Errors, abort and threads

The API throws `tbccl::Error` (a `std::runtime_error` carrying an `ErrorCode`); messages start with a lowercase tag such as `invalid_argument:`. `comm->abort(reason)` is communicator-wide. After a failure `failed()` is true and further operations fail immediately. See [Failure handling](../concepts/failure-handling.md).

A single `Communicator` may be used from several application threads concurrently for submission. Collective **ordering** across ranks remains the caller's responsibility: if two threads on one rank submit collectives concurrently, the peers must issue the matching collectives in the same relative order.

## Limits

- World sizes 1 to 4 are validated (up to 8 accepted); only `Sum` reductions through `Communicator::all_reduce`; Float16/BFloat16 reductions at world size 2 only.
- Out-of-place `all_reduce` is not supported for `MemoryKind::Cuda`.
- IPv4 endpoints only; TCP is the only transport.
