# Memory providers and buffers

TBCCL moves bytes that live in memory it does not own. A **memory provider** knows how to make a particular kind of memory readable and writable by the transport.

## Memory kinds

```cpp
enum class tbccl::MemoryKind { Host, Cuda, MetalShared };
```

| Kind | What it is | Requirements |
|---|---|---|
| `Host` | an ordinary CPU pointer from `malloc`, `new` or `std::vector` | none |
| `MetalShared` | the CPU-visible `contents` pointer of an `MTLBuffer` created with `MTLResourceStorageModeShared` | none; once `contents` is taken it is ordinary CPU-visible memory for the transport. `MTLResourceStorageModePrivate` buffers are not supported (not CPU-addressable) |
| `Cuda` | a `cudaMalloc` device pointer | the optional CUDA component (`-DTBCCL_ENABLE_CUDA=ON`, `TBCCL::tbccl_cuda`) and one call to `tbccl::register_cuda_support()` per process (`tbcclRegisterCudaSupport()` in C) |

An unregistered kind is an `Unsupported` error, as for any other unsupported kind.

## BufferView

```cpp
tbccl::BufferView view{tbccl::MemoryKind::Host, ptr, bytes, /*device_ordinal=*/0};
```

A `BufferView` **never owns memory**. The caller keeps the allocation valid, and unmodified for sends, until the returned `Work` is terminal; TBCCL never frees, reallocates or copies it to relax this contract. The byte count is explicit and is checked against `count * datatype_size` with overflow-safe arithmetic. `device_ordinal` only matters for `Cuda`. See {doc}`../adr/0002-non-owning-buffer-view`.

## CUDA staging and stream readiness

For CUDA buffers TBCCL stages chunks through persistent pinned host scratch memory on its own copy stream, overlapping device copies with network transfer. If the producer writes the buffer asynchronously on its own stream, pass that stream as an `ExecutionContext`: TBCCL records an event on it and makes its copy stream wait on that event before reading, so no host-side device synchronization is needed for correctness. Omitting the context means the buffer is already synchronized.

Out-of-place `all_reduce` (`send` buffer different from `recv` buffer) is not supported for `Cuda`; pass the same view for both.

## Provider interface

Providers implement a small internal interface (`ExternalMemoryProvider`): staging a chunk in and out, optional direct transport access, and an optional `reduce_backend_range` that reduces a received sub-range into a buffer. A provider without it makes the ring all-reduce unavailable and the planner falls back to another algorithm. Host and CUDA providers implement the reduction.

Which reduction types each memory kind supports is listed in [Platform capabilities](../reference/platform-capabilities.md).
