# C API quickstart

A complete, compiled and tested program is `examples/c_quickstart.c` (CTest `c_quickstart`); the code below is that program. Specification: [C ABI v1](../reference/c-abi.md). Bootstrap model: [C API bootstrap](../reference/c-abi-bootstrap.md). Threading rules: [C API thread safety](../reference/c-abi-thread-safety.md).

```cmake
find_package(TBCCL CONFIG REQUIRED)               # a C-only project is fine: no C++ compiler, no -lstdc++/-lc++
add_executable(app app.c)
target_link_libraries(app PRIVATE TBCCL::tbccl_c)
```

## The two rules that bite

> **Destroying a Work handle does NOT cancel the operation.** It only drops your handle.
>
> **Destroying a Work handle is NOT permission to free or modify the buffer.** `tbcclBuffer` does not own memory: the host, CUDA or MetalShared allocation must stay valid, and (for sends and collective inputs) unmodified, **until the operation is terminal**: until a Work reports done, or until the communicator is aborted and the Work is done. If you drop handles early, use another way to know the operation finished (for P2P: a later operation on the same peer and direction completing, since the order per peer and direction is FIFO).

Posting never waits: `tbcclSend`/`tbcclRecv`/collectives return a Work immediately (or fail immediately), so "post a hundred sends, then a hundred receives, then wait" is fine. There are no tags: the application matches the logical order of P2P operations per peer pair, and all ranks must submit collectives in the same order.

`tbcclWorkWait*`'s return value is the status of the **query**; the operation's own result is the out-parameter. `tbcclWorkWaitFor` expiring returns `TBCCL_SUCCESS` with `done == 0` and changes nothing (it never reports `TBCCL_TIMEOUT`).

## The program

```{literalinclude} ../../examples/c_quickstart.c
:language: c
```

## What each step needs from you

1. **Unique id**: generate it once (`tbcclGetUniqueId`) and give the same 16 bytes to every rank (environment variable, launcher argument, MPI broadcast...). It contains no address.
2. **Endpoint blobs**: each rank calls `tbcclBootstrapGetEndpoint` and the application all-gathers the fixed-size 256-byte blobs in rank order. For several machines pass `bind_host`/`advertise_host` in `tbcclBootstrapOptions` (the advertised address must be reachable by the peers; never a wildcard).
3. **Complete**: all ranks call `tbcclBootstrapComplete` concurrently (it connects them); a rank that never arrives makes the others fail with `TBCCL_TIMEOUT`.
4. **CUDA**: call `tbcclRegisterCudaSupport()` once (it returns `TBCCL_UNSUPPORTED` in a host-only build), describe device memory with `memory_kind = TBCCL_MEMORY_CUDA`, and pass your `cudaStream_t` as `{struct_size, TBCCL_EXEC_CUDA_STREAM, (void *)stream}` when the operand's producer runs on a non-default stream. `tbccl.h` includes no CUDA header.
