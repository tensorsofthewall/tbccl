# Your first program

A TBCCL program needs two things the library deliberately does not provide: a way to start the ranks, and a way to exchange each rank's endpoint ([Communicators](../concepts/communicators.md)). Everything else is a normal asynchronous communication library.

## From C

The [C API quickstart](../guides/c-api-quickstart.md) is a complete, compiled and tested program (`examples/c_quickstart.c`): two processes on one machine exchange endpoint blobs through pipes, send a message and run a Float32 `all_reduce`. It uses only `<tbccl/tbccl.h>` and links `TBCCL::tbccl_c`.

## From C++

`examples/async_allreduce.cpp` is a standalone public-headers-only example (built as the `async_allreduce` target). Run one process per rank:

```sh
async_allreduce --rank 0 --peers HOST0:PORT0,HOST1:PORT1 --backend host
async_allreduce --rank 1 --peers HOST0:PORT0,HOST1:PORT1 --backend host
```

`--backend host` needs no device. `--backend cuda` needs a CUDA device and the CUDA component, and `--backend metal-shared` needs a Metal device on macOS, which is how a heterogeneous Linux/CUDA and macOS/Metal pair is exercised.

## What to read next

- [C++ API overview](../reference/cpp-api-overview.md) and [Ordering domains](../concepts/ordering-domains.md) for the programming model.
- [Failure handling](../concepts/failure-handling.md) before you use TBCCL in a long-running process.
- [Thunderbolt link guide](../guides/thunderbolt-link.md) for a direct Mac-to-Linux connection.
