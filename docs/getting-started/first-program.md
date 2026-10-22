# Your first program

A TBCCL program needs two things the library deliberately does not provide: a way to start the ranks, and a way to exchange each rank's endpoint ([Communicators](../concepts/communicators.md)). Everything else is a normal asynchronous communication library.

## Smallest possible programs

Before writing real communication code, build and run the two tiny examples from [installing](install.md): `examples/version_check.c` (links `TBCCL::tbccl_c` and prints the version and C ABI) and `examples/two_rank_threads.cpp` (two ranks as two threads of one process, one `all_reduce`, on loopback). If both run, your installation, headers and linking are correct.

## From C

The [C API quickstart](../guides/c-api-quickstart.md) is a complete, compiled and tested program (`examples/c_quickstart.c`): two processes on one machine exchange endpoint blobs through pipes, send a message and run a Float32 `all_reduce`. It uses only `<tbccl/tbccl.h>` and links `TBCCL::tbccl_c`.

## From C++

`examples/async_allreduce.cpp` is a standalone public-headers-only example (built as the `async_allreduce` target). Run one process per rank:

```sh
async_allreduce --rank 0 --peers HOST0:PORT0,HOST1:PORT1 --backend host
async_allreduce --rank 1 --peers HOST0:PORT0,HOST1:PORT1 --backend host
```

`--backend host` needs no device. `--backend cuda` needs a CUDA device and the CUDA component, and `--backend metal-shared` needs a Metal device on macOS, which is how a heterogeneous Linux/CUDA and macOS/Metal pair is exercised.

Each rank's data endpoint is its control port plus 1000, so keep both ports free.

## What success looks like

`version_check` prints `TBCCL 0.7.0, C ABI 1`, and `two_rank_threads` prints `TBCCL 0.7.0: all_reduce(Sum) of 1 and 2 -> 3 and 3` and exits with status 0 (the program returns non-zero if either rank's result is not 3). **This showed** a working two-rank `all_reduce` through the real transport on one machine. **It did not show** behavior across machines or over a physical link, and these two programs use host memory only (`async_allreduce` can use device memory with `--backend cuda` or `--backend metal-shared`). **Read next:** the list below. **If it fails:** [Troubleshooting](../guides/troubleshooting.md), starting at "Start here".

## What to read next

- [C++ API overview](../reference/cpp-api-overview.md) and [Ordering domains](../concepts/ordering-domains.md) for the programming model.
- [Failure handling](../concepts/failure-handling.md) before you use TBCCL in a long-running process.
- [Thunderbolt link guide](../guides/thunderbolt-link.md) for a direct Mac-to-Linux connection.
- [Troubleshooting](../guides/troubleshooting.md) and [diagnostics](../guides/diagnostics.md) if something does not connect.
