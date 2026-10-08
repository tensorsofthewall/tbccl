# Public API boundary

What you may rely on, what is internal, and what is only for debugging.

## Supported public API

| Surface | Where | Stability |
|---|---|---|
| **C ABI v1** | `include/tbccl/tbccl.h`, target `TBCCL::tbccl_c` | frozen: constants are never renumbered, struct fields are never reordered or removed, exported functions are never removed or changed. Grows only compatibly (guarded by `struct_size`). The exported symbols are pinned in `tests/c_api/abi_symbols_v1.txt` |
| **C++ API** | `Communicator`, `Work`, `BufferView`, `CommunicatorOptions`, `CommunicatorListeners`, `RankDirectory`, `ExecutionContext`, error types (`include/tbccl/communicator.hpp`, `work.hpp`, `buffer.hpp`, `rank_directory.hpp`, `error.hpp`, `types.hpp`, `cuda_support.hpp`) | source-compatible within a development line where possible; **not an ABI promise**. Prefer the C ABI for anything that must survive a library upgrade without recompiling |
| **Version information** | `tbccl-info`, `tbccl/version.h`, `tbcclGetAbiVersion`, `tbcclGetPackageVersion` | stable |
| **CMake package** | `find_package(TBCCL CONFIG)`, targets `TBCCL::tbccl`, `TBCCL::tbccl_c`, optional `TBCCL::tbccl_cuda`; `tbccl.pc` | stable target names |

Introductions: [C++ API overview](cpp-api-overview.md), [C ABI v1](c-abi.md), [C API quickstart](../guides/c-api-quickstart.md). The generated [C++ reference](cpp-api.md) and [C reference](c-api.md) list the declarations.

## Internal implementation

Everything under `src/` is internal and may change without notice: the transport, the collective planner and algorithms, the staging pool, the chunk pipeline, the worker threads, the wire format and the rendezvous internals. Do not include or link against anything outside the installed headers and libraries. Consumers always link an **installed** prefix, never the source tree.

Some installed headers exist because other headers include them, or for the project's own benchmarks and tests. The legacy synchronous two-rank `World` API (`world.hpp`, `tcp_world.hpp`, `collectives.hpp`) is used by the benchmarks and is **not** part of the supported API; the transport interface (`transport.hpp`, `tcp.hpp`) is used internally. Headers such as `async_transfer.hpp`, `hetero_allreduce.hpp`, `peer_capabilities.hpp` and `reduction.hpp` are included by the public headers; only the types the [C++ API overview](cpp-api-overview.md) documents (for example `DataType`, `ReduceOp`, `PeerCapabilities`) are supported. New code uses `Communicator`.

## Experimental and debug interfaces

Environment variables marked *diagnostic* in [configuration](configuration.md) (`TBCCL_LATENCY_TRACE`, the per-collective algorithm overrides, `TBCCL_ASYNC_TIMING`, `TBCCL_ALLREDUCE_TIMING`, `TBCCL_ASYNC_NO_STAGING_THREAD`) and the `TBCCL_ENABLE_RING_TRACE` build option are for debugging and measurement. They are not stable interfaces and are not supported for production use. `TBCCL_TRACE=1` is a supported diagnostic.
