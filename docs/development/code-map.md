# Code map

Where to start reading, by topic. It is a map to the directories and the first files, not a walk through the source; the rules for the code base are in `AGENTS.md`, and the design is in the [architecture](../concepts/architecture.md) page. Every path below is checked by a test, so a rename that leaves this page stale fails the documentation checks.

```{table}
:class: tb-stack tb-stack-map

| Topic | Start at | Read with |
|---|---|---|
| The C ABI (the stable boundary) | `include/tbccl/tbccl.h`, `src/c_api/` | [C ABI](../reference/c-abi.md), `tests/c_api/` pins layouts and exported symbols |
| The C++ API | `include/tbccl/communicator.hpp`, `include/tbccl/work.hpp` | [C++ API overview](../reference/cpp-api-overview.md) |
| Runtime initialization and bootstrap | `src/core/communicator.cpp`, `src/core/bootstrap_config.cpp`, `src/core/rank_directory.cpp` | [Communicators](../concepts/communicators.md) |
| Connections and the transport | `src/core/connection_manager.cpp`, `src/transport/tcp.cpp`, `include/tbccl/transport.hpp` | [Transports](../concepts/transports.md) |
| The wire protocol and its version | `src/core/wire_protocol.cpp`, `src/core/wire_protocol.hpp` | [Wire protocol](../reference/wire-protocol.md), [versioning](../reference/versioning.md) |
| Collective algorithms and selection | `src/collectives/`, `src/collectives/algorithm_selector.cpp`, `src/core/collective_plan.cpp` | [Collective algorithms](../reference/collective-algorithms.md) |
| Memory kinds and buffers | `include/tbccl/buffer.hpp`, `src/core/buffer.cpp`, `src/core/host_pointer_backend.hpp` | [Memory providers](../concepts/memory-providers.md) |
| The CUDA component | `include/tbccl/cuda_support.hpp`, `benchmarks/tensor/cuda_memory_provider.cu` | built only with `-DTBCCL_ENABLE_CUDA=ON` |
| Compatibility facts | `compatibility.json`, `scripts/check_compatibility_manifest.py` | [Compatibility matrix](../reference/compatibility-matrix.md) |
| Release archives and the install helper | `scripts/package_native.sh`, `scripts/install_adapter.py` | [Release process](release-process.md) |
| Diagnostics for users | `tools/collect_diagnostics.py`, `tools/thunderbolt_link_doctor.py` | [Diagnostics](../guides/diagnostics.md) |
| The benchmark suite | `benchmarks/suite/` | [Benchmark suite guide](benchmark-suite.md) |
```

Metal shared memory has no separate source file: a `MetalShared` buffer is the CPU-visible pointer of a shared `MTLBuffer`, so the transport treats it like host memory (`src/core/host_pointer_backend.hpp`). The Metal device tests are `tests/communicator_metal_test.mm`.

The adapters (torch-tbccl, vllm-tbccl, exo-tbccl) are separate repositories with their own code maps; see [where should this change go?](where-should-this-change-go.md).
