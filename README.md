# TBCCL

TBCCL is a C++17 collective-communication runtime with a stable C ABI. It moves tensors between machines that differ in operating system and memory type, for example a Linux host with an NVIDIA GPU and a Mac, over TCP, including a direct Thunderbolt 4 link. It is a transport and collectives library only: it knows nothing about models, placement or any machine-learning framework.

> **Status:** development version 0.5.1, **C ABI 1**, **wire protocol 4**. No release has been published; the next release is planned as 0.6.0. Wire protocol 4 does not interoperate with wire protocol 3.

## What you can use it for

- Point-to-point `send` / `recv` and the collectives `barrier`, `broadcast`, `all_gather` and `Sum` `all_reduce` across 1 to 4 ranks (validated), with nonblocking submission: every call returns a `Work` handle immediately.
- Heterogeneous worlds: `Host`, CUDA and Metal-shared buffers in one communicator.
- Mixing point-to-point and collective traffic freely on one communicator; the two are independent ordering domains.
- A stable C ABI for other languages and for consumers that must not depend on the C++ ABI.

It is not an RDMA implementation: the only transport is TCP.

## Install

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release     # add -DTBCCL_ENABLE_CUDA=ON on Linux with CUDA
cmake --build build-release -j
(cd build-release && ctest --output-on-failure)
cmake --install build-release --prefix <prefix>
```

Consumers use `find_package(TBCCL CONFIG REQUIRED)` and link `TBCCL::tbccl` (C++) or `TBCCL::tbccl_c` (C). Requirements: a C++17 compiler, CMake 3.20 or newer, and for CUDA the CUDA Toolkit.

## Minimal example

`examples/c_quickstart.c` is a complete program: two processes bootstrap a communicator, send a message and run an `all_reduce`. It is built and run by the test suite. `examples/async_allreduce.cpp` is the C++ equivalent.

```cmake
find_package(TBCCL CONFIG REQUIRED)
add_executable(app app.c)
target_link_libraries(app PRIVATE TBCCL::tbccl_c)
```

## Supported configurations

| | Validated |
|---|---|
| Platforms | Linux x86-64 (host, CUDA) and macOS arm64 (host, Metal shared memory) |
| World sizes | 1 to 4 (up to 8 accepted) |
| Reductions | `Sum`; Float32, Float64, Int32, Int64, Int8, UInt8 at any size; Float16 and BFloat16 at world size 2 |
| Transport | TCP over IPv4, including Thunderbolt 4 |

Details and limits: [platform capabilities](docs/reference/platform-capabilities.md). Validation evidence: [validation](docs/validation/0.6.0.md) (draft).

## Documentation

The documentation is in `docs/` and builds with `make docs`:

- [Getting started](docs/getting-started/index.md), [guides](docs/guides/index.md), [concepts](docs/concepts/index.md), [reference](docs/reference/index.md), [architecture decision records](docs/adr/index.md).
- Contributing: `CONTRIBUTING.md` and `AGENTS.md`.

The hosted documentation, https://tbccl.tensorsofthewall.com/en/stable/, is published with the first release; until then build it with `make docs`. Related projects and their documentation: `docs/related-projects.md`.

## Related projects

[torch-tbccl](https://github.com/tensorsofthewall/torch-tbccl) (PyTorch backend), [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl) (vLLM integration) and [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl) (exo pipeline data plane) are separate projects that use an installed TBCCL.

## License

tbccl is licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright 2026 Sandesh Bharadwaj.

Security: see `SECURITY.md` and the security model in the documentation. Changes for users: `CHANGELOG.md`. Releasing: `RELEASE.md`.
