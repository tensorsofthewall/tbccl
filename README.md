# TBCCL

[![docs](https://github.com/tensorsofthewall/tbccl/actions/workflows/docs.yml/badge.svg)](https://github.com/tensorsofthewall/tbccl/actions/workflows/docs.yml) [![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE) [![release](https://img.shields.io/github/v/release/tensorsofthewall/tbccl?include_prereleases&label=pre-release)](https://github.com/tensorsofthewall/tbccl/releases)

TBCCL (Thunderbolt Collective Communication Library) is a C++17 runtime with a stable C ABI that moves tensors between machines that differ in operating system and memory type, for example a Linux host with an NVIDIA GPU and a Mac with Apple silicon, over TCP, including a direct Thunderbolt 4 link.

> **Status:** **0.6.0** (first release; experimental). **C ABI 1**, **wire protocol 4**. Wire protocol 4 does not interoperate with wire protocol 3.

## Why it exists

NCCL and RCCL target homogeneous clusters of one GPU vendor, and Apple-silicon collective backends target Macs. TBCCL covers the gap between them: one communicator can span a CUDA machine and a Metal machine, with host staging handled for you. It is a transport and collectives library only. It knows nothing about models, placement or any machine-learning framework; frameworks use it through adapters.

It is **not** an NCCL replacement, it does not do RDMA (the only transport is TCP), and there is no direct CUDA-to-Metal DMA: CUDA buffers are staged through pinned host memory, and Metal shared (unified) memory is handed to the transport directly.

## Which project do I need?

| I want to... | Use | Documentation |
|---|---|---|
| call TBCCL directly from C or C++ | **tbccl** (this repository) | [tbccl docs](https://tbccl.tensorsofthewall.com/en/latest/) |
| run PyTorch `torch.distributed` (DDP, collectives) across machines | [torch-tbccl](https://github.com/tensorsofthewall/torch-tbccl) | [torch-tbccl docs](https://torch-tbccl.tensorsofthewall.com/en/latest/) |
| run a vLLM pipeline across a CUDA node and a Mac | [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl) | [vllm-tbccl docs](https://vllm-tbccl.tensorsofthewall.com/en/latest/) |
| run an exo pipeline across a Mac and a Linux CUDA host | [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl) | [exo-tbccl docs](https://exo-tbccl.tensorsofthewall.com/en/latest/) |

The three adapters are independent of each other. Each one carries (or links) its own copy of libtbccl, and none of them needs another adapter.

```text
 PyTorch ---- torch-tbccl ----+
 vLLM ------- vllm-tbccl -----+---> libtbccl ---> TCP (Thunderbolt 4 or any IP network)
 exo --------- exo-tbccl -----+
```

## What is supported

- Point-to-point `send` / `recv` and the collectives `barrier`, `broadcast`, `all_gather` and `Sum` `all_reduce`, validated at 1 to 4 ranks, with nonblocking submission: every call returns a `Work` handle immediately.
- Heterogeneous worlds: `Host`, CUDA and Metal-shared buffers in one communicator.
- Point-to-point and collective traffic mixed freely on one communicator (separate ordering domains).
- A stable C ABI (v1) for other languages and for consumers that must not depend on the C++ ABI.

| | Validated |
|---|---|
| Platforms | Linux x86-64 (host, CUDA 13) and macOS arm64 (host, Metal shared memory) |
| World sizes | 1 to 4 (up to 8 accepted) |
| Reductions | `Sum`; Float32, Float64, Int32, Int64, Int8, UInt8 at any size; Float16 and BFloat16 at world size 2 |
| Transport | TCP over IPv4, including Thunderbolt 4 |

The current stable versions and framework pins of all four projects are on the [ecosystem page](docs/project/ecosystem.md) (source: [`release/ecosystem.json`](release/ecosystem.json)); the full compatibility table is the [compatibility matrix](docs/reference/compatibility-matrix.md). Maintenance and versioning policy: [MAINTENANCE.md](MAINTENANCE.md), [VERSIONING.md](VERSIONING.md).

## Quick start

Download a release archive from the [GitHub release](https://github.com/tensorsofthewall/tbccl/releases/tag/v0.6.0), verify it and check the version:

```sh
sha256sum -c SHA256SUMS --ignore-missing
tar -xzf tbccl-0.6.0-linux-x86_64-cuda13.tar.gz        # or tbccl-0.6.0-linux-x86_64.tar.gz, tbccl-0.6.0-macos-arm64.tar.gz
tbccl-0.6.0-linux-x86_64-cuda13/bin/tbccl-info
```

Then use it from CMake (`find_package(TBCCL CONFIG REQUIRED)`, target `TBCCL::tbccl_c` for C or `TBCCL::tbccl` for C++, with `-DCMAKE_PREFIX_PATH=<extracted directory>`) or through pkg-config. [Installing](docs/getting-started/install.md) covers both paths and building from source.

## Minimal example

`examples/version_check.c` is the smallest consumer: it prints the version the library reports. `examples/two_rank_threads.cpp` runs a two-rank `all_reduce` inside one process on loopback. `examples/c_quickstart.c` is a complete two-process C program (bootstrap, `send`, `all_reduce`) that the test suite builds and runs.

```c
#include <stdio.h>
#include <tbccl/tbccl.h>
#include <tbccl/version.h>

int main(void) {
    unsigned abi = 0;
    tbcclGetAbiVersion(&abi);
    printf("TBCCL %s, C ABI %u\n", TBCCL_VERSION_STRING, abi);
    return 0;
}
```

## Documentation

Hosted documentation: https://tbccl.tensorsofthewall.com/ (`latest` is development documentation built from `main`; release-candidate versions appear as numbered versions; `stable` does not exist until the final release). Build it locally with `make docs`.

Start with [getting started](docs/getting-started/index.md). The [compatibility matrix](docs/reference/compatibility-matrix.md), [troubleshooting](docs/guides/troubleshooting.md), [known limitations](docs/project/limitations.md) and the [ecosystem overview](docs/project/ecosystem.md) answer most first questions.

## Known limitations

Only TCP over IPv4 (no RDMA); no direct CUDA-to-Metal DMA; `Sum` is the only reduction; Float16 and BFloat16 reductions only at world size 2; no recovery after a communicator failure; Windows is not supported. See [known limitations](docs/project/limitations.md).

## Development

`CONTRIBUTING.md` explains the workflow and `AGENTS.md` the rules for the code base. The [development documentation](docs/development/index.md) covers building, testing and where a change belongs across the four repositories. Security: `SECURITY.md`. Changes for users: `CHANGELOG.md`. Releasing: `RELEASE.md`.

## License

tbccl is licensed under the Apache License, Version 2.0 (see `LICENSE`). Copyright 2026 Sandesh Bharadwaj.
