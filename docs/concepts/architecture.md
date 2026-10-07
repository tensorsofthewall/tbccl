# Architecture

TBCCL is a C++17 collective-communication runtime with a stable C ABI. It moves tensors between heterogeneous machines (for example a Linux host with an NVIDIA GPU and a Mac) over TCP, including a direct Thunderbolt 4 link. It is a transport and collectives library only: it has no notion of models, layer placement or any machine-learning framework.

## Layers

```
consumers (out of tree): torch-tbccl, vllm-tbccl, exo-tbccl, applications
        |
C ABI v1    include/tbccl/tbccl.h, target TBCCL::tbccl_c (C11, no CUDA/Metal/C++ in the header)
        |
core library    include/tbccl/*.hpp, target TBCCL::tbccl
   Communicator, Work, BufferView, errors
   N-rank control plane, lazily opened data connections
   collective planner and algorithms
   transfer workers, staging pool, chunk pipeline
   Transport interface, TcpTransport
        |
optional device components
   CUDA memory provider (TBCCL::tbccl_cuda)       Metal-shared buffers need no extra component
```

The outer layers depend on the inner ones, never the reverse.

## What stays out of the core

The headers and sources of the core (`include/tbccl/`, `src/core/`, `src/transport/`, `src/collectives/`, `src/c_api/`) have no dependency on CUDA, Metal, Objective-C or any ML framework. Device-specific code is optional and enabled at build time (`TBCCL_ENABLE_CUDA`, `TBCCL_ENABLE_METAL`). The C shim only calls a framework-neutral registration hook for CUDA; it never includes a CUDA header.

Framework integrations are separate projects that link an installed TBCCL package: [torch-tbccl](https://github.com/tensorsofthewall/torch-tbccl) (a PyTorch `torch.distributed` backend), [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl) (a vLLM platform integration built on torch-tbccl) and [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl) (an exo pipeline data plane over the C ABI). If a consumer seems to need a framework-specific feature, the right change is a missing generic primitive in TBCCL, never a framework concept in the core (see {doc}`../adr/0001-framework-neutral-core`).

## Public interfaces

- **C++ API** (`TBCCL::tbccl`): `Communicator`, `Work`, `BufferView`, `CommunicatorOptions`, structured errors. It is a source-level API, not an ABI promise. See the [C++ API overview](../reference/cpp-api-overview.md).
- **C ABI v1** (`TBCCL::tbccl_c`): the stable interface for other languages and for consumers that must not depend on the C++ ABI. See [C ABI v1](../reference/c-abi.md).

The package also contains a legacy synchronous two-rank `World` API used by benchmarks. New code uses `Communicator`.

## Key design properties

- **Non-owning buffers.** A `BufferView` describes memory the caller owns ({doc}`../adr/0002-non-owning-buffer-view`).
- **Nonblocking submission.** Every post returns a `Work` immediately ({doc}`../adr/0008-nonblocking-submission-without-group-calls`).
- **Explicit bootstrap.** TBCCL never discovers peers; the application supplies every rank's endpoints ({doc}`communicators`).
- **Independent ordering domains.** Collectives and point-to-point traffic never share a byte stream ({doc}`ordering-domains`).
- **Explicit compatibility.** The package version, the C ABI version and the wire protocol version are separate ([versioning](../reference/versioning.md)).
