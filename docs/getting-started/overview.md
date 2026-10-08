# Overview

## What TBCCL is

TBCCL is a collective-communication runtime: it moves bytes (tensors) between processes on different machines and runs `broadcast`, `all_gather`, `barrier` and `Sum` `all_reduce` over them. It is written in C++17, exposes a stable C ABI, and uses TCP as its only transport, so it works over a direct Thunderbolt 4 cable between two computers or over any IP network.

It is a transport and collectives library only. It does not discover peers, load models, place layers or know about any machine-learning framework. Frameworks use it through separate adapter projects ([ecosystem](../project/ecosystem.md)).

## Why it exists

A model split across a Linux machine with an NVIDIA GPU and a Mac with Apple silicon needs a way to move activations between them. NCCL and RCCL target clusters of one GPU vendor, and Apple-silicon collective backends (for example MLX's JACCL backend) target Macs. TBCCL lets a single communicator contain ranks with different operating systems, CPU architectures and memory types (`Host`, `Cuda`, `MetalShared`), and takes care of staging device memory for the transport.

## How it differs from NCCL, RCCL and JACCL

| | TBCCL |
|---|---|
| Hardware | mixed: any machine that can run the library; CUDA buffers on Linux, Metal shared buffers on macOS, host buffers everywhere |
| Transport | TCP over IPv4 only (including Thunderbolt 4 networking). No RDMA |
| Device data path | CUDA buffers are staged through pinned host memory; Metal shared (unified) memory is read and written directly by the transport. There is no direct CUDA-to-Metal DMA |
| Scope | point-to-point, `barrier`, `broadcast`, `all_gather`, `Sum` `all_reduce`; validated at 1 to 4 ranks |
| Bootstrap | explicit: the application exchanges every rank's endpoint |

NCCL, RCCL and the Apple-silicon backends remain the right choice for the homogeneous clusters they are built for. TBCCL is for the mixed case they do not cover, and it does not aim for feature parity with them.

## Which repository do I install?

See the routing table on the [home page](../index.md). In short: use **tbccl** from C or C++, **torch-tbccl** for PyTorch, **vllm-tbccl** for vLLM and **exo-tbccl** for exo. The adapters ship wheels that already contain libtbccl, so you do not install tbccl separately to use them.

## What a TBCCL program looks like

1. Start one process per rank (TBCCL does not launch anything).
2. Each rank binds its listeners and publishes its endpoint; the application exchanges the endpoints and one communicator id between ranks.
3. Each rank creates a `Communicator` and posts operations. Every call returns a `Work` handle immediately; wait on it when you need the result.

See [Your first program](first-program.md) and the [C API quickstart](../guides/c-api-quickstart.md).
