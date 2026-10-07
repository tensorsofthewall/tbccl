# ADR 0001: Framework-neutral core

- Status: accepted
- Date: 2026-10-01

## Context

TBCCL is consumed by several unrelated frameworks (PyTorch, vLLM, and exo with MLX). Each has its own concepts for tensors, streams, stores and models.

## Decision

The core library and the C ABI never depend on PyTorch, Python, vLLM, exo, MLX, CUDA headers (outside the optional CUDA component), Metal or Objective-C. Frameworks depend on TBCCL through out-of-tree adapters; TBCCL has no concept of models or layer placement. When a consumer seems to need something, the missing **generic** primitive is added to the core instead. Device support is optional and enabled at build time.

## Consequences

- Adapters carry the framework-specific glue (for example torch-tbccl's MPS dispatch and tensor adapter).
- Defects that a consumer exposes are fixed in the core as generic changes. For example, the ordering fix in {doc}`0005-independent-p2p-and-collective-ordering` was made in TBCCL, and the adapter-level guard that had refused overlapping traffic was then removed.
- The C ABI header compiles as C11 and as C++ and includes no CUDA or Metal header.
