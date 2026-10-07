# ADR 0007: Independent framework adapters

- Status: accepted
- Date: 2026-10-01

## Context

TBCCL serves PyTorch (`torch.distributed`), vLLM and exo. They differ in language, packaging, release cadence and supported platforms.

## Decision

Each integration is its own project and repository with its own documentation, version and release: torch-tbccl (a PyTorch c10d backend), vllm-tbccl (a vLLM platform integration built on torch-tbccl) and exo-tbccl (an exo pipeline data plane that links only the C ABI and uses no PyTorch). Each depends one way on an **installed** TBCCL package and never includes TBCCL sources or reimplements communication algorithms, transport code, staging or reduction. Adapters do not patch TBCCL, PyTorch, vLLM or exo in place; the one change to vllm-metal that the integration needs is carried as a patch file. A defect that an adapter exposes in TBCCL is fixed in TBCCL as a generic change.

## Consequences

- Each adapter declares its own validated compatibility tuple (framework versions, Python versions, platforms) and the C ABI and wire protocol versions it targets.
- After a wire protocol change, adapters must be rebuilt against a prefix with the new version.
- Documentation is not combined across repositories; the projects link to each other.
