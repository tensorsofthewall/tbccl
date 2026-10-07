# ADR 0002: Non-owning BufferView

- Status: accepted
- Date: 2026-10-01

## Context

Frameworks own their tensors and manage memory with their own allocators and stream semantics. The transport layer also needs to avoid copies of large payloads.

## Decision

`BufferView` (and `tbcclBuffer` in the C ABI) describes memory the caller owns. TBCCL never frees, reallocates or takes ownership of it. The caller must keep the allocation valid, and unmodified for sends, until the operation's `Work` is terminal. Destroying a `Work` handle neither cancels the operation nor releases the buffer. TBCCL does not add defensive copies to relax this contract, because doing so would silently reintroduce the staging cost that the direct, non-owning path exists to avoid. The contract is documented instead.

## Consequences

- Adapters retain strong references to every tensor until the operation completes (torch-tbccl does this in its work state).
- A `Work` becomes terminal only after no TBCCL thread can touch the buffer, including on abort and peer failure.
- Memory-kind specific rules live with the kind: `MetalShared` is a CPU-visible pointer, `Cuda` uses staging and an optional stream context.
