# PyTorch Integration Mapping (documentation only, no PyTorch code)

This maps PyTorch/`c10d` concepts onto the Phase 41 public TBCCL API,
for a future out-of-tree `torch-tbccl` `ProcessGroup` (Phase 42). No
PyTorch header, type, or dependency exists anywhere in this repository
-- this document exists so that future work has a concrete plan, per
the phase's own architectural rule: **libtbccl never depends on
PyTorch; torch-tbccl depends on libtbccl.**

| PyTorch / c10d concept | TBCCL public API | Notes |
|---|---|---|
| `c10d::Store` (rendezvous key-value store) | `tbccl::CommunicatorOptions` (rank, peer endpoint list) | `torch-tbccl` reads rank/world_size/peer addresses out of the `Store` and constructs a `CommunicatorOptions` from them. TBCCL never sees the `Store` itself. |
| `ProcessGroup` construction (`rank`, `world_size`) | `tbccl::Communicator::create()` | One `Communicator` per `ProcessGroup` instance. Expensive, created once. |
| `at::Tensor` | `tbccl::BufferView` | `torch-tbccl` extracts `tensor.data_ptr()`, `tensor.nbytes()`, and maps `tensor.device().type()` to `MemoryKind` (CPU -> Host, CUDA -> Cuda). TBCCL never sees `at::Tensor`, its strides, dtype metadata beyond byte count, or autograd state -- `torch-tbccl` is responsible for only passing contiguous, correctly-typed tensors. |
| `at::ScalarType` | `tbccl::DataType` | Maps the currently-supported subset (`kFloat`->`Float32`, `kDouble`->`Float64`, `kInt`->`Int32`, `kLong`->`Int64`). Unsupported PyTorch dtypes (`kHalf`, `kBFloat16`, etc.) are a `torch-tbccl`-side `Unsupported` error until TBCCL itself grows those types (tracked against `tbccl::DataType`'s own deliberately small initial set, not a Phase 41 gap). |
| `c10d::ReduceOp` | `tbccl::ReduceOp` | Phase 41 only implements `Sum` end-to-end for the heterogeneous N=2 engine; `torch-tbccl` surfaces `Unsupported` for `Product`/`Min`/`Max`/`BAND`/etc. until a future phase extends `n2_all_reduce_tensor()`. |
| CUDA stream (`at::cuda::getCurrentCUDAStream()`) | `tbccl::ExecutionContext{CudaStream, stream}` | `torch-tbccl` passes the tensor's producing stream as the `ExecutionContext` on `send()`/`all_reduce()` calls -- this is exactly the seam Phase 41's stream-readiness mechanism (Part L) was built for; no `cudaDeviceSynchronize()` needed on the PyTorch side either. |
| `ProcessGroup::allreduce(tensors, opts)` returning a `c10d::Work` | `Communicator::all_reduce()` returning `tbccl::Work` | `torch-tbccl`'s `c10d::Work` subclass wraps a `tbccl::Work` and forwards `wait()`/`isCompleted()` to it. `async_op=True` maps directly: return the `tbccl::Work`-wrapping object without blocking; `async_op=False` (or a plain blocking call) just calls `tbccl_work.wait()` before returning from `allreduce()`. |
| `c10d::Work::wait()` | `tbccl::Work::wait()` | Direct forward. |
| `c10d::Work` destruction before completion | `tbccl::Work` destruction before completion | Both have the same "operation survives handle destruction" semantics (Phase 41 Part Q item 96) -- no special handling needed in the adapter. |
| Collective error (`c10d` typically throws or marks the `Work` as failed) | `tbccl::Work::has_error()`/`error()` | `torch-tbccl` translates `has_error()`+`error()`'s message into whatever `c10d::Work`'s own failure convention is (an exception on `wait()`, typically). |

## Explicitly unresolved, left for Phase 42

- **Process-group-relative ranks vs. TBCCL's own rank/peer model.**
  `c10d` supports subgroups with their own rank numbering distinct from
  the global group; Phase 41's `Communicator` has no subgroup concept
  at all (one `Communicator` == one fixed 2-rank group). Phase 42 must
  decide whether `torch-tbccl` creates one `Communicator` per subgroup
  (straightforward, matches "no global singleton" from Part AD) or
  whether TBCCL itself grows a subgroup concept.
- **`async_op=False` semantics under PyTorch's own stream-ordering
  contract.** PyTorch collectives are stream-ordered: a synchronous
  call still enqueues its wait on the current stream rather than
  blocking the host. Phase 41's `Work::wait()` is a host-blocking wait.
  `torch-tbccl` will need its own stream-ordering shim (e.g. a CUDA
  event signaled from a background completion thread) -- not something
  `libtbccl` itself needs to solve, since this is PyTorch-specific
  stream-semantics glue, not a communication-runtime concern.
- **Multi-tensor collectives** (`allreduce_coalesced`,
  bucketed/flattened DDP gradient tensors spanning multiple `at::Tensor`
  objects in one call). Phase 41's `all_reduce()` takes exactly one
  `BufferView`; DDP's bucketing would need to happen in `torch-tbccl`
  (flatten into one contiguous buffer before calling TBCCL) until/unless
  TBCCL grows a native multi-buffer collective.
- **N>2 world size.** `torch-tbccl` cannot support more than 2 ranks
  until TBCCL's own `Communicator` does (explicitly out of Phase 41
  scope, tracked honestly rather than papered over).
- **`broadcast`/`barrier`/`all_gather`/`reduce_scatter` through the
  public async API.** These exist in TBCCL today only via the older,
  synchronous `World`-based `collectives.hpp` path, not through
  `Communicator`. DDP needs at least `broadcast` (initial parameter
  sync) and `barrier` before it is usable -- explicitly flagged as
  Phase 43 scope in the project's long-term roadmap, not assumed
  solved here.

## What Phase 41 already proves is NOT a problem for Phase 42

Per the phase's own gate (Part CB): Phase 42 should not need to
redesign external buffers, communicator lifetime, asynchronous Work,
bootstrap config, CUDA stream readiness, or the error model. Nothing
found during Phase 41's implementation or testing suggests otherwise --
the mapping table above is a direct, mechanical translation in every
row except the four explicitly-unresolved items listed, none of which
are "libtbccl's ownership model was wrong," only "PyTorch needs
features libtbccl doesn't have yet" (subgroups, more collectives, N>2),
which is expected, scoped, future work rather than an architectural
failure.
