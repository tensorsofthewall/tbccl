# C ABI v1: constraints derived from the Phase 50 runtime

Phase 50 does **not** ship a C ABI and does not freeze any C names or signatures. This document records what the final N-rank C++ runtime implies for a future C header (planned for Phase 52,
after Phase 51 settles the algorithms), so that header can be written without redesigning `Communicator`.

## 1. Principles the header must follow

* opaque handles; fixed-width integer types and fixed-value enums; POD structs with an explicit size/version field; explicit lengths; result codes.
* never expose `std::string`, `std::vector`, exceptions, `std::shared_ptr`, virtual classes or templates. Every C++ exception becomes a result code plus (optionally) a thread-local or handle-local message buffer.
* no hidden global state: every function takes a handle.
* the header must be usable from a host-only build (no CUDA or Metal headers); device specifics stay behind opaque stream/context values, as `ExecutionContext` already does.

## 2. Type mapping

| C++ concept | C representation | Notes |
|---|---|---|
| `Communicator` | `typedef struct tbcclComm_st *tbcclComm_t;` (opaque) | created by init, destroyed by `Destroy`; non-copyable already; destruction with outstanding work aborts (Phase 45/50), so the C contract is "destroy is always safe and bounded" |
| `Work` (`TransferWork`, shared state) | `typedef struct tbcclWork_st *tbcclWork_t;` (opaque, reference-counted wrapper around the shared state) | `Test` = `is_completed()` + `has_error()`, `Wait` = `wait()`, `Destroy` releases the handle only: **destroying a Work never cancels the operation** (already true: state is shared). A timed wait does not exist in C++ `TransferWork` today; the C API would add `WaitFor(timeout)` on top of the same condition variable |
| `BufferView` | `typedef struct { int32_t memory_kind; void *data; uint64_t bytes; int32_t device_ordinal; } tbcclBuffer_t;` (+ a leading `uint32_t struct_size`) | already a non-owning POD; `memory_kind` is an enum with fixed values |
| `ExecutionContext` | `typedef struct { int32_t kind; void *native_handle; } tbcclExecContext_t;` | `native_handle` is already an opaque `void*` (a `cudaStream_t` cast) |
| `DataType` | `int32_t` with the fixed values `Int32=0, Int64=1, Float32=2, Float64=3, Int8=4, UInt8=5, Float16=6, BFloat16=7` | the values are already append-only and ABI-visible (`datatype_metadata_test` pins them); FP8 / packed types are deliberately **not** datatypes: they travel as `UInt8` bytes |
| `ReduceOp` | `int32_t` (`Sum=0, ...`) | only `Sum` is accepted by `Communicator` collectives |
| `MemoryKind` | `int32_t` (`Host=0, Cuda=1, MetalShared=2`) | |
| `CommunicatorId` | `typedef struct { uint8_t bytes[16]; } tbcclUniqueId;` | already a 16-byte POD token with value semantics; this *is* the `ncclUniqueId` analogue |
| error / result | `typedef enum tbcclResult_t` | see section 4 |
| capabilities | `tbcclCommGetCapabilities(comm, rank, &caps)` returning a POD (`os`, memory-backend bitmask, `max_chunk`, `alignment`) | the C++ `PeerCapabilities` holds `std::vector`s of enums; the C form is bitmasks |
| `CommunicatorOptions` | not exposed as such; replaced by the bootstrap calls below | |

## 3. Bootstrap: `GetUniqueId` / `CommInitRank` can build the same `RankDirectory`

Phase 50's core takes a `RankDirectory` (every rank's control and data endpoint) plus a `CommunicatorId`, deliberately independent of how they were gathered. That is cumbersome as a C surface
(a variable-length array of strings), so the C API should not expose it directly. A NCCL-style flow maps onto it without touching `Communicator`:

```
tbcclGetUniqueId(&id)                         // rank 0 (or any one rank): CommunicatorId::generate() + its own listener endpoints
  ... the application distributes `id` to every rank (MPI, a store, a launcher: not libtbccl's business) ...
tbcclCommInitRank(&comm, nranks, id, rank)    // each rank: CommunicatorListeners::bind(), exchange {control, data} endpoints, build RankDirectory, Communicator::create()
```

The missing piece for the second step is an endpoint exchange. Two compatible designs, to be chosen in Phase 52, neither requiring a change to `Communicator`:

1. **Rendezvous inside the id.** The unique id carries the rendezvous endpoint of one rank (as NCCL's does); `CommInitRank` registers with that rank over TCP, which returns the full directory. This is a small
   bootstrap server built on the existing `tcp_listen` / handshake code and keeps the C API as simple as NCCL's.
2. **Caller-supplied exchange.** `tbcclCommBootstrapBegin(&boot, rank, nranks, id, &local_endpoint_blob)` / `tbcclCommBootstrapFinish(boot, all_blobs, &comm)`, where the blob is the serialized `{control, data}` pair of
   `CommunicatorListeners`. This is exactly what torch-tbccl already does with a `c10d::Store` (publish a record per rank, wait for all, build the directory), so it is proven.

Phase 50 gives both what they need: `CommunicatorListeners` (bind first, publish the *actual* ports), `RankDirectory` validation, `CommunicatorId`, and a handshake that rejects a wrong id, duplicate rank,
world-size mismatch or protocol-version mismatch with a clear error.

## 4. Error categories

The C++ API reports failures as `std::runtime_error` with a tagged message (`"<tag>: detail"`), and `ErrorCode` already names the categories. Phase 50 appended the two that were missing. The stable set the C API needs:

| C result | `ErrorCode` / tag | Meaning |
|---|---|---|
| `TBCCL_SUCCESS` | `Success` | |
| `TBCCL_INVALID_ARGUMENT` | `InvalidArgument` / `invalid_argument:` | bad rank, peer == self, null buffer, count too large, bad directory. Never poisons the communicator |
| `TBCCL_UNSUPPORTED` | `Unsupported` / `unsupported:` | dtype/op/memory kind/N not supported; includes the collective-capability verdict and N>2 Float16/BFloat16. Does not poison |
| `TBCCL_ABORTED` | `Aborted` / `aborted:` | the communicator is terminal (explicit abort, a peer's abort, or a fatal failure); the first reason is kept |
| `TBCCL_TIMEOUT` | `Timeout` / `timeout:` | a bounded bootstrap step expired (the C++ `Work` has no timeout yet) |
| `TBCCL_PROTOCOL_MISMATCH` | `ProtocolMismatch` / `protocol_mismatch:` | wrong communicator id, wire protocol version, world size, duplicate rank, collective descriptor mismatch (kind / root / count / dtype / bytes / sequence), P2P size mismatch |
| `TBCCL_TRANSPORT_ERROR` | `TransportError`, `PeerFailure`, `DeviceError` / `transport_error:`, `peer_failure:`, `device_error:` | socket or device failure |
| `TBCCL_INTERNAL_ERROR` | `InternalError` | a bug |

`Work` carries its own result: `tbcclWorkTest` returns "done / not done" and the final result is read with `tbcclWorkGetResult` (the text from `TransferWork::error()` is copied into a caller buffer).

## 5. Work mapping

* `tbcclWorkTest(work, &done)`: `is_completed()`; `tbcclWorkWait(work)`: `wait()` (repeated waits are safe and stable: the terminal transition happens exactly once); `tbcclWorkDestroy(work)`: drops the handle, the operation continues
  and the communicator keeps the buffers' providers alive until it completes.
* A `Work` becomes terminal only after no TBCCL thread can touch the caller's buffer (Phase 45, kept in Phase 50 by `OpGroup`, which waits for every child transfer before a collective finishes). This is the guarantee the C docs must state.
* One `Work` may aggregate several peer transfers (a collective); the C API does not expose children.
* Ordering contract to document: collectives on one communicator execute in call order on every rank (the collective sequence); P2P is FIFO per peer and direction; P2P and collectives share a peer's lane, so callers must not
  interleave a collective with P2P to the *same* peer from different threads in an order that differs between ranks.

## 6. Things the C++ API has today that the C API should not copy

* `Communicator::create` taking a `CommunicatorOptions` with `std::vector<RankEndpoint>`, `std::shared_ptr<CommunicatorListeners>` and `std::chrono::milliseconds`: use plain-data parameters and the bootstrap calls above.
* `std::function`-based memory-provider registration (`register_memory_provider_factory`): the C API should expose a fixed `tbcclRegisterCudaSupport()` (which already exists as `register_cuda_support()`), not arbitrary provider callbacks.
* exceptions as the only error channel: every C entry point must catch everything and return a code.
* `Work::wait()` blocking forever: the C API should add a timed wait.

## 6b. Phase 51 re-check

* Collective algorithms (tree, ring, recursive doubling, dissemination), the planner and lazy data connections are **internal**: no C type, call or error code depends on them. The only caller-visible effects are performance and the float-reduction contract (`numerical_reduction_semantics.md`: results identical on every rank, deterministic per version/algorithm/N, not bit-equal across algorithms).
* Bootstrap is unchanged (`RankDirectory`, `CommunicatorId`, explicit control/data endpoints); wire protocol version is 3 and must be checked at connect time, not exposed.
* Lazy data edges mean `CommInitRank` completion does not imply every data edge exists; a first-use dial failure surfaces as a `transport_error` on that operation's `Work`. The C API must not promise "init succeeded => any peer reachable on the data path".
* **Posting is not unbounded** (`docs/grouped_operations_audit.md`): more than ~9 large outstanding sends per peer block the posting call. The C API must either state a bounded-outstanding contract, or Phase 52 must first make posting non-blocking / add a group object. A `Work` poll/test call and a timed wait remain required.
* `ExternalMemoryProvider::reduce_backend_range` stays C++-only: C exposes fixed built-in providers.
* Datatype rules at N>2 (FP16/BF16 rejected) must appear in the C error documentation as `unsupported`.

## 7. Open points for Phase 52 (not decided here)

* thread-safety contract per handle (today: collectives are serialized by the executor; P2P posts may come from any thread; `abort()` from any thread);
* whether `GetUniqueId` embeds a rendezvous endpoint (design 1) or the caller exchanges endpoint blobs (design 2);
* the exact names and signatures, which this phase intentionally does not freeze.
