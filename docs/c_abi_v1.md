# TBCCL C ABI v1 (authoritative)

`include/tbccl/tbccl.h` and the symbols of `TBCCL::tbccl_c` are the stable ABI. They wrap the final asynchronous C++ runtime (`TBCCL::tbccl`); they do not reimplement it. The C++ API and its mangled symbols are **not** an ABI promise merely because a C API exists.

```
C / Rust / Swift / future bindings -> libtbccl_c (stable C ABI shim) -> libtbccl (C++ runtime)
```

## 1. Versions (independent of each other)

| what | value | meaning |
|---|---|---|
| `TBCCL_C_ABI_VERSION` | **1** | this document; changes only if the ABI is broken (never planned) |
| package version | 0.5.0 | `tbcclGetPackageVersion`; informational |
| wire protocol | 3 | rank-to-rank, internal; never visible through the C API |
| endpoint blob format | **1** | `tbcclEndpointBlob.format_version`; independent of the three above |

## 2. Rules that never change in v1

Never renumber a constant, reorder or remove a field, remove an exported function, change a signature, or reinterpret a field incompatibly. Compatible growth only: append struct fields (guarded by `struct_size`), add constants, add functions, add new struct types.

* **No C `enum` as ABI storage** (C does not fix its width). Enumerations are `int32_t` typedefs with `#define` constants.
* **`struct_size` first.** Every extensible struct starts with `uint32_t struct_size`. The caller zero-initializes the struct and sets `struct_size = sizeof(struct)`. The library validates it against the minimum supported prefix (`>=` the v1 size; smaller is `TBCCL_INVALID_ARGUMENT`), reads no field beyond `struct_size`, treats absent appended fields as zero/NULL, and **writes no field beyond the caller's `struct_size`** (output structs). There is no per-struct `abi_version`.
* **Reserved input fields must be zero** (nonzero is `TBCCL_INVALID_ARGUMENT`). Output reserved fields are written as zero.
* Strings passed in are NUL-terminated UTF-8 and are **copied** during the call; no pointer passed in is retained (except buffers, section 6). Strings returned are written into caller buffers (`char *buf, size_t capacity, size_t *required`): `required` is the length including the terminating NUL; at most `capacity - 1` characters plus a NUL are written (nothing if `capacity == 0`, where `buf` may be NULL); the call returns `TBCCL_SUCCESS` and truncation is visible as `required > capacity`.
* No exception crosses the C boundary. A failed call leaves out-parameters set to NULL/0/zero.
* `<stdint.h>` and `<stddef.h>` only; the header compiles as C11 and as C++, includes no CUDA or Metal header, and has an `extern "C"` guard. `TBCCL_API` (visibility/dllexport) and `TBCCL_CALL` (calling convention) decorate every function and are empty on the current static Linux/macOS build.

## 3. Constants

`tbcclResult_t` (`int32_t`):

| name | value | produced for |
|---|---|---|
| `TBCCL_SUCCESS` | 0 | |
| `TBCCL_INVALID_ARGUMENT` | 1 | NULL required pointer, bad `struct_size`, invalid enum value, peer/rank/root out of range or == self, size overflow, buffer too small, malformed endpoint blob, wrong id/world/duplicate rank |
| `TBCCL_UNSUPPORTED` | 2 | unsupported dtype/op/memory kind, N>2 FP16/BF16 reduction, CUDA not built in |
| `TBCCL_RESOURCE_EXHAUSTED` | 3 | allocation failed while admitting an operation (never reported by waiting) |
| `TBCCL_ABORTED` | 4 | the communicator is aborted (explicitly, by a peer, or after a fatal failure) |
| `TBCCL_TIMEOUT` | 5 | a runtime/bootstrap timeout (never a caller's `WaitFor` timeout) |
| `TBCCL_PROTOCOL_MISMATCH` | 6 | ranks disagree (communicator id, world size, wire version, collective descriptor) |
| `TBCCL_TRANSPORT_ERROR` | 7 | socket failure, peer failure |
| `TBCCL_INTERNAL_ERROR` | 8 | unexpected exception, invariant violation |
| `TBCCL_DEVICE_ERROR` | 9 | CUDA/device failure (kept separate: clearly useful to callers) |

Mapping from the C++ `ErrorCode` is one function (`PeerFailure` maps to `TRANSPORT_ERROR`). `tbcclGetResultString` returns static generic text.

`tbcclDataType_t`: `TBCCL_INT32 = 0`, `TBCCL_INT64 = 1`, `TBCCL_FLOAT32 = 2`, `TBCCL_FLOAT64 = 3`, `TBCCL_INT8 = 4`, `TBCCL_UINT8 = 5`, `TBCCL_FLOAT16 = 6`, `TBCCL_BFLOAT16 = 7`. No FP8, no INT4: those are opaque bytes through the byte-generic calls.

`tbcclReduceOp_t`: `TBCCL_SUM = 0`, `TBCCL_PRODUCT = 1`, `TBCCL_MIN = 2`, `TBCCL_MAX = 3`. `tbcclAllReduce` accepts only the valid (dtype, op) matrix of the runtime; the rest is `TBCCL_UNSUPPORTED`.

`tbcclMemoryKind_t`: `TBCCL_MEMORY_HOST = 0`, `TBCCL_MEMORY_CUDA = 1`, `TBCCL_MEMORY_METAL_SHARED = 2`.

`tbcclExecKind_t`: `TBCCL_EXEC_DEFAULT = 0` (the provider's default context), `TBCCL_EXEC_CUDA_STREAM = 1` (`native_handle` is a `cudaStream_t`).

## 4. Types

```c
typedef int32_t tbcclResult_t;   typedef int32_t tbcclDataType_t;  typedef int32_t tbcclReduceOp_t;
typedef int32_t tbcclMemoryKind_t; typedef int32_t tbcclExecKind_t;
typedef struct tbcclComm_st      *tbcclComm_t;       /* a communicator */
typedef struct tbcclWork_st      *tbcclWork_t;       /* shared operation state */
typedef struct tbcclBootstrap_st *tbcclBootstrap_t;  /* owns pre-bound listeners until Complete */

typedef struct tbcclUniqueId_st { uint8_t bytes[16]; } tbcclUniqueId;    /* fixed identity token, no addresses */

typedef struct tbcclBuffer_st {            /* sizeof 48 on LP64 */
    uint32_t struct_size;
    tbcclMemoryKind_t memory_kind;
    int32_t  device_ordinal;               /* CUDA device; -1 for host */
    uint32_t reserved0;                    /* must be 0 */
    void    *data;                         /* sends promise not to modify it */
    uint64_t bytes;
    uint64_t reserved1[2];                 /* must be 0 */
} tbcclBuffer;

typedef struct tbcclExecContext_st {       /* sizeof 32 on LP64 */
    uint32_t struct_size;
    tbcclExecKind_t kind;
    void    *native_handle;
    uint64_t reserved[1];                  /* must be 0 */
} tbcclExecContext;

typedef struct tbcclBootstrapOptions_st {  /* sizeof 56 on LP64 */
    uint32_t struct_size;
    uint32_t reserved0;                    /* must be 0 */
    const char *bind_host;                 /* NULL -> "127.0.0.1" */
    const char *advertise_host;            /* NULL -> bind_host (an error if that is a wildcard) */
    uint32_t timeout_ms;                   /* 0 -> 10000; bounds the whole bootstrap */
    uint32_t reserved1;                    /* must be 0 */
    uint64_t reserved2[3];                 /* must be 0 */
} tbcclBootstrapOptions;

typedef struct tbcclEndpointBlob_st {      /* sizeof 256, forever */
    uint32_t struct_size;                  /* 256, written by tbcclBootstrapGetEndpoint */
    uint32_t format_version;               /* 1 */
    uint32_t used_bytes;                   /* meaningful bytes of payload[] */
    uint32_t reserved;                     /* 0 */
    uint8_t  payload[240];                 /* OPAQUE: callers never parse it */
} tbcclEndpointBlob;

typedef struct tbcclCapabilities_st {      /* sizeof 48 */
    uint32_t struct_size;
    uint32_t memory_kind_mask;             /* bit (1 << tbcclMemoryKind_t) set when every rank can use that kind */
    uint64_t effective_max_chunk;          /* bytes, 0 = no limit */
    uint64_t effective_alignment;          /* bytes */
    uint64_t reserved[3];                  /* output: 0 */
} tbcclCapabilities;
```

Layouts are pinned by `sizeof`/`offsetof` static assertions on every supported ABI platform (LP64 Linux and macOS arm64/x86-64) and by `tests/c_abi_layout_test`.

## 5. Functions

```c
/* version / strings */
tbcclResult_t tbcclGetAbiVersion(uint32_t *abi_version);
tbcclResult_t tbcclGetPackageVersion(uint32_t *major, uint32_t *minor, uint32_t *patch);
const char   *tbcclGetResultString(tbcclResult_t result);              /* static text, never NULL */

/* bootstrap: caller-supplied exchange of opaque endpoint blobs (docs/c_api_bootstrap.md) */
tbcclResult_t tbcclGetUniqueId(tbcclUniqueId *id);
tbcclResult_t tbcclBootstrapBegin(uint32_t rank, uint32_t world_size, const tbcclUniqueId *id,
                                  const tbcclBootstrapOptions *options /* nullable */, tbcclBootstrap_t *bootstrap);
tbcclResult_t tbcclBootstrapGetEndpoint(tbcclBootstrap_t bootstrap, tbcclEndpointBlob *blob);
tbcclResult_t tbcclBootstrapComplete(tbcclBootstrap_t bootstrap, const tbcclEndpointBlob *blobs /* world_size, in rank order */,
                                     uint32_t blob_count, tbcclComm_t *comm);
tbcclResult_t tbcclBootstrapDestroy(tbcclBootstrap_t bootstrap);        /* NULL -> SUCCESS */

/* communicator */
tbcclResult_t tbcclCommDestroy(tbcclComm_t comm);                       /* NULL -> SUCCESS */
tbcclResult_t tbcclCommGetRank(tbcclComm_t comm, uint32_t *rank);
tbcclResult_t tbcclCommGetSize(tbcclComm_t comm, uint32_t *world_size);
tbcclResult_t tbcclCommAbort(tbcclComm_t comm, const char *reason /* nullable */);
tbcclResult_t tbcclCommIsAborted(tbcclComm_t comm, int32_t *aborted);
tbcclResult_t tbcclCommGetAbortReason(tbcclComm_t comm, char *buf, size_t capacity, size_t *required);
tbcclResult_t tbcclCommGetCapabilities(tbcclComm_t comm, tbcclCapabilities *capabilities);
tbcclResult_t tbcclCommSupportsAllReduce(tbcclComm_t comm, uint32_t rank, tbcclMemoryKind_t kind,
                                         tbcclDataType_t dtype, tbcclReduceOp_t op, int32_t *supported);

/* operations: every call returns a Work immediately and never waits for progress */
tbcclResult_t tbcclSend(tbcclComm_t comm, const tbcclBuffer *buffer, uint32_t peer, const tbcclExecContext *ctx, tbcclWork_t *work);
tbcclResult_t tbcclRecv(tbcclComm_t comm, const tbcclBuffer *buffer, uint32_t peer, const tbcclExecContext *ctx, tbcclWork_t *work);
tbcclResult_t tbcclBarrier(tbcclComm_t comm, tbcclWork_t *work);
tbcclResult_t tbcclBroadcast(tbcclComm_t comm, const tbcclBuffer *buffer, uint32_t root, const tbcclExecContext *ctx, tbcclWork_t *work);
tbcclResult_t tbcclAllGather(tbcclComm_t comm, const tbcclBuffer *send, const tbcclBuffer *recv, const tbcclExecContext *ctx, tbcclWork_t *work);
tbcclResult_t tbcclAllReduce(tbcclComm_t comm, const tbcclBuffer *send, const tbcclBuffer *recv, uint64_t count,
                             tbcclDataType_t dtype, tbcclReduceOp_t op, const tbcclExecContext *ctx, tbcclWork_t *work);

/* work: the API return is the status of the QUERY; the operation's own result is an out-parameter */
tbcclResult_t tbcclWorkTest(tbcclWork_t work, int32_t *done, tbcclResult_t *operation_result);
tbcclResult_t tbcclWorkWait(tbcclWork_t work, tbcclResult_t *operation_result);
tbcclResult_t tbcclWorkWaitFor(tbcclWork_t work, uint64_t timeout_ms, int32_t *done, tbcclResult_t *operation_result);
tbcclResult_t tbcclWorkGetErrorString(tbcclWork_t work, char *buf, size_t capacity, size_t *required);
tbcclResult_t tbcclWorkDestroy(tbcclWork_t work);                       /* NULL -> SUCCESS */

/* CUDA: the same symbol exists in host-only builds, where it returns TBCCL_UNSUPPORTED */
tbcclResult_t tbcclRegisterCudaSupport(void);                           /* idempotent, process-wide */
```

## 6. Semantics

**Buffers.** `tbcclBuffer` does not own memory. The application must keep the host, CUDA or MetalShared allocation valid and unmodified (sends) / untouched (receives) **until the operation is terminal**: until `Work` reports done, or the communicator is terminal (aborted) and the Work is done. Destroying the `Work` handle does **not** cancel the operation and does **not** release this obligation. Collective in-place buffers (broadcast, all_reduce with `send == recv`) follow the same rule.

**Submission never waits.** A successful call means TBCCL accepted responsibility and returned a `Work`. It does not mean a lane became available, a socket progressed, a peer posted a matching operation, or staging was acquired. A call either accepts immediately or fails immediately (`INVALID_ARGUMENT`, `UNSUPPORTED`, `RESOURCE_EXHAUSTED`, `ABORTED`). FIFO per (peer, direction) is the runtime's local linearization order. There are no tags: the application matches the logical P2P order between sender and receiver.

**P2P is byte-based.** `buffer->bytes` is the payload size; a send and its receive must use the same byte count (a different size fails with `PROTOCOL_MISMATCH`). `peer` must be in range and not the caller. **Zero bytes** is a legal asynchronous no-op: it completes successfully without communicating and does not synchronize (`data` may be NULL).

**Collectives.** `tbcclBarrier` has no payload. `tbcclBroadcast` is byte-generic and in place on every rank (same `bytes` everywhere). `tbcclAllGather` is byte-generic: `recv->bytes` must equal `send->bytes * world_size` (overflow-checked); rank r's contribution lands at byte offset `r * send->bytes`. `tbcclAllReduce` is typed: `count` elements of `dtype`, `send->bytes` and `recv->bytes` at least `count * sizeof(dtype)` (overflow-checked); FP16/BF16 only at world size 2; float N>2 results follow `docs/numerical_reduction_semantics.md`. Algorithm selection is internal and not exposed. All ranks must submit collectives in the same logical order (`docs/c_api_thread_safety.md`).

**Execution context.** `ctx == NULL` is the default context for the buffer's memory provider. For a non-default CUDA stream pass `{struct_size, TBCCL_EXEC_CUDA_STREAM, (void *)cudaStream}`; the stream must stay valid until the Work is done.

**Work.** `Test`: non-blocking; `*done` = 0 or 1; `*operation_result` is meaningful only if `*done == 1` (it is initialized to SUCCESS). `Wait`: blocks until the Work is terminal; API return SUCCESS and `*operation_result` the final result, unless the handle is invalid. `WaitFor`: waits at most `timeout_ms`; on expiry the API returns SUCCESS with `*done == 0`, and the Work is neither cancelled, poisoned nor made terminal; **a caller timeout is never reported as `TBCCL_TIMEOUT`**. `GetErrorString` returns the detailed terminal text (empty for a successful or not-yet-terminal Work). Repeated queries are stable. `WorkDestroy` drops only the caller's handle. Test/Wait/WaitFor may race with one another; `WorkDestroy` must not race with another call using the same handle.

**Abort and destruction.** `tbcclCommAbort` is communicator-wide, idempotent, safe from any thread; every queued operation becomes terminal with `ABORTED` without touching its buffer, the active one after the transport is interrupted. The abort reason is queryable. `tbcclCommDestroy` with outstanding operations is safe and bounded (existing C++ abort/destruction semantics) but must not race with new calls on the same handle: the caller quiesces API entry first.

**Capabilities.** `tbcclCommSupportsAllReduce` answers per (rank, kind, dtype, op), avoiding a giant matrix in a struct.

**CUDA.** `tbcclRegisterCudaSupport` registers the CUDA memory provider process-wide (idempotent); host-only builds return `UNSUPPORTED`. `tbccl.h` never includes CUDA; the application passes its `cudaStream_t` through `native_handle`.

## 7. Handle rules

`NULL` is accepted by `WorkDestroy`, `CommDestroy`, `BootstrapDestroy` (SUCCESS) and is `INVALID_ARGUMENT` for every other function. Only `NULL` and currently valid handles are validated; use of a stale (already destroyed) pointer is application undefined behaviour and is not promised to be detected.

## 8. Not in v1

Hidden rendezvous bootstrap, `GroupStart`/`GroupEnd` (`grouped_operations_audit.md`), algorithm or topology controls, user tags, communicator split/shrink/grow, fault recovery, FP8/INT4 reductions, FP16/BF16 reduction at N>2, custom provider callbacks, Python/Rust/Swift bindings, and any non-IPv4 endpoint (the transport is IPv4 numeric in this release).
