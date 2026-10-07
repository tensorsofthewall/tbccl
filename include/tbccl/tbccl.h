/*
 * TBCCL C ABI v1 (docs/reference/c-abi.md is authoritative).
 *
 * Plain C11, also valid C++. Only <stdint.h> and <stddef.h>; no CUDA, Metal or C++ header. The symbols live in the dedicated TBCCL::tbccl_c target;
 * the C++ runtime (TBCCL::tbccl) is not an ABI promise. Never renumber a constant, reorder or remove a field, remove a function or change a
 * signature in v1; growth is by appended struct fields (struct_size), new constants, new functions and new structs.
 */
#ifndef TBCCL_TBCCL_H
#define TBCCL_TBCCL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Decoration for exported functions. Empty on the static Linux/macOS build; a future Windows DLL defines TBCCL_SHARED / TBCCL_BUILDING_SHARED without
 * changing any signature. */
#ifndef TBCCL_API
#  if defined(_WIN32) && defined(TBCCL_SHARED)
#    ifdef TBCCL_BUILDING_SHARED
#      define TBCCL_API __declspec(dllexport)
#    else
#      define TBCCL_API __declspec(dllimport)
#    endif
#  else
#    define TBCCL_API
#  endif
#endif
#ifndef TBCCL_CALL
#  if defined(_WIN32) && !defined(_WIN64)
#    define TBCCL_CALL __cdecl
#  else
#    define TBCCL_CALL
#  endif
#endif

#define TBCCL_C_ABI_VERSION 1u

/* ---- fixed-width ABI storage (never a C enum) ------------------------------------------------------------------------------------------ */
typedef int32_t tbcclResult_t;
typedef int32_t tbcclDataType_t;
typedef int32_t tbcclReduceOp_t;
typedef int32_t tbcclMemoryKind_t;
typedef int32_t tbcclExecKind_t;

#define TBCCL_SUCCESS             0
#define TBCCL_INVALID_ARGUMENT    1
#define TBCCL_UNSUPPORTED         2
#define TBCCL_RESOURCE_EXHAUSTED  3
#define TBCCL_ABORTED             4
#define TBCCL_TIMEOUT             5
#define TBCCL_PROTOCOL_MISMATCH   6
#define TBCCL_TRANSPORT_ERROR     7
#define TBCCL_INTERNAL_ERROR      8
#define TBCCL_DEVICE_ERROR        9

#define TBCCL_INT32     0
#define TBCCL_INT64     1
#define TBCCL_FLOAT32   2
#define TBCCL_FLOAT64   3
#define TBCCL_INT8      4
#define TBCCL_UINT8     5
#define TBCCL_FLOAT16   6
#define TBCCL_BFLOAT16  7

#define TBCCL_SUM      0
#define TBCCL_PRODUCT  1
#define TBCCL_MIN      2
#define TBCCL_MAX      3

#define TBCCL_MEMORY_HOST          0
#define TBCCL_MEMORY_CUDA          1
#define TBCCL_MEMORY_METAL_SHARED  2

#define TBCCL_EXEC_DEFAULT      0
#define TBCCL_EXEC_CUDA_STREAM  1

#define TBCCL_ENDPOINT_BLOB_SIZE 256u

/* ---- opaque handles ---------------------------------------------------------------------------------------------------------------------- */
typedef struct tbcclComm_st      *tbcclComm_t;
typedef struct tbcclWork_st      *tbcclWork_t;
typedef struct tbcclBootstrap_st *tbcclBootstrap_t;

/* ---- structs. Every extensible struct starts with struct_size (zero-initialize, then set it to sizeof). Reserved fields must be zero. ----- */

/* The session identity: 16 random bytes, no addresses. One process generates it, the application distributes it to every rank. */
typedef struct tbcclUniqueId_st {
    uint8_t bytes[16];
} tbcclUniqueId;

/* A region of memory. Does not own it: the application keeps it valid (and unmodified for sends) until the operation is terminal, even if the Work
 * handle is destroyed first. */
typedef struct tbcclBuffer_st {
    uint32_t          struct_size;
    tbcclMemoryKind_t memory_kind;
    int32_t           device_ordinal; /* CUDA device; -1 for host memory */
    uint32_t          reserved0;      /* must be 0 */
    void             *data;
    uint64_t          bytes;
    uint64_t          reserved1[2];   /* must be 0 */
} tbcclBuffer;

/* NULL means "the default context of the buffer's memory provider". */
typedef struct tbcclExecContext_st {
    uint32_t        struct_size;
    tbcclExecKind_t kind;
    void           *native_handle;    /* kind == TBCCL_EXEC_CUDA_STREAM: a cudaStream_t */
    uint64_t        reserved[1];      /* must be 0 */
} tbcclExecContext;

typedef struct tbcclBootstrapOptions_st {
    uint32_t    struct_size;
    uint32_t    reserved0;            /* must be 0 */
    const char *bind_host;            /* NULL -> "127.0.0.1"; dotted-quad IPv4 */
    const char *advertise_host;       /* NULL -> bind_host; must be reachable by the peers and must not be a wildcard */
    uint32_t    timeout_ms;           /* 0 -> 10000; bounds the whole bootstrap */
    uint32_t    reserved1;            /* must be 0 */
    uint64_t    reserved2[3];         /* must be 0 */
} tbcclBootstrapOptions;

/* One rank's published endpoints. Fixed size so an all-gather needs no length exchange. The payload is opaque: callers never parse it. */
typedef struct tbcclEndpointBlob_st {
    uint32_t struct_size;             /* TBCCL_ENDPOINT_BLOB_SIZE, written by tbcclBootstrapGetEndpoint */
    uint32_t format_version;          /* 1 */
    uint32_t used_bytes;
    uint32_t reserved;
    uint8_t  payload[240];
} tbcclEndpointBlob;

typedef struct tbcclCapabilities_st {
    uint32_t struct_size;
    uint32_t memory_kind_mask;        /* bit (1 << tbcclMemoryKind_t): every rank can use that kind */
    uint64_t effective_max_chunk;     /* bytes; 0 = no limit */
    uint64_t effective_alignment;     /* bytes */
    uint64_t reserved[3];             /* output: 0 */
} tbcclCapabilities;

/* Layout pins (64-bit pointers): a wrong size or offset is a compile error for every consumer. */
#if defined(UINTPTR_MAX) && UINTPTR_MAX == 0xffffffffffffffffu
#  define TBCCL_ASSERT_LAYOUT(cond, name) typedef char tbccl_layout_assert_##name[(cond) ? 1 : -1]
TBCCL_ASSERT_LAYOUT(sizeof(tbcclUniqueId) == 16, unique_id);
TBCCL_ASSERT_LAYOUT(sizeof(tbcclBuffer) == 48, buffer);
TBCCL_ASSERT_LAYOUT(offsetof(tbcclBuffer, data) == 16 && offsetof(tbcclBuffer, bytes) == 24, buffer_offsets);
TBCCL_ASSERT_LAYOUT(sizeof(tbcclExecContext) == 24, exec_context);
TBCCL_ASSERT_LAYOUT(offsetof(tbcclExecContext, native_handle) == 8, exec_context_offsets);
TBCCL_ASSERT_LAYOUT(sizeof(tbcclBootstrapOptions) == 56, bootstrap_options);
TBCCL_ASSERT_LAYOUT(offsetof(tbcclBootstrapOptions, bind_host) == 8 && offsetof(tbcclBootstrapOptions, timeout_ms) == 24, bootstrap_options_offsets);
TBCCL_ASSERT_LAYOUT(sizeof(tbcclEndpointBlob) == TBCCL_ENDPOINT_BLOB_SIZE, endpoint_blob);
TBCCL_ASSERT_LAYOUT(offsetof(tbcclEndpointBlob, payload) == 16, endpoint_blob_offsets);
TBCCL_ASSERT_LAYOUT(sizeof(tbcclCapabilities) == 48, capabilities);
#endif

/* ---- version and strings ---------------------------------------------------------------------------------------------------------------- */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclGetAbiVersion(uint32_t *abi_version);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclGetPackageVersion(uint32_t *major, uint32_t *minor, uint32_t *patch);
/* Static generic text for a result value; never NULL. */
TBCCL_API const char *TBCCL_CALL tbcclGetResultString(tbcclResult_t result);

/* ---- bootstrap: the application exchanges opaque endpoint blobs (docs/reference/c-abi-bootstrap.md) --------------------------------------------- */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclGetUniqueId(tbcclUniqueId *id);
/* Binds this rank's listeners (ports chosen by the kernel) and owns them. `options` may be NULL. */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclBootstrapBegin(uint32_t rank, uint32_t world_size, const tbcclUniqueId *id,
                                                       const tbcclBootstrapOptions *options, tbcclBootstrap_t *bootstrap);
/* Fills `blob` (set blob->struct_size to sizeof(tbcclEndpointBlob) first) with this rank's endpoints for the application to all-gather. */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclBootstrapGetEndpoint(tbcclBootstrap_t bootstrap, tbcclEndpointBlob *blob);
/* `blobs` holds world_size blobs in rank order (index i is rank i). May be called once; afterwards the bootstrap can only be destroyed. */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclBootstrapComplete(tbcclBootstrap_t bootstrap, const tbcclEndpointBlob *blobs, uint32_t blob_count,
                                                          tbcclComm_t *comm);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclBootstrapDestroy(tbcclBootstrap_t bootstrap); /* NULL -> SUCCESS */

/* ---- communicator ----------------------------------------------------------------------------------------------------------------------- */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclCommDestroy(tbcclComm_t comm); /* NULL -> SUCCESS; must not race with other calls on this handle */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclCommGetRank(tbcclComm_t comm, uint32_t *rank);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclCommGetSize(tbcclComm_t comm, uint32_t *world_size);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclCommAbort(tbcclComm_t comm, const char *reason);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclCommIsAborted(tbcclComm_t comm, int32_t *aborted);
/* Copies the abort reason (empty if none) into buf: at most capacity-1 characters plus a NUL; *required is the full length including the NUL. */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclCommGetAbortReason(tbcclComm_t comm, char *buf, size_t capacity, size_t *required);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclCommGetCapabilities(tbcclComm_t comm, tbcclCapabilities *capabilities);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclCommSupportsAllReduce(tbcclComm_t comm, uint32_t rank, tbcclMemoryKind_t kind, tbcclDataType_t dtype,
                                                              tbcclReduceOp_t op, int32_t *supported);

/* ---- operations: every call returns a Work immediately and never waits for socket, peer, lane or staging progress ----------------------- */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclSend(tbcclComm_t comm, const tbcclBuffer *buffer, uint32_t peer, const tbcclExecContext *ctx,
                                             tbcclWork_t *work);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclRecv(tbcclComm_t comm, const tbcclBuffer *buffer, uint32_t peer, const tbcclExecContext *ctx,
                                             tbcclWork_t *work);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclBarrier(tbcclComm_t comm, tbcclWork_t *work);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclBroadcast(tbcclComm_t comm, const tbcclBuffer *buffer, uint32_t root, const tbcclExecContext *ctx,
                                                  tbcclWork_t *work);
/* recv->bytes must equal send->bytes * world_size; rank r's contribution lands at byte offset r * send->bytes. */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclAllGather(tbcclComm_t comm, const tbcclBuffer *send, const tbcclBuffer *recv, const tbcclExecContext *ctx,
                                                  tbcclWork_t *work);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclAllReduce(tbcclComm_t comm, const tbcclBuffer *send, const tbcclBuffer *recv, uint64_t count,
                                                  tbcclDataType_t dtype, tbcclReduceOp_t op, const tbcclExecContext *ctx, tbcclWork_t *work);

/* ---- work: the API return is the status of the QUERY; *operation_result is the terminal result of the asynchronous operation ------------ */
/* *done is 0 or 1; *operation_result is meaningful only if *done == 1 (it is initialized to TBCCL_SUCCESS). */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclWorkTest(tbcclWork_t work, int32_t *done, tbcclResult_t *operation_result);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclWorkWait(tbcclWork_t work, tbcclResult_t *operation_result);
/* Waits at most timeout_ms. Expiry returns TBCCL_SUCCESS with *done == 0 and changes nothing: it does not cancel, poison or complete the Work. */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclWorkWaitFor(tbcclWork_t work, uint64_t timeout_ms, int32_t *done, tbcclResult_t *operation_result);
TBCCL_API tbcclResult_t TBCCL_CALL tbcclWorkGetErrorString(tbcclWork_t work, char *buf, size_t capacity, size_t *required);
/* Drops only the caller's handle: does NOT cancel the operation and does NOT permit freeing or modifying its buffer. NULL -> SUCCESS. */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclWorkDestroy(tbcclWork_t work);

/* ---- CUDA: present in every build; host-only builds return TBCCL_UNSUPPORTED. Idempotent, process-wide. ---------------------------------- */
TBCCL_API tbcclResult_t TBCCL_CALL tbcclRegisterCudaSupport(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* TBCCL_TBCCL_H */
