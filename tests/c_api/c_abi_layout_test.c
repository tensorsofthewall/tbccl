/* Pins the C ABI v1 constants and struct layouts (docs/c_abi_v1.md) and the struct_size evolution rules. A failure here means the ABI was changed. */

#include "c_test_support.h"

#include <stddef.h>
#include <stdint.h>

#define PIN(name, value) CHECK((int)(name) == (value))

static void test_constants(void)
{
    PIN(TBCCL_SUCCESS, 0); PIN(TBCCL_INVALID_ARGUMENT, 1); PIN(TBCCL_UNSUPPORTED, 2); PIN(TBCCL_RESOURCE_EXHAUSTED, 3); PIN(TBCCL_ABORTED, 4);
    PIN(TBCCL_TIMEOUT, 5); PIN(TBCCL_PROTOCOL_MISMATCH, 6); PIN(TBCCL_TRANSPORT_ERROR, 7); PIN(TBCCL_INTERNAL_ERROR, 8); PIN(TBCCL_DEVICE_ERROR, 9);
    PIN(TBCCL_INT32, 0); PIN(TBCCL_INT64, 1); PIN(TBCCL_FLOAT32, 2); PIN(TBCCL_FLOAT64, 3); PIN(TBCCL_INT8, 4); PIN(TBCCL_UINT8, 5);
    PIN(TBCCL_FLOAT16, 6); PIN(TBCCL_BFLOAT16, 7);
    PIN(TBCCL_SUM, 0); PIN(TBCCL_PRODUCT, 1); PIN(TBCCL_MIN, 2); PIN(TBCCL_MAX, 3);
    PIN(TBCCL_MEMORY_HOST, 0); PIN(TBCCL_MEMORY_CUDA, 1); PIN(TBCCL_MEMORY_METAL_SHARED, 2);
    PIN(TBCCL_EXEC_DEFAULT, 0); PIN(TBCCL_EXEC_CUDA_STREAM, 1);
    CHECK(TBCCL_C_ABI_VERSION == 1u);
    CHECK(TBCCL_ENDPOINT_BLOB_SIZE == 256u);
    /* fixed-width storage, never a C enum */
    CHECK(sizeof(tbcclResult_t) == 4 && sizeof(tbcclDataType_t) == 4 && sizeof(tbcclReduceOp_t) == 4 && sizeof(tbcclMemoryKind_t) == 4 && sizeof(tbcclExecKind_t) == 4);
    CHECK((tbcclResult_t)-1 < 0); /* signed */
    printf("[PASS] result, datatype, reduce-op, memory-kind and exec-kind constants are pinned; all enumerations are int32_t\n");
}

static void test_layout(void)
{
    CHECK(sizeof(tbcclUniqueId) == 16);
    CHECK(sizeof(tbcclBuffer) == 48 && offsetof(tbcclBuffer, struct_size) == 0 && offsetof(tbcclBuffer, memory_kind) == 4 && offsetof(tbcclBuffer, device_ordinal) == 8 &&
          offsetof(tbcclBuffer, reserved0) == 12 && offsetof(tbcclBuffer, data) == 16 && offsetof(tbcclBuffer, bytes) == 24 && offsetof(tbcclBuffer, reserved1) == 32);
    CHECK(sizeof(tbcclExecContext) == 24 && offsetof(tbcclExecContext, struct_size) == 0 && offsetof(tbcclExecContext, kind) == 4 &&
          offsetof(tbcclExecContext, native_handle) == 8 && offsetof(tbcclExecContext, reserved) == 16);
    CHECK(sizeof(tbcclBootstrapOptions) == 56 && offsetof(tbcclBootstrapOptions, struct_size) == 0 && offsetof(tbcclBootstrapOptions, reserved0) == 4 &&
          offsetof(tbcclBootstrapOptions, bind_host) == 8 && offsetof(tbcclBootstrapOptions, advertise_host) == 16 && offsetof(tbcclBootstrapOptions, timeout_ms) == 24 &&
          offsetof(tbcclBootstrapOptions, reserved1) == 28 && offsetof(tbcclBootstrapOptions, reserved2) == 32);
    CHECK(sizeof(tbcclEndpointBlob) == 256 && offsetof(tbcclEndpointBlob, struct_size) == 0 && offsetof(tbcclEndpointBlob, format_version) == 4 &&
          offsetof(tbcclEndpointBlob, used_bytes) == 8 && offsetof(tbcclEndpointBlob, reserved) == 12 && offsetof(tbcclEndpointBlob, payload) == 16);
    CHECK(sizeof(tbcclCapabilities) == 48 && offsetof(tbcclCapabilities, struct_size) == 0 && offsetof(tbcclCapabilities, memory_kind_mask) == 4 &&
          offsetof(tbcclCapabilities, effective_max_chunk) == 8 && offsetof(tbcclCapabilities, effective_alignment) == 16 && offsetof(tbcclCapabilities, reserved) == 24);
    printf("[PASS] struct sizes and field offsets are pinned (LP64)\n");
}

/* struct_size evolution against a live communicator: exact size, a larger zero-filled struct, and a too-small prefix. */
static void evolution_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)rank; (void)world; (void)user;

    /* output struct: exact size */
    tbcclCapabilities exact;
    memset(&exact, 0xCD, sizeof(exact));
    exact.struct_size = sizeof(exact);
    CHECK_OK(tbcclCommGetCapabilities(comm, &exact));
    CHECK(exact.struct_size == sizeof(exact));

    /* output struct: caller's struct is LARGER than v1 and zero-filled beyond: the library writes nothing beyond the v1 fields */
    struct { tbcclCapabilities caps; uint64_t appended[4]; } larger;
    memset(&larger, 0x5A, sizeof(larger)); /* sentinel everywhere */
    memset(&larger.caps, 0, sizeof(larger.caps));
    larger.caps.struct_size = (uint32_t)sizeof(larger);
    CHECK_OK(tbcclCommGetCapabilities(comm, &larger.caps));
    for (int i = 0; i < 4; ++i) CHECK(larger.appended[i] == 0x5A5A5A5A5A5A5A5Aull); /* untouched */
    CHECK(larger.caps.memory_kind_mask == exact.memory_kind_mask);

    /* output struct: prefix smaller than v1 is rejected, the memory is untouched */
    tbcclCapabilities tiny;
    memset(&tiny, 0x77, sizeof(tiny));
    tiny.struct_size = 8;
    CHECK_RESULT(tbcclCommGetCapabilities(comm, &tiny), TBCCL_INVALID_ARGUMENT);
    CHECK(tiny.memory_kind_mask == 0x77777777u);

    /* input struct: a larger zero-filled tbcclBuffer is accepted (the appended bytes are ignored); a smaller prefix is rejected */
    uint8_t data[16] = {0}, out[16] = {0}; /* world size 1: recv bytes = send bytes * 1 */
    struct { tbcclBuffer b; uint64_t appended[2]; } big_buf;
    memset(&big_buf, 0, sizeof(big_buf));
    big_buf.b = host_buffer(data, sizeof(data));
    big_buf.b.struct_size = (uint32_t)sizeof(big_buf);
    tbcclBuffer out_buf = host_buffer(out, sizeof(out));
    tbcclWork_t w = NULL;
    CHECK_OK(tbcclAllGather(comm, &big_buf.b, &out_buf, NULL, &w));
    CHECK(wait_result(w) == TBCCL_SUCCESS);
    CHECK_OK(tbcclWorkDestroy(w));
    big_buf.b.struct_size = (uint32_t)(sizeof(tbcclBuffer) - 8);
    CHECK_RESULT(tbcclAllGather(comm, &big_buf.b, &out_buf, NULL, &w), TBCCL_INVALID_ARGUMENT);
}

static void test_struct_size_rules(void)
{
    run_world(1, evolution_body, NULL, NULL);

    /* input struct (bootstrap options) and endpoint blob, at world size 1 where no sockets are involved */
    tbcclUniqueId id;
    CHECK_OK(tbcclGetUniqueId(&id));
    struct { tbcclBootstrapOptions o; uint64_t appended[3]; } big_opt;
    memset(&big_opt, 0, sizeof(big_opt));
    big_opt.o.struct_size = (uint32_t)sizeof(big_opt);
    tbcclBootstrap_t bs = NULL;
    CHECK_OK(tbcclBootstrapBegin(0, 1, &id, &big_opt.o, &bs));
    CHECK_OK(tbcclBootstrapDestroy(bs));
    big_opt.o.struct_size = (uint32_t)(sizeof(tbcclBootstrapOptions) - 1);
    CHECK_RESULT(tbcclBootstrapBegin(0, 1, &id, &big_opt.o, &bs), TBCCL_INVALID_ARGUMENT);

    /* the endpoint blob is written whole (fixed 256 bytes); a caller struct_size below 256 is rejected, one above is accepted and nothing beyond 256 is written */
    CHECK_OK(tbcclBootstrapBegin(0, 1, &id, NULL, &bs));
    struct { tbcclEndpointBlob b; uint8_t after[16]; } bb;
    memset(&bb, 0x3C, sizeof(bb));
    bb.b.struct_size = (uint32_t)sizeof(bb);
    CHECK_OK(tbcclBootstrapGetEndpoint(bs, &bb.b));
    CHECK(bb.b.struct_size == TBCCL_ENDPOINT_BLOB_SIZE && bb.b.format_version == 1);
    for (int i = 0; i < 16; ++i) CHECK(bb.after[i] == 0x3C);
    CHECK_OK(tbcclBootstrapDestroy(bs));
    printf("[PASS] struct_size rules: exact, larger zero-filled (nothing written beyond v1), below-minimum rejected (input and output structs)\n");
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    test_constants();
    test_layout();
    test_struct_size_rules();
    printf("All C ABI layout tests passed.\n");
    return 0;
}
