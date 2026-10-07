/* pure-C functional test of C ABI v1 (docs/c_abi_v1.md): bootstrap at N=1..4, P2P, every collective, Work semantics, opaque byte payloads, abort,
 * capabilities and the argument-validation matrix. Plain C11; the only TBCCL header is tbccl.h. */

#include "c_test_support.h"

#include <stdint.h>

/* ------------------------------------------------------------------------------------------------------------------------------------------ */
static void test_version_and_strings(void)
{
    uint32_t abi = 0, major = 99, minor = 99, patch = 99;
    CHECK_OK(tbcclGetAbiVersion(&abi));
    CHECK(abi == TBCCL_C_ABI_VERSION && abi == 1u);
    CHECK_OK(tbcclGetPackageVersion(&major, &minor, &patch));
    CHECK(major == 0 && minor >= 4);
    CHECK_RESULT(tbcclGetAbiVersion(NULL), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclGetPackageVersion(&major, NULL, &patch), TBCCL_INVALID_ARGUMENT);
    for (int r = TBCCL_SUCCESS; r <= TBCCL_DEVICE_ERROR; ++r) {
        const char *s = tbcclGetResultString(r);
        CHECK(s != NULL && s[0] != '\0');
    }
    CHECK(tbcclGetResultString(12345) != NULL && tbcclGetResultString(-1) != NULL);
    printf("[PASS] version, package version and result strings\n");
}

/* ------------------------------------------------------------------------------------------------------------------------------------------ */
typedef struct {
    TestBarrier gate;
} Shared;

static void exchange_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)user;
    uint32_t r = 99, n = 99;
    CHECK_OK(tbcclCommGetRank(comm, &r));
    CHECK_OK(tbcclCommGetSize(comm, &n));
    CHECK((int)r == rank && (int)n == world);

    /* barrier, twice (sequence ordering) */
    for (int i = 0; i < 2; ++i) {
        tbcclWork_t w = NULL;
        CHECK_OK(tbcclBarrier(comm, &w));
        CHECK(w != NULL);
        CHECK(wait_result(w) == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(w));
    }

    if (world > 1) {
        /* P2P ring with distinct per-rank content, send and receive posted before either is waited on */
        enum { N = 4097 };
        uint8_t out[N], in[N];
        const int next = (rank + 1) % world, prev = (rank + world - 1) % world;
        for (int i = 0; i < N; ++i) out[i] = (uint8_t)(rank * 37 + i * 11);
        memset(in, 0, sizeof(in));
        tbcclBuffer sb = host_buffer(out, N), rb = host_buffer(in, N);
        tbcclWork_t s = NULL, rcv = NULL;
        CHECK_OK(tbcclSend(comm, &sb, (uint32_t)next, NULL, &s));
        CHECK_OK(tbcclRecv(comm, &rb, (uint32_t)prev, NULL, &rcv));
        CHECK(wait_result(s) == TBCCL_SUCCESS && wait_result(rcv) == TBCCL_SUCCESS);
        for (int i = 0; i < N; ++i) CHECK(in[i] == (uint8_t)(prev * 37 + i * 11));
        CHECK_OK(tbcclWorkDestroy(s));
        CHECK_OK(tbcclWorkDestroy(rcv));
    }

    /* broadcast from every root: byte-generic, odd size */
    for (int root = 0; root < world; ++root) {
        uint8_t buf[1001];
        for (int i = 0; i < 1001; ++i) buf[i] = (rank == root) ? (uint8_t)(root * 53 + i * 3 + 1) : 0;
        tbcclBuffer b = host_buffer(buf, sizeof(buf));
        tbcclWork_t w = NULL;
        CHECK_OK(tbcclBroadcast(comm, &b, (uint32_t)root, NULL, &w));
        CHECK(wait_result(w) == TBCCL_SUCCESS);
        for (int i = 0; i < 1001; ++i) CHECK(buf[i] == (uint8_t)(root * 53 + i * 3 + 1));
        CHECK_OK(tbcclWorkDestroy(w));
    }

    /* all_gather: rank r's block lands at offset r * send_bytes */
    {
        enum { B = 333 };
        uint8_t mine[B], all[8 * B];
        for (int i = 0; i < B; ++i) mine[i] = (uint8_t)(rank * 17 + i);
        memset(all, 0xEE, sizeof(all));
        tbcclBuffer s = host_buffer(mine, B), r = host_buffer(all, (size_t)B * (size_t)world);
        tbcclWork_t w = NULL;
        CHECK_OK(tbcclAllGather(comm, &s, &r, NULL, &w));
        CHECK(wait_result(w) == TBCCL_SUCCESS);
        for (int q = 0; q < world; ++q)
            for (int i = 0; i < B; ++i) CHECK(all[q * B + i] == (uint8_t)(q * 17 + i));
        CHECK_OK(tbcclWorkDestroy(w));
    }

    /* typed all_reduce, in place and out of place; exact integer-valued sums */
    {
        float f[1000];
        for (int i = 0; i < 1000; ++i) f[i] = (float)(rank + 1);
        tbcclBuffer b = host_buffer(f, sizeof(f));
        tbcclWork_t w = NULL;
        CHECK_OK(tbcclAllReduce(comm, &b, &b, 1000, TBCCL_FLOAT32, TBCCL_SUM, NULL, &w));
        CHECK(wait_result(w) == TBCCL_SUCCESS);
        const float want = (float)(world * (world + 1) / 2);
        for (int i = 0; i < 1000; ++i) CHECK(f[i] == want);
        CHECK_OK(tbcclWorkDestroy(w));

        int64_t src[300], dst[300];
        for (int i = 0; i < 300; ++i) { src[i] = (int64_t)(rank + 1) * 1000000007LL + i; dst[i] = -1; }
        tbcclBuffer sb = host_buffer(src, sizeof(src)), db = host_buffer(dst, sizeof(dst));
        CHECK_OK(tbcclAllReduce(comm, &sb, &db, 300, TBCCL_INT64, TBCCL_SUM, NULL, &w));
        CHECK(wait_result(w) == TBCCL_SUCCESS);
        for (int i = 0; i < 300; ++i) {
            int64_t t = 0;
            for (int q = 0; q < world; ++q) t += (int64_t)(q + 1) * 1000000007LL + i;
            CHECK(dst[i] == t);
        }
        CHECK_OK(tbcclWorkDestroy(w));
    }

    /* capabilities */
    {
        tbcclCapabilities caps;
        memset(&caps, 0xAB, sizeof(caps));
        caps.struct_size = (uint32_t)sizeof(caps);
        CHECK_OK(tbcclCommGetCapabilities(comm, &caps));
        CHECK(caps.struct_size == sizeof(caps) && (caps.memory_kind_mask & 1u) != 0);
        CHECK(caps.reserved[0] == 0 && caps.reserved[1] == 0 && caps.reserved[2] == 0);
        int32_t sup = -1;
        CHECK_OK(tbcclCommSupportsAllReduce(comm, (uint32_t)rank, TBCCL_MEMORY_HOST, TBCCL_FLOAT32, TBCCL_SUM, &sup));
        CHECK(sup == 1);
        CHECK_OK(tbcclCommSupportsAllReduce(comm, (uint32_t)rank, TBCCL_MEMORY_HOST, TBCCL_FLOAT32, TBCCL_PRODUCT, &sup));
        CHECK(sup == 0);
        CHECK_OK(tbcclCommSupportsAllReduce(comm, (uint32_t)rank, TBCCL_MEMORY_HOST, TBCCL_FLOAT16, TBCCL_SUM, &sup));
        CHECK(sup == (world <= 2 ? 1 : 0));
        CHECK_RESULT(tbcclCommSupportsAllReduce(comm, (uint32_t)world, TBCCL_MEMORY_HOST, TBCCL_FLOAT32, TBCCL_SUM, &sup), TBCCL_INVALID_ARGUMENT);
        CHECK_RESULT(tbcclCommSupportsAllReduce(comm, 0, 77, TBCCL_FLOAT32, TBCCL_SUM, &sup), TBCCL_INVALID_ARGUMENT);
    }

    /* an unsupported reduction fails the call (before anything is submitted) and the communicator stays usable */
    {
        float f[4] = {1, 1, 1, 1};
        tbcclBuffer b = host_buffer(f, sizeof(f));
        tbcclWork_t w = (tbcclWork_t)(uintptr_t)1;
        CHECK_RESULT(tbcclAllReduce(comm, &b, &b, 4, TBCCL_FLOAT32, TBCCL_PRODUCT, NULL, &w), TBCCL_UNSUPPORTED);
        CHECK(w == NULL);
        int32_t aborted = 1;
        CHECK_OK(tbcclCommIsAborted(comm, &aborted));
        CHECK(aborted == 0);
    }
}

static void test_worlds(void)
{
    for (int n = 1; n <= 4; ++n) {
        run_world(n, exchange_body, NULL, NULL);
        printf("[PASS] N=%d: bootstrap, barrier, P2P ring, broadcast (every root), all_gather, all_reduce, capabilities\n", n);
    }
}

/* ------------------------------------------------------------------------------------------------------------------------------------------ */
/* FP16 and BF16 SUM at N=2 (bit patterns; exact) and the N>2 rejection. */
static void lowprec_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)user;
    uint16_t h[64], bf[64];
    for (int i = 0; i < 64; ++i) {
        h[i] = (rank == 0) ? 0x3C00u : 0x4000u;  /* 1.0, 2.0 in binary16 */
        bf[i] = (rank == 0) ? 0x3F80u : 0x4000u; /* 1.0, 2.0 in bfloat16 */
    }
    tbcclBuffer hb = host_buffer(h, sizeof(h)), bb = host_buffer(bf, sizeof(bf));
    tbcclWork_t w = NULL;
    if (world == 2) {
        CHECK_OK(tbcclAllReduce(comm, &hb, &hb, 64, TBCCL_FLOAT16, TBCCL_SUM, NULL, &w));
        CHECK(wait_result(w) == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(w));
        CHECK_OK(tbcclAllReduce(comm, &bb, &bb, 64, TBCCL_BFLOAT16, TBCCL_SUM, NULL, &w));
        CHECK(wait_result(w) == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(w));
        for (int i = 0; i < 64; ++i) {
            CHECK(h[i] == 0x4200u);  /* 3.0 in binary16 */
            CHECK(bf[i] == 0x4040u); /* 3.0 in bfloat16 */
        }
    } else {
        CHECK_RESULT(tbcclAllReduce(comm, &hb, &hb, 64, TBCCL_FLOAT16, TBCCL_SUM, NULL, &w), TBCCL_UNSUPPORTED);
        CHECK_RESULT(tbcclAllReduce(comm, &bb, &bb, 64, TBCCL_BFLOAT16, TBCCL_SUM, NULL, &w), TBCCL_UNSUPPORTED);
        CHECK(w == NULL);
    }
}

static void test_lowprec(void)
{
    run_world(2, lowprec_body, NULL, NULL);
    run_world(3, lowprec_body, NULL, NULL);
    printf("[PASS] FP16 and BF16 SUM exact at N=2; rejected as UNSUPPORTED at N=3\n");
}

/* ------------------------------------------------------------------------------------------------------------------------------------------ */
/* Opaque quantized payloads: nothing is interpreted, everything is bit exact (FP8 bytes, packed INT4, scale words, zero points, odd sizes). */
static void fill_pattern(uint8_t *p, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)((i * 131u + seed * 17u) ^ (i >> 5));
}

static void opaque_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)user;
    static const size_t sizes[] = {1, 2, 3, 7, 255, 256, 257, 4097, 65537};
    for (size_t s = 0; s < sizeof(sizes) / sizeof(sizes[0]); ++s) {
        const size_t n = sizes[s];
        uint8_t *a = (uint8_t *)malloc(n), *b = (uint8_t *)calloc(n, 1), *all = (uint8_t *)calloc(n * (size_t)world, 1), *want = (uint8_t *)malloc(n);
        CHECK(a && b && all && want);
        const int next = (rank + 1) % world, prev = (rank + world - 1) % world;
        /* send/recv ring */
        fill_pattern(a, n, (unsigned)rank);
        tbcclBuffer ab = host_buffer(a, n), bb = host_buffer(b, n);
        tbcclWork_t ws = NULL, wr = NULL;
        if (world > 1) {
            CHECK_OK(tbcclSend(comm, &ab, (uint32_t)next, NULL, &ws));
            CHECK_OK(tbcclRecv(comm, &bb, (uint32_t)prev, NULL, &wr));
            CHECK(wait_result(ws) == TBCCL_SUCCESS && wait_result(wr) == TBCCL_SUCCESS);
            fill_pattern(want, n, (unsigned)prev);
            CHECK(memcmp(b, want, n) == 0);
            CHECK_OK(tbcclWorkDestroy(ws));
            CHECK_OK(tbcclWorkDestroy(wr));
        }
        /* broadcast from the last rank */
        const int root = world - 1;
        if (rank == root) fill_pattern(b, n, 99u); else memset(b, 0, n);
        CHECK_OK(tbcclBroadcast(comm, &bb, (uint32_t)root, NULL, &ws));
        CHECK(wait_result(ws) == TBCCL_SUCCESS);
        fill_pattern(want, n, 99u);
        CHECK(memcmp(b, want, n) == 0);
        CHECK_OK(tbcclWorkDestroy(ws));
        /* all_gather */
        tbcclBuffer gb = host_buffer(all, n * (size_t)world);
        fill_pattern(a, n, (unsigned)rank + 5u);
        CHECK_OK(tbcclAllGather(comm, &ab, &gb, NULL, &ws));
        CHECK(wait_result(ws) == TBCCL_SUCCESS);
        for (int q = 0; q < world; ++q) {
            fill_pattern(want, n, (unsigned)q + 5u);
            CHECK(memcmp(all + (size_t)q * n, want, n) == 0);
        }
        CHECK_OK(tbcclWorkDestroy(ws));
        free(a); free(b); free(all); free(want);
    }
    /* every byte value 0..255 (FP8 / INT4 payloads can be any byte) */
    {
        uint8_t src[256], dst[256];
        for (int i = 0; i < 256; ++i) src[i] = (uint8_t)i;
        memset(dst, 0, sizeof(dst));
        tbcclBuffer sb = host_buffer(src, 256), db = host_buffer(dst, 256);
        tbcclWork_t w = NULL;
        if (rank == 0) memcpy(dst, src, 256);
        CHECK_OK(tbcclBroadcast(comm, &db, 0, NULL, &w));
        CHECK(wait_result(w) == TBCCL_SUCCESS);
        CHECK(memcmp(dst, src, 256) == 0);
        CHECK_OK(tbcclWorkDestroy(w));
        (void)sb;
    }
}

static void test_opaque_payloads(void)
{
    for (int n = 1; n <= 4; ++n) run_world(n, opaque_body, NULL, NULL);
    printf("[PASS] opaque bytes (FP8/INT4/scale words/odd sizes/all 256 byte values): send/recv, broadcast and all_gather are bit exact at N=1..4\n");
}

/* ------------------------------------------------------------------------------------------------------------------------------------------ */
/* Zero-byte semantics: a legal asynchronous no-op, data may be NULL. */
static void zero_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)user;
    if (world < 2) return;
    tbcclBuffer z = host_buffer(NULL, 0);
    tbcclWork_t s = NULL, r = NULL;
    const uint32_t peer = (uint32_t)(1 - rank);
    CHECK_OK(tbcclSend(comm, &z, peer, NULL, &s));
    CHECK_OK(tbcclRecv(comm, &z, peer, NULL, &r));
    CHECK(wait_result(s) == TBCCL_SUCCESS && wait_result(r) == TBCCL_SUCCESS);
    CHECK_OK(tbcclWorkDestroy(s));
    CHECK_OK(tbcclWorkDestroy(r));
    /* a non-empty message right behind it still arrives intact */
    uint8_t a[8] = {1, 2, 3, 4, 5, 6, 7, (uint8_t)(rank + 8)}, b[8] = {0};
    tbcclBuffer ab = host_buffer(a, 8), bb = host_buffer(b, 8);
    CHECK_OK(tbcclSend(comm, &ab, peer, NULL, &s));
    CHECK_OK(tbcclRecv(comm, &bb, peer, NULL, &r));
    CHECK(wait_result(s) == TBCCL_SUCCESS && wait_result(r) == TBCCL_SUCCESS);
    CHECK(b[0] == 1 && b[7] == (uint8_t)((1 - rank) + 8));
    CHECK_OK(tbcclWorkDestroy(s));
    CHECK_OK(tbcclWorkDestroy(r));
}

static void test_zero_bytes(void)
{
    run_world(2, zero_body, NULL, NULL);
    printf("[PASS] zero-byte send/recv: legal asynchronous no-op, NULL data accepted, FIFO unaffected\n");
}

/* ------------------------------------------------------------------------------------------------------------------------------------------ */
/* Work semantics: Test / Wait / WaitFor never conflate the query status with the operation result; WaitFor expiry changes nothing; Destroy does not cancel. */
static void work_body(int rank, int world, tbcclComm_t comm, void *user)
{
    Shared *sh = (Shared *)user;
    CHECK(world == 2);
    uint8_t buf[64] = {0}, payload[64];
    for (int i = 0; i < 64; ++i) payload[i] = (uint8_t)(i + 1);
    tbcclBuffer bb = host_buffer(buf, sizeof(buf)), pb = host_buffer(payload, sizeof(payload));
    tbcclWork_t w = NULL;
    int32_t done = -1;
    tbcclResult_t op = -1;
    if (rank == 1) {
        /* nobody has sent yet: a receive stays pending */
        CHECK_OK(tbcclRecv(comm, &bb, 0, NULL, &w));
        CHECK_OK(tbcclWorkTest(w, &done, &op));
        CHECK(done == 0 && op == TBCCL_SUCCESS);
        for (int i = 0; i < 3; ++i) { /* repeated timed waits: done=0 every time, the Work is untouched, no TIMEOUT result anywhere */
            done = -1; op = -1;
            CHECK_OK(tbcclWorkWaitFor(w, 20, &done, &op));
            CHECK(done == 0 && op == TBCCL_SUCCESS);
        }
        barrier_wait(&sh->gate); /* rank 0 sends only now */
        CHECK_OK(tbcclWorkWaitFor(w, 10000, &done, &op));
        CHECK(done == 1 && op == TBCCL_SUCCESS);
        for (int i = 0; i < 64; ++i) CHECK(buf[i] == (uint8_t)(i + 1));
        /* repeated queries after completion are stable */
        for (int i = 0; i < 3; ++i) {
            done = -1; op = -1;
            CHECK_OK(tbcclWorkTest(w, &done, &op));
            CHECK(done == 1 && op == TBCCL_SUCCESS);
            op = -1;
            CHECK_OK(tbcclWorkWait(w, &op));
            CHECK(op == TBCCL_SUCCESS);
        }
        /* an empty error string for a successful Work */
        char text[16] = "xx";
        size_t need = 0;
        CHECK_OK(tbcclWorkGetErrorString(w, text, sizeof(text), &need));
        CHECK(need == 1 && text[0] == '\0');
        CHECK_OK(tbcclWorkDestroy(w));
    } else {
        barrier_wait(&sh->gate);
        CHECK_OK(tbcclSend(comm, &pb, 1, NULL, &w));
        /* destroy the handle immediately: the send continues (the payload buffer stays alive until the peer has it) */
        CHECK_OK(tbcclWorkDestroy(w));
    }
    barrier_wait(&sh->gate);

    /* a pending receive, then abort: the QUERY succeeds, the operation result is ABORTED, the text is available */
    if (rank == 0) {
        CHECK_OK(tbcclRecv(comm, &bb, 1, NULL, &w));
        CHECK_OK(tbcclWorkWaitFor(w, 30, &done, &op));
        CHECK(done == 0);
        CHECK_OK(tbcclCommAbort(comm, "work-test abort"));
        CHECK_OK(tbcclWorkWait(w, &op));
        CHECK(op == TBCCL_ABORTED);
        done = -1;
        op = -1;
        CHECK_OK(tbcclWorkWaitFor(w, 0, &done, &op));
        CHECK(done == 1 && op == TBCCL_ABORTED);
        size_t need = 0;
        char small[8];
        CHECK_OK(tbcclWorkGetErrorString(w, NULL, 0, &need)); /* sizing call */
        CHECK(need > 8);
        CHECK_OK(tbcclWorkGetErrorString(w, small, sizeof(small), &need)); /* truncation is safe and NUL-terminated */
        CHECK(strlen(small) == 7);
        char *full = (char *)malloc(need);
        size_t need2 = 0;
        CHECK_OK(tbcclWorkGetErrorString(w, full, need, &need2));
        CHECK(need2 == need && strlen(full) == need - 1);
        free(full);
        CHECK_OK(tbcclWorkDestroy(w));
        int32_t aborted = 0;
        CHECK_OK(tbcclCommIsAborted(comm, &aborted));
        CHECK(aborted == 1);
        char reason[64];
        CHECK_OK(tbcclCommGetAbortReason(comm, reason, sizeof(reason), &need));
        CHECK(strstr(reason, "work-test abort") != NULL);
        /* new calls on an aborted communicator report ABORTED */
        tbcclWork_t w2 = (tbcclWork_t)(uintptr_t)1;
        CHECK_RESULT(tbcclBarrier(comm, &w2), TBCCL_ABORTED);
        CHECK(w2 == NULL);
        CHECK_RESULT(tbcclSend(comm, &pb, 1, NULL, &w2), TBCCL_ABORTED);
    }
    barrier_wait(&sh->gate);
}

static void test_work_semantics(void)
{
    Shared sh;
    barrier_init(&sh.gate, 2);
    run_world(2, work_body, &sh, NULL);
    printf("[PASS] Work: Test/Wait/WaitFor separate query status from operation result; WaitFor expiry changes nothing; stable repeats; Destroy does not cancel; text truncation safe; abort -> ABORTED\n");
}

/* ------------------------------------------------------------------------------------------------------------------------------------------ */
/* Argument validation: NULL pointers, bad struct_size, invalid enums, ranges, overflow, small buffers, reserved fields. */
static void validation_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)user;
    (void)rank;
    CHECK(world == 3);
    uint8_t buf[64];
    float fbuf[16];
    tbcclBuffer b = host_buffer(buf, sizeof(buf)), fb = host_buffer(fbuf, sizeof(fbuf));
    tbcclWork_t w = (tbcclWork_t)(uintptr_t)1;
    const uint32_t self = (uint32_t)rank;

    CHECK_RESULT(tbcclSend(NULL, &b, 1, NULL, &w), TBCCL_INVALID_ARGUMENT);
    CHECK(w == NULL);
    CHECK_RESULT(tbcclSend(comm, NULL, 1, NULL, &w), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclSend(comm, &b, 1, NULL, NULL), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclSend(comm, &b, self, NULL, &w), TBCCL_INVALID_ARGUMENT);   /* peer == self */
    CHECK_RESULT(tbcclRecv(comm, &b, 3, NULL, &w), TBCCL_INVALID_ARGUMENT);      /* peer out of range */
    CHECK_RESULT(tbcclRecv(comm, &b, 0xFFFFFFFFu, NULL, &w), TBCCL_INVALID_ARGUMENT);

    tbcclBuffer bad = b;
    bad.struct_size = 8; /* below the minimum prefix */
    CHECK_RESULT(tbcclSend(comm, &bad, (self + 1) % 3, NULL, &w), TBCCL_INVALID_ARGUMENT);
    bad = b;
    bad.memory_kind = 42; /* invalid enum value */
    CHECK_RESULT(tbcclSend(comm, &bad, (self + 1) % 3, NULL, &w), TBCCL_INVALID_ARGUMENT);
    bad = b;
    bad.reserved0 = 1; /* reserved input must be zero */
    CHECK_RESULT(tbcclSend(comm, &bad, (self + 1) % 3, NULL, &w), TBCCL_INVALID_ARGUMENT);
    bad = b;
    bad.reserved1[1] = 1;
    CHECK_RESULT(tbcclSend(comm, &bad, (self + 1) % 3, NULL, &w), TBCCL_INVALID_ARGUMENT);
    bad = b;
    bad.data = NULL; /* non-empty buffer without data */
    CHECK_RESULT(tbcclSend(comm, &bad, (self + 1) % 3, NULL, &w), TBCCL_INVALID_ARGUMENT);
    bad = b;
    bad.memory_kind = TBCCL_MEMORY_CUDA; /* a registered kind is required: unsupported without registration, or a device error path; never a crash */
    {
        const tbcclResult_t r = tbcclSend(comm, &bad, (self + 1) % 3, NULL, &w);
        CHECK(r == TBCCL_UNSUPPORTED || r == TBCCL_INVALID_ARGUMENT);
    }

    tbcclExecContext ec;
    memset(&ec, 0, sizeof(ec));
    ec.struct_size = sizeof(ec);
    ec.kind = 77;
    CHECK_RESULT(tbcclSend(comm, &b, (self + 1) % 3, &ec, &w), TBCCL_INVALID_ARGUMENT);
    ec.kind = TBCCL_EXEC_DEFAULT;
    ec.reserved[0] = 5;
    CHECK_RESULT(tbcclSend(comm, &b, (self + 1) % 3, &ec, &w), TBCCL_INVALID_ARGUMENT);
    ec.reserved[0] = 0;
    ec.struct_size = 4;
    CHECK_RESULT(tbcclSend(comm, &b, (self + 1) % 3, &ec, &w), TBCCL_INVALID_ARGUMENT);

    CHECK_RESULT(tbcclBroadcast(comm, &b, 3, NULL, &w), TBCCL_INVALID_ARGUMENT); /* root out of range */
    CHECK_RESULT(tbcclBroadcast(comm, NULL, 0, NULL, &w), TBCCL_INVALID_ARGUMENT);

    uint8_t big[64 * 3];
    tbcclBuffer gb = host_buffer(big, sizeof(big)), gsmall = host_buffer(big, 64 * 2);
    CHECK_RESULT(tbcclAllGather(comm, &b, &gsmall, NULL, &w), TBCCL_INVALID_ARGUMENT); /* recv must be send_bytes * world */
    tbcclBuffer huge = host_buffer(buf, (size_t)-1 / 2 + 2);
    CHECK_RESULT(tbcclAllGather(comm, &huge, &gb, NULL, &w), TBCCL_INVALID_ARGUMENT);  /* send_bytes * world overflows or mismatches */

    CHECK_RESULT(tbcclAllReduce(comm, &fb, &fb, 17, TBCCL_FLOAT32, TBCCL_SUM, NULL, &w), TBCCL_INVALID_ARGUMENT);          /* buffer too small */
    CHECK_RESULT(tbcclAllReduce(comm, &fb, &fb, (uint64_t)-1, TBCCL_FLOAT64, TBCCL_SUM, NULL, &w), TBCCL_INVALID_ARGUMENT); /* count*size overflows */
    CHECK_RESULT(tbcclAllReduce(comm, &fb, &fb, 16, 99, TBCCL_SUM, NULL, &w), TBCCL_INVALID_ARGUMENT);                      /* invalid dtype */
    CHECK_RESULT(tbcclAllReduce(comm, &fb, &fb, 16, TBCCL_FLOAT32, 99, NULL, &w), TBCCL_INVALID_ARGUMENT);                  /* invalid op */
    CHECK_RESULT(tbcclAllReduce(comm, &fb, &fb, 16, TBCCL_FLOAT16, TBCCL_SUM, NULL, &w), TBCCL_UNSUPPORTED);                /* N>2 FP16 */
    CHECK_RESULT(tbcclAllReduce(comm, &fb, &fb, 16, TBCCL_BFLOAT16, TBCCL_SUM, NULL, &w), TBCCL_UNSUPPORTED);
    CHECK_RESULT(tbcclBarrier(NULL, &w), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclBarrier(comm, NULL), TBCCL_INVALID_ARGUMENT);

    /* queries and Work functions on NULL */
    int32_t done;
    tbcclResult_t op;
    CHECK_RESULT(tbcclWorkTest(NULL, &done, &op), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclWorkWait(NULL, &op), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclWorkWaitFor(NULL, 1, &done, &op), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclWorkGetErrorString(NULL, NULL, 0, NULL), TBCCL_INVALID_ARGUMENT);
    CHECK_OK(tbcclWorkDestroy(NULL));
    CHECK_RESULT(tbcclCommGetRank(NULL, (uint32_t[1]){0}), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclCommGetRank(comm, NULL), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclCommAbort(NULL, "x"), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclCommIsAborted(comm, NULL), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclCommGetAbortReason(comm, NULL, 4, NULL), TBCCL_INVALID_ARGUMENT);
    tbcclCapabilities caps;
    memset(&caps, 0, sizeof(caps));
    caps.struct_size = 8; /* below the minimum prefix */
    CHECK_RESULT(tbcclCommGetCapabilities(comm, &caps), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclCommGetCapabilities(comm, NULL), TBCCL_INVALID_ARGUMENT);

    /* the communicator survived every rejected call */
    int32_t aborted = 1;
    CHECK_OK(tbcclCommIsAborted(comm, &aborted));
    CHECK(aborted == 0);
    CHECK_OK(tbcclBarrier(comm, &w));
    CHECK(wait_result(w) == TBCCL_SUCCESS);
    CHECK_OK(tbcclWorkDestroy(w));
}

static void test_validation(void)
{
    run_world(3, validation_body, NULL, NULL);
    printf("[PASS] argument validation: NULL pointers, bad struct_size, invalid enums, reserved fields, peer/root ranges, overflow, small buffers, N>2 FP16/BF16; no rejected call poisons the communicator\n");
}

/* ------------------------------------------------------------------------------------------------------------------------------------------ */
/* Bootstrap failures: malformed blob, wrong format version, wrong communicator id, duplicate rank, wrong blob count, double Complete, destroy states. */
static void test_bootstrap_failures(void)
{
    tbcclUniqueId id, other;
    CHECK_OK(tbcclGetUniqueId(&id));
    CHECK_OK(tbcclGetUniqueId(&other));
    CHECK(memcmp(id.bytes, other.bytes, 16) != 0);

    tbcclBootstrap_t bs[2] = {NULL, NULL};
    tbcclBootstrap_t foreign = NULL;
    CHECK_RESULT(tbcclBootstrapBegin(2, 2, &id, NULL, &bs[0]), TBCCL_INVALID_ARGUMENT); /* rank out of range */
    CHECK(bs[0] == NULL);
    CHECK_RESULT(tbcclBootstrapBegin(0, 0, &id, NULL, &bs[0]), TBCCL_INVALID_ARGUMENT); /* world_size 0 */
    CHECK_RESULT(tbcclBootstrapBegin(0, 9, &id, NULL, &bs[0]), TBCCL_INVALID_ARGUMENT); /* world_size above the mesh limit */
    CHECK_RESULT(tbcclBootstrapBegin(0, 2, NULL, NULL, &bs[0]), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclBootstrapBegin(0, 2, &id, NULL, NULL), TBCCL_INVALID_ARGUMENT);

    tbcclBootstrapOptions opt;
    memset(&opt, 0, sizeof(opt));
    opt.struct_size = sizeof(opt);
    opt.bind_host = "0.0.0.0"; /* a wildcard bind with no advertised address would serialize an unusable endpoint */
    CHECK_RESULT(tbcclBootstrapBegin(0, 2, &id, &opt, &bs[0]), TBCCL_INVALID_ARGUMENT);
    opt.advertise_host = "0.0.0.0";
    CHECK_RESULT(tbcclBootstrapBegin(0, 2, &id, &opt, &bs[0]), TBCCL_INVALID_ARGUMENT);
    opt.advertise_host = "not-an-address";
    CHECK_RESULT(tbcclBootstrapBegin(0, 2, &id, &opt, &bs[0]), TBCCL_INVALID_ARGUMENT);
    opt.advertise_host = "127.0.0.1"; /* bind to the wildcard, advertise loopback: fine */
    CHECK_OK(tbcclBootstrapBegin(0, 2, &id, &opt, &bs[0]));
    tbcclBootstrapDestroy(bs[0]);
    bs[0] = NULL;
    memset(&opt, 0, sizeof(opt));
    opt.struct_size = 8; /* below the minimum prefix */
    CHECK_RESULT(tbcclBootstrapBegin(0, 2, &id, &opt, &bs[0]), TBCCL_INVALID_ARGUMENT);
    opt.struct_size = sizeof(opt);
    opt.reserved1 = 3;
    CHECK_RESULT(tbcclBootstrapBegin(0, 2, &id, &opt, &bs[0]), TBCCL_INVALID_ARGUMENT);

    CHECK_OK(tbcclBootstrapBegin(0, 2, &id, NULL, &bs[0]));
    CHECK_OK(tbcclBootstrapBegin(1, 2, &id, NULL, &bs[1]));
    CHECK_OK(tbcclBootstrapBegin(1, 2, &other, NULL, &foreign));
    tbcclEndpointBlob blobs[2], mutated[2];
    memset(blobs, 0, sizeof(blobs));
    blobs[0].struct_size = blobs[1].struct_size = (uint32_t)sizeof(tbcclEndpointBlob);
    CHECK_OK(tbcclBootstrapGetEndpoint(bs[0], &blobs[0]));
    CHECK_OK(tbcclBootstrapGetEndpoint(bs[1], &blobs[1]));
    CHECK(blobs[0].format_version == 1 && blobs[0].struct_size == TBCCL_ENDPOINT_BLOB_SIZE && blobs[0].used_bytes > 0 && blobs[0].used_bytes <= sizeof(blobs[0].payload));
    tbcclEndpointBlob small;
    memset(&small, 0, sizeof(small));
    small.struct_size = 16;
    CHECK_RESULT(tbcclBootstrapGetEndpoint(bs[0], &small), TBCCL_INVALID_ARGUMENT); /* caller's blob smaller than v1 */

    tbcclComm_t comm = (tbcclComm_t)(uintptr_t)1;
    CHECK_RESULT(tbcclBootstrapComplete(bs[0], blobs, 1, &comm), TBCCL_INVALID_ARGUMENT); /* wrong blob count */
    CHECK(comm == NULL);
    memcpy(mutated, blobs, sizeof(blobs));
    mutated[1].payload[0] ^= 0xFF; /* malformed: bad magic */
    CHECK_RESULT(tbcclBootstrapComplete(bs[0], mutated, 2, &comm), TBCCL_INVALID_ARGUMENT);
    memcpy(mutated, blobs, sizeof(blobs));
    mutated[1].format_version = 2; /* another blob format version */
    CHECK_RESULT(tbcclBootstrapComplete(bs[0], mutated, 2, &comm), TBCCL_INVALID_ARGUMENT);
    memcpy(mutated, blobs, sizeof(blobs));
    mutated[1] = blobs[0]; /* duplicate rank 0 */
    CHECK_RESULT(tbcclBootstrapComplete(bs[0], mutated, 2, &comm), TBCCL_INVALID_ARGUMENT);
    memcpy(mutated, blobs, sizeof(blobs));
    mutated[0] = blobs[1];
    mutated[1] = blobs[0]; /* out of rank order */
    CHECK_RESULT(tbcclBootstrapComplete(bs[0], mutated, 2, &comm), TBCCL_INVALID_ARGUMENT);
    {
        tbcclEndpointBlob foreign_blob;
        memset(&foreign_blob, 0, sizeof(foreign_blob));
        foreign_blob.struct_size = (uint32_t)sizeof(foreign_blob);
        CHECK_OK(tbcclBootstrapGetEndpoint(foreign, &foreign_blob));
        memcpy(mutated, blobs, sizeof(blobs));
        mutated[1] = foreign_blob; /* a rank of a different communicator id */
        CHECK_RESULT(tbcclBootstrapComplete(bs[0], mutated, 2, &comm), TBCCL_INVALID_ARGUMENT);
    }
    CHECK_RESULT(tbcclBootstrapComplete(bs[0], NULL, 2, &comm), TBCCL_INVALID_ARGUMENT);
    CHECK_RESULT(tbcclBootstrapComplete(NULL, blobs, 2, &comm), TBCCL_INVALID_ARGUMENT);

    /* every failure above left bs[0] intact; destroy is safe in the incomplete state and for NULL */
    CHECK_OK(tbcclBootstrapDestroy(bs[0]));
    CHECK_OK(tbcclBootstrapDestroy(bs[1]));
    CHECK_OK(tbcclBootstrapDestroy(foreign));
    CHECK_OK(tbcclBootstrapDestroy(NULL));
    CHECK_OK(tbcclCommDestroy(NULL));
    CHECK_OK(tbcclWorkDestroy(NULL));
    printf("[PASS] bootstrap negatives: ranges, wildcard advertise, struct_size, reserved fields, blob count, malformed blob, wrong format version, duplicate/out-of-order rank, wrong communicator id; Destroy safe incomplete and for NULL\n");
}

typedef struct {
    int done_twice;
} Twice;

static void complete_twice_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)rank; (void)world; (void)comm; (void)user;
}

static void test_double_complete(void)
{
    /* a second Complete on a completed bootstrap is invalid; run by hand with one rank */
    tbcclUniqueId id;
    CHECK_OK(tbcclGetUniqueId(&id));
    tbcclBootstrap_t bs = NULL;
    CHECK_OK(tbcclBootstrapBegin(0, 1, &id, NULL, &bs));
    tbcclEndpointBlob blob;
    memset(&blob, 0, sizeof(blob));
    blob.struct_size = sizeof(blob);
    CHECK_OK(tbcclBootstrapGetEndpoint(bs, &blob));
    tbcclComm_t comm = NULL;
    CHECK_OK(tbcclBootstrapComplete(bs, &blob, 1, &comm));
    CHECK(comm != NULL);
    tbcclComm_t again = (tbcclComm_t)(uintptr_t)1;
    CHECK_RESULT(tbcclBootstrapComplete(bs, &blob, 1, &again), TBCCL_INVALID_ARGUMENT);
    CHECK(again == NULL);
    CHECK_OK(tbcclBootstrapDestroy(bs)); /* completed state */
    CHECK_OK(tbcclCommDestroy(comm));
    complete_twice_body(0, 1, NULL, NULL);
    printf("[PASS] a second Complete is invalid; Destroy is safe in the completed state\n");
}

/* Two communicators coexist in one process (no global rendezvous state). */
static void coexist_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)rank; (void)world;
    tbcclComm_t *other = (tbcclComm_t *)user; /* unused slot: each world has its own handles */
    (void)other;
    tbcclWork_t w = NULL;
    CHECK_OK(tbcclBarrier(comm, &w));
    CHECK(wait_result(w) == TBCCL_SUCCESS);
    CHECK_OK(tbcclWorkDestroy(w));
}

typedef struct {
    pthread_t t;
    int world;
} Runner;

static void *runner(void *p)
{
    Runner *r = (Runner *)p;
    for (int i = 0; i < 5; ++i) run_world(r->world, coexist_body, NULL, NULL);
    return NULL;
}

static void test_coexisting_communicators(void)
{
    Runner a = {.world = 2}, b = {.world = 3};
    CHECK(pthread_create(&a.t, NULL, runner, &a) == 0);
    CHECK(pthread_create(&b.t, NULL, runner, &b) == 0);
    pthread_join(a.t, NULL);
    pthread_join(b.t, NULL);
    printf("[PASS] independent worlds bootstrap concurrently in one process (no global rendezvous state)\n");
}

/* ------------------------------------------------------------------------------------------------------------------------------------------ */
int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    test_version_and_strings();
    test_worlds();
    test_lowprec();
    test_opaque_payloads();
    test_zero_bytes();
    test_work_semantics();
    test_validation();
    test_bootstrap_failures();
    test_double_complete();
    test_coexisting_communicators();
    printf("All C API functional tests passed.\n");
    return 0;
}
