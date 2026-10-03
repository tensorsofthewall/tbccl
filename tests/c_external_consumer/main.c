/* Phase 52: pure-C consumer of the INSTALLED C ABI (include/tbccl/tbccl.h + TBCCL::tbccl_c). Real processes: a parent generates the unique id, forks one child per rank,
 * and ferries the opaque endpoint blobs between them over pipes: a test-only stand-in for "the application all-gathers the blobs" (MPI, a launcher, files...).
 * libtbccl contains no bootstrap service. N = 1..4. Built with CONSUMER_WITH_CUDA it also exercises CUDA buffers (the CUDA runtime is included HERE; tbccl.h is CUDA-free). */

#define _POSIX_C_SOURCE 200809L

#include <tbccl/tbccl.h>

#ifdef CONSUMER_WITH_CUDA
#include <cuda_runtime_api.h>
#endif

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#define CHECK(cond)                                                                                         \
    do {                                                                                                    \
        if (!(cond)) {                                                                                      \
            fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, #cond);                               \
            _exit(1);                                                                                       \
        }                                                                                                   \
    } while (0)

#define CHECK_OK(call)                                                                                      \
    do {                                                                                                    \
        tbcclResult_t r_ = (call);                                                                          \
        if (r_ != TBCCL_SUCCESS) {                                                                          \
            fprintf(stderr, "[FAIL] %s:%d: %s -> %d (%s)\n", __FILE__, __LINE__, #call, (int)r_, tbcclGetResultString(r_)); \
            _exit(1);                                                                                       \
        }                                                                                                   \
    } while (0)

static void write_all(int fd, const void *p, size_t n)
{
    const char *c = (const char *)p;
    while (n > 0) {
        ssize_t w = write(fd, c, n);
        if (w < 0 && errno == EINTR) continue;
        CHECK(w > 0);
        c += w;
        n -= (size_t)w;
    }
}

static void read_all(int fd, void *p, size_t n)
{
    char *c = (char *)p;
    while (n > 0) {
        ssize_t r = read(fd, c, n);
        if (r < 0 && errno == EINTR) continue;
        CHECK(r > 0);
        c += r;
        n -= (size_t)r;
    }
}

static tbcclBuffer host_buf(void *p, size_t n)
{
    tbcclBuffer b;
    memset(&b, 0, sizeof(b));
    b.struct_size = (uint32_t)sizeof(b);
    b.memory_kind = TBCCL_MEMORY_HOST;
    b.device_ordinal = -1;
    b.data = p;
    b.bytes = n;
    return b;
}

static tbcclResult_t wait_op(tbcclWork_t w)
{
    tbcclResult_t op = TBCCL_INTERNAL_ERROR;
    CHECK_OK(tbcclWorkWait(w, &op));
    if (op != TBCCL_SUCCESS) {
        char text[256];
        size_t need = 0;
        if (tbcclWorkGetErrorString(w, text, sizeof(text), &need) == TBCCL_SUCCESS)
            fprintf(stderr, "[work failed] result %d (%s): %s\n", (int)op, tbcclGetResultString(op), text);
    }
    return op;
}

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* ---- the per-rank program -------------------------------------------------------------------------------------------------------------------- */
static void host_exercise(tbcclComm_t comm, int rank, int world, int to_parent, int from_parent)
{
    uint32_t r = 99, n = 99;
    CHECK_OK(tbcclCommGetRank(comm, &r));
    CHECK_OK(tbcclCommGetSize(comm, &n));
    CHECK((int)r == rank && (int)n == world);

    tbcclWork_t w = NULL;
    CHECK_OK(tbcclBarrier(comm, &w));
    CHECK(wait_op(w) == TBCCL_SUCCESS);
    CHECK_OK(tbcclWorkDestroy(w));

    if (world > 1) {
        uint8_t out[1000], in[1000];
        for (int i = 0; i < 1000; ++i) out[i] = (uint8_t)(rank * 29 + i);
        memset(in, 0, sizeof(in));
        const uint32_t next = (uint32_t)((rank + 1) % world), prev = (uint32_t)((rank + world - 1) % world);
        tbcclBuffer sb = host_buf(out, sizeof(out)), rb = host_buf(in, sizeof(in));
        tbcclWork_t s = NULL, rv = NULL;
        CHECK_OK(tbcclSend(comm, &sb, next, NULL, &s));
        CHECK_OK(tbcclRecv(comm, &rb, prev, NULL, &rv));
        /* poll with Test, then a timed wait, then Wait */
        int32_t done = 0;
        tbcclResult_t op = -1;
        for (int spin = 0; spin < 100000 && !done; ++spin) CHECK_OK(tbcclWorkTest(rv, &done, &op));
        CHECK_OK(tbcclWorkWaitFor(s, 10000, &done, &op));
        CHECK(done == 1 && op == TBCCL_SUCCESS);
        CHECK(wait_op(rv) == TBCCL_SUCCESS);
        for (int i = 0; i < 1000; ++i) CHECK(in[i] == (uint8_t)((int)prev * 29 + i));
        CHECK_OK(tbcclWorkDestroy(s));
        CHECK_OK(tbcclWorkDestroy(rv));
    }

    for (int root = 0; root < world; ++root) {
        uint8_t buf[257];
        for (int i = 0; i < 257; ++i) buf[i] = (rank == root) ? (uint8_t)(root * 61 + i) : 0;
        tbcclBuffer b = host_buf(buf, sizeof(buf));
        CHECK_OK(tbcclBroadcast(comm, &b, (uint32_t)root, NULL, &w));
        CHECK(wait_op(w) == TBCCL_SUCCESS);
        for (int i = 0; i < 257; ++i) CHECK(buf[i] == (uint8_t)(root * 61 + i));
        CHECK_OK(tbcclWorkDestroy(w));
    }

    {
        uint8_t mine[100], all[400];
        for (int i = 0; i < 100; ++i) mine[i] = (uint8_t)(rank * 13 + i);
        tbcclBuffer s = host_buf(mine, sizeof(mine)), g = host_buf(all, (size_t)100 * (size_t)world);
        CHECK_OK(tbcclAllGather(comm, &s, &g, NULL, &w));
        CHECK(wait_op(w) == TBCCL_SUCCESS);
        for (int q = 0; q < world; ++q)
            for (int i = 0; i < 100; ++i) CHECK(all[q * 100 + i] == (uint8_t)(q * 13 + i));
        CHECK_OK(tbcclWorkDestroy(w));
    }

    {
        static float f[4096];
        for (int i = 0; i < 4096; ++i) f[i] = (float)(rank + 1);
        tbcclBuffer b = host_buf(f, sizeof(f));
        CHECK_OK(tbcclAllReduce(comm, &b, &b, 4096, TBCCL_FLOAT32, TBCCL_SUM, NULL, &w));
        CHECK(wait_op(w) == TBCCL_SUCCESS);
        const float want = (float)(world * (world + 1) / 2);
        for (int i = 0; i < 4096; ++i) CHECK(f[i] == want);
        CHECK_OK(tbcclWorkDestroy(w));
    }

    if (world == 2) {
        uint16_t h[32], bf[32];
        for (int i = 0; i < 32; ++i) {
            h[i] = rank == 0 ? 0x3C00u : 0x4000u;  /* 1.0 / 2.0 binary16 */
            bf[i] = rank == 0 ? 0x3F80u : 0x4000u; /* 1.0 / 2.0 bfloat16 */
        }
        tbcclBuffer hb = host_buf(h, sizeof(h)), bb = host_buf(bf, sizeof(bf));
        CHECK_OK(tbcclAllReduce(comm, &hb, &hb, 32, TBCCL_FLOAT16, TBCCL_SUM, NULL, &w));
        CHECK(wait_op(w) == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(w));
        CHECK_OK(tbcclAllReduce(comm, &bb, &bb, 32, TBCCL_BFLOAT16, TBCCL_SUM, NULL, &w));
        CHECK(wait_op(w) == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(w));
        for (int i = 0; i < 32; ++i) CHECK(h[i] == 0x4200u && bf[i] == 0x4040u);
    } else if (world > 2) {
        uint16_t h[8] = {0};
        tbcclBuffer hb = host_buf(h, sizeof(h));
        CHECK(tbcclAllReduce(comm, &hb, &hb, 8, TBCCL_FLOAT16, TBCCL_SUM, NULL, &w) == TBCCL_UNSUPPORTED);
    }

    /* Process-level gate (N=2): rank 0 posts 100 large sends while rank 1 has posted NOTHING; every post must return, rank 0 reports POSTED, only then does the
     * parent release rank 1, which posts the 100 receives. A submission call that waited for the peer would never return and the parent would time out. */
    if (world == 2) {
        enum { COUNT = 100 };
        const size_t bytes = (size_t)4 << 20;
        uint8_t *buf = (uint8_t *)malloc(bytes);
        CHECK(buf != NULL);
        tbcclWork_t *works = (tbcclWork_t *)calloc(COUNT, sizeof(tbcclWork_t));
        CHECK(works != NULL);
        if (rank == 0) {
            memset(buf, 0xA5, bytes);
            tbcclBuffer b = host_buf(buf, bytes);
            const double t0 = now_seconds();
            for (int i = 0; i < COUNT; ++i) CHECK_OK(tbcclSend(comm, &b, 1, NULL, &works[i]));
            const double elapsed = now_seconds() - t0;
            CHECK(elapsed < 20.0);
            char posted = 'P';
            write_all(to_parent, &posted, 1); /* POSTING_COMPLETE */
            /* the peer has not released its receives yet: no send can have completed all 100 */
            int32_t done = 1;
            tbcclResult_t op = -1;
            CHECK_OK(tbcclWorkTest(works[COUNT - 1], &done, &op));
            CHECK(done == 0);
        } else {
            char go = 0;
            read_all(from_parent, &go, 1); /* released by the parent only after POSTING_COMPLETE */
            CHECK(go == 'G');
            memset(buf, 0, bytes);
            tbcclBuffer b = host_buf(buf, bytes);
            for (int i = 0; i < COUNT; ++i) CHECK_OK(tbcclRecv(comm, &b, 0, NULL, &works[i]));
        }
        for (int i = 0; i < COUNT; ++i) {
            CHECK(wait_op(works[i]) == TBCCL_SUCCESS);
            CHECK_OK(tbcclWorkDestroy(works[i]));
        }
        if (rank == 1) CHECK(buf[0] == 0xA5 && buf[bytes - 1] == 0xA5);
        free(works);
        free(buf);
    }

    /* abort, then everything reports ABORTED and the reason is queryable. An abort kills operations still in flight on the peers, so no rank aborts before the
     * parent has heard that EVERY rank finished its last barrier. */
    CHECK_OK(tbcclBarrier(comm, &w));
    CHECK(wait_op(w) == TBCCL_SUCCESS);
    CHECK_OK(tbcclWorkDestroy(w));
    {
        char done_mark = 'D', go_abort = 0;
        write_all(to_parent, &done_mark, 1);
        read_all(from_parent, &go_abort, 1);
        CHECK(go_abort == 'A');
    }
    CHECK_OK(tbcclCommAbort(comm, "consumer finished"));
    int32_t aborted = 0;
    CHECK_OK(tbcclCommIsAborted(comm, &aborted));
    CHECK(aborted == 1);
    char reason[64];
    size_t need = 0;
    CHECK_OK(tbcclCommGetAbortReason(comm, reason, sizeof(reason), &need));
    CHECK(strstr(reason, "consumer finished") != NULL);
    tbcclWork_t after = NULL;
    CHECK(tbcclBarrier(comm, &after) == TBCCL_ABORTED);
}

#ifdef CONSUMER_WITH_CUDA
#define CU(call)                                                                                            \
    do {                                                                                                    \
        cudaError_t e_ = (call);                                                                            \
        if (e_ != cudaSuccess) {                                                                            \
            fprintf(stderr, "[FAIL] %s:%d: %s -> %s\n", __FILE__, __LINE__, #call, cudaGetErrorString(e_)); \
            _exit(1);                                                                                       \
        }                                                                                                   \
    } while (0)

static tbcclBuffer cuda_buf(void *p, size_t n)
{
    tbcclBuffer b = host_buf(p, n);
    b.memory_kind = TBCCL_MEMORY_CUDA;
    b.device_ordinal = 0;
    return b;
}

static void cuda_exercise(tbcclComm_t comm, int rank, int world)
{
    tbcclWork_t w = NULL;
    uint32_t next = (uint32_t)((rank + 1) % world), prev = (uint32_t)((rank + world - 1) % world);

    /* CUDA buffer Send/Recv (default context) */
    enum { N = 1 << 20 };
    uint8_t *host = (uint8_t *)malloc(N);
    void *dev_out = NULL, *dev_in = NULL;
    CU(cudaMalloc(&dev_out, N));
    CU(cudaMalloc(&dev_in, N));
    for (int i = 0; i < N; ++i) host[i] = (uint8_t)(rank * 7 + i);
    CU(cudaMemcpy(dev_out, host, N, cudaMemcpyHostToDevice));
    CU(cudaMemset(dev_in, 0, N));
    if (world > 1) {
        tbcclBuffer sb = cuda_buf(dev_out, N), rb = cuda_buf(dev_in, N);
        tbcclWork_t s = NULL, rv = NULL;
        CHECK_OK(tbcclSend(comm, &sb, next, NULL, &s));
        CHECK_OK(tbcclRecv(comm, &rb, prev, NULL, &rv));
        CHECK(wait_op(s) == TBCCL_SUCCESS && wait_op(rv) == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(s));
        CHECK_OK(tbcclWorkDestroy(rv));
        CU(cudaMemcpy(host, dev_in, N, cudaMemcpyDeviceToHost));
        for (int i = 0; i < N; ++i) CHECK(host[i] == (uint8_t)((int)prev * 7 + i));
    }

    /* Float32 AllReduce on CUDA buffers */
    {
        enum { COUNT = 100000 };
        float *hf = (float *)malloc(COUNT * sizeof(float));
        void *df = NULL;
        CU(cudaMalloc(&df, COUNT * sizeof(float)));
        for (int i = 0; i < COUNT; ++i) hf[i] = (float)(rank + 1);
        CU(cudaMemcpy(df, hf, COUNT * sizeof(float), cudaMemcpyHostToDevice));
        tbcclBuffer b = cuda_buf(df, COUNT * sizeof(float));
        CHECK_OK(tbcclAllReduce(comm, &b, &b, COUNT, TBCCL_FLOAT32, TBCCL_SUM, NULL, &w));
        CHECK(wait_op(w) == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(w));
        CU(cudaMemcpy(hf, df, COUNT * sizeof(float), cudaMemcpyDeviceToHost));
        const float want = (float)(world * (world + 1) / 2);
        for (int i = 0; i < COUNT; ++i) CHECK(hf[i] == want);
        CU(cudaFree(df));
        free(hf);
    }

    /* the producer of the operand runs on a NON-DEFAULT, non-blocking stream that this program never synchronizes: the library must order itself after it */
    {
        cudaStream_t stream;
        CU(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        enum { COUNT = 65536 };
        float *hf = (float *)malloc(COUNT * sizeof(float));
        void *df = NULL;
        CU(cudaMalloc(&df, COUNT * sizeof(float)));
        for (int i = 0; i < COUNT; ++i) hf[i] = (float)(10 * (rank + 1));
        CU(cudaMemcpyAsync(df, hf, COUNT * sizeof(float), cudaMemcpyHostToDevice, stream)); /* pageable source: ordered on the stream */
        tbcclExecContext ctx;
        memset(&ctx, 0, sizeof(ctx));
        ctx.struct_size = (uint32_t)sizeof(ctx);
        ctx.kind = TBCCL_EXEC_CUDA_STREAM;
        ctx.native_handle = (void *)stream;
        tbcclBuffer b = cuda_buf(df, COUNT * sizeof(float));
        CHECK_OK(tbcclAllReduce(comm, &b, &b, COUNT, TBCCL_FLOAT32, TBCCL_SUM, &ctx, &w));
        CHECK(wait_op(w) == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(w));
        CU(cudaStreamSynchronize(stream));
        float *back = (float *)malloc(COUNT * sizeof(float));
        CU(cudaMemcpy(back, df, COUNT * sizeof(float), cudaMemcpyDeviceToHost));
        const float want = (float)(10 * (world * (world + 1) / 2));
        for (int i = 0; i < COUNT; ++i) CHECK(back[i] == want);
        free(back);
        CU(cudaFree(df));
        CU(cudaStreamDestroy(stream));
        free(hf);
    }

    /* N=2: FP16 and BF16 SUM on CUDA buffers (bit patterns, exact) */
    if (world == 2) {
        uint16_t h[256], bf[256];
        for (int i = 0; i < 256; ++i) {
            h[i] = rank == 0 ? 0x3C00u : 0x4000u;
            bf[i] = rank == 0 ? 0x3F80u : 0x4000u;
        }
        void *dh = NULL, *db = NULL;
        CU(cudaMalloc(&dh, sizeof(h)));
        CU(cudaMalloc(&db, sizeof(bf)));
        CU(cudaMemcpy(dh, h, sizeof(h), cudaMemcpyHostToDevice));
        CU(cudaMemcpy(db, bf, sizeof(bf), cudaMemcpyHostToDevice));
        tbcclBuffer hb = cuda_buf(dh, sizeof(h)), bb = cuda_buf(db, sizeof(bf));
        CHECK_OK(tbcclAllReduce(comm, &hb, &hb, 256, TBCCL_FLOAT16, TBCCL_SUM, NULL, &w));
        CHECK(wait_op(w) == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(w));
        CHECK_OK(tbcclAllReduce(comm, &bb, &bb, 256, TBCCL_BFLOAT16, TBCCL_SUM, NULL, &w));
        CHECK(wait_op(w) == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(w));
        CU(cudaMemcpy(h, dh, sizeof(h), cudaMemcpyDeviceToHost));
        CU(cudaMemcpy(bf, db, sizeof(bf), cudaMemcpyDeviceToHost));
        for (int i = 0; i < 256; ++i) CHECK(h[i] == 0x4200u && bf[i] == 0x4040u);
        CU(cudaFree(dh));
        CU(cudaFree(db));
    }
    CU(cudaFree(dev_out));
    CU(cudaFree(dev_in));
    free(host);
}
#endif

static int child_main(int rank, int world, const tbcclUniqueId *id, int to_parent, int from_parent)
{
#ifdef CONSUMER_WITH_CUDA
    CHECK_OK(tbcclRegisterCudaSupport());
    CHECK_OK(tbcclRegisterCudaSupport()); /* idempotent */
#else
    CHECK(tbcclRegisterCudaSupport() == TBCCL_UNSUPPORTED || tbcclRegisterCudaSupport() == TBCCL_SUCCESS);
#endif
    tbcclBootstrap_t bs = NULL;
    CHECK_OK(tbcclBootstrapBegin((uint32_t)rank, (uint32_t)world, id, NULL, &bs));
    tbcclEndpointBlob mine;
    memset(&mine, 0, sizeof(mine));
    mine.struct_size = (uint32_t)sizeof(mine);
    CHECK_OK(tbcclBootstrapGetEndpoint(bs, &mine));
    write_all(to_parent, &mine, sizeof(mine));
    tbcclEndpointBlob *all = (tbcclEndpointBlob *)malloc(sizeof(tbcclEndpointBlob) * (size_t)world);
    CHECK(all != NULL);
    read_all(from_parent, all, sizeof(tbcclEndpointBlob) * (size_t)world); /* the application's all-gather */
    tbcclComm_t comm = NULL;
    CHECK_OK(tbcclBootstrapComplete(bs, all, (uint32_t)world, &comm));
    CHECK_OK(tbcclBootstrapDestroy(bs));
    free(all);
#ifdef CONSUMER_WITH_CUDA
    cuda_exercise(comm, rank, world);
#endif
    host_exercise(comm, rank, world, to_parent, from_parent);
    CHECK_OK(tbcclCommDestroy(comm));
    return 0;
}

/* ---- the parent: id generation, blob ferry, process management ---------------------------------------------------------------------------------- */
static int run_world(int world)
{
    tbcclUniqueId id;
    CHECK_OK(tbcclGetUniqueId(&id));
    int up[4][2], down[4][2];
    pid_t pid[4];
    for (int r = 0; r < world; ++r) CHECK(pipe(up[r]) == 0 && pipe(down[r]) == 0);
    for (int r = 0; r < world; ++r) {
        pid[r] = fork();
        CHECK(pid[r] >= 0);
        if (pid[r] == 0) {
            for (int q = 0; q < world; ++q) {
                if (q != r) { close(up[q][0]); close(up[q][1]); close(down[q][0]); close(down[q][1]); }
            }
            close(up[r][0]);
            close(down[r][1]);
            _exit(child_main(r, world, &id, up[r][1], down[r][0]));
        }
        close(up[r][1]);
        close(down[r][0]);
    }
    tbcclEndpointBlob blobs[4];
    for (int r = 0; r < world; ++r) read_all(up[r][0], &blobs[r], sizeof(blobs[r]));
    for (int r = 0; r < world; ++r) write_all(down[r][1], blobs, sizeof(tbcclEndpointBlob) * (size_t)world);
    if (world == 2) {
        char posted = 0;
        read_all(up[0][0], &posted, 1); /* rank 0 finished posting 100 large sends while rank 1 posted nothing */
        CHECK(posted == 'P');
        char go = 'G';
        write_all(down[1][1], &go, 1); /* now release rank 1 */
    }
    for (int r = 0; r < world; ++r) {
        char done_mark = 0;
        read_all(up[r][0], &done_mark, 1);
        CHECK(done_mark == 'D');
    }
    for (int r = 0; r < world; ++r) {
        char go_abort = 'A';
        write_all(down[r][1], &go_abort, 1);
    }
    int failed = 0;
    for (int r = 0; r < world; ++r) {
        int status = 0;
        CHECK(waitpid(pid[r], &status, 0) == pid[r]);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "[FAIL] rank %d of %d exited abnormally (status %d)\n", r, world, status);
            failed = 1;
        }
        close(up[r][0]);
        close(down[r][1]);
    }
    return failed;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    uint32_t abi = 0, major = 0, minor = 0, patch = 0;
    CHECK_OK(tbcclGetAbiVersion(&abi));
    CHECK(abi == TBCCL_C_ABI_VERSION);
    CHECK_OK(tbcclGetPackageVersion(&major, &minor, &patch));
    printf("TBCCL C ABI %u, package %u.%u.%u\n", abi, major, minor, patch);
    int failed = 0;
    for (int world = 1; world <= 4; ++world) {
        const int rc = run_world(world);
        printf("[%s] N=%d: bootstrap through caller-exchanged blobs, barrier, P2P, broadcast, all_gather, all_reduce, FP16/BF16 (N=2), Work test/wait/wait_for, abort%s\n",
               rc ? "FAIL" : "PASS", world,
#ifdef CONSUMER_WITH_CUDA
               ", CUDA buffers and a non-default stream"
#else
               ""
#endif
        );
        failed |= rc;
    }
    if (failed) return 1;
    printf("pure C consumer ok\n");
    return 0;
}
