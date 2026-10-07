/* TBCCL C ABI cross-host correctness probe (the C ABI v1 work): one process per rank, possibly on different hosts. NOT a benchmark: payloads <= 1 MiB and about a dozen
 * operations. The "application" that exchanges the endpoint blobs is the operator (file + scp):
 *
 *   c_link_probe gen-id                                   print a fresh 32-hex-digit unique id
 *   c_link_probe <rank> <world> <idhex> <bind> <advertise> <dir>
 *       writes <dir>/blob.<rank>, waits for <dir>/blobs.all (the world's blobs concatenated in rank order), then runs the checks.
 *
 * Every check is exact (integer-valued Float32 sums, patterned bytes). */

#define _POSIX_C_SOURCE 200809L

#include <tbccl/tbccl.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)
#define OK(call) do { tbcclResult_t r_ = (call); if (r_ != TBCCL_SUCCESS) { fprintf(stderr, "[FAIL] %s:%d: %s -> %d (%s)\n", __FILE__, __LINE__, #call, (int)r_, tbcclGetResultString(r_)); exit(1); } } while (0)

static tbcclBuffer buf(void *p, size_t n)
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

static void finish(tbcclWork_t w, const char *what)
{
    tbcclResult_t op = -1;
    OK(tbcclWorkWait(w, &op));
    if (op != TBCCL_SUCCESS) {
        char t[256];
        size_t need;
        tbcclWorkGetErrorString(w, t, sizeof(t), &need);
        fprintf(stderr, "[FAIL] %s: %s: %s\n", what, tbcclGetResultString(op), t);
        exit(1);
    }
    OK(tbcclWorkDestroy(w));
}

static int hexval(char c) { return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1; }

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc == 2 && strcmp(argv[1], "gen-id") == 0) {
        tbcclUniqueId id;
        OK(tbcclGetUniqueId(&id));
        for (int i = 0; i < 16; ++i) printf("%02x", id.bytes[i]);
        printf("\n");
        return 0;
    }
    if (argc != 7) { fprintf(stderr, "usage: %s gen-id | %s <rank> <world> <idhex> <bind> <advertise> <dir>\n", argv[0], argv[0]); return 2; }
    const int rank = atoi(argv[1]), world = atoi(argv[2]);
    tbcclUniqueId id;
    CHECK(strlen(argv[3]) == 32);
    for (int i = 0; i < 16; ++i) { int hi = hexval(argv[3][2 * i]), lo = hexval(argv[3][2 * i + 1]); CHECK(hi >= 0 && lo >= 0); id.bytes[i] = (uint8_t)(hi * 16 + lo); }

    tbcclBootstrapOptions opt;
    memset(&opt, 0, sizeof(opt));
    opt.struct_size = (uint32_t)sizeof(opt);
    opt.bind_host = argv[4];
    opt.advertise_host = argv[5];
    opt.timeout_ms = 60000;
    tbcclBootstrap_t bs;
    OK(tbcclBootstrapBegin((uint32_t)rank, (uint32_t)world, &id, &opt, &bs));
    tbcclEndpointBlob mine;
    memset(&mine, 0, sizeof(mine));
    mine.struct_size = (uint32_t)sizeof(mine);
    OK(tbcclBootstrapGetEndpoint(bs, &mine));
    char path[512];
    snprintf(path, sizeof(path), "%s/blob.%d.tmp", argv[6], rank);
    FILE *f = fopen(path, "wb");
    CHECK(f && fwrite(&mine, sizeof(mine), 1, f) == 1);
    fclose(f);
    char final_path[512];
    snprintf(final_path, sizeof(final_path), "%s/blob.%d", argv[6], rank);
    CHECK(rename(path, final_path) == 0);
    printf("rank %d: endpoint published\n", rank);

    snprintf(path, sizeof(path), "%s/blobs.all", argv[6]);
    tbcclEndpointBlob *all = (tbcclEndpointBlob *)malloc(sizeof(tbcclEndpointBlob) * (size_t)world);
    CHECK(all != NULL);
    for (int waited = 0;; ++waited) {
        f = fopen(path, "rb");
        if (f) {
            const size_t n = fread(all, sizeof(tbcclEndpointBlob), (size_t)world, f);
            fclose(f);
            if (n == (size_t)world) break;
        }
        CHECK(waited < 1200);
        struct timespec ts = {0, 100 * 1000 * 1000};
        nanosleep(&ts, NULL);
    }
    tbcclComm_t comm;
    OK(tbcclBootstrapComplete(bs, all, (uint32_t)world, &comm));
    OK(tbcclBootstrapDestroy(bs));
    printf("rank %d: communicator up (world %d)\n", rank, world);

    tbcclWork_t w;
    OK(tbcclBarrier(comm, &w)); finish(w, "barrier");
    const uint32_t next = (uint32_t)((rank + 1) % world), prev = (uint32_t)((rank + world - 1) % world);

    { /* P2P ring, 64 KiB */
        enum { N = 65536 };
        uint8_t *out = (uint8_t *)malloc(N), *in = (uint8_t *)calloc(N, 1);
        for (int i = 0; i < N; ++i) out[i] = (uint8_t)(rank * 41 + i * 7);
        tbcclBuffer sb = buf(out, N), rb = buf(in, N);
        tbcclWork_t s, r;
        OK(tbcclSend(comm, &sb, next, NULL, &s));
        OK(tbcclRecv(comm, &rb, prev, NULL, &r));
        finish(s, "ring send"); finish(r, "ring recv");
        for (int i = 0; i < N; ++i) CHECK(in[i] == (uint8_t)((int)prev * 41 + i * 7));
        free(out); free(in);
    }
    for (int root = 0; root < world; ++root) { /* broadcast 256 KiB from every root */
        enum { N = 262144 };
        uint8_t *b = (uint8_t *)malloc(N);
        for (int i = 0; i < N; ++i) b[i] = rank == root ? (uint8_t)(root * 19 + i) : 0;
        tbcclBuffer bb = buf(b, N);
        OK(tbcclBroadcast(comm, &bb, (uint32_t)root, NULL, &w)); finish(w, "broadcast");
        for (int i = 0; i < N; ++i) CHECK(b[i] == (uint8_t)(root * 19 + i));
        free(b);
    }
    { /* all_gather 64 KiB per rank */
        enum { N = 65536 };
        uint8_t *mine_b = (uint8_t *)malloc(N), *gathered = (uint8_t *)malloc((size_t)N * (size_t)world);
        for (int i = 0; i < N; ++i) mine_b[i] = (uint8_t)(rank * 23 + i);
        tbcclBuffer sb = buf(mine_b, N), gb = buf(gathered, (size_t)N * (size_t)world);
        OK(tbcclAllGather(comm, &sb, &gb, NULL, &w)); finish(w, "all_gather");
        for (int q = 0; q < world; ++q)
            for (int i = 0; i < N; ++i) CHECK(gathered[(size_t)q * N + i] == (uint8_t)(q * 23 + i));
        free(mine_b); free(gathered);
    }
    for (int pass = 0; pass < 2; ++pass) { /* Float32 all_reduce: 4 KiB and 1 MiB, twice each */
        const int counts[2] = {1024, 262144};
        for (int c = 0; c < 2; ++c) {
            float *x = (float *)malloc((size_t)counts[c] * sizeof(float));
            for (int i = 0; i < counts[c]; ++i) x[i] = (float)(rank + 1);
            tbcclBuffer xb = buf(x, (size_t)counts[c] * sizeof(float));
            OK(tbcclAllReduce(comm, &xb, &xb, (uint64_t)counts[c], TBCCL_FLOAT32, TBCCL_SUM, NULL, &w)); finish(w, "all_reduce");
            const float want = (float)(world * (world + 1) / 2);
            for (int i = 0; i < counts[c]; ++i) CHECK(x[i] == want);
            free(x);
        }
    }
    OK(tbcclBarrier(comm, &w)); finish(w, "final barrier");
    OK(tbcclCommDestroy(comm));
    free(all);
    printf("rank %d: all checks passed\n", rank);
    return 0;
}
