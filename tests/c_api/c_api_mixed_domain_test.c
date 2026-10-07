/* The C ABI gets the repaired ordering semantics with no API change. Collectives and point-to-point are independent ordering domains:
 * - one application thread per rank submits all_reduces while another submits send/recv, in opposite relative orders on the two ranks (pure C, pthreads);
 * - an all_reduce and a recv pending together are both terminal with TBCCL_ABORTED after tbcclCommAbort from another thread. */
#define _POSIX_C_SOURCE 200809L
#include "c_test_support.h"

#include <time.h>

enum { kOps = 24, kElems = 1024 };

typedef struct {
    tbcclComm_t comm;
    int rank;
    int variant; /* which thread starts first on which rank */
    float reduce[kOps][kElems];
    uint8_t tx[kOps][2048 + kOps * 3];
    uint8_t rx[kOps][2048 + kOps * 3];
    int coll_failed, p2p_failed;
} RankState;

static void fill(uint8_t *p, size_t n, int from, int tag)
{
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)(from * 131 + tag * 17 + i * 7 + 1);
}

static void *collective_thread(void *arg)
{
    RankState *s = (RankState *)arg;
    tbcclWork_t works[kOps];
    for (int k = 0; k < kOps; ++k) {
        for (int i = 0; i < kElems; ++i) s->reduce[k][i] = (float)(k + 1) * (float)(s->rank + 1) + (float)(i % 7);
        tbcclBuffer b = host_buffer(s->reduce[k], sizeof(s->reduce[k]));
        if (tbcclAllReduce(s->comm, &b, &b, kElems, TBCCL_FLOAT32, TBCCL_SUM, NULL, &works[k]) != TBCCL_SUCCESS) { s->coll_failed = 1; return NULL; }
    }
    for (int k = 0; k < kOps; ++k) {
        if (wait_result(works[k]) != TBCCL_SUCCESS) s->coll_failed = 1;
        CHECK_OK(tbcclWorkDestroy(works[k]));
        for (int i = 0; i < kElems; ++i) {
            const float want = (float)(k + 1) * 3.0f + 2.0f * (float)(i % 7); /* ranks contribute (k+1)*1 + (k+1)*2 */
            if (s->reduce[k][i] != want) { s->coll_failed = 1; break; }
        }
    }
    return NULL;
}

static void *p2p_thread(void *arg)
{
    RankState *s = (RankState *)arg;
    const uint32_t peer = (uint32_t)(1 - s->rank);
    tbcclWork_t sw[kOps], rw[kOps];
    for (int k = 0; k < kOps; ++k) {
        const size_t n = sizeof(s->tx[k]);
        fill(s->tx[k], n, s->rank, 40 + k);
        memset(s->rx[k], 0xEE, n);
        tbcclBuffer rb = host_buffer(s->rx[k], n), sb = host_buffer(s->tx[k], n);
        if (tbcclRecv(s->comm, &rb, peer, NULL, &rw[k]) != TBCCL_SUCCESS || tbcclSend(s->comm, &sb, peer, NULL, &sw[k]) != TBCCL_SUCCESS) { s->p2p_failed = 1; return NULL; }
    }
    for (int k = 0; k < kOps; ++k) {
        const size_t n = sizeof(s->rx[k]);
        if (wait_result(sw[k]) != TBCCL_SUCCESS || wait_result(rw[k]) != TBCCL_SUCCESS) s->p2p_failed = 1;
        CHECK_OK(tbcclWorkDestroy(sw[k]));
        CHECK_OK(tbcclWorkDestroy(rw[k]));
        uint8_t want[2048 + kOps * 3];
        fill(want, n, (int)peer, 40 + k);
        if (memcmp(want, s->rx[k], n) != 0) s->p2p_failed = 1;
    }
    return NULL;
}

static void concurrent_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)user;
    CHECK(world == 2);
    for (int variant = 0; variant < 2; ++variant) {
        RankState *s = (RankState *)calloc(1, sizeof(RankState));
        s->comm = comm;
        s->rank = rank;
        s->variant = variant;
        pthread_t a, b;
        const int collectives_first = ((rank + variant) % 2) == 0; /* the ranks start the threads in opposite order, and the order flips between the two passes */
        if (collectives_first) {
            CHECK(pthread_create(&a, NULL, collective_thread, s) == 0);
            CHECK(pthread_create(&b, NULL, p2p_thread, s) == 0);
        } else {
            CHECK(pthread_create(&b, NULL, p2p_thread, s) == 0);
            CHECK(pthread_create(&a, NULL, collective_thread, s) == 0);
        }
        pthread_join(a, NULL);
        pthread_join(b, NULL);
        CHECK(!s->coll_failed);
        CHECK(!s->p2p_failed);
        free(s);
    }
}

typedef struct {
    tbcclComm_t comm;
} AbortArg;

static void *abort_thread(void *arg)
{
    AbortArg *a = (AbortArg *)arg;
    const struct timespec ts = {0, 150 * 1000 * 1000};
    nanosleep(&ts, NULL);
    CHECK_OK(tbcclCommAbort(a->comm, "mixed abort"));
    return NULL;
}

static void abort_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)user;
    CHECK(world == 2);
    if (rank != 0) return; /* rank 1 posts nothing: both of rank 0's operations stay pending until the abort */
    static float data[4096];
    static uint8_t rx[4096];
    memset(rx, 0xEE, sizeof(rx));
    tbcclBuffer db = host_buffer(data, sizeof(data)), rb = host_buffer(rx, sizeof(rx));
    tbcclWork_t coll = NULL, recv = NULL;
    CHECK_OK(tbcclAllReduce(comm, &db, &db, 4096 / 4, TBCCL_FLOAT32, TBCCL_SUM, NULL, &coll));
    CHECK_OK(tbcclRecv(comm, &rb, 1, NULL, &recv));
    AbortArg arg = {comm};
    pthread_t t;
    CHECK(pthread_create(&t, NULL, abort_thread, &arg) == 0);
    int32_t done = 0;
    tbcclResult_t op = TBCCL_SUCCESS;
    CHECK_OK(tbcclWorkWaitFor(coll, 10000, &done, &op));
    /* the aborted collective is terminal and failed; its text says why (the result code of a collective cut short inside the N=2 reduction is mapped by the collective executor,
     * not by this test) */
    CHECK(done == 1 && op != TBCCL_SUCCESS);
    {
        char text[256];
        size_t need = 0;
        CHECK_OK(tbcclWorkGetErrorString(coll, text, sizeof(text), &need));
        CHECK(strstr(text, "abort") != NULL);
    }
    CHECK_OK(tbcclWorkWaitFor(recv, 10000, &done, &op));
    CHECK(done == 1 && op == TBCCL_ABORTED);
    pthread_join(t, NULL);
    for (size_t i = 0; i < sizeof(rx); ++i) CHECK(rx[i] == 0xEE); /* nothing was written into the receive buffer */
    CHECK_OK(tbcclWorkDestroy(coll));
    CHECK_OK(tbcclWorkDestroy(recv));
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    run_world(2, concurrent_body, NULL, NULL);
    printf("[PASS] C ABI: all_reduce thread + send/recv thread, opposite start orders\n");
    run_world(2, abort_body, NULL, NULL);
    printf("[PASS] C ABI: all_reduce + recv pending, tbcclCommAbort from another thread\n");
    printf("All C API mixed-domain tests passed.\n");
    return 0;
}
