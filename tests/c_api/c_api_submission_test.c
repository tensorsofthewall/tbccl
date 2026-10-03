/* Phase 52 (Part D through the C ABI): submission is nonblocking. tbccl_test_pause_progress (test-only) holds all transport progress; every submission call must
 * return while it is held, and only after every rank has signalled ALL_POSTED is progress released. Also: WaitFor without cancelling, Work handles destroyed
 * early while the buffers stay alive, concurrent submitters, concurrent Work queries, and an abort from another thread with hundreds of operations queued. */

#define _POSIX_C_SOURCE 200809L /* clock_gettime / nanosleep under strict C11 */

#include "c_test_support.h"

#include <signal.h>
#include <stdint.h>
#include <time.h>

void tbccl_test_pause_progress(tbcclComm_t comm, int paused);

static void *watchdog(void *arg)
{
    struct timespec ts = {(time_t)(intptr_t)arg, 0};
    nanosleep(&ts, NULL);
    fprintf(stderr, "[FAIL] watchdog: the C submission test did not finish (a submission call or a wait is blocking)\n");
    fflush(stderr);
    _exit(2);
}

static double seconds_since(const struct timespec *t0)
{
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec - t0->tv_sec) + (double)(t1.tv_nsec - t0->tv_nsec) / 1e9;
}

static void fill(uint8_t *p, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; ++i) p[i] = (uint8_t)(seed * 31u + i * 7u + (i >> 8));
}
static int check(const uint8_t *p, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; ++i)
        if (p[i] != (uint8_t)(seed * 31u + i * 7u + (i >> 8))) return 0;
    return 1;
}

typedef struct {
    TestBarrier gate;
    int count;
    size_t bytes;
} Ctx;

/* ---- 100 sends THEN 100 receives, progress gated on both ranks until both signalled ALL_POSTED --------------------------------------------- */
static void symmetric_body(int rank, int world, tbcclComm_t comm, void *user)
{
    Ctx *c = (Ctx *)user;
    CHECK(world == 2);
    const int n = c->count;
    const size_t bytes = c->bytes;
    const uint32_t peer = (uint32_t)(1 - rank);
    uint8_t **out = (uint8_t **)calloc((size_t)n, sizeof(uint8_t *)), **in = (uint8_t **)calloc((size_t)n, sizeof(uint8_t *));
    tbcclWork_t *works = (tbcclWork_t *)calloc((size_t)n * 2, sizeof(tbcclWork_t));
    CHECK(out && in && works);
    for (int i = 0; i < n; ++i) {
        out[i] = (uint8_t *)malloc(bytes);
        in[i] = (uint8_t *)calloc(bytes, 1);
        CHECK(out[i] && in[i]);
        fill(out[i], bytes, (unsigned)(i * 2 + rank));
    }
    tbccl_test_pause_progress(comm, 1); /* the peer's transport makes no progress at all until ALL_POSTED */
    for (int i = 0; i < n; ++i) {
        tbcclBuffer b = host_buffer(out[i], bytes);
        CHECK_OK(tbcclSend(comm, &b, peer, NULL, &works[i]));
    }
    for (int i = 0; i < n; ++i) {
        tbcclBuffer b = host_buffer(in[i], bytes);
        CHECK_OK(tbcclRecv(comm, &b, peer, NULL, &works[n + i]));
    }
    /* nothing can have completed while progress is held */
    for (int i = 0; i < 2 * n; i += 7) {
        int32_t done = -1;
        tbcclResult_t op = -1;
        CHECK_OK(tbcclWorkTest(works[i], &done, &op));
        CHECK(done == 0);
    }
    /* a timed wait expires without cancelling anything */
    {
        int32_t done = -1;
        tbcclResult_t op = -1;
        CHECK_OK(tbcclWorkWaitFor(works[0], 20, &done, &op));
        CHECK(done == 0 && op == TBCCL_SUCCESS);
    }
    barrier_wait(&c->gate); /* ALL_POSTED on both ranks */
    tbccl_test_pause_progress(comm, 0);
    for (int i = 0; i < 2 * n; ++i) CHECK(wait_result(works[i]) == TBCCL_SUCCESS);
    for (int i = 0; i < n; ++i) CHECK(check(in[i], bytes, (unsigned)(i * 2 + (int)peer)));
    for (int i = 0; i < 2 * n; ++i) CHECK_OK(tbcclWorkDestroy(works[i]));
    for (int i = 0; i < n; ++i) { free(out[i]); free(in[i]); }
    free(out); free(in); free(works);
}

static void test_symmetric(int count, size_t bytes, const char *label)
{
    Ctx c;
    barrier_init(&c.gate, 2);
    c.count = count;
    c.bytes = bytes;
    run_world(2, symmetric_body, &c, NULL);
    printf("[PASS] C ABI %s: %d sends then %d receives posted on both ranks with progress held, all calls returned before release, all completed in order\n", label, count, count);
}

/* ---- N=4: hundreds of outstanding Works per rank; handles destroyed early; the buffers stay alive until the FIFO tail marker completes ------------------ */
static void n4_body(int rank, int world, tbcclComm_t comm, void *user)
{
    Ctx *c = (Ctx *)user;
    CHECK(world == 4);
    enum { OPS = 150, BYTES = 4096 };
    const uint32_t next = (uint32_t)((rank + 1) % world), prev = (uint32_t)((rank + world - 1) % world);
    static __thread uint8_t out_store[OPS][BYTES], in_store[OPS][BYTES];
    tbcclWork_t keep[2] = {NULL, NULL};
    tbccl_test_pause_progress(comm, 1);
    for (int i = 0; i < OPS; ++i) {
        fill(out_store[i], BYTES, (unsigned)(i * 4 + rank));
        tbcclBuffer sb = host_buffer(out_store[i], BYTES), rb = host_buffer(in_store[i], BYTES);
        tbcclWork_t s = NULL, r = NULL;
        CHECK_OK(tbcclSend(comm, &sb, next, NULL, &s));
        CHECK_OK(tbcclRecv(comm, &rb, prev, NULL, &r));
        if (i == OPS - 1) { keep[0] = s; keep[1] = r; } /* the FIFO tail marker of each direction is kept; every earlier handle is dropped at once */
        else { CHECK_OK(tbcclWorkDestroy(s)); CHECK_OK(tbcclWorkDestroy(r)); }
    }
    /* the tail markers cannot be done while progress is held; WaitFor must not cancel them */
    for (int k = 0; k < 2; ++k) {
        int32_t done = -1;
        tbcclResult_t op = -1;
        CHECK_OK(tbcclWorkWaitFor(keep[k], 20, &done, &op));
        CHECK(done == 0 && op == TBCCL_SUCCESS);
    }
    barrier_wait(&c->gate);
    tbccl_test_pause_progress(comm, 0);
    /* later: both tail markers complete with SUCCESS, which (FIFO per peer and direction) means everything before them did too */
    for (int k = 0; k < 2; ++k) {
        int32_t done = 0;
        tbcclResult_t op = -1;
        while (!done) CHECK_OK(tbcclWorkWaitFor(keep[k], 1000, &done, &op));
        CHECK(op == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(keep[k]));
    }
    for (int i = 0; i < OPS; ++i) CHECK(check(in_store[i], BYTES, (unsigned)(i * 4 + (int)prev)));
}

static void test_n4(void)
{
    Ctx c;
    barrier_init(&c.gate, 4);
    run_world(4, n4_body, &c, NULL);
    printf("[PASS] C ABI N=4: 150 sends + 150 receives per rank posted with progress held, 298 handles destroyed immediately while their buffers stayed alive, tail markers WaitFor done=0 then SUCCESS, data intact\n");
}

/* ---- several application threads submit concurrently; the runtime linearizes them ------------------------------------------------------------------ */
typedef struct {
    tbcclComm_t comm;
    int send;
    int count;
    uint8_t *buf;
    size_t bytes;
    tbcclWork_t *works;
} Submitter;

static void *submit_main(void *p)
{
    Submitter *s = (Submitter *)p;
    for (int i = 0; i < s->count; ++i) {
        tbcclBuffer b = host_buffer(s->buf, s->bytes);
        if (s->send) CHECK_OK(tbcclSend(s->comm, &b, 1, NULL, &s->works[i]));
        else CHECK_OK(tbcclRecv(s->comm, &b, 0, NULL, &s->works[i]));
    }
    return NULL;
}

static void concurrent_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)user;
    CHECK(world == 2);
    enum { THREADS = 4, PER = 60, BYTES = 2048 };
    pthread_t t[THREADS];
    Submitter sub[THREADS];
    tbcclWork_t *all = (tbcclWork_t *)calloc((size_t)THREADS * PER, sizeof(tbcclWork_t));
    static uint8_t send_buf[BYTES], recv_buf[THREADS][BYTES];
    memset(send_buf, 0x6B, sizeof(send_buf));
    for (int i = 0; i < THREADS; ++i) {
        sub[i].comm = comm;
        sub[i].send = (rank == 0);
        sub[i].count = PER;
        sub[i].buf = (rank == 0) ? send_buf : recv_buf[i];
        sub[i].bytes = BYTES;
        sub[i].works = all + (size_t)i * PER;
        CHECK(pthread_create(&t[i], NULL, submit_main, &sub[i]) == 0);
    }
    for (int i = 0; i < THREADS; ++i) pthread_join(t[i], NULL);
    for (int i = 0; i < THREADS * PER; ++i) CHECK(wait_result(all[i]) == TBCCL_SUCCESS);
    if (rank == 1)
        for (int i = 0; i < THREADS; ++i) CHECK(recv_buf[i][0] == 0x6B && recv_buf[i][BYTES - 1] == 0x6B);
    for (int i = 0; i < THREADS * PER; ++i) CHECK_OK(tbcclWorkDestroy(all[i]));
    free(all);
}

static void test_concurrent_submitters(void)
{
    run_world(2, concurrent_body, NULL, NULL);
    printf("[PASS] C ABI: 4 application threads submit 240 sends (and the matching 240 receives) concurrently; all complete\n");
}

/* ---- Work queries race with each other while the operation is active ---------------------------------------------------------------------------- */
typedef struct {
    tbcclWork_t w;
    int mode;
    tbcclResult_t result;
    int32_t done;
} Querier;

static void *query_main(void *p)
{
    Querier *q = (Querier *)p;
    if (q->mode == 0) { CHECK_OK(tbcclWorkWait(q->w, &q->result)); q->done = 1; }
    else if (q->mode == 1) { do CHECK_OK(tbcclWorkTest(q->w, &q->done, &q->result)); while (!q->done); }
    else { do CHECK_OK(tbcclWorkWaitFor(q->w, 5, &q->done, &q->result)); while (!q->done); }
    return NULL;
}

static void query_body(int rank, int world, tbcclComm_t comm, void *user)
{
    Ctx *c = (Ctx *)user;
    CHECK(world == 2);
    static uint8_t buf[1 << 20];
    tbcclBuffer b = host_buffer(buf, sizeof(buf));
    tbcclWork_t w = NULL;
    pthread_t t[6];
    Querier q[6];
    if (rank == 1) {
        CHECK_OK(tbcclRecv(comm, &b, 0, NULL, &w));
        for (int i = 0; i < 6; ++i) { q[i].w = w; q[i].mode = i % 3; q[i].result = -1; q[i].done = 0; CHECK(pthread_create(&t[i], NULL, query_main, &q[i]) == 0); }
        barrier_wait(&c->gate); /* the senders starts only after the queriers are running */
        for (int i = 0; i < 6; ++i) { pthread_join(t[i], NULL); CHECK(q[i].done == 1 && q[i].result == TBCCL_SUCCESS); }
        CHECK_OK(tbcclWorkDestroy(w));
    } else {
        memset(buf, 1, sizeof(buf));
        barrier_wait(&c->gate);
        CHECK_OK(tbcclSend(comm, &b, 1, NULL, &w));
        CHECK(wait_result(w) == TBCCL_SUCCESS);
        CHECK_OK(tbcclWorkDestroy(w));
    }
}

static void test_concurrent_queries(void)
{
    Ctx c;
    barrier_init(&c.gate, 2);
    run_world(2, query_body, &c, NULL);
    printf("[PASS] C ABI: Wait, Test and WaitFor race on one Work from six threads while the operation runs; every thread observes SUCCESS\n");
}

/* ---- abort from a separate thread with hundreds of operations queued ---------------------------------------------------------------------------- */
typedef struct {
    tbcclComm_t comm;
} Aborter;

static void *abort_main(void *p)
{
    Aborter *a = (Aborter *)p;
    struct timespec ts = {0, 20 * 1000 * 1000};
    nanosleep(&ts, NULL);
    CHECK_OK(tbcclCommAbort(a->comm, "abort from another thread"));
    return NULL;
}

static void abort_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)user;
    CHECK(world == 2);
    enum { OPS = 300, BYTES = 1 << 20 };
    if (rank != 0) return; /* rank 1 posts nothing: rank 0's sends stay queued / blocked */
    uint8_t *bufs = (uint8_t *)malloc((size_t)OPS * BYTES);
    CHECK(bufs != NULL);
    memset(bufs, 3, (size_t)OPS * BYTES);
    tbcclWork_t *works = (tbcclWork_t *)calloc(OPS, sizeof(tbcclWork_t));
    tbccl_test_pause_progress(comm, 1);
    for (int i = 0; i < OPS; ++i) {
        tbcclBuffer b = host_buffer(bufs + (size_t)i * BYTES, BYTES);
        CHECK_OK(tbcclSend(comm, &b, 1, NULL, &works[i]));
    }
    Aborter a = {comm};
    pthread_t t;
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    CHECK(pthread_create(&t, NULL, abort_main, &a) == 0);
    int aborted_count = 0;
    for (int i = 0; i < OPS; ++i) {
        tbcclResult_t op = -1;
        CHECK_OK(tbcclWorkWait(works[i], &op)); /* every operation becomes terminal */
        CHECK(op == TBCCL_ABORTED);
        ++aborted_count;
    }
    pthread_join(t, NULL);
    CHECK(aborted_count == OPS && seconds_since(&t0) < 10.0);
    for (int i = 0; i < OPS; ++i) CHECK_OK(tbcclWorkDestroy(works[i]));
    free(works);
    free(bufs); /* every Work is terminal: no transport thread touches the buffers any more (ASan checks this) */
}

static void test_abort_from_thread(void)
{
    run_world(2, abort_body, NULL, NULL);
    printf("[PASS] C ABI: abort from another thread with 300 queued 1 MiB sends: every Work terminal with ABORTED, bounded; buffers freed afterwards\n");
}

/* ---- collectives queued before any wait ------------------------------------------------------------------------------------------------------- */
static void collectives_body(int rank, int world, tbcclComm_t comm, void *user)
{
    (void)user;
    enum { ROUNDS = 10 };
    float vals[ROUNDS][100];
    tbcclWork_t works[ROUNDS * 2];
    for (int i = 0; i < ROUNDS; ++i) {
        for (int k = 0; k < 100; ++k) vals[i][k] = (float)(rank + 1 + i);
        tbcclBuffer b = host_buffer(vals[i], sizeof(vals[i]));
        CHECK_OK(tbcclBarrier(comm, &works[i * 2]));
        CHECK_OK(tbcclAllReduce(comm, &b, &b, 100, TBCCL_FLOAT32, TBCCL_SUM, NULL, &works[i * 2 + 1]));
    }
    for (int i = 0; i < ROUNDS * 2; ++i) CHECK(wait_result(works[i]) == TBCCL_SUCCESS);
    for (int i = 0; i < ROUNDS; ++i) {
        float want = 0;
        for (int r = 0; r < world; ++r) want += (float)(r + 1 + i);
        CHECK(vals[i][0] == want && vals[i][99] == want);
    }
    for (int i = 0; i < ROUNDS * 2; ++i) CHECK_OK(tbcclWorkDestroy(works[i]));
}

static void test_queued_collectives(void)
{
    for (int n = 2; n <= 4; ++n) run_world(n, collectives_body, NULL, NULL);
    printf("[PASS] C ABI: 20 collectives queued before any wait execute in submission order at N=2..4\n");
}

int main(void)
{
    pthread_t dog;
    pthread_create(&dog, NULL, watchdog, (void *)(intptr_t)240);
    pthread_detach(dog);
    setvbuf(stdout, NULL, _IONBF, 0);
    test_symmetric(100, 1u << 20, "symmetric (1 MiB)");
    test_symmetric(1000, 256, "symmetric small operations");
    test_n4();
    test_concurrent_submitters();
    test_concurrent_queries();
    test_abort_from_thread();
    test_queued_collectives();
    printf("All C API submission tests passed.\n");
    return 0;
}
