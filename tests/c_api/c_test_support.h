/* Shared helpers for the pure-C ABI tests: assertions, a portable barrier (macOS has no pthread_barrier_t), and an in-process N-rank world whose
 * ranks exchange endpoint blobs through a shared array (a test-only exchange mechanism: libtbccl contains no bootstrap service). */
#ifndef TBCCL_C_TEST_SUPPORT_H
#define TBCCL_C_TEST_SUPPORT_H

#include <tbccl/tbccl.h>

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(cond)                                                                                         \
    do {                                                                                                    \
        if (!(cond)) {                                                                                      \
            fprintf(stderr, "[FAIL] %s:%d: %s\n", __FILE__, __LINE__, #cond);                               \
            fflush(stderr);                                                                                 \
            _exit(1);                                                                                       \
        }                                                                                                   \
    } while (0)

#define CHECK_OK(call)                                                                                      \
    do {                                                                                                    \
        tbcclResult_t check_ok_r_ = (call);                                                                 \
        if (check_ok_r_ != TBCCL_SUCCESS) {                                                                 \
            fprintf(stderr, "[FAIL] %s:%d: %s returned %d (%s)\n", __FILE__, __LINE__, #call, (int)check_ok_r_, tbcclGetResultString(check_ok_r_)); \
            fflush(stderr);                                                                                 \
            _exit(1);                                                                                       \
        }                                                                                                   \
    } while (0)

#define CHECK_RESULT(call, expected)                                                                        \
    do {                                                                                                    \
        tbcclResult_t check_res_r_ = (call);                                                                \
        if (check_res_r_ != (expected)) {                                                                   \
            fprintf(stderr, "[FAIL] %s:%d: %s returned %d (%s), expected %d\n", __FILE__, __LINE__, #call, (int)check_res_r_, tbcclGetResultString(check_res_r_), (int)(expected)); \
            fflush(stderr);                                                                                 \
            _exit(1);                                                                                       \
        }                                                                                                   \
    } while (0)

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    unsigned count, waiting, generation;
} TestBarrier;

static inline void barrier_init(TestBarrier *b, unsigned count)
{
    pthread_mutex_init(&b->mutex, NULL);
    pthread_cond_init(&b->cond, NULL);
    b->count = count;
    b->waiting = 0;
    b->generation = 0;
}

static inline void barrier_wait(TestBarrier *b)
{
    pthread_mutex_lock(&b->mutex);
    const unsigned gen = b->generation;
    if (++b->waiting == b->count) {
        b->waiting = 0;
        ++b->generation;
        pthread_cond_broadcast(&b->cond);
    } else {
        while (gen == b->generation) pthread_cond_wait(&b->cond, &b->mutex);
    }
    pthread_mutex_unlock(&b->mutex);
}

static inline tbcclBuffer host_buffer(void *data, size_t bytes)
{
    tbcclBuffer b;
    memset(&b, 0, sizeof(b));
    b.struct_size = (uint32_t)sizeof(b);
    b.memory_kind = TBCCL_MEMORY_HOST;
    b.device_ordinal = -1;
    b.data = data;
    b.bytes = bytes;
    return b;
}

/* Waits for a Work and returns its operation result, checking the query itself succeeded. */
static inline tbcclResult_t wait_result(tbcclWork_t w)
{
    tbcclResult_t op = TBCCL_INTERNAL_ERROR;
    CHECK_OK(tbcclWorkWait(w, &op));
    return op;
}

/* ---- an in-process world -------------------------------------------------------------------------------------------------------------- */
typedef struct World World;
typedef void (*RankBody)(int rank, int world, tbcclComm_t comm, void *user);

struct World {
    int size;
    tbcclUniqueId id;
    tbcclEndpointBlob blobs[8];
    TestBarrier barrier;
    RankBody body;
    void *user;
    const tbcclBootstrapOptions *options;
};

typedef struct {
    World *w;
    int rank;
} RankArg;

static void *rank_main(void *p)
{
    RankArg *a = (RankArg *)p;
    World *w = a->w;
    tbcclBootstrap_t bs = NULL;
    CHECK_OK(tbcclBootstrapBegin((uint32_t)a->rank, (uint32_t)w->size, &w->id, w->options, &bs));
    memset(&w->blobs[a->rank], 0, sizeof(w->blobs[a->rank]));
    w->blobs[a->rank].struct_size = (uint32_t)sizeof(tbcclEndpointBlob);
    CHECK_OK(tbcclBootstrapGetEndpoint(bs, &w->blobs[a->rank]));
    barrier_wait(&w->barrier); /* every blob is published: the application's "all-gather" */
    tbcclComm_t comm = NULL;
    CHECK_OK(tbcclBootstrapComplete(bs, w->blobs, (uint32_t)w->size, &comm));
    CHECK(comm != NULL);
    CHECK_OK(tbcclBootstrapDestroy(bs));
    w->body(a->rank, w->size, comm, w->user);
    barrier_wait(&w->barrier); /* nobody tears down a socket a peer still uses */
    CHECK_OK(tbcclCommDestroy(comm));
    return NULL;
}

static inline void run_world(int size, RankBody body, void *user, const tbcclBootstrapOptions *options)
{
    World *w = (World *)calloc(1, sizeof(World));
    pthread_t threads[8];
    RankArg args[8];
    CHECK(size >= 1 && size <= 4);
    w->size = size;
    w->body = body;
    w->user = user;
    w->options = options;
    CHECK_OK(tbcclGetUniqueId(&w->id));
    barrier_init(&w->barrier, (unsigned)size);
    for (int r = 0; r < size; ++r) {
        args[r].w = w;
        args[r].rank = r;
        CHECK(pthread_create(&threads[r], NULL, rank_main, &args[r]) == 0);
    }
    for (int r = 0; r < size; ++r) pthread_join(threads[r], NULL);
    free(w);
}

#endif
