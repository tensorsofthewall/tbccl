# C API quickstart

A complete, compiled and tested program is `examples/c_quickstart.c` (CTest `c_quickstart`); the code below is that program. Specification: `docs/c_abi_v1.md`. Bootstrap model: `docs/c_api_bootstrap.md`. Threading rules: `docs/c_api_thread_safety.md`.

```cmake
find_package(TBCCL CONFIG REQUIRED)               # a C-only project is fine: no C++ compiler, no -lstdc++/-lc++
add_executable(app app.c)
target_link_libraries(app PRIVATE TBCCL::tbccl_c)
```

## The two rules that bite

> **Destroying a Work handle does NOT cancel the operation.** It only drops your handle.
>
> **Destroying a Work handle is NOT permission to free or modify the buffer.** `tbcclBuffer` does not own memory: the host, CUDA or MetalShared allocation must stay valid, and (for sends and collective inputs) unmodified, **until the operation is terminal**: until a Work reports done, or until the communicator is aborted and the Work is done. If you drop handles early, use another way to know the operation finished (for P2P: a later operation on the same peer and direction completing, since the order per peer and direction is FIFO).

Posting never waits: `tbcclSend`/`tbcclRecv`/collectives return a Work immediately (or fail immediately), so "post a hundred sends, then a hundred receives, then wait" is fine. There are no tags: the application matches the logical order of P2P operations per peer pair, and all ranks must submit collectives in the same order.

`tbcclWorkWait*`'s return value is the status of the **query**; the operation's own result is the out-parameter. `tbcclWorkWaitFor` expiring returns `TBCCL_SUCCESS` with `done == 0` and changes nothing (it never reports `TBCCL_TIMEOUT`).

## The program

```c
/* TBCCL C ABI quickstart: two ranks (two processes on one machine), caller-exchanged endpoint blobs, a P2P message and a Float32 all_reduce.
 * Only <tbccl/tbccl.h> is needed. The "application all-gather" is a pair of pipes between a parent and its children here; in a real program it is
 * whatever your framework already has (MPI_Allgather, a launcher, files, a key-value store). TBCCL itself has no bootstrap service.
 *
 *   cmake: find_package(TBCCL CONFIG REQUIRED)  target_link_libraries(app PRIVATE TBCCL::tbccl_c)   (a C-only project is fine) */

#define _POSIX_C_SOURCE 200809L

#include <tbccl/tbccl.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

#define TRY(call)                                                                                     \
    do {                                                                                              \
        tbcclResult_t r_ = (call);                                                                    \
        if (r_ != TBCCL_SUCCESS) {                                                                    \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__, #call, tbcclGetResultString(r_)); \
            exit(1);                                                                                  \
        }                                                                                             \
    } while (0)

/* Waits for an operation and returns ITS result; the call itself returning SUCCESS only says the query worked. */
static tbcclResult_t wait_for_work(tbcclWork_t work)
{
    tbcclResult_t operation_result = TBCCL_INTERNAL_ERROR;
    TRY(tbcclWorkWait(work, &operation_result));
    if (operation_result != TBCCL_SUCCESS) {
        char text[256];
        size_t required = 0;
        tbcclWorkGetErrorString(work, text, sizeof(text), &required);
        fprintf(stderr, "operation failed (%s): %s\n", tbcclGetResultString(operation_result), text);
    }
    return operation_result;
}

static void rank_main(int rank, int world, const tbcclUniqueId *id, int to_parent, int from_parent)
{
    /* 1. Bootstrap. Begin binds this rank's listeners on kernel-chosen ports. */
    tbcclBootstrap_t bootstrap;
    TRY(tbcclBootstrapBegin((uint32_t)rank, (uint32_t)world, id, NULL /* defaults: bind and advertise 127.0.0.1 */, &bootstrap));

    /* 2. Publish this rank's opaque endpoint blob, receive everyone's (the application's all-gather: blobs in RANK ORDER). */
    tbcclEndpointBlob mine, all[2];
    memset(&mine, 0, sizeof(mine));
    mine.struct_size = (uint32_t)sizeof(mine);
    TRY(tbcclBootstrapGetEndpoint(bootstrap, &mine));
    if (write(to_parent, &mine, sizeof(mine)) != (ssize_t)sizeof(mine)) exit(1);
    for (size_t got = 0; got < sizeof(all);) {
        ssize_t n = read(from_parent, (char *)all + got, sizeof(all) - got);
        if (n <= 0) exit(1);
        got += (size_t)n;
    }

    /* 3. Complete: connects the ranks. The bootstrap handle may now only be destroyed. */
    tbcclComm_t comm;
    TRY(tbcclBootstrapComplete(bootstrap, all, (uint32_t)world, &comm));
    TRY(tbcclBootstrapDestroy(bootstrap));

    /* 4. Point to point. Posting NEVER waits for the peer or the network: both calls return a Work at once. The buffers belong to YOU and must stay valid
     *    and unmodified until the operations are done, even if you destroy the Work handles earlier (destroying a Work does not cancel anything). */
    char out[64], in[64];
    snprintf(out, sizeof(out), "hello from rank %d", rank);
    memset(in, 0, sizeof(in));
    tbcclBuffer send_buffer = {0}, recv_buffer = {0};
    send_buffer.struct_size = recv_buffer.struct_size = (uint32_t)sizeof(tbcclBuffer);
    send_buffer.memory_kind = recv_buffer.memory_kind = TBCCL_MEMORY_HOST;
    send_buffer.device_ordinal = recv_buffer.device_ordinal = -1;
    send_buffer.data = out;
    send_buffer.bytes = sizeof(out);
    recv_buffer.data = in;
    recv_buffer.bytes = sizeof(in);
    const uint32_t peer = (uint32_t)(1 - rank);
    tbcclWork_t send_work, recv_work;
    TRY(tbcclSend(comm, &send_buffer, peer, NULL, &send_work));
    TRY(tbcclRecv(comm, &recv_buffer, peer, NULL, &recv_work));
    if (wait_for_work(send_work) != TBCCL_SUCCESS || wait_for_work(recv_work) != TBCCL_SUCCESS) exit(1);
    TRY(tbcclWorkDestroy(send_work));
    TRY(tbcclWorkDestroy(recv_work));
    printf("rank %d received: \"%s\"\n", rank, in);

    /* 5. A collective: in-place Float32 all_reduce. Every rank must submit collectives in the same order. */
    float values[8];
    for (int i = 0; i < 8; ++i) values[i] = (float)(rank + 1);
    tbcclBuffer vb = {0};
    vb.struct_size = (uint32_t)sizeof(vb);
    vb.memory_kind = TBCCL_MEMORY_HOST;
    vb.device_ordinal = -1;
    vb.data = values;
    vb.bytes = sizeof(values);
    tbcclWork_t reduce_work;
    TRY(tbcclAllReduce(comm, &vb, &vb, 8, TBCCL_FLOAT32, TBCCL_SUM, NULL, &reduce_work));
    if (wait_for_work(reduce_work) != TBCCL_SUCCESS) exit(1);
    TRY(tbcclWorkDestroy(reduce_work));
    printf("rank %d all_reduce(sum) = %.1f (expected %.1f)\n", rank, (double)values[0], (double)(world * (world + 1) / 2));

    /* 6. Cleanup: no operation is outstanding. Destroy handles in this order; none of them may race with another call on the same handle. */
    TRY(tbcclCommDestroy(comm));
    exit(values[0] == (float)(world * (world + 1) / 2) ? 0 : 1);
}

int main(void)
{
    const int world = 2;
    tbcclUniqueId id;
    TRY(tbcclGetUniqueId(&id)); /* generated once, then distributed to every rank by the application (here: inherited through fork) */

    int up[2][2], down[2][2];
    pid_t pid[2];
    for (int r = 0; r < world; ++r)
        if (pipe(up[r]) != 0 || pipe(down[r]) != 0) return 1;
    for (int r = 0; r < world; ++r) {
        pid[r] = fork();
        if (pid[r] == 0) rank_main(r, world, &id, up[r][1], down[r][0]);
    }
    tbcclEndpointBlob blobs[2];
    for (int r = 0; r < world; ++r)
        for (size_t got = 0; got < sizeof(blobs[r]);) {
            ssize_t n = read(up[r][0], (char *)&blobs[r] + got, sizeof(blobs[r]) - got);
            if (n <= 0) return 1;
            got += (size_t)n;
        }
    for (int r = 0; r < world; ++r)
        if (write(down[r][1], blobs, sizeof(blobs)) != (ssize_t)sizeof(blobs)) return 1;
    int failed = 0;
    for (int r = 0; r < world; ++r) {
        int status = 0;
        waitpid(pid[r], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) failed = 1;
    }
    puts(failed ? "quickstart FAILED" : "quickstart ok");
    return failed;
}
```

## What each step needs from you

1. **Unique id**: generate it once (`tbcclGetUniqueId`) and give the same 16 bytes to every rank (environment variable, launcher argument, MPI broadcast...). It contains no address.
2. **Endpoint blobs**: each rank calls `tbcclBootstrapGetEndpoint` and the application all-gathers the fixed-size 256-byte blobs in rank order. For several machines pass `bind_host`/`advertise_host` in `tbcclBootstrapOptions` (the advertised address must be reachable by the peers; never a wildcard).
3. **Complete**: all ranks call `tbcclBootstrapComplete` concurrently (it connects them); a rank that never arrives makes the others fail with `TBCCL_TIMEOUT`.
4. **CUDA**: call `tbcclRegisterCudaSupport()` once (it returns `TBCCL_UNSUPPORTED` in a host-only build), describe device memory with `memory_kind = TBCCL_MEMORY_CUDA`, and pass your `cudaStream_t` as `{struct_size, TBCCL_EXEC_CUDA_STREAM, (void *)stream}` when the operand's producer runs on a non-default stream. `tbccl.h` includes no CUDA header.
