# Staged transfers: failure handling and cancellation

This page describes how a staged (chunked) transfer stops when one of its stages fails, and what the tests check. It is for people changing `src/core/tensor_comm_worker.cpp` or `src/core/staging_pool.cpp`.

## The pipeline

Memory that the transport cannot read or write directly (CUDA device memory, or any provider that is not directly accessible) moves through host staging buffers. A `TensorCommWorker` lane runs a staged request with two threads that share a `StagingPool` of `depth` slots (`min(pipeline_depth, chunk count)`):

| direction | staging thread | network thread |
|---|---|---|
| send | acquire a slot, `stage_source_chunk`, publish the chunk | wait for the chunk, `Transport::send`, release the slot |
| receive | wait for the chunk, `commit_destination_chunk`, release the slot | acquire a slot, `Transport::recv`, publish the chunk |

The network thread submits the staging role, runs its own role, then waits for the staging job (`wait_job()`) before it completes the `Work`. `Communicator::abort()` and the worker destructor wait for the lane to become idle.

With the default one chunk per transfer the pool has one slot and each stage acquires at most once, so the pipeline cannot run out of slots. A transfer split into three or more chunks (`TBCCL_STAGED_CHUNK_BYTES`) can.

## What used to go wrong

When a stage failed it gave its slot back to the pool and only then recorded the failure. In between, the other stage could take that slot, publish one more chunk and block in `StagingPool::acquire()` with every slot occupied. Recording the failure woke the stage waiting for a chunk, but nothing woke a stage waiting for a slot. That stage slept forever, the lane never became idle, and `abort()` and the destructor never returned. It was seen on the send side (failing `Transport::send`, including the send failing because of an abort) and the same window exists on the receive side (failing destination commit).

## The failure transition

Every failure site calls one function, `fail_transfer`, which does these steps in order:

1. record the first error (later errors, including the cancellation a woken stage sees, never replace it);
2. cancel the pool: `StagingPool::cancel()` wakes every thread blocked in `acquire()`, and `acquire()` throws `Error(Aborted)` from then on without handing out a slot;
3. give back the slot the failing stage owns (this does not undo the cancellation).

The other stage then ends on its next step: it observes the recorded failure, or its `acquire()` throws. `wait_job()` returns, the `Work` completes once with the original error, and the cancelled pool is discarded. A pool is cancelled at most once and is never reset or reused (a pool that reports `cancelled()` is rebuilt even if it were still cached); successful requests keep using the cached pool as before.

An exception that is not a `std::exception` escaping the network stage takes the same transition through `fail_transfer_unknown`, which records the failure (code `InternalError`) without constructing an exception object, so it cannot fail for lack of memory and still notifies both kinds of waiters.

`fail_transfer` does not throw: each step is attempted on its own, so an exception inside it (for example an allocation failure while copying the error text) cannot leave the other stage asleep. The network thread always reaches `wait_job()` once the staging job is submitted, including when its own stage throws, because the staging job refers to request-local state.

## Limits

* Cancelling the pool does not interrupt a stage that is inside a backend call (a device copy) or a transport call. Those have to return; an abort interrupts the transport (`Transport::abort`), which makes a blocked `send`/`recv` throw.
* Locks: no code path holds two of the pool mutex, the progress mutex, the job mutex and the queue mutex at once, and no backend or transport call is made with an internal lock held. (The two early exits of the publishing stages, which find the transfer already failed, decide under the progress mutex and release their slot to the pool only after dropping it.)

## Tests

* `staging_pool_cancel_test`: the pool in isolation (wake on release, wake on cancel, no slot after cancel, idempotence, several waiters, cancel racing with release).
* `tensor_comm_worker_failure_test`: a fake transport and staging backend. A test-only probe (`src/core/tensor_comm_worker_probe.hpp`, a function pointer that is null unless a test sets it and is read only on the failure path) stalls the failing stage after each step of the transition while the other stage keeps running. With the previous behaviour (slot released first, no cancellation) the send and receive scenarios stall; the test reports that as a failure with a thread dump instead of hanging. It also injects failures at the first, a middle and the last chunk for transport, staging and commit errors at pipeline depths 1, 2 and 4, and checks one terminal completion, the original error, and that the next request succeeds. The same matrix runs with an exception thrown from inside each step of the transition, and with a transport that throws something that is not a `std::exception` (send and receive).
* `communicator_staged_chunk_test` and `communicator_cuda_stream_chunk_test` cover the same paths through the `Communicator`, including an abort during a chunked send.
