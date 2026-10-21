# Two-rank AllGather exchange

With two ranks, `Communicator::all_gather()` posts its send and its receive together and lets them run at the same time. Before this change each rank sent and received one after the other (rank 0 sent first, rank 1 received first), so the second direction could not start until the first had finished.

## What runs

The collective job runs on the collective executor thread, as before. After the local copy of the rank's own contribution it calls `run_exchange()`, which:

1. enqueues the send and then the receive on the peer's collective worker (`coll_worker`), both with `shared_lane = false`;
2. waits for both to finish;
3. reports one error if either failed.

The collective worker already has two independent lanes (a queue, a network thread and a staging pool each) and unbounded admission. A send and a receive therefore land on different lanes, neither can sit in front of the other, and posting the second never waits for the first. The exchange uses the same collective connection as before. It adds no thread, no connection, no frame and no public API, and the bytes on the wire are unchanged: one raw payload each way. A rank running the previous sequential schedule can exchange with a rank running this one.

## Ordering

AllGather is still one collective job. The executor runs the next collective only after the job returns, and the job returns only after both transfers have finished, so no later collective can start on the collective connection while a direction of this one is in flight. Point-to-point traffic uses its own connection and worker and is unaffected.

## Failure handling

* **Both transfers always finish.** The job never returns while a transfer can still read or write the caller's buffers. If the first failure is seen while the other transfer is blocked, the communicator is aborted (the abort is idempotent) so the other transfer is interrupted, and then both are waited for.
* **The first cause is reported.** A transfer that failed only because of the abort is not reported in place of the transfer that caused it. The abort text carries the original error.
* **Partial admission.** If the second enqueue fails after the first was accepted, the peer would wait for the missing leg, so the communicator is aborted, the accepted transfer is waited for, and the admission error is rethrown.
* The staged-transfer cancellation described in [staged-transfer-cancellation](staged-transfer-cancellation.md) applies to each transfer unchanged.

## Staged chunking

Each direction takes its chunk size from the memory it moves, exactly as the sequential schedule did. Host and Metal-shared memory is read and written in place. For memory that must be staged, `TBCCL_STAGED_CHUNK_BYTES` applies to the send and the receive independently; the two ranks may use different values.

## Buffers

`outputs[rank()]` is filled by a local copy unless it is the same pointer as `input`, which is the only supported alias. An output slot that overlaps the input of the other slot was never supported (the two ranks used to read and write it in different orders) and still is not.

## Tests

* `n2_exchange_unit_test` drives the exchange over a controlled transport in which a send cannot finish until a receive has started and the reverse. It completes only if the two transfers genuinely overlap; the same transport run with the old sequential schedule is a negative control. It also injects failures into each leg, into the staged source copy and the staged destination commit, aborts before and during the exchange, and fails the second admission.
* `communicator_n2_all_gather_exchange_test` runs real two-rank worlds: contents of both slots byte by byte, the in-place alias, ordering against broadcast, all-reduce, barrier and point-to-point traffic, descriptor/thread stability over thousands of collectives, abort, a silent peer, a peer that exits, and destruction with an outstanding AllGather.

## Performance

On one Mac and Linux pair connected by Thunderbolt 4 (host memory, CUDA with Metal-shared memory), AllGather latency fell by roughly 20 to 75 percent from 64 bytes to 16 MiB, and the simultaneous point-to-point control did not change. The gain depends on the payload size, the memory kinds and the host; on a single host over loopback the benefit is smaller, and above a few MiB it can be negative because both directions then compete for the same memory bandwidth. Measure your own workload. The benchmark method is in [benchmark-methodology](benchmark-methodology.md).
