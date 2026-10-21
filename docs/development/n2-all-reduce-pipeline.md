# Two-rank AllReduce pipeline

With two ranks, `Communicator::all_reduce()` normally runs sequentially: the root (rank 0) receives the whole contribution, reduces it, then sends the whole result, and the other rank sends everything before it receives anything. The two directions of the link are never busy together, and the reduction sits between them. Setting `TBCCL_ALLREDUCE_PIPELINE_BYTES` (opt-in, off by default) cuts the payload into chunks so the stages overlap. Nothing changes when it is unset.

## What runs

The collective job runs on the collective executor thread, as before. The payload is cut into chunks of a whole number of elements (a multiple of 64 bytes, so a reduction never splits an element). Each rank keeps one direction admitted up front and releases the other chunk by chunk, all on the collective worker's two independent lanes (`shared_lane = false`):

* **Root.** The receive of every chunk of its scratch buffer is admitted at once. The executor waits for chunk *k*, reduces exactly that range with the provider's `reduce_backend_range()`, and then admits the send of the result of chunk *k*. Later chunks keep arriving while earlier ones are reduced and sent.
* **Non-root.** The send of every chunk of its input is admitted at once. The executor waits for the send of chunk *k* to finish and then admits the receive of the result of chunk *k*, into the same range (in place).

No thread, connection, frame or public API is added. The bytes on the wire are the same as in the sequential schedule: the same number of bytes, in the same order, in each direction.

## Which payloads pipeline

`TBCCL_ALLREDUCE_PIPELINE_BYTES=<n>` sets the largest chunk. A smaller payload gets smaller chunks (an eighth of it, but not below 32 KiB), so every pipelined payload has at least eight chunks, and never more than 16384 however small the setting (each chunk costs about half a kilobyte of request state while the call runs). These keep the sequential schedule, byte for byte as before:

* payloads under 256 KiB (a few chunks overlap too little to pay for the extra requests; measured neutral to slower there);
* a buffer larger than `count` elements (the sequential schedule moves the whole buffer but reduces only `count` elements) and `count == 0`;
* a root whose memory provider cannot reduce a sub-range (decided before any data moves).

## Compatibility with a rank on the other schedule

No negotiation is needed, and the two ranks may use different values or leave it unset on one side. The reason is that neither schedule waits for the other's:

* The root's receive lane never waits for a reduction or for the send lane, and the executor does not wait for a result send before the last chunk. A peer that sends everything before it receives (the sequential non-root) only fills the root's socket buffers while the root keeps receiving.
* The non-root never waits for a receive before its sends are admitted. A peer that returns nothing until it has everything (the sequential root) is waited for only after this rank's own sends were admitted.

This is tested with ranks on different schedules and chunk sizes in separate processes (`n2_pipeline_mixed_test`), against a model of a sequential peer in the unit test, and on the physical link with the previous release's binary on one side. A rank on the pipelined schedule opposite a sequential peer is correct but gains little, and the non-root can be somewhat slower than the sequential schedule in that pairing (it issues many small receives for a result that arrives all at once), so enable it on both ranks.

## Ordering and buffer ownership

* **Reduction before result.** A result range is posted only after the matching contribution range was received and reduced.
* **In place.** The non-root's receive of a range is not even admitted until the send that reads that range has finished. This is this rank's own ordering and does not depend on the peer returning bytes only after it has received them (and it is visible to ThreadSanitizer, which cannot see ordering that only exists on the wire).
* **One collective at a time.** The job returns only after every admitted transfer is terminal, so no later collective can begin on the collective connection, and the provider and scratch outlive every transfer. Point-to-point traffic uses its own connection and is unaffected.
* **Device memory.** The CUDA reduction runs on the default stream and synchronises with the staging copies; the ranges that are reduced, staged out and committed at the same time are disjoint.

## Failure handling

The policy is the AllGather exchange's (see [n2-all-gather-exchange](n2-all-gather-exchange.md)), and the abort helper is shared:

* **Every admitted transfer finishes** before the job returns or throws, whatever threw (a transfer, the reduction, an admission, formatting an error).
* **The first failure aborts** the communicator with the non-blocking abort request (never `Communicator::abort()` from the executor thread, which would wait for itself). If building the reason or the request throws, the transport and the worker's lanes are interrupted directly.
* **The first cause is reported**, with the failing leg and chunk (`transport_error: all_reduce recv chunk 3: ...`). A transfer that only failed because of the abort does not replace it. A failure of the reduction is rethrown as it was.
* **Partial admission** (a chunk cannot be admitted) aborts the exchange, because the peer would wait for the missing chunks.

## Tests

* `n2_pipeline_unit_test` drives each rank over a controlled transport with fake direct and staged backends and a real duplex worker: causal gates prove that receive, reduction and send overlap (the sequential algorithm is the negative control), that no leg queues behind the other against a sequential peer, and (non-root) that a receive never starts on a range whose send has not finished. It injects failures into every leg at the first, middle and final chunk, the staged copies, the reduction, admission, the abort notification and cleanup, and checks that nothing is running and nothing touches the buffers afterwards.
* `communicator_n2_all_reduce_pipeline_test` runs real two-rank worlds: every datatype with partial final chunks, out-of-place host buffers, ordering against other collectives and point-to-point traffic, resource stability, abort, a partial admission, a silent peer, a peer that exits and destruction.
* `n2_pipeline_mixed_test` runs ranks on different schedules and chunk sizes as separate processes and fails if they do not finish.
* `communicator_n2_all_reduce_cuda_test` (CUDA builds) repeats contents, abort, peer exit and destruction with device memory and checks with a sentinel that nothing touches a buffer after its Work is terminal.

## Performance

The method is in [benchmark-methodology](benchmark-methodology.md).

On one Mac and Linux pair connected by Thunderbolt 4, with `TBCCL_ALLREDUCE_PIPELINE_BYTES=131072` on both ranks, five independent process-pair launches per arm in each orientation (interleaved with the sequential schedule), median latency of an in-place Float32 sum:

| payload | Host / Host | CUDA / Metal-shared |
|---|---|---|
| 256 KiB | -19% to -29% | -25% to -28% |
| 512 KiB | -35% to -38% | -34% to -49% |
| 1 MiB | -41% to -46% | -64% to -65% |
| 4 MiB | -49% to -53% | -64% to -66% |
| 16 MiB | -42% to -45% | -56% to -60% |

With the existing opt-in staged chunking (`TBCCL_STAGED_CHUNK_BYTES=131072`) already enabled on the sequential side, the CUDA / Metal-shared figures at 1 to 16 MiB are -35% to -55%. Payloads under 256 KiB run the unchanged schedule; the differences measured there (within about 10%) are launch-to-launch noise, as the unchanged AllGather control shows. Stage timing (`TBCCL_ALLREDUCE_TIMING`) shows the overlap directly: at 16 MiB on Host / Host the root spends about 64% of its time with result sends in flight while contributions are still arriving.

CPU time per operation is about the same or lower (1 MiB: roughly a third lower; 16 MiB: up to 10% higher), because the operation finishes sooner on more cores at once. Voluntary context switches per operation rise, by about 2.5 times at 1 MiB and by 1.7 (Host) to 3.5 times (CUDA) at 16 MiB. No payload-sized allocation is added: the root's scratch buffer is the one it already had, and each chunk costs a small request object.

Not measured: payloads under 64 bytes on the pipelined path (they never use it), chunk sizes other than 128 KiB in the final comparison (a sweep from 8 KiB to 1 MiB on Host / Host and 64 to 256 KiB on CUDA / Metal-shared informed the choice), a single host over loopback (where both directions share memory bandwidth and the gain is smaller), CUDA / Host pairs on the physical link, and application-level throughput. A rank on the pipelined schedule opposite a sequential rank gains little and can be slower (the non-root by about 10 to 27 percent at 1 to 16 MiB on Host / Host in the pairings measured), so enable it on both ranks.

It is off by default. Enabling it by default is deliberately left for a later change, after review and wider use.
