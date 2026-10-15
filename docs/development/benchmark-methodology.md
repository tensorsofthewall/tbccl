# Benchmark methodology conventions

## Never put expensive verification inside a timed streaming loop

A permanent convention for every benchmark in this repository that does
a timed loop of network transfers.

**Rule**: a timed measured loop must never include full byte-pattern
verification, an expensive destination fill, or pattern recomputation
between rounds. Instead:

1. An (untimed) warmup loop.
2. A measured loop with **no correctness checking** between rounds.
3. One additional, explicitly untimed, fully-verified transfer after
   the measured loop.

**Why this matters**: a receiver's ack for round N is typically sent
only after its own local work for round N completes. If the receiver
does real CPU work (a fill, a per-byte pattern check) *before*
re-posting its next `recv()`, and the payload is large enough to
exceed typical TCP window/socket-buffer sizes, the *sender's* next
`send()` call will block on that delayed receiver. This silently
inflates every measured round's completion time for reasons that have
nothing to do with the thing actually being measured (transport,
staging, or scheduling cost) -- it manifests as TCP backpressure fed
back through the timing region.

This was found and fixed in `benchmarks/async_transfer_bench.cpp`,
where a per-round `std::fill()` + full byte-pattern verify inflated
measured completion time by tens of milliseconds and was mistaken, for
a long time, for a scheduling/thread-execution-context
regression. `benchmarks/tensor_transfer_bench.cpp` (the original
synchronous benchmark) never had this problem -- it always verified
exactly once, in an explicitly untimed round after its whole measured
loop, which is the convention above.

**Before trusting timing data from any new benchmark**, check it
against this convention:

- grep the benchmark for `verify`, `fill`, `checksum`, or pattern
  generation/comparison code, and confirm none of it runs inside the
  timed measured loop.
- If in doubt, run the benchmark once with verification entirely
  disabled (or moved outside the timed region) and confirm the timing
  does not change meaningfully -- if it does, verification was
  contaminating the measurement.

## Reporting rules for any TBCCL measurement

Added with the shared harness `tools/benchmark_allreduce.py`, which follows the rule above and writes a machine-readable result file.

1. **Say what it was.** Record the TBCCL, adapter and framework versions, the platform, the backend, the dtype, the sizes, the warmup, iteration and repeat counts, and whether the ranks shared one machine. The harness writes all of this into its JSON file; a number without it is not a result.
2. **Name the transport honestly.** TBCCL is TCP over IPv4, and over a Thunderbolt link it is IP over Thunderbolt. It is not RDMA, and results must never be described as RDMA or compared to RDMA figures.
3. **Loopback is for comparing builds, not for claiming speed.** Two processes on one host measure that host's software and memory overhead. They can show that a change made a loop faster or slower; they say nothing about a network.
4. **Compare like with like.** The baseline runs the same sizes, the same loop, the same machine and the same moment (for example `--backend gloo`), not a figure from another setup or a vendor's page.
5. **Report spread, not one number.** Use the median of several timed loops and keep the minimum and maximum. Repeat on another day before believing a difference.
6. **Physical runs follow the hardware rules.** Check the link's health and the GPU's thermal state before timed runs and the link again afterwards; stop on new errors. A throttled GPU or a degraded link invalidates the numbers, and a run that stopped early is reported as stopped.
7. **No numbers from verification loops or from runs that failed.** A result file whose verified `all_reduce` did not match is a failure, not a slow result.
