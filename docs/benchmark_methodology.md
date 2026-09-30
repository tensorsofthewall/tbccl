# Benchmark methodology conventions

## Never put expensive verification inside a timed streaming loop

Established by Phase 34 (`docs/phase34_report.md`), carried forward as a
permanent convention for every benchmark in this repository that does
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
most of two phases, for a scheduling/thread-execution-context
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
