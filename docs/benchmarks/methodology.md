# Benchmark methodology

The rules below apply to every benchmark the suite drives. The full developer version is in [benchmark methodology](../development/benchmark-methodology.md) and the [benchmark suite guide](../development/benchmark-suite.md).

## What is timed

* **The replication unit is an independent run.** One repetition is a fresh launch of the benchmark processes (a fresh communicator, a fresh process group, a fresh engine or cluster). Iterations inside a repetition are samples of that repetition and are not independent of each other. Spread, tests and confidence statements are computed over repetitions, never over iterations or requests.
* **Warm-up, then timed iterations.** Warm-up iterations are discarded. Raw per-iteration samples are kept where the program provides them. Percentiles use linear interpolation, and a tail percentile is withheld when there are too few samples.
* **Completion means something specific.** Every record states what completion is: the call returned, the data arrived and an acknowledgement came back, or the work handle was waited on. A completion-confirmed time includes the acknowledgement's return trip, and a ping-pong is a round trip. Round trips are never halved into a one-way latency, and submission time is never reported as transfer time.
* **One clock per interval.** Durations use one machine's monotonic clock. Timestamps from different machines are never subtracted.
* **Correctness outside the timer.** Data are verified after the timed loop, or in a separate phase, never inside a timed loop. A run whose verification fails records no measurement; it is listed as a failed point.
* **Actual quantities.** Throughput uses the bytes or tokens that were actually moved or generated, not a configured maximum.

## What a result records

Each result states its payload size and what it means, the memory kinds on each rank, the topology and interconnect label, the transport, the direction and data source, the timing method, the completion semantics, the warm-up and iteration counts, the host hardware, the build identity (commit, executable hashes, build type), the TBCCL, C ABI and wire versions of every host and, for framework benchmarks, the provider and framework versions. Anything the suite could not determine is recorded as unavailable, not guessed.

## What may be compared

`compare` and the baseline registry compare two runs only when they measured the same thing: the same benchmark and operation, payload and its meaning, datatype, rank count, memory kinds, direction, timing method, completion semantics, topology, transport, hardware and options. Differences that could explain a change (a different commit, executable, build type, library version or provider) are reported next to the numbers. Pairs that are not equivalent are listed with the reason and are not compared; a labelled `--contextual` view exists for exploration. A different machine is a different benchmark environment: do not compare your numbers with someone else's as if they were the same experiment.

A difference smaller than the repeat-to-repeat spread of the baseline is not reported as a change, and a verdict needs enough independent repetitions for the exact test used (for example at least four against four for an unpaired comparison). The classifications are `improvement`, `regression`, `indeterminate`, `incompatible`, `insufficient_evidence` and `invalid_measurement`. See [Comparing runs](comparing.md).

## Observed difference versus cause

A measured difference between two programs or configurations is an observation. Calling it an overhead of one component requires an experiment that isolates that component. The suite labels comparisons by purpose (`regression`, `ab_transport`, `ab_variant`, `ab_placement`, `native_vs_framework`) so that reports do not blur the two.

## Limits that apply to every result

* A single machine pair, a single link and a handful of repetitions describe that setup only.
* Loopback runs measure the software path on one machine.
* The transport is TCP; results are not link-rate measurements and say nothing about RDMA.
* Where PCIe AER is disabled the suite reports AER as unavailable instead of reporting zero errors.
* Performance numbers are never CI gates.
