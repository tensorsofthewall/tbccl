# Cold progress model: which threads sleep, which wakes are cold, what polling does

Phase 56. Complements `docs/progress_model.md` (Phase 55, the hot path and the thread inventory). Raw data and scripts: `docs/data/phase56/`.
Everything here is the Host-memory N=2 path; the experiment switches described below were removed again (commits `2b76e64`, `55e200f`, reverted by `8de589b`, `90cf977`) and exist only in git history.

## Who sleeps and who wakes whom (one rank, N=2)

| thread | sleeps in | woken by |
|---|---|---|
| caller | `TransferWork::wait` (condition variable) | the thread that completes the Work (a lane worker, or the executor for a collective) |
| CollectiveExecutor | `cv_.wait` between jobs; `TransferWork::wait` on each child | `submit()` (caller); the lane worker completing a child |
| lane worker (two per peer) | `queue_cv.wait` between requests; **inside `recvmsg`** for a posted receive | `enqueue()` (executor or caller); the kernel when data arrives |
| control watcher | blocking read on the control socket | peer loss / abort only |

An N=2 AllGather on one rank is a chain of **cross-thread wakes**: caller -> executor, executor -> lane (child 0), lane -> executor, executor -> lane (child 1), lane -> executor, executor -> caller, plus the kernel waking the lane thread blocked in `recvmsg` when the peer's data arrives. `docs/phase55_latency_audit.md` has the exact call chain. The Phase 56 trace (`TBCCL_LATENCY_TRACE`, sites 11-15; `tools/coll_trace_report.py`) timestamps each of these on one process's clock.

## Where the cold penalty comes from (Linux)

`docs/data/phase56/cold_decomposition.md`, 2 KiB N=2 AllGather, medians in microseconds (trace ON, so absolute values include the trace's cost; the deltas are the point):

| stage (rank 0) | hot | 1 ms gap | 4 ms gap |
|---|---:|---:|---:|
| submit -> executor running | 5.0 | 11.6 | 94.8 |
| child 0 posted -> lane worker dequeued | 3.7 | 8.1 | 76.2 |
| child 0 dequeued -> terminal (recv; includes the peer's whole path) | 14.1 | 26.6 | 170.0 |
| child 0 terminal -> executor observed | 5.0 | 10.2 | 85.1 |
| child 1 posted -> lane worker dequeued | 3.7 | 7.2 | 55.3 |
| child 1 dequeued -> terminal (send) | 7.5 | 15.0 | 65.8 |
| child 1 terminal -> executor observed | 5.1 | 10.3 | 79.8 |
| collective terminal -> caller awake | 4.4 | 8.8 | 20.5 |
| whole collective | 52.9 | 109.8 | 638.5 |

Every one of the 6-8 wakes costs 4-5 us hot and **55-95 us after a 4 ms idle gap**: the penalty is uniform across the handoffs, not concentrated in one. The executor's own wakes (entry and the two child observations) are about 260 us of the 640; the lane threads' wakes are about 130 us; the recv socket stage (kernel wake plus the peer's path) is the rest. So there is no single dominant handoff: the cold chain is the sum of many wakes that are each 10-20x slower than when hot.

**Mechanism: CPU idle-state exit, not the library.** The machine uses `intel_idle` with the `menu` governor and states POLL, C1E (2 us exit), C6 (220 us), C8 (280 us), C10 (680 us), residency targets 4/600/800/2000 us. `docs/data/phase56/idle_state_counters.txt` counts entries over cpus 2-7 for 1000 decode chains: hot (28 us each) 117 entries into C6/C8/C10; after a 4 ms gap (843 us each) 18,912, about 19 per chain, i.e. nearly every wake lands on a core in a deep state. The cliffs of the fixed-gap curve (about 100 us, 1 ms, 2 ms and 4 ms) line up with the governor's residency targets. TBCCL's wakes are ordinary futex/kernel wakes; what makes them slow is the core they land on.

## The same decomposition on macOS

Same tool, Mac mini, normal scheduler (no affinity API): each wake costs 1-3 us hot and **5-16 us** after 1 or 4 ms; the whole 2 KiB AllGather goes from 47 us hot to 141 us (1 ms) and 194 us (4 ms), and of the 147 us cold delta about 75 us is the recv socket stage (the blocked `recvmsg` thread being woken by the peer's data). The Mac cold curve (`cold_curve_tables.txt`, timer-calibrated, see `docs/phase56_cadence_audit.md`) steps at 250 us (chain, AllGather) and 500 us (P2P), then rises slowly: P2P 28 us hot, 130 us at 500 us, 166 us at 2 ms, 206 us at 4 ms, 246 us at 8 ms. That is roughly a third of Linux's penalty. Tails are heavy at 4 ms and above (p95 0.8-6 ms), which is the Mac's timer/scheduler slack acting on the peer's wake-up from its own sleep, not a TBCCL cost; medians are the comparable number.

## What operation-scoped polling can and cannot reach

The plan's principle: consume CPU only while a communication operation is outstanding. Under that constraint a poll can only replace a **block that starts inside an operation** by a spin:
* the caller's wait for its Work (`TransferWork::wait`);
* the executor's wait for a child transfer (an N=2 AllGather has two);
* a posted receive's `recvmsg` (poll the socket before blocking);
* the lane worker waiting for the next request just after finishing one.

It cannot reach a wake of a thread that was idle *before* the operation began: the executor's entry wake (`submit -> running`), a lane thread's wake for the first child of an operation, or any wake that happens after the peer's multi-millisecond compute (the receive that completes when the peer's data arrives). Those are exactly the wakes onto cores that went deep during the gap, and spinning through the gap is not allowed.

### The experiments (each independent, then combined)

Environment-gated switches (default off, no public counterpart), 2 KiB/16 KiB, Linux loopback, ranks pinned to 3 or 4 cores each, three interleaved rounds: `variants_tables.txt`.
* **W** (caller polls the terminal flag for X us before the condition variable), **E** (the executor does the same on its child waits), **RX** (a posted receive polls the socket with `MSG_DONTWAIT` for X us), **TX** (an idle lane worker spins X us after its last request), **E+W**, **E+W+TX+RX**, and the Phase 55 **direct send** kept strictly as a control.
* **Hot loop**: each lowers hot latency (P2P 16 -> 12 us with W; chain 30 -> 17 us with all four) but at 1.4-2.5 busy cores in the hot loop (base 0.7).
* **4 ms gap** (the real-cadence range's upper end): the all-four combination gives P2P -52% and chain -41% at 0.16-0.25 busy cores, but AllGather +6 to +12%; W alone: P2P -33% at W=1000 us, AllGather +/-; E alone: chain -17%, AllGather +16% at 250 us, -17% at 1000 us.
* **1 ms gap**: **most variants are worse**, AllGather +100% to +400% (base 114 us, e.g. 450 us with E+W), chain +40% to +230%. This is not CPU contention: it persists with pinning to four cores and with executor polling alone. The cause is the governor: with polling, the cores' recent idle intervals look long and regular, `menu` selects deeper states, and the wakes that polling cannot remove get slower. At 1 ms gap, executor polling shifted entry counts from C1E (11,846 -> 4,657) to C10 (2,797 -> 4,168) and the median from 115 to 482 us (`idle_state_counters.txt`). **Polling changes the idle pattern and can make the remaining wakes slower.** A fast path that is a win at one cadence and a loss at a neighbouring one is not a safe default.
* **TX spin** additionally made `communicator_nrank_failure_test` time out (300 s) when enabled, so it fails the semantics gate regardless of speed. All other switches passed that test and `communicator_allreduce_ring_test` one at a time; with all of them on at `-j4` two tests timed out from CPU oversubscription.

### Executor-inline (the structural candidate)

If the polling cannot remove the entry wakes, remove the handoffs themselves: let the executor thread run its N=2 child transfers (send and receive, at most 64 KiB) on an idle lane instead of posting them to the lane worker and blocking (`TBCCL_P56_EXEC_INLINE`, commit `55e200f`). It needs no spinning and no extra CPU (0.05-0.12 busy cores cold, same as base), submission still never blocks (the executor, not the caller, blocks in the socket call), and the lane FIFO order is kept by an `inline_busy` flag taken under the lane mutex.
* Hot (gap 0): chain -19%, AllGather -30%. 4 ms gap: chain -27%, AllGather -32%, P2P unchanged (it never touches the executor).
* **Between 750 us and 2 ms it is worse** (AllGather 468 us vs 249 us at 1.5 ms; 136 vs 116 at 1 ms; recovers by 2 ms): the same governor effect. The curve is in `executor_inline_gap_curve.jsonl`.
* Replaying the recorded Linux cadences with a sleeping application: **-3.1% to -4.4% of the decode step** (about 0.2-0.28 ms of 6.3-9.3 ms), zero extra CPU, tight (range +/-0.03 ms), on all three Linux-involving profile pairs. On the Mac replay: -0.4% to +0.1%.
* Replaying the same cadences with a spinning application (the real decode keeps 1.3-1.9 cores busy per rank while waiting for the GPU; the sleeping replay sat at 0.02-0.07): -2.2% to -3.4%.
* **Real model, Linux loopback, 20 interleaved runs per variant, tokens identical to the unsharded reference every time:** base 5.920 ms mean TPOT (sd 0.28), INL 5.891 (-0.5%, paired difference -0.03 ms, se 0.10), E+W polling 5.909 (-0.2%, se 0.09). The 0.2-0.28 ms the replay predicts is about two to three standard errors below what this run could resolve, and the measured effect is not distinguishable from zero.

## Why the replay predicts more than the real model delivers

Both replays and the real run were checked against each other. The sleeping replay reproduces the real step to within 3-10% (Linux 6.3 ms vs 5.7-6.1; Mac 10.9 vs 11.0-11.2) but not the real thread behavior: the real process keeps 1.3-1.9 CPU cores busy per rank during decode (the CUDA wait spins; sampled once a second over a 900-token run), the replay 0.02-0.07. A busy application thread keeps part of the package out of the deepest states and gives the scheduler warm cores to wake onto, which is why the spin-mode replay predicts a smaller gain (2.2-3.4%) and the real model shows none beyond noise. **The recorded gaps alone do not determine the cold penalty; what the application does during them does.** Whether the real Linux<->Mac pair behaves like loopback is untested: the TB4 link was not used this phase beyond the Phase 55 follow-up numbers (hot 128 us; 4 ms gap 471 us with the Mac as rank 0 and 1029 us with Linux as rank 0, 2 KiB P2P round trip, one run each, both ranks idling the gap; `docs/phase56_results.md` and `docs/data/phase56/tb4/`).

## CPU cost and idle behavior

* An idle communicator (30 s) uses 0.0000 busy cores, with and without every experiment switch on (`small_message_latency --modes idle`).
* During the compute gap the TBCCL threads are asleep in every variant: the fixed-gap runs show 0.02-0.07 busy cores at 4-16 ms gaps for the base and 0.05-0.25 for the polling variants (their budget is spent only while an operation is outstanding).

## Conclusion

The cold-cadence penalty on Linux is real, per-wake and uniform, and it is the cores' idle-state exit. It is large in a sleeping benchmark and in a fixed-gap sweep and small in the real decode. No operation-scoped progress strategy found here is a win at every cadence, and none moves real-model TPOT beyond noise; the only no-CPU-cost variant (executor-inline) is the one worth revisiting if a real Linux<->Mac measurement shows the wakes matter (it removes 4 of the 8 handoffs of an AllGather without spinning). Nothing is retained in the runtime. See `docs/phase56_results.md`.
