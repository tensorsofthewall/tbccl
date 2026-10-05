# Progress model of the N=2 Host path (Phase 55)

How an operation makes progress, where it can wait, and what an adaptive-spin or direct-send change would have to preserve.

## Who makes progress

Nothing progresses in the caller's thread. `send`/`recv` build a descriptor and return a `Work` (Phase 52: O(1), never waits for socket, peer, lane or staging). Per peer there are two lane threads (duplex) running one
transfer at a time in FIFO order, and one `CollectiveExecutor` thread per communicator running one collective at a time. A control watcher thread blocks on the control socket for peer loss and abort.

## Where each thread waits

| waiter | primitive | woken by |
|---|---|---|
| idle lane thread | `queue_cv.wait` (futex) | `enqueue` (`notify_all`), `stopping`, abort |
| lane thread mid-receive | blocking `recvmsg` | the kernel when the segment arrives; `shutdown(SHUT_RDWR)` on abort |
| lane thread mid-send | blocking `sendmsg` (rarely blocks for small messages) | socket buffer space; abort |
| executor thread | `cv_.wait`; inside a job `TransferWork::wait` | job submit; transfer completion |
| application waiter | `TransferWork::wait` (`state->cv`) | `complete_ok` / `complete_error` |

All of them are blocking waits that cost a futex wake plus a scheduler wakeup. Measured per hop on the real link (2 KiB, per-host trace): caller -> lane thread wake 4-8 us, completion wake 7-10 us, `sendmsg` 11-12 us,
submit+admission 3-4 us; about 25-35 us of TBCCL software on the critical path per hop of about 128 us (loopback with hot cores: ~12 us per hop). 12 futex calls per ping-pong iteration, 1 `sendmsg`, 1 `recvmsg`, no `poll`.

## What was tried (commit `3fe9a3b`, reverted in `b3aee03`, results in `phase55_results.md`)

Environment-gated, default-off experiments: `TBCCL_DIAG_DIRECT_SEND` (send on the caller's thread when the lane is idle: removes the enqueue wake and the lane wake), `TBCCL_DIAG_WORKER_SPIN_US` (idle lane thread spins on an
atomic before blocking), `TBCCL_DIAG_WAIT_SPIN_US` (`Work::wait` spins on an atomic before blocking), `TBCCL_DIAG_RECV_SPIN_US` (`recvmsg(MSG_DONTWAIT)` polling before the blocking call),
`TBCCL_DIAG_ACTIVITY_WINDOW_US`, `TBCCL_DIAG_QUICKACK`, `TBCCL_DIAG_SOCKOPTS`.

Loopback, pinned to three cores per rank, 2 KiB round trip (baseline 15.6 us): direct send -15% at no extra CPU; each single spin -15..-25% at 0.9-1.4 busy cores; all spins together -56% at 1.8 cores; QUICKACK +23% (worse).
Spinning is only a latency win because the cores are otherwise idle; pinning all of a rank's threads to ONE core makes a spinning thread starve its siblings (a 14x slowdown), so any spin needs a core per progress thread.

## Why none of it was kept

No candidate could be validated on the real link: the interleaved TB4 A/B stopped at the plan's hard stop (AER Timeout 0 -> 1 after the second variant). The hot-loop software ceiling measured on the link is 25-35 us per hop of ~128 us.
**That ceiling understates the problem**: with an idle gap between operations (the decode cadence) loopback TBCCL costs 5-30x more than in a hot loop (P2P 16 -> ~550-640 us, chain 27 -> ~850-895 us at a 4 ms gap), because each handoff then wakes a thread that has been
asleep for milliseconds. Under that cadence, direct send plus waiter and receive spin (1 ms windows) cut loopback P2P by 64% and the chain by 36% at ~0.06-0.22 average busy cores (a spin only runs while an operation is in flight). Results at a 1 ms gap were
mixed or worse, so the effect is not monotonic. An unconditional spin costs 1-2.5 busy cores per communicator and is not acceptable as a default. Nothing is retained until it is validated on TB4 and the Mac (macOS wake behavior is unmeasured).

## Requirements for any future progress change

- Phase 52: submission stays O(1) and nonblocking; no caller waits for progress.
- Every spin loop checks `stopping` and the abort flag each iteration and exits within a bounded time; peer death, explicit abort and communicator destruction must still complete every Work (terminal exactly once).
- Direct send on the caller's thread must hold the lane against the lane thread (`inline_busy` in the experiment) so a send never overlaps another send on the same socket and FIFO order holds; a failure must poison the
  communicator exactly as the lane thread would, and a blocking `sendmsg` on a full socket buffer must not run on the caller's thread for large payloads (the experiment showed unstable 64 KiB results).
- Activity-aware spin (spin only within a window after recent activity) is the only form worth considering; never an unconditional spin.

## Phase 56 addendum

The cold-wake behaviour of the hot-versus-cadence findings above was decomposed through the CollectiveExecutor, measured on macOS, and tested against the real decode cadence: see `docs/cold_progress_model.md` and `docs/phase56_results.md`. The conditional spin windows suggested here were measured as operation-scoped polling (waiter, executor child wait, posted receive, idle lane worker) and executor-inline; none is retained, because none moves real-model TPOT and the polling variants are slower at 0.75-1.5 ms gaps. The rule above that every spin loop checks the abort flag stands for any future attempt.
