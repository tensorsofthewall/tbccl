# Phase 51 grouped-operations audit

Question: can a caller post an arbitrary pattern of asynchronous point-to-point operations through the current `Communicator` without a grouped-call API (NCCL `ncclGroupStart/End`, MPI `Waitall`-style batches), and what must a C ABI promise?

## Method

`benchmarks/communicator_posting_probe.cpp`: every rank posts K sends of B bytes to its peer **before** posting any receive, then K receives, then waits for everything (the classic Isend...Irecv exchange that deadlocks if a post blocks). A 12 s watchdog reports `BLOCKED`. Loopback, Host memory, release build (`docs/data/phase51/posting_probe.txt`).

## Result

| world | bytes per op | K=4 | K=8 | K=9 | K=10 | K=16 |
|---|---|---|---|---|---|---|
| 2 | 16 MiB | ok | ok | ok | **BLOCKED** | **BLOCKED** |
| 3 | 16 MiB | ok | ok | ok | **BLOCKED** | **BLOCKED** |
| 2 | 4 KiB | n/a | n/a | ok | n/a | ok |

Cause (read from the code, matches the numbers): each peer lane has a bounded request queue (depth 8) and `send`/`recv` enqueue **blocks** when it is full. With K sends posted before any receive and payloads larger than the socket buffers, the worker is stuck in the first send (the peer is also stuck posting), the queue fills at 1 in flight + 8 queued = 9, and the 10th post blocks the caller forever: a posting deadlock that exists independently of the algorithm layer. Small payloads fit the socket buffers, so the worker drains and nothing blocks.

## Consequences

1. Async `Work` plus independent per-peer progress is sufficient for the **internal** algorithms (they post at most one send and one receive per peer per round, and wait via `OpGroup`), and for any caller that keeps at most ~8 large operations outstanding per peer and direction.
2. It is **not** sufficient as a general contract: an unbounded "post everything, then wait" pattern (what a grouped-call API or an MPI-style batch implies) can deadlock inside the post call. A C ABI must therefore not promise non-blocking posts without bound.
3. No fix is made in Phase 51 (scope: algorithms, selector, sparse topology). Candidate for Phase 52: make `enqueue` non-blocking with an unbounded (or accounted) queue, or return a `would_block`/capacity error; alternatively add an explicit group object that posts a batch atomically. The decision belongs with the C-ABI design; the probe is kept as the regression test for whichever is chosen.
4. Collectives never post unmatched operations, so the finding does not affect Phase 51 algorithm correctness; `docs/c_abi_v1_constraints.md` records the constraint.
