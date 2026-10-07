# Execution and progress

## Who makes progress

Nothing progresses in the caller's thread. `send`, `recv` and collectives build a small descriptor and return a `Work` in constant time; they never wait for socket progress, a matching operation on the peer, staging or queue capacity. A call either accepts the operation immediately or fails immediately (`InvalidArgument`, `Unsupported`, `ResourceExhausted`).

Per communicator:

- **Per peer, two duplex workers** (one for each data domain), each running one transfer at a time in FIFO order with a network thread per direction; a staging thread is created lazily when a transfer needs staging.
- **One collective executor thread**, running one collective at a time.
- **One control watcher thread per peer**, blocked on the control socket to notice peer loss and abort.

Per-peer queues are unbounded queues of lightweight descriptors, so "post many sends, then post the receives, then wait" works without a grouped-call API ({doc}`../adr/0008-nonblocking-submission-without-group-calls`).

## Where threads wait

| Waiter | Primitive | Woken by |
|---|---|---|
| idle worker thread | condition variable | enqueue, stop, abort |
| worker mid-receive | blocking `recvmsg` | data arriving; `shutdown` on abort |
| worker mid-send | blocking `sendmsg` | socket buffer space; abort |
| executor thread | condition variable; `Work::wait` on each child transfer | job submit; transfer completion |
| application waiter | `Work::wait` | the completing thread |

All of these are blocking waits: each handoff costs a futex wake plus a scheduler wakeup.

## Latency behavior

On a hot back-to-back loop, TBCCL software adds roughly 25 to 35 microseconds per hop on a hop of about 128 microseconds over Thunderbolt 4 (2 KiB messages). With an idle gap between operations, as in a decode loop, every handoff wakes a thread that has been asleep for milliseconds, and the cost per hop grows several-fold: the cause measured on Linux is CPU idle-state exit, not the library. A framed P2P message is already a single `sendmsg` and a single `recvmsg`; the remaining cost is thread handoff.

TBCCL deliberately includes no spin-polling or direct-send fast path: the variants measured (waiter, receive, executor polling and direct send) lowered latency in some regimes but were worse in others, none moved a real model's per-token time beyond noise, and an unconditional spin costs one to two busy cores per communicator. Any future progress change must keep submission O(1) and nonblocking, must exit every spin on stop or abort within bounded time, and must still complete every `Work` exactly once on abort, peer death and destruction.

Tracing: `TBCCL_LATENCY_TRACE=<path>` records per-operation stage timestamps (internal, off by default, free when off); `tools/latency_trace_report.py` and `tools/coll_trace_report.py` decompose them. Follow the [benchmark methodology](../development/benchmark-methodology.md) when measuring.
