# ADR 0008: Nonblocking submission, and no group calls in C ABI v1

- Status: accepted
- Date: 2026-10-03

## Context

The question was whether a caller can post an arbitrary pattern of asynchronous point-to-point operations without a grouped-call API (like `ncclGroupStart/End` or MPI wait-all batches). A probe that posts K sends before any receive found that with a bounded per-peer queue the posting call itself could block: with payloads larger than the socket buffers, the 10th outstanding large send to one peer blocked the caller, at world sizes 2 and 3.

## Decision

Submission never waits. Every post returns a `Work` immediately and does O(1) work: it admits a lightweight descriptor into an unbounded per-peer queue, with no payload copy, staging or device scratch at admission; persistent worker threads move the data. A call either accepts immediately or fails immediately. Because individual nonblocking `send` and `recv` calls then express every explicit-peer pattern, including "post everything, then wait", the C ABI v1 does not include `GroupStart`/`GroupEnd`.

Evidence recorded with the decision: with progress gated by explicit barriers (not timing), a peer held back while 100 sends of 8 MiB were submitted still let every call return; symmetric "100 sends, then 100 receives, then wait" completed at 1 MiB and 8 MiB; 1,000 small operations each way, mixed patterns at world size 4, fairness across peers, 36 collectives admitted before any wait, and abort or destroy with 300 to 400 queued operations all passed; restoring the old queue depth made the first test block, as a negative control.

## Consequences

- Groups remain a possible future optimization (launch aggregation), not a correctness requirement.
- The remaining obligations are the application's: with no tags, matching operations must be posted in a compatible order per peer pair; every rank issues the same collectives in the same order; buffers stay valid until the operation is terminal.
