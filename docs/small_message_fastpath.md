# Small-message fast path: investigation outcome (Phase 55)

**No small-message fast path is retained in the runtime.** This page records what was examined and why, so a later phase does not repeat it.

## The framing path is already minimal

A framed P2P message is one 16-byte header and the payload in a single `sendmsg` (two iovecs) and, on receive, a single `recvmsg` (two iovecs) with the header validated as soon as it arrives. There is no chunk planner, staging pool,
or scratch allocation on the Host/Metal-shared direct path (`chunk_hint == 0 && supports_direct_transport_access()`), no per-frame allocation in the transport, no request/ack/control round trip. Syscalls per message: 1 (`strace`).
The plan's candidate "one frame, one descriptor, no chunk loop, direct buffer send/recv" is what the code already does; the remaining cost is thread handoff, not framing (see `progress_model.md`).

## What a fast path could still remove, and how much

Per hop on the real link in a HOT loop (2 KiB): about 4-8 us (caller -> lane thread wake), 7-10 us (completion wake), 3-4 us (submit/admission). The measured pure-TBCCL numbers over TB4: P2P ping-pong 256 us (64 B - 16 KiB), AllGather 172/324 us (2/16 KiB),
decode chain 273/379 us. Direct send (the only variant that costs no CPU) would remove roughly the enqueue + lane wake from the send side only: ~10 us of a ~128 us hop. Receive-side and completion handoffs need spinning.

## The cold-wake regime (added after the cadence measurement)

The hot-loop numbers above are a lower bound. With milliseconds of idleness between operations, every thread handoff pays a cold wake-up: loopback P2P 16 us (hot) -> ~550-640 us (4 ms gap); decode chain 27 -> ~850-895 us. In that regime the handoffs
a fast path removes are worth far more than in a hot loop (direct send + waiter + receive spin: P2P -64%, chain -36% at ~0.1-0.2 average busy cores), but the results at a 1 ms gap are mixed, the collective executor's handoffs were not addressed, and nothing
was validated on the link or the Mac. See `phase55_results.md`.

## Thresholds

Not derived: no variant was validated on the real link, so no crossover size was fixed. Loopback (pinned) direct send wins at 64 B-2 KiB, ties at 16 KiB and was unstable at 64 KiB (blocking `sendmsg` on the caller thread);
that is evidence for a small threshold (<= 16 KiB) only, not a decision. Collectives need no separate threshold decision: they use the same lanes.

## If revisited

Reproduce with commit `3fe9a3b` (`git show 3fe9a3b`), `benchmarks/small_message_latency.cpp`, `tools/latency_trace_report.py` and `TBCCL_LATENCY_TRACE`; validate with interleaved A/B on the real link under the AER gate before keeping anything.
