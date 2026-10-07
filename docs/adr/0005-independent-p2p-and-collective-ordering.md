# ADR 0005: Independent ordering domains for point-to-point and collectives

- Status: accepted
- Date: 2026-10-06

## Context

Point-to-point messages and collective payloads used to share one byte stream and one worker per peer. Only P2P messages carried a header. When one rank issued a collective then a send while its peer issued a receive then the collective, the bytes of one kind were consumed as the other, producing `protocol_mismatch` errors or, for large messages, silently wrong data. The cross-rank order of the two kinds on the shared stream was decided by thread timing. The framework adapter worked around this by refusing overlapping P2P and collective calls.

## Decision

P2P and collectives are independent ordering domains. They may overlap on one communicator, to the same peer, from any threads, and their relative submission order may differ between ranks. Within each domain the existing guarantees hold: P2P is FIFO per (peer, direction); every rank issues the same collectives in the same order. A failure in either domain fails the communicator.

## Consequences

- Valid mixed asynchronous traffic is correct and does not deadlock. The adapter-level guard was removed.
- The guarantee holds through the C++ API, the C ABI and the framework adapters.
- Matching rules the application must still follow are unchanged: no tags, so P2P calls must match per peer pair; the same collective sequence everywhere.
- Mechanism: {doc}`0006-separate-connections-per-traffic-domain`.
