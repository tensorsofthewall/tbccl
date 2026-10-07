# ADR 0006: Separate data connections and workers per traffic domain

- Status: accepted
- Date: 2026-10-06

## Context

{doc}`0005-independent-p2p-and-collective-ordering` needs the two kinds of traffic to be structurally unable to interfere, with no timing-dependent behavior.

## Decision

Each peer pair gets a second data connection, `CollectiveData`, served by its own duplex worker. Collective payloads use it; P2P keeps the first. The same advertised data endpoint accepts both roles and the role in the handshake decides which connection a socket is. Because the two kinds of bytes travel on different TCP connections read by different workers, a P2P receive can never consume collective bytes and the reverse. Within a domain, ordering is unchanged. The wire protocol version went from 3 to 4 ({doc}`0004-wire-protocol-compatibility`). The C ABI and C++ API are unchanged.

## Alternatives considered (as recorded when the decision was made)

| Alternative | Outcome |
|---|---|
| Separate collective connection and worker per peer pair | chosen: structural, low-moderate complexity, one extra socket per pair |
| Framed multiplexing on one connection | rejected: high complexity (receive-destination lifetime, backpressure, partial-frame abort) and head-of-line blocking between domains, to save one socket |
| A scheduler arbitrating the shared stream | rejected: arbitration by local submission order cannot make opposite relative orders legal; it would need a global protocol |
| Rejecting overlapping traffic in the core | kept only as a fallback; unnecessary once the chosen design worked |

## Consequences

- Footprint per peer pair: data sockets 1 to 2, duplex workers 1 to 2, network threads 2 to 4. The collective connection is lazy above world size 2.
- A peer that connects only one data role fails with a bounded timeout naming the missing role.
- Wire 3 and wire 4 peers do not interoperate.
