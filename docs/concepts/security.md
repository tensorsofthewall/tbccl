# Security model

TBCCL is designed for **trusted peers on a trusted network**: ranks that belong to one job run by one party, joined by loopback, a private network or a direct link such as Thunderbolt 4. It does not defend against a malicious peer or an attacker on the path. This page states what the current code does and does not provide.

## What TBCCL provides

| Property | Status | Notes |
|---|---|---|
| Peer authentication | **No** | A connecting peer is accepted if its handshake matches the expected communicator id, rank, world size, role and wire protocol version. Those values are identifiers, not credentials, and are sent in clear text. |
| Transport encryption | **No** | All traffic, including tensor payloads, is plain TCP. |
| Message integrity and authentication | **No** | TBCCL adds no checksum, signature or MAC to its frames. It relies on TCP. A party on the path can read and modify traffic undetected. |
| Authorization | **No** | There are no users, roles or permissions. Any rank that completes the handshake may send and receive. |
| Isolation from a malicious peer | **No** | A peer that has joined the communicator is trusted: it can send wrong data, stall, or abort the communicator. |
| Resource-exhaustion controls | **Limited** | See below. |

The communicator id is 16 random bytes from `std::random_device`. It distinguishes one communicator from another, so that a stray connection or a rank from a different job is rejected at the handshake. It is exchanged in clear text and must not be treated as a secret.

## Resource use

What bounds a peer's influence on a receiving rank today:

- Connection setup is bounded by the bootstrap timeout, and a connection that does not complete a valid hello within the handshake deadline is dropped without stalling the acceptor.
- A receiver fixes the size of every receive in advance. A framed point-to-point message must carry exactly the posted byte count, or the receive fails with `protocol_mismatch`; the peer cannot make a rank allocate more memory by claiming a larger payload.
- Control-frame text and payload fields are bounded by a fixed frame size.

What is **not** controlled: there is no limit on the number of incoming connections, no rate limit, no memory quota and no limit on how long an established peer may stay silent. After setup the data-plane reads do not have an I/O timeout, so a stalled peer blocks the operation until the caller's `Work::wait` timeout, an abort, or a closed connection ends it.

## Robustness is not security

TBCCL checks the messages it parses and reports defined errors for malformed input. That is a **robustness** property: a mistaken or incompatible peer, a port scanner or another service on the port produces a clean error instead of corrupted data or a crash. It is not **security against a malicious peer**. A clean `protocol_mismatch` error says nothing about what a determined attacker who knows the protocol can do, and TBCCL has not been fuzzed or reviewed against that threat.

Malformed-input cases that the test suite covers today:

| Case | Where it is tested |
|---|---|
| Bad handshake magic or a connection that is not TBCCL | `wire_protocol_test` (garbage bytes, an old-style short hello) |
| Unsupported wire protocol version | `wire_protocol_test` |
| Rank outside the world, or a rank that connects twice | `wire_protocol_test`, `world_test` |
| Wrong communicator id or world size | `wire_protocol_test`, `communicator_mesh_test` |
| Unexpected, unknown or out-of-range connection role | `wire_protocol_test` |
| A first message that is not a hello | `wire_protocol_test` |
| A hello that is cut short, or never arrives | `wire_protocol_test` |
| A control frame that is cut short by a close | `wire_protocol_test` |
| An unknown control frame type | `wire_protocol_test` |
| A point-to-point payload length that differs from the posted receive | `communicator_p2p_nrank_test`, `mixed_domain_ordering_test` |
| A peer that closes unexpectedly | `tcp_transport_test`, `world_test` |

Behavior that is defined in the code but has no dedicated test: a point-to-point frame with a wrong magic number is rejected with `protocol_mismatch` (the length case above is tested). A fuzzing campaign over the handshake and frame parsers has not been done; it is future hardening work.

## Supported deployment

- Run all ranks of a job inside one trust domain. Treat every rank as able to read all the data of the job and to disrupt it.
- Bind each rank's listeners to the address of the intended link. The C ABI binds to `127.0.0.1` unless the caller sets `bind_host`; a wildcard bind is accepted only together with an explicit `advertise_host`. Do not expose listeners to untrusted networks.
- Use host firewall rules to limit who can reach the listener ports.
- If traffic must cross a network you do not control, run it inside an authenticated and encrypted tunnel (a VPN or an encrypted overlay). TBCCL does not provide one.

Authentication and encryption are not part of the current design.

## Reporting a vulnerability

See the security policy (`SECURITY.md` in the repository, also reproduced under [security policy](../development/security-policy.md)).
