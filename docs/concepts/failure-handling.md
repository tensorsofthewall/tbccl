# Failure handling

## Errors

The C++ API throws `tbccl::Error` (derived from `std::runtime_error`), which carries an `ErrorCode`. `Work::error_code()` is the structured terminal result of an operation; `Work::error()` is its human-readable text, formatted as `tag: detail` (for example `invalid_argument: ...`, `unsupported: ...`, `protocol_mismatch: ...`, `transport_error: ...`, `peer_failure: ...`). Codes are chosen where the failure happens. The C layer maps codes, never message text.

| `ErrorCode` | Meaning |
|---|---|
| `Success` | no error |
| `InvalidArgument` | a bad argument; rejected at submission |
| `Unsupported` | an unsupported datatype, op, memory kind or world size |
| `TransportError`, `PeerFailure` | socket or peer failure |
| `Timeout` | a bootstrap or runtime timeout (never a caller's `wait_for` expiring) |
| `DeviceError` | a CUDA or device failure |
| `InternalError` | an unexpected exception or violated invariant |
| `Aborted` | the communicator was aborted |
| `ProtocolMismatch` | ranks disagree (communicator id, world size, wire version, collective descriptor) |
| `ResourceExhausted` | an admission resource was unavailable |

The C ABI result codes and their mapping are in [C ABI v1](../reference/c-abi.md).

## Which failures poison a communicator

- A **rejected submission** (bad argument, unregistered memory kind, `ResourceExhausted`) does not poison the communicator.
- A **rank-local capability problem** in a collective fails the collective on every rank with `unsupported:` and does not poison the communicator.
- Any **failure after admission** (transport error, protocol mismatch, descriptor disagreement, device failure) fails the communicator on **every** rank: `failed()` becomes true and all later operations fail immediately instead of hanging. There is no automatic reconnection.

## Abort

`abort(reason)` is communicator-wide, idempotent and safe from any thread. The aborting rank tells every peer over the control connections before it interrupts its own work; each peer aborts and relays. Every queued operation becomes terminal with `Aborted` without touching its buffer, and the active operation completes after its transport is interrupted. A `Work` becomes terminal only once no TBCCL thread can touch its buffer.

A peer that dies abruptly (its control connection closes without a goodbye) aborts the others. A clean destruction sends a goodbye, so an idle rank finishing early does not abort anyone. A rank that is alive but silent cannot be detected by the library; the caller's own timeout can abort, and that reaches the blocked ranks.

Aborting or destroying a communicator while a peer is still finishing its last collective fails that peer. Quiesce all ranks first; destroying a communicator with nothing in flight needs no abort. Destroying a communicator with outstanding operations is safe and bounded: it aborts them.

There is no recovery, shrinking or renumbering. If a communicator fails, every rank discards it and creates a new one.

## Bounded bootstrap

Connection setup, handshakes and the capability exchange are bounded by `bootstrap_timeout`. A peer that never connects, connects with the wrong identity or version, or connects only some of its connections produces a `Timeout` or `ProtocolMismatch` error that names the cause, not a hang.

## Known detection limits

- At world size 2, a collective whose element count or byte size differs between the two ranks is not reported as a mismatch (worlds of size 3 and above are descriptor-checked).
- An `all_reduce` aborted inside the world-size-2 reduction reports the generic internal-error code through the C ABI; the error text says it was aborted.
