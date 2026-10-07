# Troubleshooting

## A rank fails during bootstrap

| Symptom | Likely cause | What to do |
|---|---|---|
| `protocol_mismatch: ... wire protocol ...` | two ranks, or a rank and an adapter, were built against different TBCCL versions (wire 3 versus wire 4) | rebuild every rank and every adapter against the same installed prefix ([Versioning](../reference/versioning.md)) |
| `protocol_mismatch: ... communicator` / `duplicate rank` / `rank outside the world` | ranks were given different communicator ids, the same rank twice, or a wrong world size | share one communicator id and give each process a unique rank |
| `timeout: ... waiting for the connection(s)` | a peer never connected, or connected only some of its connections | check that the peer is running and that its published endpoint is reachable from this host |
| macOS rank times out, Linux is fine | Local Network permission, application firewall, or a `nohup` job outside a live session | [Thunderbolt link guide](thunderbolt-link.md) |

Bootstrap is bounded: it fails with one of these errors within `bootstrap_timeout` instead of hanging.

## A collective fails or hangs

- Every rank must issue the same collectives in the same order. At world size 3 and above a disagreement fails on every rank with `protocol_mismatch:`. At world size 2 a count or size mismatch is not detected and can hang or corrupt; compare the arguments on both ranks.
- `unsupported:` on every rank names the rank whose memory kind or datatype cannot do the reduction ([Platform capabilities](../reference/platform-capabilities.md)).
- Set `TBCCL_TRACE=1` to log each collective's sequence, kind, bytes, datatype, peer and verdict on every rank and compare the logs.

## After a failure

A failure after admission fails the communicator on every rank and there is no recovery: discard the communicator and create a new one ([Failure handling](../concepts/failure-handling.md)). If a peer may still be finishing its last collective, quiesce all ranks before aborting or destroying.

## Performance looks wrong

Check the [benchmark methodology](../development/benchmark-methodology.md): never verify results inside a timed loop, and note that an idle gap between operations makes every thread handoff slower ([Execution and progress](../concepts/execution-and-progress.md)).
