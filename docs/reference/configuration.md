# Configuration

## Build options

| CMake option | Default | Effect |
|---|---|---|
| `TBCCL_ENABLE_CUDA` | `OFF` | build the CUDA memory provider component (`TBCCL::tbccl_cuda`); needs the CUDA Toolkit |
| `TBCCL_ENABLE_METAL` | `OFF` | build the Metal device backends; Apple platforms only |
| `TBCCL_ENABLE_RING_TRACE` | `OFF` | diagnostic per-thread ring event tracing; leave off for normal builds |

## Communicator options

Set on `tbccl::CommunicatorOptions` (C: `tbcclBootstrapOptions`): `rank`, `world_size`, `communicator_id`, `rank_directory` (or the two-rank `peers` shorthand), optional pre-bound `listeners`, and `bootstrap_timeout` (default 10 seconds; bounds connect, accept, handshakes and the capability exchange). The C bootstrap options are `bind_host`, `advertise_host` and `timeout_ms`; see [C API bootstrap](c-abi-bootstrap.md).

## Environment variables

None of these are required. Variables marked *diagnostic* are for debugging and measurement and are not stable interfaces.

| Variable | Scope | Effect |
|---|---|---|
| `TBCCL_TRACE=1` | `Communicator` | per-rank log of every collective: sequence, kind, bytes, datatype, peer and verdict |
| `TBCCL_LATENCY_TRACE=<path>` | `Communicator`, *diagnostic* | records per-operation stage timestamps to a file; free when unset |
| `TBCCL_BARRIER_ALGORITHM`, `TBCCL_BROADCAST_ALGORITHM`, `TBCCL_ALLGATHER_ALGORITHM`, `TBCCL_ALLREDUCE_ALGORITHM` | `Communicator`, *diagnostic* | force an algorithm (`reference`, `tree`, `recursive`, `ring`, `dissemination`, `auto`) for that collective. A typo is an error; an algorithm that cannot run at this world size fails the collective with `unsupported:` on every rank. Ranks that force different algorithms get `protocol_mismatch` |
| `TBCCL_ALGORITHM`, `TBCCL_ALL_GATHER_ALGORITHM`, `TBCCL_REDUCE_SCATTER_ALGORITHM`, `TBCCL_ALL_REDUCE_ALGORITHM` | legacy synchronous `World` API only | choose `auto`, `reference` or `ring`. All ranks in a `World` must use identical values; a mismatch is unsupported and may deadlock |
| `TBCCL_ASYNC_TIMING`, `TBCCL_ALLREDUCE_TIMING` | *diagnostic* | print stage timing of asynchronous transfers or the two-rank all_reduce to standard error |
| `TBCCL_ASYNC_NO_STAGING_THREAD` | *diagnostic* | disable the staging thread; staged transfers then fail with `unsupported:` |
| `TBCCL_TEST_WORLD_PACE_MS` | tests only | pace multi-rank test setup to avoid ephemeral-port exhaustion on macOS |

The Communicator exposes no public algorithm controls: rank 0 plans each collective and tells every rank, so ranks cannot disagree ([Collectives](../concepts/collectives.md)).

## Ports

For the two-rank `peers` shorthand, each rank's data endpoint is the same host at the control port plus 1000; keep both free. With a `RankDirectory` every endpoint is explicit.
