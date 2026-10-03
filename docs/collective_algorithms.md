# N>2 collective algorithms (Phase 51)

`Communicator::barrier()`, `broadcast()`, `all_gather()` and `all_reduce()` keep their Phase 50 semantics. Behind them an internal **CollectivePlanner** picks the algorithm; rank 0 decides once and the verdict of the
collective descriptor exchange carries the algorithm id to every rank, so all ranks always run the same plan (a rank-local environment override cannot make ranks disagree: ranks that force different algorithms are a
`protocol_mismatch`). `world_size == 1` is local and `world_size == 2` is the specialised N=2 fast path, which no generic algorithm ever replaces. The Phase 50 root-based algorithms stay available as `reference`
(correctness fallback, test oracle, benchmark baseline).

Debug overrides (not a stable interface): `TBCCL_BARRIER_ALGORITHM`, `TBCCL_BROADCAST_ALGORITHM`, `TBCCL_ALLGATHER_ALGORITHM`, `TBCCL_ALLREDUCE_ALGORITHM` = `reference | tree | recursive | ring | dissemination | auto`.
A typo is an error; an algorithm that cannot run the collective at this world size (e.g. `recursive` at N=3) fails the collective with `unsupported:` on every rank before any data moves and poisons nothing.

## Control plane and data plane

* Every rank pair has a control connection from bootstrap (abort, Goodbye, **collective descriptors and verdicts**). The control plane is the only full mesh.
* Data connections (world_size > 2) are created **lazily** by the first send/receive over them (the higher rank dials, the lower rank's acceptor installs it). A collective therefore only needs the data edges its algorithm
  uses; the "required edges" below are exactly what the tests assert. world_size 2 keeps its eager data connection.
* Per rank at world size N: control sockets N-1, control watcher threads N-1, and per *used* data peer one socket and one duplex worker (2 threads); plus one executor thread and (N>2, rank < N-1) one data acceptor thread.

## Algorithms

| collective | algorithm | world sizes | rounds / critical path | data per rank | required data edges | datatypes | notes |
|---|---|---|---|---|---|---|---|
| barrier | `reference` | any > 1 | one control round trip at rank 0 | none (control frames only) | none | n/a | gather of N-1 descriptors at rank 0, verdict back |
| barrier | **`dissemination`** | > 2 | ceil(log2 N) rounds | 16-byte token per round | rank +-2^k (mod N) | n/a | no coordinator round trip; descriptors go to rank 0 on the control plane and are validated as they arrive |
| broadcast | `reference` | > 2 | N-1 sequential sends at the root | root sends (N-1) x bytes | root <-> every rank | any bytes | |
| broadcast | **`tree`** (binomial) | > 2 | ceil(log2 N) levels | root sends ceil(log2 N) x bytes; a node sends (#children) x bytes | parent, children | any bytes (byte-generic: FP8, packed INT4, anything) | children served concurrently when the backend allows direct access, otherwise one by one |
| all_gather | `reference` | > 2 | N-1 receives then N-1 sends of N-1 chunks at rank 0 | rank 0 receives (N-1) x b, sends (N-1)^2 x b | rank 0 <-> every rank | any bytes | |
| all_gather | **`ring`** | > 2 | N-1 steps | (N-1) x b sent and received | predecessor, successor | any bytes | send and receive of each step overlapped; output slot q always holds rank q's input |
| all_reduce | `reference` | > 2 | 2(N-1) sequential transfers at rank 0 | rank 0 receives and sends (N-1) x bytes | rank 0 <-> every rank | Float32/64, Int32/64, Int8/UInt8 | folds ranks 1..N-1 in order |
| all_reduce | **`tree`** (latency) | > 2 | 2 ceil(log2 N) levels of whole-buffer transfers | each node receives (#children) x bytes and sends bytes up, (#children) x bytes down | parent, children (root 0) | same | fixed combination order |
| all_reduce | `recursive` | power-of-two N only | log2 N rounds | log2 N x bytes sent and received | rank XOR 2^k | same | symmetric commutative sums: identical bits on both partners; unsupported (clean verdict) at other N |
| all_reduce | **`ring`** (bandwidth) | > 2 | 2(N-1) steps | 2(N-1)/N x bytes sent and received | predecessor, successor | same | any count (uneven chunks differ by at most one element, count < N allowed); reduce-scatter reduces each received chunk through `ExternalMemoryProvider::reduce_backend_range` |

Datatypes: reductions accept the Phase 49/50 set at N>2 (Float32, Float64, Int32, Int64, Int8, UInt8). **Float16/BFloat16 SUM is rejected at N>2 before the planner runs**, whatever algorithm is forced
(`unsupported: ... N>2 reduction semantics are not defined`). Numerical contract for reordered floating-point sums: `docs/numerical_reduction_semantics.md`.

## Failure behavior (all algorithms)

A collective's `Work` is its executor job. Each round posts its child transfers and waits for **all** of them (`OpGroup`) before looking at errors, so the Work never becomes terminal while a transport thread can still
touch the caller's buffer (the Phase 41/45 guarantee). The first child error is kept, a transport/protocol failure aborts the communicator (Abort frames reach every rank over the control plane, so ranks blocked on
unrelated peers also leave), and a rank-local capability problem or an unsupported forced algorithm fails the collective on every rank without poisoning. Collective descriptor mismatches (sequence, kind, root, count,
dtype, op, bytes, forced algorithm) abort the communicator on every rank; they are detected on the control plane, independently of the data algorithm, so a mismatch cannot hang a barrier or a ring.

## Default selection

See `docs/phase51_results.md` for the measured crossovers. The thresholds are a generic local-loopback heuristic, **not** Thunderbolt-tuned constants.
