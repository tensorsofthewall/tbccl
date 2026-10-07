# tbccl

TBCCL is a C++17 collective-communication runtime with a stable C ABI for
framework-neutral use, built to move tensors between heterogeneous machines
(for example Linux/NVIDIA and a Mac) over a direct Thunderbolt 4 link or any TCP
path. It is a transport and collective library only: it knows nothing about models,
placement or any ML framework.

**Version 0.5.1 (development, ordering-domain changes unreleased) - C ABI v1 - wire protocol 4** (P2P and collectives are independent ordering domains; wire-3 peers are not compatible).

- N-rank `Communicator` (control-plane mesh, lazily opened data channels, rank-0 planned collectives).
- Byte-generic `send` / `recv` / `broadcast` / `all_gather`; SUM/PRODUCT/MIN/MAX `all_reduce`
  (float16/bfloat16 reductions are two-rank only).
- Nonblocking submission: every post returns a `Work` immediately; `Work` has test / wait / wait_for.
- Structured errors (`tbccl::Error`, `ErrorCode`); the C API maps result codes, never message text.
- Memory kinds: host, CUDA (optional component), Metal shared (macOS).
- C ABI `include/tbccl/tbccl.h` (target `TBCCL::tbccl_c`): opaque handles, 16-byte unique id, caller-exchanged
  256-byte endpoint blobs (no hidden rendezvous), 28 stable exported symbols.
- Abort (communicator-wide, no recovery). No GroupStart/GroupEnd (see `docs/grouped_operations_audit.md`).

Out-of-tree consumers: [torch-tbccl](../torch-tbccl) (PyTorch backend), [vllm-tbccl](../vllm-tbccl),
[exo-tbccl](../exo-tbccl) (exo / MLX pipeline parallelism over the C ABI).

## Build, test, install

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release            # add -DTBCCL_ENABLE_CUDA=ON on Linux/NVIDIA
cmake --build build-release -j && (cd build-release && ctest --output-on-failure)
cmake --install build-release --prefix <prefix>                    # consumers link the INSTALLED prefix
```

Consumers use `find_package(TBCCL CONFIG)`; C-only consumers link `TBCCL::tbccl_c`
(`scripts/check_c_external_consumer.sh` builds and runs a pure-C consumer against an install).

## Documentation map

| topic | where |
|---|---|
| C++ API | `docs/public_api.md` |
| C ABI v1, bootstrap, thread safety, quickstart | `docs/c_abi_v1.md`, `docs/c_api_bootstrap.md`, `docs/c_api_thread_safety.md`, `docs/c_api_quickstart.md` |
| algorithms and the planner | `docs/collective_algorithms.md` |
| numerical semantics, low precision | `docs/numerical_reduction_semantics.md`, `docs/quantized_payloads.md` |
| hardware, the Thunderbolt link, safety | `docs/mac_thunderbolt_access.md`, `docs/tb4_recovery.md`, `docs/tb4_busy_poll.md` |
| contributing, repository guidance | `CONTRIBUTING.md`, `AGENTS.md` |

### Algorithm selection: `TBCCL_ALGORITHM` (synchronous `World` API)

> This section describes the legacy two-rank synchronous `World` collectives. The N-rank `Communicator`
> chooses its algorithm internally (rank 0 plans and tells every rank, see `docs/collective_algorithms.md`);
> it exposes no public algorithm controls.

`all_gather()`, `reduce_scatter()`, and `all_reduce()` each have a
centralized ("reference") implementation and an optimized ring
implementation. By default, every call resolves which one to use via
`auto`: a conservative, transport-agnostic policy based only on world
size and message/segment/tensor size — the same value on every rank,
since it never looks at host identity, interface, OS, or measured
local latency. No collective outside these three (`barrier`,
`broadcast`, `reduce`) participates in algorithm selection.

Force a specific implementation with environment variables:

```bash
TBCCL_ALGORITHM=reference ./app   # force reference for all three
TBCCL_ALGORITHM=ring ./app        # force ring for all three
TBCCL_ALL_GATHER_ALGORITHM=ring ./app
```

Per-collective overrides (`TBCCL_ALL_GATHER_ALGORITHM`,
`TBCCL_REDUCE_SCATTER_ALGORITHM`, `TBCCL_ALL_REDUCE_ALGORITHM`) take
precedence over `TBCCL_ALGORITHM`, which takes precedence over `auto`.
Accepted values are `auto`, `reference`, `ring` (case-insensitive); an
explicitly-set unrecognized value throws rather than falling back
silently. Ring AllReduce additionally requires the element count to be
divisible by the world size; `auto` falls back to reference when a
tensor is large enough to want ring but isn't divisible, while a
forced `ring` request throws that requirement's error instead of
silently using reference.

**All ranks in a `World` must use identical `TBCCL_ALGORITHM`/
per-collective override values.** The current synchronous World
protocol has no channel to negotiate or detect a mismatched algorithm
choice across ranks, so a mismatch is unsupported and may deadlock.

Current `auto` thresholds — "v2", recalibrated after a persistent
ring worker thread (see below) substantially cut ring's fixed
per-invocation cost — (every unlisted world size uses reference):

```text
AllGather (threshold: contribution bytes per rank)
  N=2  : 64 B (effectively always ring above the zero-size fast path)
  N=3  : 64 KiB
  N=4  : 32 KiB
  N=8  : 32 KiB

ReduceScatter (threshold: output segment bytes per rank)
  N=2  : 128 KiB
  N=3  : 64 KiB
  N=4  : 64 KiB
  N=8  : 32 KiB

AllReduce (threshold: total tensor bytes per rank; N=2 and N=3
always use reference regardless of size)
  N=4  : 2 MiB
  N=8  : 512 KiB
```

An early version of this library (v1) added a persistent, World-owned
ring worker thread that replaced a per-invocation `std::thread`
create/join with a reusable one, cutting ring's fixed overhead
substantially. The original thresholds above (set before that change)
had gone stale as a result — several were one to two orders of
magnitude too conservative — so they were recalibrated using both
median *and* p95 (tail) latency, not median crossover points alone: a
size where ring's median improves but its p95 gets meaningfully worse
does not automatically get a lower threshold. AllGather at N=2 changed
policy outright (was always reference; real-hardware A/B testing now
shows ring consistently faster, including at the smallest tested
size, with no p95 regression). AllReduce's thresholds are unchanged —
recalibration measurements did not find repeatable evidence to lower
them; composing two ring phases per call (reduce-scatter then
all-gather) appears to need a larger message to amortize than either
phase alone.

These thresholds come from local-loopback and real Thunderbolt A/B
benchmark data gathered so far and are deliberately conservative (a
missed ring opportunity is preferred over a regression) — they are
**performance heuristics, not an API/ABI guarantee, and may change
between versions** as implementations and benchmark data improve.
Applications that need a deterministic implementation choice should
use the explicit overrides above rather than relying on the current
thresholds.

### macOS Local Network permission

On macOS, `tb_pingpong` requires Local Network access.

If the client connects from Linux to macOS but stalls, or macOS returns
`No route to host` despite working ping/routing, check:

System Settings → Privacy & Security → Local Network

and allow `tb_pingpong`.

When developing over Remote SSH, macOS may not present the permission
dialog in the remote session. A local GUI session may be required to
approve it.

### Low-latency mode: `--busy-poll`

`tb_pingpong` supports Linux's `SO_BUSY_POLL` socket option to reduce
receive latency by spinning briefly on the NIC driver instead of
sleeping for an interrupt. It's disabled by default and only takes
effect on Linux; macOS ignores it.

```text
--busy-poll 0     normal blocking behavior, lowest CPU usage (default)
--busy-poll 100   historically measured low-latency setting on Linux
```

Apply it on whichever socket is on the Linux side (server, client, or
both):

```bash
./build/tb_pingpong server --bind 192.168.3.2 --busy-poll 100
./build/tb_pingpong client --host 192.168.3.1 --busy-poll 100 --mode pingpong --sizes 64
```

Historical 64 B ping-pong RTT, Mac ↔ Linux over Thunderbolt:

```text
normal          ~146 µs
busy-poll 100   ~55 µs
```

This is hardware- and system-specific (CPU, NIC driver, kernel
scheduler); treat it as a starting point to tune for your own setup,
not a guaranteed result. Busy polling can consume additional CPU on the Linux receive path.
Measure latency tails, CPU cost and bulk throughput under the same conditions
before choosing a value; see the controlled characterization below.

### Controlled TB4 latency characterization

The busy-poll sweep runner compares busy-poll settings with raw samples,
per-rank steady-state CPU accounting and read-only link-health checks. It keeps
the existing default and transport unchanged. See
[the audited semantics and reproduction procedure](docs/tb4_busy_poll.md) and
[the TB4 diagnosis/recovery procedure](docs/tb4_recovery.md).
