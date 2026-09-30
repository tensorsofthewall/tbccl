# Phase 20 busy-poll audit and measurement procedure

The library default remains **0**. This phase measures the existing control;
it does not select a new default automatically.

## Inspected code path

`tensor_transfer_bench.cpp` parses `--busy-poll` as an integer in [0, 1000000]
and passes it through `TcpWorldOptions::tcp.busy_poll_us`. `create_tcp_world`
passes those options to `tcp_listen` and `tcp_connect`. In `src/transport/tcp.cpp`:

- On Linux builds with `SO_BUSY_POLL`, a positive value becomes a
  `setsockopt(SOL_SOCKET, SO_BUSY_POLL)` request in **microseconds**.
- Options are applied after outbound `connect` and to each accepted connection.
  The listening socket itself is not configured. In a two-rank World, rank 1
  connects to rank 0; changing the source rank does not change that topology.
  Both machines create listeners during bootstrap, including rank 1.
- A zero request skips the setsockopt call. It therefore retains the socket's
  initial kernel value; it does not explicitly override a system-wide default.
  The sweep requires observed Linux values to match the requested value even at
  zero and stops if they differ.
- A failed positive Linux request throws and fails bootstrap (outgoing bootstrap
  retries remain subject to World's existing timeout). There is no fallback
  that can honestly be described as enabled.
- On macOS the transport compiles a no-op. `busy_poll_supported()` is a build
  capability check, not runtime driver capability detection. The benchmark now
  prints an explicit unsupported diagnostic for a positive request on macOS.
- Data receives remain blocking `recv` calls. There is no application polling
  loop, socket timeout change, framing change, or adaptive tuning. Both payload
  and ACK reception can be affected on the Linux endpoint.

With `--diagnostics`, the benchmark inspects its connected IPv4 TCP file
descriptors through `/proc/self/fd` and calls `getsockopt(SO_BUSY_POLL)` before
warmup. Its ordinary execution owns one connected World socket per rank. Each
record contains the descriptor, local/peer ports and observed value. Failure to
observe exactly one matching value rejects a Linux sweep result. macOS records
`unsupported`, an empty socket list and an unavailable effective value.

The observed socket value is **configuration, not proof of active driver
polling**. Phase 21 additionally samples Linux's system-wide
`TcpExtBusyPollRxPackets` before and after each run. It stayed unchanged in the
poll-0 controls and increased in matching positive-poll runs, providing kernel
counter evidence that busy-poll receive work occurred. The counter is not
socket-attributed and does not expose time spent in NAPI, so it remains separate
from requested/effective socket configuration and latency.
See [socket(7)](https://man7.org/linux/man-pages/man7/socket.7.html) and
[Linux NAPI documentation](https://docs.kernel.org/networking/napi.html).

Phase 21 also showed that the 64 KiB result depends on arrival cadence. With
Mac→Linux host transfers, poll 200 improved the zero-gap median from 226.6 to
192.1 µs in traced controls, but the benefit disappeared with requested gaps of
100–500 µs. CUDA→Metal retained a poll-200 median benefit at gaps 0 and 500 µs;
Metal→CUDA did not. Treat a busy-poll setting as workload-specific. The default
remains zero.

## Instrumentation and timing

The new flag is optional and limited to `end-to-end` and `ack-calibration`.
Existing CSV columns and the benchmark wire protocol are unchanged. Each rank
writes `TBCCL_DIAGNOSTIC` JSON lines to stderr. They contain:

- One requested/observed socket configuration record after bootstrap.
- One process user/system/wall-time record per payload, plus thread CPU time
  where supported and voluntary/involuntary context switches. CPU migrations
  are explicitly unavailable (`null`).
- On the source, the unsorted, individual completion-confirmed latency samples.

CPU snapshots use `getrusage(RUSAGE_SELF)`, local monotonic wall time and, where
available, `CLOCK_THREAD_CPUTIME_ID`. They bracket only the measured iteration
batch, excluding setup, allocation, warmup, statistics, verification and output.
There are two snapshots per batch, not per iteration.

`process_cpu_pct = 100 * (user_seconds + system_seconds) / wall_seconds`.
This is per process, not normalized to machine core count. CPU time includes
producer work in the iteration batch even in **ready** scope, whereas the ready
latency timer excludes producer work. Thus CPU percentages describe the full
steady-state workload, not just the communication interval. Kernel interrupt
work on other CPUs is not included. Short batches have coarse CPU-time rounding;
use independent repetitions and inspect absolute CPU seconds as well.

Completion-confirmed time includes destination completion and the ACK's return
trip; it is not a directly observed one-way latency. No cross-machine timestamp
subtraction is used. Ready and produce are grouped separately. The legacy
latency-floor mode still reports RTT and a labeled half-RTT approximation.
Changing source seeds and full-byte verification remain enabled. As before,
end-to-end checks a separate untimed full tensor after every size; it does not
read back every measured GPU iteration.

## Runner

`scripts/run_tb4_busy_poll_sweep.py` uses only the existing tensor-transfer
benchmark and the new diagnostic flag. Python 3.9+ is sufficient. The remote
binary must already be built; the runner never syncs, builds or fixes a machine.

Example after an explicitly authorized Mac sync/build:

```sh
python3 scripts/run_tb4_busy_poll_sweep.py \
  --mac-root /Users/ragnarok/projects/tbccl \
  --mac-binary /Users/ragnarok/projects/tbccl/build-release/tbccl_tensor_transfer_bench \
  --sizes 65536,1048576 --busy-poll 0,50,100,150,200 \
  --repetitions 3 --rank0 mac --source both \
  --output results/phase20-local/host-focused
```

Use the default size list for 64 B through 4 MiB. Default measured iterations
are 200, falling to 50 at 4 MiB; warmup is 30. Explicitly set `--repetitions 5`
for ambiguous results. Start with host-only runs, then `--backend-pair cuda-metal`
and `--busy-poll 0,50,100,200`. Start accelerator runs at 64 KiB and 1 MiB, then
cover 256 KiB and 4 MiB after checking correctness and health. Use
`--timing-scope produce` for a separately labeled control. Use
`--rank0 both --busy-poll 0,100,200 --sizes 65536,1048576` for rank reversal;
`--source` selects the physical source independently, or `--source-rank` selects
it relative to the assignment.

`--order rotate` rotates candidate order across independent repetitions within
each size/direction/rank-assignment group. `--order abba --busy-poll 0,200`
runs A-B-B-A in every independent repetition. The plan records actual execution
indices. No candidate is compared with an older session's baseline.

`--dry-run` writes the planned matrix without SSH or measurement.
`--loopback` exercises local host-only processes, records the environment as
loopback and skips physical-link diagnostics. Its `linux`/`mac` fields are logical
endpoint labels, **not measurements of two operating systems**.

Output directories must be new. Each run preserves commands, physical mapping,
source/backend/payload/poll/scope, timestamp, machine identity, Git HEAD, tracked
working-diff hash, binary SHA256, rank stdout/stderr, exit codes, CSV and health
deltas. The plan records runner arguments. `status.json` distinguishes complete,
failed and partial sweeps. Failed configurations remain as run metadata; they
are not substituted with successful-looking zeroes or silently retried.

`runs.csv` retains each successful independent run. `summary.csv` takes the
median of per-run statistics, **not percentiles of pooled samples**. It includes
min/median/p95/p99/max, throughput, both CPUs and paired median-latency percentage
improvements within each repetition. Ranges and positive-repetition counts
expose variability. Read raw samples before interpreting a p99 from 50 samples.

## Cleanup and health gates

Each endpoint runs a small supervisor with a bounded lifetime and controller
heartbeats. EOF, lost heartbeats, signals or timeout terminate only its own
benchmark process group. A failed peer closes the other supervisor's input and
waits for cleanup. The SSH client is terminated only after that grace period;
the remote deadline remains a fallback if connectivity is lost. Both rank logs
are regular files, so large sample exports cannot deadlock on undrained pipes.
An uncatchable supervisor kill is outside these guarantees.

Ports are selected by binding a temporary socket to the correct interface with
port zero, then releasing it before World binds. There remains a bind race;
that is reported as a failed run. No unrelated port owner is ever killed.

Real TB4 runs require both expected addresses, MTU 9000, Linux carrier, Mac
active bridge, bidirectional ping, discovered PCI topology and readable kernel
logs. The initial snapshots precede the first run; every run's post snapshot
is the next run's pre snapshot. A separate live journal monitor watches for
fatal/uncorrectable events, repeated recovery failures and rapid Thunderbolt
error growth, while the runner checks interface presence/carrier every 25 ms.
Snapshots and the live journal are retained even when a run fails. Monitoring
latency is bounded by scheduling and journal delivery, not hard real-time.

A boot change requires a new baseline. Known unrelated PCI BDFs, including the
NVMe root port, are retained as `other-pci`, not attributed to Thunderbolt.
Correctable growth of at least ten error-bearing journal records in a run or
monitored sweep is a conservative stop threshold (one incident can produce
several records). A same-boot increase of ten in a device correctable AER
counter also stops; any increase in its fatal/nonfatal counters stops.
For Phase 21, `--stop-on-any-tb-timeout` applies a stricter group rule: one new
Timeout continuation record or one same-boot increase in the named sysfs
`Timeout` counter stops the group. Other correctable classes are not mislabeled
as Timeout.
Snapshots retain controller-specific continuation records such as `[12] Timeout`
and raw per-device AER counters where available. macOS `netstat` link-layer
error/drop counters are also recorded; address-specific duplicate rows are ignored. The tool never writes PCI power
settings, resets hardware, or performs automatic recovery.
