# Benchmark suite inventory and integration design

This page is the audit that the benchmark suite in `benchmarks/suite/` is built on. It lists what each existing TBCCL benchmark measures, how it is invoked, what it prints, where its timer starts and stops, and what it can and cannot be trusted for. The suite drives these executables; it does not replace them. Nothing here is part of the libtbccl runtime API.

How to read the audit: everything below was read from the benchmark sources in this repository when the suite was written, and the six core executables were also run once on loopback to capture their real output.

## Canonical implementation per workload

| Workload | Canonical executable | Why this one |
|---|---|---|
| Raw TCP latency and throughput (transport reference) | `tb_pingpong` (`benchmarks/tcp_pingpong.cpp`) | Only benchmark with no TBCCL protocol in the path. |
| Small-message latency through the Communicator | `small_message_latency` | The only one that uses the nonblocking Communicator and records process CPU use. |
| Point-to-point tensor transfer (latency floor, streaming, device end-to-end) | `tbccl_tensor_transfer_bench` | Rich CSV, verification outside timed loops, per-stage staging breakdown, per-rank backend choice (heterogeneous). |
| all-reduce / all-gather / reduce-scatter, algorithm comparison | `tbccl_collective_bench` | The only one with `reference`/`ring`/`auto`/`ring-pipelined` and a recorded resolved algorithm. |
| Async submission versus completion, overlap | `tbccl_async_transfer_bench`, `tbccl_async_overlap_bench`, `tbccl_cuda_overlap_bench` | Integrated. |
| Hetero all-reduce (sync vs async substrate) | `tbccl_hetero_allreduce_bench` | Integrated. |
| Health snapshots of the link | `scripts/tb4_health_snapshot.py` | Read-only, no sudo, with a stop policy the suite reuses. |

`scripts/run_collective_bench.py` and `scripts/latency_sweep.py` are launchers over the executables above. The suite supersedes them for orchestration but does not modify them; their output formats are accepted by the importers.

## Per-benchmark audit

### `tb_pingpong` (TCP ping-pong and stream)

| Item | Detail |
|---|---|
| Operation | Server/client; modes `pingpong` (RTT of one message each way) and `stream` (one-way stream, completion by a final ready byte). |
| CLI | `server [--bind IPv4] [--port N] [--busy-poll US]`; `client --host IPv4 [--port N] [--mode all\|pingpong\|stream] [--sizes LIST] [--output FILE] [--busy-poll US]`. Sizes accept suffixes (`4K`, `1M`). |
| Output | Human text on stdout; CSV `mode,size_bytes,iterations,seconds,rtt_us,gbps,messages_per_second` only with `--output`. |
| Units | `rtt_us` microseconds; **`gbps` is Gbit/s (decimal, bits)**, despite the column name. For `pingpong` the figure counts payload in **both directions** (`2 * size * iterations * 8 / seconds / 1e9`); for `stream` one direction. |
| Timing boundary | `steady_clock` around the whole loop; `rtt_us` is the **mean** over the loop (`seconds / iterations`). No per-iteration samples, so no median and no percentiles. |
| Warmup | **None.** Not configurable. |
| Iterations | Not configurable: pingpong `clamp(64 MiB / size, 10, 10000)`, stream `clamp(4 GiB / size, 16, 100000)`. |
| Verification | **None.** The buffer is filled with `0xA5` and never compared. |
| Process model | The server **never exits**: it loops on `accept` until killed. The runner must terminate it after the client finishes and must not count that as a failure. |
| Platforms | Linux and macOS (`SO_BUSY_POLL` only on Linux). Host memory only. |
| Multi-host | Yes (server on one host, client on the other). Two parties only. |
| Limitations | Mean only; iteration count fixed by size; no data check; busy-poll is Linux-only. |

### `small_message_latency`

| Item | Detail |
|---|---|
| Operation | N=2 Communicator on Host memory; modes `p2p` (ping-pong round trip), `oneway` (blocks of 32 back-to-back sends, per-message time at the receiver), `simul`, `allgather`, `allreduce` (Float32 SUM), `chain`, `idle`, `replay`. |
| CLI | `--rank R --peers ip:port,ip:port --modes LIST --sizes LIST --iters N --warmup N [--cpu CORES] [--gap-us N] [--gap-mode sleep\|spin] [--label S] [--profile FILE --passes N]`. Defaults: 2000 iterations, 200 warmup. |
| Output | One JSON object per line on **rank 0's** stdout (`TBCCL_BENCH_BOTH` adds rank 1 with label suffix `/r1`): `median_us p25_us p75_us p95_us n cpu_cores sleep_ratio`. |
| Units | microseconds; `cpu_cores` is busy cores (user+system CPU / wall) for the timed loop. |
| Timing boundary | `steady_clock` around each operation including `.wait()`; the optional idle gap is outside the sample. A barrier precedes warmup and the timed loop. |
| Percentile definition | **Nearest rank by floor**: `sorted[min(n-1, floor(q*n))]`. This differs from `tbccl_collective_bench` and `latency_sweep.py` (linear interpolation). |
| Raw samples | Not emitted. Only the aggregates above. No p99. |
| Verification | One untimed verified round for `p2p` only (exit code 3 on mismatch). `simul`, `oneway`, `allgather`, `allreduce`, `chain` are **not verified**. |
| Payload semantics | `p2p`/`oneway`/`simul`: message bytes. `allgather`: bytes contributed per rank. `allreduce`: `bytes/4` Float32 elements (sizes under 4 skipped; sizes are truncated to a multiple of 4). `p2p` measures a **round trip**, not a one-way latency. |
| Connection order | Rank 0 listens, rank 1 connects. Rank 1 started first fails with `connect failed`; no retry. |
| Platforms | Linux, macOS. Host memory only. N=2 only. |
| Limitations | Aggregates only; rank-0 view; per-mode verification gaps. |

### `communicator_probe`

| Item | Detail |
|---|---|
| Operation | N=2 Communicator, one process per rank; `--ops` any of `p2p`, `simul`, `stream01`, `stream10`, `allreduce`, `allgather`, `broadcast0`, `broadcast1`, `bcastpp`, `barrier`. `--mem host\|metal\|cuda` selects the buffer kind; `communicator_probe_cuda` (built with `TBCCL_ENABLE_CUDA`) is needed for `cuda`. |
| CLI | `--rank R --peers ip:port,ip:port --ops LIST [--sizes LIST] [--iters N] [--label S] [--samples PATH] [--mem KIND] [--timeout-s N]`. `--peers` has no default, so a run always names its endpoints. Default `--timeout-s` is 1800; the process exits with status 5 when it expires. Exit status: 0 success, 2 invalid arguments (including `--mem cuda` on a build without CUDA), 3 a content check failed, 4 a device allocation failed, 5 timeout. |
| Output | One JSON object per line on **each rank's** stdout: `meta` (version, build type, units, `TBCCL_STAGED_CHUNK_BYTES` as seen by the process), `init`, `calibration`, then one line per (op, size) with `total_us`, `submit_us`, `wait_us` (`median p25 p75 p95 p99 min max mean`), CPU use, context switches. `--samples PATH` also writes raw per-iteration samples to `PATH.r<rank>.csv`. |
| Units | microseconds; sizes in bytes; clocks are per process and never compared across hosts. |
| Timing boundary | `steady_clock`. `submit_us` is the posting call, `wait_us` is `Work::wait`, `total_us` covers both. `p2p` and `bcastpp` are round trips. `broadcast0`/`broadcast1` in a free-running loop are the root's local completion, a pipelined throughput figure, not a destination-complete latency. `streamXY` reports sender-complete, acknowledged, and acknowledged minus half a calibrated 64-byte round trip. Initialisation is reported separately and is not part of any sample. |
| Percentile definition | Nearest rank by floor, as `small_message_latency`. |
| Verification | Every point runs an untimed content check before and after the timed loop (exit code 3 on mismatch); nothing is verified inside the timed loop. `barrier` also checks that it waits for a rank that enters 30 ms late. |
| Platforms | Linux and macOS; `communicator_probe_cuda` on Linux/NVIDIA. N=2 only. |
| Limitations | Not a pass/fail gate: results depend on the host, link and CPU state and are never compared against thresholds in CI. Physical-link runs are manual. |

### `tbccl_tensor_transfer_bench`

| Item | Detail |
|---|---|
| Operation | Modes `staging-only` (single process, no network), `network-only` (sender streams, one-way), `latency-floor` (RTT ping-pong, host), `end-to-end` (device-to-device with destination-completion ACK), `ack-calibration` (host-only control for the ACK cost). |
| CLI | `--mode M --rank R --peers H:P,H:P [--source-rank N] [--local-backend host\|cuda-pageable\|cuda-pinned\|metal-shared\|metal-private-staged] --sizes LIST --warmup N --iterations N [--timing-scope ready\|produce] [--busy-poll US] [--diagnostics] [--trace] [--run-id ID] [--physical-machine NAME] [--source-gap-us N]`. |
| Output | CSV on the source rank's stdout (header always printed): `mode,stage,source_rank,source_backend,destination_backend,payload_bytes,iterations,warmup,median_us,p95_us,p99_us,min_us,max_us,half_rtt_us,...,completion_confirmed_us,effective_GBps,effective_Gbps,...,timing_scope,producer_enqueue_us`. Unavailable columns are the literal `NA`. |
| Units | microseconds; `effective_GBps` decimal gigabytes/s, `effective_Gbps` decimal gigabits/s. |
| Timing boundary | `ready` (default): source generation and producer sync happen **before** the timer. `produce`: the timer includes the producer. These are different quantities and are never compared by the suite. `end-to-end` completion includes the ACK's return trip and is **not** a one-way latency. `half_rtt_us` is an approximation, never a measured one-way time. |
| Verification | `end-to-end`, `ack-calibration`, `staging-only`: one untimed full-byte verified round after the timed loop (failure aborts, nonzero exit). `latency-floor` and `network-only`: none. |
| Percentile definition | Linear interpolation (see source, `percentile` lambda). Statistic fields are computed over the iterations the source rank timed. |
| Raw samples | Not emitted. |
| Platforms / backends | Host everywhere; CUDA and Metal only when built with `TBCCL_ENABLE_CUDA`/`TBCCL_ENABLE_METAL`. Backends are chosen **per rank**, so CUDA to Metal is a normal invocation. |
| Connection order | Rank 0 listens, rank 1 connects. |
| Limitations | N=2 only; aggregates only. |

### `tbccl_collective_bench`

| Item | Detail |
|---|---|
| Operation | `--collective all-gather\|reduce-scatter\|all-reduce`, `--algorithm reference\|ring\|ring-pipelined\|auto`, `--datatype int32\|int64\|float32\|float64`, `--op sum\|product\|min\|max`. Uses the legacy synchronous `World` API. |
| CLI | `--rank R --peers HOST:PORT,... [--bind A] [--busy-poll US] --sizes LIST [--chunk-bytes N] [--iterations N] [--warmup N]`. Defaults: 100 iterations, 20 warmup; default sizes 64 B to 16 MiB. |
| Output | CSV on rank 0's stdout: `collective,requested_algorithm,algorithm,selection_reason,datatype,op,world_size,segment_bytes,input_bytes_per_rank,output_bytes_per_rank,chunk_bytes,chunks_per_segment,iterations,warmup,median_us,p95_us,min_us,max_us,algbw_GBps,busbw_GBps,busy_poll_us`. |
| Units | microseconds; decimal GB/s (1e9 bytes/s). |
| Timing boundary | Per iteration: barrier, then only the collective call. The duration used is the **maximum across ranks** for that iteration, exchanged by plain `World::send/recv` outside the timed region. Statistics are over those per-iteration maxima. |
| Percentile definition | Linear interpolation `p*(n-1)`. |
| Raw samples | Not emitted. |
| Verification | Untimed correctness gate before the loop and again after it; a mismatch throws (nonzero exit). |
| Bandwidth | `algbw = N*B/T`, `busbw = algbw*(N-1)/N` where `B` is the per-rank segment (see below) and `T` is the per-iteration median of the max-across-ranks time. `busbw` is an **estimate for an ideal ring**, not a measured bus rate: `algbw*(N-1)/N`, times 2 for all-reduce (normalised as reduce-scatter plus all-gather volume, `busbw_multiplier` in the source). For all-reduce `B` is `tensor_bytes/N`, so `algbw` is the tensor bytes over the time. The suite records these as reported and does not reinterpret them. |
| Payload semantics | `all-gather`: bytes **contributed per rank** (output is N times that). `reduce-scatter`: **output segment** bytes per rank (input is N times that). `all-reduce`: **total tensor bytes per rank**; the element count must divide the world size, which the tool enforces for **every** algorithm (checked: `reference`, `ring` and `auto` all exit with an error otherwise). The CSV columns `segment_bytes`, `input_bytes_per_rank`, `output_bytes_per_rank` carry the exact figures and are used by the suite. |
| Resolved algorithm | `algorithm` column; `selection_reason` explains `auto`. Observed on loopback at world size 2: `auto` resolved to `reference` (`auto_unmeasured_world_size`). `auto` is resolved by the pure selector, never by an environment variable. |
| Ports | Rank `i` listens on `peers[i]`. |
| Platforms | Linux, macOS; host memory. |
| Limitations | Aggregates only; legacy World API (not the nonblocking Communicator). |

### `tbccl_async_transfer_bench`

Integrated (`async_transfer`). One JSON object on stdout per rank; only the source (sender) rank reports timing. Arguments: `--rank --peers --source-rank --bytes --chunk-bytes (0 = one chunk) --depth --warmup --iterations --backend --metal-async-path adapter|direct --verify on|off --label --output`. `--backend`: `host`, `cuda-pageable`, `cuda-pinned`, `cuda-chunked` (CUDA device memory with per-chunk staging; async only), `metal-shared`, `metal-private-staged`; each rank names its own, so CUDA to Metal is a normal invocation. Rank 0 listens, rank 1 connects.

* `enqueue_median_us` is the time for `enqueue()` to return (submission only); it is the only enqueue statistic and there are no enqueue samples.
* `completion_*_us` is measured from the start of `enqueue()` until the destination's 1-byte ACK has been received (so it includes the ACK return trip, like the synchronous tool's `completion_confirmed_us`); median, p95, p99, max, mean and the raw `completion_samples_us` are printed. Source generation and `prepare_source()` are outside both.
* `gib_per_s` is **binary GiB/s over the mean** completion time; the suite converts it x1.073741824 to decimal GB/s and keeps the native figure.
* Verification (`--verify on`, default) runs once after the timed loop on the destination (exit status 1 on mismatch).
* Limits: no per-iteration enqueue samples; the Metal `direct` path applies to `metal-shared` only; the tool does not itself prove a transfer finished from `enqueue()`.

### `tbccl_async_overlap_bench`

Integrated (`async_overlap`) and labelled **CPU-compute overlap**. One JSON object (the source rank) with `compute_only_us`, `comm_only_us`, `serial_us`, `async_overlap_us`, `hidden_us`, `hidden_fraction`, `speedup`: four single-shot phases per run (no warmup, no iterations, no data verification), host buffers only, compute = a deterministic CPU hash loop on the sender (`--compute-rounds`). `hidden_fraction` is not bounded by the tool: loopback runs gave 1.41 (not physical); the adapter records the value as printed and flags a value outside [0, 1] instead of clipping it.

### `tbccl_cuda_overlap_bench`

Integrated (`cuda_overlap`). Real CUDA kernel compute on the sender (`--source-rank`, must be a CUDA build; a CPU or sleep substitute is not offered) with `--bucket-count` buckets of `--bucket-bytes`, `--compute-rounds` kernel rounds, sent with the chunked async CUDA staging (`--chunk-bytes`, `--depth`) to a receiver that does no compute (`--receiver-backend host|metal-shared`; the receiver needs no CUDA). One run measures one `--schedule`: `serial` (compute all, synchronize, then communicate all), `overlap` (compute bucket i, synchronize its stream, enqueue it, go on to bucket i+1), `compute-only`, `comm-only`. The compute stream is synchronized before a bucket is enqueued, so data dependencies hold. Output: `total_median_us`, `total_p95_us`, `enqueue_median_us`, `enqueue_max_us` (no p99, no raw samples; `--timeline-output` writes one iteration's per-bucket CSV, not integrated). `--calibrate` (no network) picks `--compute-rounds` for a target kernel time. The total ends when every transfer's `work()` completed on the sender (no destination ACK). The receiver verifies the computed bytes after the loop. Overlap is derived afterwards from the four schedules' totals; it is not a native output.

### `tbccl_hetero_allreduce_bench`

Integrated (`hetero_allreduce`). N=2 float32 SUM all-reduce; `--count` is in **elements**, `--root`, `--backend host|cuda|metal-shared`, `--algo sync|async|both`, `--chunk-bytes`, `--metal-async-path`. Both ranks print one JSON object with `{sync,async}_{median,p95,p99,mean}_us` and `{sync,async}_verify_ok`; the full result vector is compared after one untimed round. `sync` is the legacy `reduce()` + `broadcast()` over a TensorBackend's host staging pointer (CUDA means pinned staging, filled on the host and staged host-to-device after the broadcast; there is no device-to-host step), `async` is `n2_all_reduce_tensor()` over the async substrate (CUDA means `CudaChunkedAsyncBackend` device memory), so the two differ in more than the substrate. Time is per rank on its own clock (the suite reports the slower rank's median). Rank `i` listens on `peers[i]`; the async substrate uses `peers[0].port + 1000`.

### `tbccl_tensor_transfer_bench` additional modes

`staging-only` (integrated as `tensor_staging`: one process, no network; allocation, source generation and verification outside the timer; one `d2h` and one `h2d` row per size; a shared-memory backend's median can fall below the clock resolution) and `ack-calibration` (host buffers whatever `--local-backend` says; the implicit 0-byte row is kept but is not an ACK floor: over a physical link it completes in under 1 us, so it holds no network round trip; the mode's other rows equal the host `end-to-end` figures within 1% and serve as an independent repeat of the host control).

### Scripts and tools

| File | Role | Suite use |
|---|---|---|
| `scripts/run_collective_bench.py` | Launches N local ranks of the collective bench, kills all on failure. | Behaviour reproduced by the suite runner; format accepted by the importers. |
| `scripts/latency_sweep.py` | Repeated `tb_pingpong` trials into raw CSV and a summary. | Raw files accepted by the CSV importer. |
| `scripts/tb4_health_snapshot.py` | Read-only snapshot and same-boot comparison with a stop policy; no sudo. | Reused for physical preflight and before/after snapshots. |
| `scripts/run_tb4_sustained_session.py` | Sustained session with ftrace, hard-coded SSH host and passwordless sudo. | **Not used**: it assumes sudo and a fixed host, both of which the suite must not. |
| `tools/benchmark_allreduce.py` | torch-tbccl all_reduce harness (needs PyTorch). Writes `tbccl-benchmark/1` JSON. | Result files accepted by the JSON importer; not executed by the suite. |
| `tools/thunderbolt_link_doctor.py` | Read-only link diagnosis. | Optional extra check from `doctor` for a physical peer. |
| `results/*.csv`, `results/latency-v1/`, iperf3 JSON | Historical measurements. | Importable with provenance `imported`. |

## Overlaps and conflicts found

1. **Two percentile definitions** (floor index in `small_message_latency`; linear interpolation elsewhere). The suite records the definition per result and never compares percentiles across definitions.
2. **Three throughput units**: Gbit/s (`tb_pingpong`), decimal GB/s (collective and tensor benches), binary GiB/s (`async_transfer_bench`). Units are carried through unchanged and converted only by an explicit function.
3. **`tb_pingpong` reports a mean without samples**; the others report medians. The suite therefore tracks which statistic is primary for each result and compares like with like only.
4. **`tb_pingpong` pingpong throughput counts both directions**; stream counts one. These are not the same quantity.
5. **No benchmark here takes a "resolved algorithm" from the environment**; `collective_bench` resolves it itself. The suite also leaves `TBCCL_ALGORITHM`-style variables out of the child environment so a user's shell cannot change a run.
6. **Connection order**: several benchmarks have rank 1 connect without retry. The runner starts rank 0 first and waits for its listener.

## Timing-method findings (checked against the methodology page)

| Benchmark | Verification inside a timed loop? | Result |
|---|---|---|
| `tb_pingpong` | No verification at all | Compliant, but nothing is checked. |
| `small_message_latency` | One untimed round after the loop, `p2p` only | Compliant. |
| `tbccl_tensor_transfer_bench` | One untimed round after the loop | Compliant. |
| `tbccl_collective_bench` | Gate before and after the loop | Compliant. |
| `tbccl_async_transfer_bench` | Once after the loop (the older per-round fill was removed) | Compliant. |

No violation was found, so nothing in the suite needs to disable or work around a native check. The suite does not change any benchmark's methodology.

## Adapter repositories (read only)

| Repository | What exists | Relation to this suite |
|---|---|---|
| `torch-tbccl` | `examples/cross_host_collectives.py`, `examples/real_link_overlap.py`, `examples/ddp_train.py`, `examples/ddp_mlp.py`; `tools/p71_physical_ops.py`, `tools/p72_hetero_ops.py`; recorded JSON and logs under `docs/data/`. Real-link harnesses use one process per host, monotonic timestamps from one process only, and verify outside timed regions. | Not executed by the suite. The suite's own PyTorch adapter (`torch_collectives`, `torch_p2p`, `torch_async`, `torch_ddp`) runs `torch.distributed` over torch-tbccl through its own wrapper scripts. The schema holds per-host timelines and never subtracts timestamps across hosts. |
| `vllm-tbccl` | `tools/p70_perf.py` (TTFT and decode time per token against a running server), `examples/p48_bench.py`, `examples/pp_latency.py`, `tools/perf_table.py`, JSONL results. | Future adapter. These are application-level latencies, not collective timings; the schema has a free-form category for them. |
| `exo-tbccl` | About 47 files under `benchmarks/` (critical-path decomposition, `clock_sync.py` cross-host offset estimation, `compare_runs.py`, emulator profiles). | Future adapter. `clock_sync.py` shows how cross-host alignment is done; the suite does not attempt it. No exo adapter exists yet, and systematic exo measurements with the suite have not been made. |

None of these were modified.

## Integration design

```
              config (YAML) + optional local overlay
                           |
                  config.py  -> validated matrix (workload points)
                           |
 inventory.py / preflight.py  (binaries, versions, ports, link health)
                           |
                      runner.py  -- one measurement group at a time
                           |    start rank 0, wait for listener, start others
                           |    own process groups, timeouts, cleanup
                           v
              adapters/{network,core}.py   plan() argv per rank, parse() output
                           |
          schema.py: result records (one per repetition per point)
                           |
   <output>/<run_id>/{manifest.json, results.jsonl, raw/..., health/...}
                           |
        reporting/{summary,compare,plots}.py   importers/* (historical data)
```

Design decisions:

* **Adapters, not rewrites.** An adapter builds the command line for each rank, says what the executable can and cannot do (iterations, warmup, verification, memory backends, ports), and parses its existing output. No algorithm is reimplemented and no C++ was added or changed.
* **One record per point per repetition.** Repetitions are separate native invocations (a fresh process group each time), so repeat-to-repeat variation is observable. A record never merges ranks' percentiles.
* **Statistics provenance.** Every statistic is tagged `reported` (native), `computed` (suite, from raw samples) or absent. The suite never computes a percentile from medians.
* **Comparability key.** Results compare only when benchmark, operation, requested algorithm, payload semantics and bytes, datatype, rank count and order, per-rank memory backend, topology, transport, timing method, completion semantics, percentile definition and primary statistic all match. Version differences are shown, not blocked.
* **Physical runs are opt-in.** A configuration marked physical needs `--allow-physical`, a clean preflight and an unchanged before/after snapshot. A changed snapshot invalidates the run and nothing is retried.
* **No new C++ API.** The suite only uses command lines, stdout and exit codes that exist today.

## Integrated, deferred, not integrated

| Benchmark | Status |
|---|---|
| `tb_pingpong` | Integrated |
| `small_message_latency` | Integrated |
| `tbccl_tensor_transfer_bench` (`latency-floor`, `network-only`, `end-to-end`) | Integrated |
| `tbccl_collective_bench` (all-reduce, all-gather, reduce-scatter) | Integrated |
| `tbccl_tensor_transfer_bench` (`staging-only`, `ack-calibration`) | Integrated |
| `tbccl_async_transfer_bench`, `tbccl_async_overlap_bench`, `tbccl_cuda_overlap_bench`, `tbccl_hetero_allreduce_bench` | Integrated |
| Other `communicator_*` benches, bucketed all-reduce, raw transport bench | Not integrated |
| iperf3 | Detected only (baseline interface; no execution) |
| torch-tbccl, vllm-tbccl, exo-tbccl | Deferred |
