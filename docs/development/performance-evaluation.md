# Performance evaluation

This page states what has been measured with the [benchmark suite](benchmark-suite.md), on which systems, and what the evidence does and does not establish. It is deliberately short. Every number below was recomputed from the saved raw results of the campaign it came from and agrees with that campaign's analysis; nothing here is a claim about other machines, other links or other software versions. Performance numbers are never a CI gate in this project.

```{important}
All measurements were made on **one pair of machines connected by one Thunderbolt 4 cable**, on 2026-10-16 and 2026-10-17, with the software versions listed below. TBCCL's transport is **TCP over IPv4, not RDMA**. Treat the figures as a worked example of what the suite reports, not as expected performance on your hardware.
```

## A. Benchmark environment

| | Linux host | macOS host |
|---|---|---|
| Machine | laptop, Intel Core i7-12800HX, NVIDIA RTX 3070 Ti Laptop GPU (8 GB VRAM, PCIe 4.0 x8) | Mac mini, Apple M4 (10 cores), 16 GB unified memory |
| OS | EndeavourOS, Linux 6.18 LTS, GCC 16.1.1, CUDA toolkit 13.3, driver 610.43 | macOS 26.5.1, Apple clang 21.0.0, Metal 4 |
| Data interface | `thunderbolt0` (IP over Thunderbolt), MTU 9000 | `bridge0`, MTU 9000 |
| TBCCL | 0.6.0, C ABI 1, wire protocol 4, Release build, providers `host` and `cuda` | the same version, Release build, providers `host` and `metal_shared` |
| Python | CPython 3.13 for the suite and the PyTorch runs; 3.12 for vLLM | the same |

* One direct Thunderbolt 4 cable between the two machines carried all benchmark traffic. Management traffic (ssh, heartbeats, relayed output) used a separate Ethernet interface; the suite checks before every run that the route to the peer's benchmark address leaves through the data interface and the management path does not.
* Nothing on either machine was tuned for the campaigns: MTU, offloads, CPU governors, power management, IRQ affinity, firewall rules and GPU clocks were left as found. The suite reads link health (AER counters, kernel events, interface errors, GPU throttling) before and after every measurement point and stops on any new error; no stop condition occurred.
* Frameworks (sections C and D): PyTorch 2.13.0 (`+cu130` on Linux), torch-tbccl built from its `main` branch (version 0.2.0.dev0); vLLM 0.31.0, vllm-tbccl 0.2.0, the vllm-metal fork tag `v0.31.0-tbccl1`, MLX 0.32.1.
* Statistics are medians of independent repetitions (each repetition a fresh process pair) unless the row says otherwise. "Spread" is (max - min) / median of those repetitions.

## B. Native TBCCL performance

Host memory, 2 ranks, `tbccl_*` executables driven by the suite. "Linux to Mac" and "Mac to Linux" name the physical direction of the data. These rows are round trips or completion times that include the receiver's acknowledgement; **none is a one-way latency** and none is halved.

| Operation | Size | Result | Statistic, repetitions | Caveats |
|---|---|---|---|---|
| Raw TCP ping-pong round trip (`tb_pingpong`) | 64 B | 128.2 µs (rank 0 on Linux), 128.5 µs (rank 0 on Mac) | mean over the tool's 10,000 round trips; median of 5 runs | no warmup option; mean, not median |
| TBCCL point-to-point round trip (`small_message_latency`, Communicator API) | 64 B | 128.1 µs (rank 0 on Linux), 128.5 µs (rank 0 on Mac) | median of 1,500 round trips per repetition; median of 5 repetitions | statistics differ from the raw TCP row; the two agree within 1 µs at 64 B and 4 KiB |
| Raw TCP, single stream (`iperf3` 3.21) | bulk | 15.81 Gbit/s Linux to Mac, 18.81 Gbit/s Mac to Linux | receiver-side goodput, 3 s runs, median of 3 | TCP payload, not the link rate; the direction asymmetry has no determined cause |
| TBCCL `end_to_end` transfer (`tbccl_tensor_transfer_bench`) | 16 MiB | 8,669 µs (1.93 GB/s, 15.5 Gbit/s) Linux to Mac; 7,294 µs (2.31 GB/s, 18.4 Gbit/s) Mac to Linux | median of 3 repetitions of 30 iterations; spread under 1% | completion confirmed by the receiver's ACK, so it includes the ACK's return trip; about 98% of the raw TCP rate in each direction |
| all_gather, `ring` versus `reference` algorithm (`tbccl_collective_bench`, float32, legacy synchronous API) | 1 MiB / 16 MiB | `ring` takes 0.42x / 0.49x the time of `reference` (694 vs 1,663 µs; 12.0 vs 24.4 ms) | median of 3 repetitions | 2 ranks only; at 64 KiB the ratio is 0.76 |
| all_reduce, `ring` versus `reference` | 64 KiB / 1 MiB / 16 MiB | 1.26x / 0.68x / 0.89x the time of `reference` | median of 3 repetitions | `ring` is slower below about 256 KiB; the `auto` selector chose `reference` at every size, so it was the slower choice from 1 MiB; 2 ranks only |

**Device memory (CUDA and Metal).** The same `end_to_end` transfer between the two GPUs' memory kinds, 16 MiB, median of 3 repetitions of 30 iterations, together with a host-memory control taken in the same run:

| Path | Result | Note |
|---|---|---|
| host to host (control), Linux to Mac / Mac to Linux | 8,667 µs / 7,318 µs | agrees with the host-memory row above within 0.3% |
| CUDA pinned to Metal shared (synchronous) | 9,968 µs | +15% over the control; the difference (1.30 ms) equals the isolated CUDA device-to-host staging time (1.28 ms), so staging and sending are serialized |
| Metal shared to CUDA pinned (synchronous) | 9,791 µs | the Mac-to-Linux advantage of host memory (16%) shrinks to 2% because the CUDA destination's host-to-device copy costs more |
| CUDA to Metal, asynchronous chunked staging (256 KiB chunks), Linux to Mac / Mac to Linux | 8,677 µs / 7,748 µs | within 0.1% of the host control (Linux to Mac) and 5% of the asynchronous host control (Mac to Linux, 7,376 µs): pipelining hides the CUDA staging at this size; whole-tensor asynchronous staging was *slower* than the synchronous tool (11,894 µs) |
| local staging only, no network (`tensor_staging`), CUDA pinned, device to host | 3.9 µs at 4 KiB; 1,279 µs at 16 MiB (13.1 GB/s) | 200 iterations, median of 3 repetitions; PCIe 4.0 x8 |
| local staging only, Metal private-staged, device to host | 152 µs at 4 KiB; 675 µs at 16 MiB (24.8 GB/s) | fixed cost per command buffer; Metal *shared* and host memory stage by no copy |

**Small-message bimodality.** At 4 KiB the host-memory transfer completes in about 125 µs. In the device-memory runs the per-repetition median was either about 127 µs or about 245 µs, differing between repetitions of the same configuration. This was first attributed to CUDA staging. Later PyTorch-adapter campaigns saw the same split with **host memory only**, in a system state with extra background load, so it is not a property of CUDA memory; its cause was not determined. Consequently no small-message *overhead* is stated as a single number anywhere on this page.

## C. PyTorch integration

PyTorch `torch.distributed` over torch-tbccl, 2 ranks, float32, tensors on CPU, CUDA (Linux) and MPS (Mac), all four pairings CPU/CPU, CUDA/CPU, CPU/MPS and CUDA/MPS.

* **Supported operations exercised and verified bit for bit** on all pairings: `all_reduce` (SUM), `broadcast`, `all_gather`, `send`/`recv`, `isend`/`irecv`, `Work.wait()`, `Work.get_future()` callbacks, and `DistributedDataParallel` training (checked against a CPU reference). Operations the backend rejects (for example `reduce_scatter`, `scatter`, `all_to_all`, non-SUM reductions, FSDP) were not benchmarked and there is no CPU fallback. Gloo could not be a baseline: the Linux and macOS PyTorch 2.13.0 Gloo builds did not rendezvous with each other on this pair.
* **Framework versus native.** Matched pairs, interleaved (native, torch, torch, native) in 3 blocks (6 repetitions per arm), same hosts, direction, size, memory kinds and completion semantics (send/recv followed by a 1-byte ACK):

| Operation | Size | Native (`tbccl_tensor_transfer_bench`) | torch-tbccl | Observed difference |
|---|---|---|---|---|
| send/recv + ACK, CPU/CPU, Linux to Mac | 1 MiB / 16 MiB | 758 µs / 8,697 µs | 892 µs / 10,740 µs | +18% / +23% |
| send/recv + ACK, CPU/CPU, Mac to Linux | 16 MiB | 7,286 µs | 7,569 µs | +4% |
| send/recv + ACK, CUDA/MPS, Linux to Mac | 16 MiB | 9,952 µs | 14,083 µs | +42% |
| send/recv + ACK, CUDA/MPS, Mac to Linux | 16 MiB | 8,736 µs | 11,772 µs | +35% |
| all_reduce, CPU/CPU (native: `small_message_latency`) | 16 MiB | 17,187 µs | 21,437 µs | +25% |

These are *observed differences between two measurements*, not an isolated adapter cost: the native tool times the whole loop and the framework each call, the PyTorch path includes c10d and worker hand-offs, and the CUDA and MPS rows also include the adapter's whole-buffer staging. At 4 KiB the native baseline itself was two-mode (see section B), so no small-message overhead is claimed.

* **Blocking and asynchronous collectives take about the same time.** `all_reduce(async_op=True)` returned in tens of microseconds (10 to 91 µs across sizes), was never complete at return, and its submission-to-`wait()` lifetime was 0.8 to 1.1 times the blocking call. Returning from the call is not completion.
* **DDP training** (synthetic 8-layer MLP, 33.6 MB of gradients, 64 samples per rank per step, 25 MB bucket cap, 3 repetitions of 20 steps): the step took 54 to 66 ms depending on the pairing (CUDA/MPS 57.4 ms, CPU/CPU 64.2 ms), with each rank's compute-only step at 1 to 11 ms. With a larger bucket cap the CUDA/MPS step was shorter (82.2 ms at 1 MB, 54.4 ms at 100 MB). With 32 times the batch size the step is set by the slowest rank's compute (CUDA/MPS 96.9 ms, CUDA/CPU 128.3 ms). The split between "compute" and "exposed communication" in the underlying analysis is an *inference* (backward with DDP minus backward under `no_sync()`), not an exact decomposition of communication time, and on the faster rank it includes waiting for the slower one.
* **Thermal context.** During CPU-heavy phases the Linux CPU ran near its thermal limit and its throttle counter rose; the GPU never throttled. This is recorded as context only: no experiment isolated its effect on any figure above.

## D. vLLM integration

Validated deployment: vLLM 0.31.0 pipeline parallelism, PP=2, TP=1, with vllm-tbccl 0.2.0 carrying the stage-to-stage tensors over TBCCL; Qwen3-0.6B in bf16 on every device; the settings the vllm-tbccl release validates (eager mode, no async scheduling, no prefix caching, `max-model-len` 4096). Three stage assignments: **CUDA then Metal**, **Metal then CUDA**, **CUDA then the Mac CPU**; the API node is stage 0. The suite proves, per run, that TBCCL carried the pipeline (both workers logged the `tbccl` backend, no NCCL, Gloo or MLX-ring marker, TCP connections only between the two link addresses, interface counters moved), that a single-device CUDA reference matches the pipeline's outputs (every gate phase passed; outputs that differ from the reference only where the reference's own top-1/top-2 logit margin is at most 0.251, a near-tie of bf16 logits, are classified, not hidden, and a single-node server shows them too), and that failed or truncated requests fail the run.

Client-observed, single request (concurrency 1), 128 prompt tokens, 32 output tokens; medians over the pooled requests of three passes (30 requests per cell):

| Configuration | TTFT | TPOT |
|---|---:|---:|
| CUDA then Metal | 58 ms | 21.3 ms |
| Metal then CUDA | 40 ms | 15.9 ms |
| CUDA then Mac CPU | 329 ms | 19.7 ms |
| single device, CUDA only | 11.0 ms | 7.9 ms |
| single device, Metal only | 48.9 ms | 14.5 ms |
| single device, Mac CPU only | 659 ms | 23.5 ms |

TTFT is the first server-sent event after submission; TPOT is (last event - first event) / (tokens after the first); percentiles are withheld below 10 (p90), 20 (p95) and 100 (p99) observations. A pipeline cannot lower latency for a model that fits on every device; it exists for models that do not.

* **Concurrency scaling.** In a closed loop the output rate grew almost linearly from 1 to 12 concurrent requests: CUDA then Metal 43.0 to 373 tokens/s, Metal then CUDA 58.2 to 453 tokens/s (3 passes; single-device CUDA 125 to 1,140 tokens/s). TPOT barely moved with concurrency (about 21 ms and 16 to 18 ms), so nothing saturated inside the validated limit of 12; no saturation point is claimed. The CUDA then CPU pipeline saturated early because the Mac CPU stage prefills at about 2.3 ms per prompt token.
* **Pipeline stage order matters.** Metal then CUDA had a TPOT 25% lower than CUDA then Metal (median of five lifecycle medians each, interleaved: 15.83 ms with a range of 15.61 to 16.10 ms, against 21.19 ms with a range of 20.98 to 21.46 ms; the ranges do not overlap) and higher throughput at concurrency 12 (+21% in the main sweep, 453 against 373 tokens/s; +32% in the interleaved confirmation lifecycles, 452 against 341 tokens/s, ten measurements each). The TPOT difference persisted with the client on the other machine and with tracing enabled. **The cause is not known.** The API node is stage 0 by construction, so stage order, API host and the location of the passive sampler cannot be separated by this experiment, and no mechanism has been shown.
* **A pipeline-boundary ping-pong** through vllm-tbccl's process group with no model (vLLM's own tensor-dict exchange, 1 token = 4 KiB per direction, 300 round trips after 50 warm-up, 3 repetitions, payloads verified) took 0.64 to 0.69 ms *round trip* with a CUDA rank 0 and 0.79 to 0.82 ms with a CPU rank 0. This is the time of the whole exchange the engine performs, which consists of several point-to-point messages; it is **not** a one-way latency and is not divided by two. Counted in full, it is roughly 3% to 5% of a 16 to 21 ms decode step, consistent with (but not proof of) the transport not being the dominant per-step cost; a trace of the engine's own stages would be needed to say more.
* **Not explained:** TTFT in CUDA then Metal is about 10 ms higher when a request asks for 32 or more output tokens than for 8 or fewer; it was absent in every other configuration.
* **Lifecycle.** A worker killed on the headless node was detected by the suite's external sampler within about 1.3 s, the group failed in bounded time and cleanup was confirmed on both hosts; vLLM's own leader does not notice such a worker.

## E. Exo integration

Systematic benchmark-suite measurement of the exo integration has **not** been done. There is no exo adapter in the suite and no exo result on this page. Do not read anything above as an exo performance statement.

## F. Measurement limitations

* **One hardware pair.** One Linux laptop and one Mac mini over one cable. Other machines, cables, operating-system versions, link speeds and driver states were not tested; results are not portable across setups.
* **Software versions.** TBCCL 0.6.0 (C ABI 1, wire protocol 4), PyTorch 2.13.0, vLLM 0.31.0, vllm-tbccl 0.2.0. Other versions were not measured.
* **Small model.** The vLLM measurements use a 0.6 B-parameter model that fits on every device used. Behaviour of models large enough to need the pipeline, of models other than Qwen3-0.6B, of sampling other than greedy, and of more than 12 concurrent requests is unmeasured.
* **TCP over Thunderbolt, not RDMA.** Nothing here compares TBCCL with RDMA or with a vendor's interconnect figures.
* **Synthetic versus application.** The native transfers and collectives are synthetic; the PyTorch rows use synthetic tensors and a synthetic MLP; only the vLLM rows are an application, with a client's timing semantics.
* **Different timing semantics.** Native tools time loops or single calls with their own statistics (mean, median, `floor(q*n)` or interpolated percentiles). Completion times include acknowledgements, ping-pongs are round trips, submission times are not completion times, and client-observed latencies include HTTP and scheduling. The suite records the semantics and refuses to compare records that differ; this page does not blend them.
* **Few repetitions.** Three to six repetitions per point (five lifecycles for the stage-order comparison). With 30 iterations per repetition p95 and p99 are not tail estimates.
* **Unsupported baseline comparisons.** No Gloo (it does not interoperate across this OS pair), no NCCL (no macOS build), no RDMA, and the single-node vLLM rows are context, not a like-for-like pipeline baseline (they differ in layer split, scheduler path and device mix).
* **System state.** The Linux CPU ran hot during some CPU-bound and API-node phases and the desktop session shared the GPU. Telemetry was recorded passively; its effect was not isolated and no benchmark was gated on CPU temperature. macOS provides no Thunderbolt or PCIe error telemetry, so link-health stop rules relied on the Linux side and interface counters.

## G. Reproduction

* How to run the suite, configure two hosts and enable physical-link runs: [benchmark suite guide](benchmark-suite.md).
* What each native benchmark times and reports: [benchmark inventory](benchmark-suite-inventory.md).
* Measurement conventions: [benchmark methodology](benchmark-methodology.md).
* Presets for every campaign mentioned here are in `benchmarks/suite/configs/` (`b3b-*` host memory over a link, `b3c-*` device memory, `b4-*` PyTorch, `b5-*` vLLM, `two-host.example.yaml` and `tb4-core.example.yaml` as templates). They contain `<PLACEHOLDER>` values; real addresses, interpreters and launchers go into a git-ignored `*.local.yaml` overlay.
* The result record format is `tbccl-bench-result/1`, defined by `benchmarks/suite/schemas/result-v1.json` and described in the guide.
* Reproducing a number means running the same preset on the same kind of hardware with the same versions and then using `compare` (which refuses to pair records that differ in hardware, memory backend, timing method, algorithm or other settings); it does not mean expecting these values on different hardware.
