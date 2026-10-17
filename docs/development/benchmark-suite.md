# Benchmark suite

`benchmarks/suite/` is optional tooling that runs the benchmark executables TBCCL already has, normalises their results into one schema, and compares runs without mixing experiments that are not equivalent. It is **not part of the libtbccl runtime API**: it adds nothing to the C ABI, the wire protocol or the CMake build, changes no framework adapter, and needs no PyTorch, vLLM or Exo to import. Performance numbers are never a pass/fail gate in CI.

What it does not do: implement a communication algorithm, replace a native benchmark, change a benchmark's methodology, change a machine's settings, or run anything on a physical link unless you opt in. The audit it is built on is in [the benchmark inventory](benchmark-suite-inventory.md); the measurement conventions it follows are in [benchmark methodology](benchmark-methodology.md); what has been measured with it, and on which machines, is in [the performance overview](performance-evaluation.md).

## What the suite benchmarks

| Area | Benchmarks (names as `list` prints them) | Memory kinds |
|---|---|---|
| Raw TCP reference | `tcp_pingpong` (`tb_pingpong`), `iperf3` (a tool you install; not a dependency) | host |
| TBCCL point-to-point | `small_message_latency` (Communicator), `tensor_transfer` (`tbccl_tensor_transfer_bench`) | host; CUDA pageable/pinned and Metal shared/private-staged for `tensor_transfer` |
| TBCCL collectives | `collective_bench` (all_reduce, all_gather, reduce_scatter; algorithms `reference`, `ring`, `auto`, experimental `ring-pipelined`) | host |
| Device staging and asynchrony | `tensor_staging`, `async_transfer`, `async_overlap`, `cuda_overlap`, `hetero_allreduce` | host, CUDA, Metal |
| PyTorch over torch-tbccl | `torch_collectives`, `torch_p2p`, `torch_async`, `torch_ddp` | CPU, CUDA and MPS tensors |
| vLLM over vllm-tbccl | `vllm_startup`, `vllm_correctness`, `vllm_latency`, `vllm_throughput`, `vllm_concurrency`, `vllm_pipeline`, `vllm_backend` | CUDA, Metal, CPU stages |
| Exo over exo-tbccl (MlxTbccl) and Exo MlxRing | `exo_correctness`, `exo_latency`, `exo_throughput`, `exo_transport_ab`, `exo_stage_profile`, `exo_large_model`, `exo_lifecycle` | CUDA and Metal stages |

`python3 -m benchmarks.suite list` prints the authoritative table: operations, algorithms, memory backends, rank limits, native verification and whether each benchmark is integrated.

## Adapters and optional dependencies

The suite is plain CPython; each adapter drives one existing executable or script and parses its output.

| Adapter family | Drives | Needs |
|---|---|---|
| native (`adapters/core.py`, `device.py`, `network.py`) | the executables built from this repository | a CMake build of the targets you want; `iperf3` only for `iperf3` |
| PyTorch (`adapters/torch.py`; provider: torch-tbccl `benchmarks/provider/`, compatibility copy `torch_bench/`) | wrapper scripts that call `torch.distributed` | per host: a Python environment with a matching PyTorch and torch-tbccl (the same libtbccl as the native executables it is compared with), named by `executables.torch_python` |
| vLLM (`adapters/vllm.py`; provider: vllm-tbccl `benchmarks/provider/`, compatibility copy `vllm_bench/`) | `vllm serve` processes and a streaming client | per host: an environment with vLLM and vllm-tbccl (and vllm-metal for a Metal stage), a local copy of the model |
| Exo (`adapters/exo.py`; provider: exo-tbccl `benchmarks/provider/`, compatibility copy `exo_bench/`) | one `exo` node per host and a streaming client | per host: an Exo environment with the exo-tbccl extra (and MLX for the device), a local copy of the model |

The framework workload scripts belong to the repository that owns the framework integration and are configured per host as a [provider](benchmark-providers.md) (`executables.<name>_bench_dir`); the copies under `torch_bench/`, `vllm_bench/` and `exo_bench/` are used only when a host configures no provider directory. The suite never imports PyTorch, vLLM, Exo or MLX itself, never downloads a model, and never modifies TBCCL, torch-tbccl, vllm-tbccl, exo-tbccl, vLLM or Exo. The framework adapters are optional: without those environments the native benchmarks work unchanged, and a framework preset whose environment is not configured is reported as `cannot plan` in the dry run (unresolved placeholders are refused outright), so nothing is launched.


## Install and build

The suite is not installed: run it from the repository root with CPython 3.13, the version the adapters use (`uv venv --python 3.13`), and use the same interpreter on every host. Only the standard library is required to import it; PyYAML is needed to read the shipped presets, which are YAML.

| Need | For | Required? |
|---|---|---|
| Python standard library | everything | yes |
| `pyyaml` | the shipped presets and any `.yaml` configuration (`.json` configurations work without it) | yes for the presets (`pip install pyyaml`) |
| `matplotlib` | static plots | optional; reports are written without it |
| `jsonschema` | cross-checking the built-in schema validator in the tests | optional |
| `psutil` | the vLLM benchmarks' passive sampler, which runs inside each host's vLLM environment (vLLM brings it) | only in those environments |
| built TBCCL executables | `run` | yes; see below |

Build the executables you want to drive (the suite never builds anything):

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target tb_pingpong small_message_latency tbccl_tensor_transfer_bench tbccl_collective_bench tbccl-info
```

The suite looks in `build_dirs` from the configuration (default `build-release`, `build-bench`, `build`) and in `$TBCCL_BUILD_DIR` first. It warns when the build type is not an optimised one, because such timings do not represent the library.

## Running on one machine (loopback)

All commands run as `python3 -m benchmarks.suite COMMAND` from the repository root and have `--help`.

| Command | Purpose |
|---|---|
| `list` | registered benchmarks, operations, algorithms, memory backends, rank limits, native verification, and whether each is integrated |
| `inventory` | detected executables with build type and version, local hardware, optional modules, iperf3 |
| `doctor [CONFIG]` | read-only readiness checks; with a configuration also its hosts, executables and physical opt-in |
| `run CONFIG` | validate, expand the matrix and run; `--dry-run`, `--output-dir`, `--label`, `--overlay`, `--allow-physical`, `--benchmark`, `--operation`, `--workload`, `--orientation` |
| `provenance stamp --build-dir BUILD` | record the repository commit and the executables' SHA-256 next to a build (run it on the host that built the binaries) |
| `import PATH...` | normalise historical CSV/JSON/JSONL results; `--set PATH=VALUE` adds context the file lacks |
| `providers [CONFIG]` | which framework providers are available on each host, with the reason when one is not; read-only |
| `profiles` | the ready-made configuration profiles (quick, native, device, pytorch, vllm, exo, custom) |
| `compare RUN_A RUN_B` | compare equivalent experiments; `--contextual` also shows labelled non-equivalent ones |
| `report RUN_ID` | `report.md`, `summary.csv`, `report.json`, a self-contained offline `report.html` and plots (`--format` selects files); `--baseline RUN` adds baseline-versus-candidate figures |

A run id, a run directory or a `results.jsonl` path all work wherever a run is expected. Output goes under `results/bench-suite-local/` by default, which is git-ignored.

### One working end-to-end example

```sh
python3 -m benchmarks.suite run benchmarks/suite/configs/smoke.yaml --dry-run      # the matrix and exact commands; nothing starts
python3 -m benchmarks.suite run benchmarks/suite/configs/smoke.yaml --label first  # about ten seconds on loopback
python3 -m benchmarks.suite run benchmarks/suite/configs/smoke.yaml --label second
python3 -m benchmarks.suite report <first run id>
python3 -m benchmarks.suite compare <first run id> <second run id>
```

The smoke preset runs four payloads (4 B, 4 KiB, 64 KiB, 1 MiB) of every integrated benchmark with two ranks on this machine. Points a benchmark cannot run (for example a 4 B all_reduce with two ranks, which has one element) are listed in the matrix as `SKIPPED` with the reason and recorded as skipped. The full standard sweep (`core-loopback.yaml`, 13 payloads from 4 B to 64 MiB, five repetitions) is never run automatically.

## Configuration

A YAML (or JSON) file with these sections. Unknown keys are errors, so typos are caught.

```yaml
schema: tbccl-bench-config/1
name: smoke
defaults:   {warmup: 5, iterations: 30, repetitions: 2, timeout_s: 60, verification: preferred}
limits:     {max_processes: 8}              # most processes of one measurement
topology:   {kind: loopback, interconnect: loopback, transport: tcp, physical: false}
hosts:                                        # physical machines
  local: {launcher: {kind: local}, address: 127.0.0.1, interface: lo}
ranks:                                        # rank identity, separate from the host
  - {rank: 0, host: local, memory_backend: host}
  - {rank: 1, host: local, memory_backend: host}
workloads:
  - {name: collectives, benchmark: collective_bench, operations: [all_reduce, all_gather],
     algorithms: [reference, ring, auto], datatype: float32, sizes: smoke, rank_counts: [2]}
```

* `sizes` is a list (`4`, `4KiB`, `1MiB`; `K/M/G` and `KiB` are binary like the native tools, `KB` is decimal) or a preset (`smoke`, `standard`). The meaning of a size depends on the benchmark and is recorded with each result (see the inventory).
* `rank_counts` uses the first N ranks of `ranks`. A workload may set its own `memory_backend` (one name, one per rank, or a `{host: backend}` map that keeps each backend with its machine when `orientations` swaps the ranks), for example `host` for a TCP baseline on a rank whose own backend is CUDA.
* A host has an `address`, `interface`, optional declared hardware (`os`, `arch`, `cpu`, `gpu`) and a `launcher`. `kind: local` runs commands directly; `kind: prefix` puts your argument prefix (an ssh command, for instance) in front of the supervised command (see [Two hosts](#running-on-two-machines)). The suite does **not** assume SSH exists. More host keys: `interfaces` (name to address), `env` (environment for every rank on the host), `executables` / `executables_dir` (where the benchmark binaries are, per host), `python`, `health_script` / `health_command`, `expect` (expected `mtu`, `gpu_max_temp_c`, `allow_unavailable`) and `build` (a declared build commit).
* `options` are passed through where the benchmark supports them: `chunk_bytes` (required for `ring-pipelined`), `busy_poll_us`, `timing_scope` (`ready` or `produce`, tensor end_to_end), `env` (extra environment for the children).
* `verification: required` turns points whose benchmark cannot verify them into `unsupported` records instead of running them; `preferred` records the verification status; `off` does not care.

**Tracked files never hold private values.** Put hostnames, addresses, remote paths and launcher commands into `CONFIG.local.yaml` next to the file (picked up automatically, git-ignored) or into any file passed with `--overlay`. Tracked examples use `<PLACEHOLDER>` values; a run with unresolved placeholders is refused, a dry run lists them.

Child processes receive the suite's environment **without any `TBCCL_*` variable**, so a variable in your shell cannot change algorithm selection or tracing. Add what you want explicitly with `options.env`; the names that were stripped are in the manifest.

## Running on two machines

A *host* is a machine; a *rank* is a place a benchmark process runs. Ranks are listed in `ranks` with their host, memory backend and optionally the `interface` (or explicit `address`) peers use to reach them, so the same two machines can be used in either orientation and over several interfaces. `two-host.example.yaml` and `two-host-reversed.example.yaml` are the two orientations of one setup (rank 0 on host-a, or rank 0 on host-b); a workload can also say `orientations: [forward, reverse]` to run both in one go (the rank numbers stay, the hosts and backends swap, and the two are never compared as equivalent).

Setup, once per machine:

1. Build the benchmark executables on that machine (the same TBCCL package version, C ABI and wire protocol on both: the run is refused otherwise) and stamp the build: `python3 -m benchmarks.suite provenance stamp --build-dir BUILD`.
2. Put a copy of `scripts/tb4_health_snapshot.py` where the host can read it and give its path as `health_script`.
3. Copy the example to `NAME.local.yaml` (git-ignored) and fill the placeholders: addresses, interfaces, the launcher prefix, the remote build directory. Nothing private belongs in tracked files.
4. `python3 -m benchmarks.suite doctor NAME.local.yaml --probe`: read-only; probes each host (hardware, executables, build provenance) and evaluates the health signals. Fix everything it marks FAIL.
5. `run NAME.local.yaml --dry-run` shows every command, including the supervised ones; then `run ... --allow-physical`.

**macOS** needs the same CPython 3.13 (a uv-managed one: set the host's `python:` to its path so the supervised helper runs under it too), `lsof` (part of macOS) for listener ownership, and a build with `-DTBCCL_ENABLE_METAL=ON` for the `metal-*` backends. Keep the rank in a live session (a terminal multiplexer, not a detached `nohup`) as the [Thunderbolt link guide](../guides/thunderbolt-link.md) explains, and allow the firewall prompt for the benchmark binaries yourself: the suite never changes the firewall.

**The launcher.** Remote commands are `PREFIX python3 -c <supervisor> run ... -- <benchmark command>`; with `shell_join: true` (the default) the command after the prefix is one properly quoted string, which is what ssh-like launchers expect. The supervisor is a standard-library helper that the suite ships inside the command, so nothing is installed on the remote and nothing stays running. The local host can also be supervised (`launcher: {kind: local, supervised: true}`).

## Physical (Thunderbolt) runs

A run is physical when any host has a non-loopback address. Nothing physical starts unless the configuration declares it (`topology.physical: true`) **and** the command line says `--allow-physical`. The usual sequence is `doctor --probe` (read-only), `run --dry-run`, then `run --allow-physical`, one preset at a time.

The physical presets in `benchmarks/suite/configs/` are named for the campaign they were written for: `b3b-*` (host memory over the link: latency, throughput, collectives), `b3c-*` (CUDA and Metal device memory), `b4-*` (PyTorch communication and DDP) and `b5-*` (vLLM, generated by `configs/generate_b5.py`; `b4-*` by `generate_b4.py`). `two-host.example.yaml`, `two-host-reversed.example.yaml` and `tb4-core.example.yaml` are the starting points for your own configuration; `smoke.yaml` and `core-loopback.yaml` are loopback.

### Checklist

1. **Same interpreter everywhere**: CPython 3.13 from uv on every host; set each remote host's `python:` to it.
2. **Build and stamp on each host** (`provenance stamp`), with the backends you will measure: `-DTBCCL_ENABLE_CUDA=ON` on the CUDA host and `-DTBCCL_ENABLE_METAL=ON` on the Mac. `doctor --probe` must show `stamped` for every executable and matching package, ABI and wire versions.
3. **Backends**: `tbccl-info` lists `cuda` and `metal_shared`; a rank asking for a backend its host's build lacks is `UNSUPPORTED` at planning time. Validate `end_to_end` with a 4 KiB payload on loopback on each host first, then across a management network, before the real link.
4. **Management path**: the launcher must not reach a host over the data link. The preflight warns if it does.
5. **Firewalls**: open the `port_range` on both hosts yourself (Linux `firewalld` rejects inbound connections by default; macOS asks once per new binary, and a silent hang until someone clicks Allow is normal). The suite never changes a firewall; a refused connection fails the group and is cleaned up.
6. **Health expectations** (`expect`): the link's MTU, a GPU temperature limit; read the baseline signals in `doctor --probe` and check that none is unexpectedly `unavailable` (macOS reports Thunderbolt/PCIe error events as unavailable).
7. **Orientations**: run both (`orientations: [forward, reverse]`); they are separate experiments and `compare` will not treat them as equivalent.
8. **Dry run, then run** with `--allow-physical`. Remove any runtime firewall rule you added when finished.

## Hardware safety behaviour

### Process lifecycle and failure handling

Every supervised rank gets a unique ownership token (in its environment) and its own process group. The supervisor on the host:

* starts the process, reports `started` with its pid, process group and start time, and relays stdout and stderr to per-rank files on the controller (`raw/.../rankN.stdout`, `.stderr`; the launcher's own stderr is `rankN.launcher.stderr`);
* needs a heartbeat from the controller every `lease_s / 4`; with no heartbeat for `lease_s`, a closed connection (stdin EOF, which is what a crashed or killed controller and a dropped ssh session look like), a cancel request or the runtime cap (`timeout_s + 10`) it terminates the **whole process group** (SIGTERM, then SIGKILL after `term_grace_s`), and anything outside the group that still carries the token, so grandchildren go too, including ones that started their own session. The token is found in `/proc/*/environ` on Linux and with `ps -E` on macOS (the user's own processes, no privileges); on each host the helper first proves the scan works by finding a process it just started with a test token, and reports whether it is proven;
* runs a watchdog, forked first into its own session, that does the same if the supervisor itself is killed, or when the hard deadline passes; on Linux the child also gets `PR_SET_PDEATHSIG`;
* never signals by name: no `pkill -f`, no `killall`, no matching of process names, only the exact group and token it created.

After the rank ends the controller runs a *separate* `check-clean` through the launcher. The per-rank record says `cleanup_confirmed: true` only when both the supervisor and that independent check agree **and** the token scan is proven on that host; if the scan cannot be proven the process group may be empty but descendants that left it cannot be ruled out, so the record says `null` with that reason; if the host cannot be asked again it says `null` ("unverified"), the measurement keeps a warning, and the manifest lists it under `cleanup_unconfirmed`. A failure, nonzero exit, timeout, lost connection or unparseable output on any rank invalidates the whole measurement; the other ranks are cancelled, nothing is retried, and the record keeps each rank's host, role, exit code, termination flag and raw output. With `failure_policy: {on_failure: abort_run}` (or `max_failures`) the first failed measurement stops the run.

Listener readiness is proven, not slept for: the process the suite started must hold a LISTEN socket on the expected port, found in `/proc` (Linux) or with `lsof` (macOS) and tied to that process by process group or token. The suite never connects to the port, because that could consume the benchmark's own `accept()`. A process that exits first, a port held by something else, a timeout, or a platform where ownership cannot be established all fail the group with the reason; nothing proceeds on a guess.

### Health monitoring

For a physical run every used host needs a health source. Before the run each host is probed independently and these signals are evaluated and stored in the manifest: interface present, link state, IP configuration (the expected address is on the interface), MTU (if `expect.mtu` is set), boot identity, error counters, Thunderbolt/PCIe error visibility (Linux kernel log; on macOS it is reported unavailable), and, for hosts with a CUDA rank, GPU availability and thermal state (`expect.gpu_max_temp_c`, default 83 C, and any active throttle reason). A signal is `ok`, `bad` or `unavailable`. A bad signal, or an unavailable critical one (interface, link, IP, boot identity), refuses the run unless `expect.allow_unavailable` lists it. GPU throttling is decoded from NVML's reason bitmask: idle is normal, a software power cap is a note, thermal and hardware-slowdown reasons are bad. After each workload point every host is snapshotted again and compared with the baseline using the stop policy of `scripts/tb4_health_snapshot.py` plus the suite's own checks (interface gone, link down, address removed, MTU changed, reboot, rx/tx error counters up, GPU throttling). Any hit invalidates that point, skips the rest and aborts the run; a missing snapshot is itself a stop. Nothing is retried. Whether the kernel can report PCIe errors at all is a separate signal, `aer_observability`: `observable`, `unavailable` (for example `pci=noaer` on the kernel command line, which keeps every AER counter at zero, or a host that is not Linux) or `unknown`. Unavailable is a warning in every preflight and a limitation to state in every campaign summary; a run on such a host must not be described as having "zero AER errors", only as having no detected link-state or interface-health regression under the available checks. The suite only reads the kernel command line; it never changes it. The suite never changes PCI power management, Thunderbolt authorisation, MTU, offloads, IRQ affinity, CPU governors, GPU clocks or kernel parameters, and needs no sudo.

### Physical benchmark safety

Physical means any host with a non-loopback address. Such a configuration must declare `topology.physical: true`, and `run` refuses it without `--allow-physical`. It also refuses unresolved placeholders, hosts without a health source, a bad baseline and inconsistent builds. Check GPU thermals before timed runs yourself as well, and follow the [Thunderbolt link guide](../guides/thunderbolt-link.md) and [troubleshooting](../guides/thunderbolt-troubleshooting.md); a throttled GPU or a degraded link invalidates the numbers.

### Safe test execution

The test suite never touches a network interface or another machine: remote behaviour is exercised with a stand-in launcher that runs commands locally through a shell, and the macOS paths with canned tool output. `python3 -m unittest discover -s benchmarks/suite/tests -t .` is safe anywhere. Do not point `run` at a physical configuration to "try it": use `doctor --probe` first, then `--dry-run`.

### Validating a real host

`python3 -m benchmarks.suite.validation.real_launcher --launcher-argv '["ssh","-o","BatchMode=yes","ALIAS"]' --python PATH_TO_PYTHON_ON_THE_HOST` runs 21 checks of the supervision through a real launcher (completion, forwarding of stdout and stderr, exit codes, runtime cap, cancellation, child, grandchild and escaped-descendant cleanup, a dropped connection, a killed controller, lease expiry, a killed supervisor, independent cleanup confirmation, and that an unrelated process is untouched). Only throwaway `python -c` processes run on the host, the helper travels in the command so nothing is written there, and no benchmark traffic is generated. Processes are tracked by exact pid, group and token. The same harness runs in the unit tests against the stand-in launcher.

**Firewalls.** A benchmark listener is reachable only if the host's firewall allows it. The suite never changes a firewall. If a connection is refused or hangs, check `firewalld`/`ufw` on Linux (open a narrow range with `defaults.port_range` and your own runtime rule) and the macOS application firewall (a prompt for each new binary, which only the user can accept; a silent hang until then is normal). The run fails within its timeouts and cleans up.

## How results are recorded

A run directory contains `manifest.json` (configuration name, repository commit and dirty flag, preflight outcome, counts, health snapshots, final status), `results.jsonl` (one record per workload point per independent repetition), `raw/` (each rank's stdout and stderr, the command lines, files a benchmark wrote) and, for physical runs, `health/`. Raw files are not sanitised; read them before sharing a run. Records and the manifest replace the configured non-loopback host addresses (in values and in keys) and the controller's repository and home paths; text that comes from a remote host, such as a failure message quoting a remote path or user name, is not rewritten. Treat the run directory of a physical campaign as private.

The record schema is `tbccl-bench-result/1` (`benchmarks/suite/schemas/result-v1.json`; a validator is built in and cross-checked against the JSON schema when `jsonschema` is installed). Sections: `benchmark` (name, operation, requested and resolved algorithm, backend, repository commit, executable hash and version, build type, provenance `native` or `imported`), `environment` (hosts with hardware, per-rank memory backend, memory domain such as `cpu-cpu`, `cuda-cpu`, `cuda-metal`, topology, transport, interface, MTU, versions), `configuration` (payload and its **semantics**, datatype, rank count, warmup, iterations, repetitions, timing method, completion semantics), `measurements`, `verification` and `status`.

* Statistics carry their provenance: `reported` (what the native tool printed) or `computed` (from raw samples). The native tools print aggregates, so today everything is `reported`. A percentile is never derived from medians and per-rank aggregates are never merged into a global percentile (`statistics.py` refuses).
* Each timing names its percentile definition. `small_message_latency` uses `floor(q*n)`; the others use linear interpolation. Results with different definitions are not compared.
* Only a record with status `success` has measurements. `failure`, `timeout`, `invalid` (output unparseable or failed schema validation), `skipped` and `unsupported` keep their raw output and reason and no numbers. Verification `failed` can never be a success.
* Throughput records say what it is: `tcp_application_payload`, `collective_algorithmic`, `collective_bus_estimate`, `device_to_device_effective`, `application_payload`, `iperf3_tcp_goodput`, each with its definition and unit as the native tool defines it (decimal GB/s, Gbit/s; `async_transfer_bench`'s binary GiB/s is converted x1.073741824 to decimal GB/s and the native figure is kept). **Application payload throughput is not physical link bandwidth.**
* TBCCL's transport is TCP over IPv4. Over Thunderbolt it is IP over Thunderbolt, never RDMA, and nothing in the suite says otherwise.

### Correctness methodology

The suite follows [the benchmark methodology conventions](benchmark-methodology.md) and changes none of them.

1. Setup, warmup, the timed loop and the verification step are the native benchmark's own: untimed warmup, a timed loop with no byte checking inside it, then one untimed verified transfer. The audit confirmed this for every integrated executable.
2. The suite passes the configured warmup and iteration counts to the executables that take them. `tb_pingpong` takes neither: it has no warmup and picks its own iteration count, and the record says so.
3. Verification is whatever the executable does. The suite records `passed` only when the executable's own check ran and every process exited 0, `failed` when its stderr reports a failed check, and `not_supported` where the executable does not verify (for example `tb_pingpong`, the host-only tensor modes, and every `small_message_latency` mode except p2p). Nothing is verified in Python.
4. Durations measured by the suite itself (the group's wall time) use `time.monotonic()`. Timestamps from different hosts are never subtracted; the latencies in a record come from one process's monotonic clock inside the native tool.
5. A nonzero exit, a timeout or an unparseable output on **any** rank invalidates the whole measurement. Every process the suite started is terminated (each rank has its own session, so unrelated processes are never signalled) and nothing is retried.
6. For asynchronous benchmarks, submission latency and completion latency are separate quantities; the schema has no field that equates them.

Ports: each measurement gets freshly allocated ports, checked free together with port+1000 (other TBCCL tools place a data plane there). Measurements run one at a time because concurrent ones disturb each other; `limits.max_processes` bounds the processes of one measurement.

### Build provenance

Per host the suite records the executable's SHA-256, build type, generator, compiler and its version, TBCCL package version, C ABI and wire protocol, enabled providers (CUDA/Metal), OS, architecture, CPU, GPU, driver, and the CUDA or Metal runtime where it can be read. The repository HEAD is **not** taken to be the source of a binary. Three statuses:

* `stamped`: `python3 -m benchmarks.suite provenance stamp --build-dir BUILD`, run on the host that built the binaries, writes `build-provenance.json`. It is read back as provenance only if (1) the executable's SHA-256 still equals the stamped one, (2) the build's CMake source directory is the stamped checkout, (3) no tracked build input (`src`, `include`, `cmake`, `CMakeLists.txt`, `benchmarks` except this suite) was modified, and (4) no build input is newer than the executables. It is an **attestation, not a cryptographic proof**: nothing in the binary names its source, so a stamp is only as good as the moment it was made and these consistency checks. Changes to this suite do not affect it.
* `declared`: the host configuration names a commit (`build: {commit: ...}`). It is the operator's statement, not checked against the binary, and is never presented as verified.
* `unverified`: everything else (no stamp, a stamp that no longer matches, a stamp from another checkout or made after sources changed). `source_commit` stays null; the hash, versions, compiler and build type still identify the binary.

The controller's own commit and dirty state are recorded separately as `repository_commit` and `repository_dirty`. These are additive fields in result schema v1; older records stay valid; the manifest holds the full `host_provenance`. Hosts whose package version, C ABI or wire protocol differ are refused (`provenance.require_same_version`, default on); `provenance.require_stamped` (the older name `require_verified` still works) additionally refuses builds that are not `stamped`; differing build commits are noted.

## How comparisons are validated

`compare` pairs records of the same benchmark, operation, requested algorithm, payload and datatype, then requires everything else to match: hardware, memory backend per rank, topology, transport, MTU, resolved algorithm, payload semantics, timing method, completion semantics, percentile definition, primary statistic and options. Version differences (commit, executable hash, build type) are allowed and printed. A pair that differs is listed under "Not compared" with each difference; `--contextual` shows it with a `CONTEXTUAL` label.

The output gives the relative difference `(candidate - baseline) / baseline` of the median over repetitions, and whether it exceeds the repeat-to-repeat spread. With fewer than three repetitions it says so and makes no judgement. It does not use the word speedup. Do not compare measurements from unrelated hardware as if they were a controlled experiment, and do not compare TBCCL with NCCL, Gloo, iperf3 or Exo's ring as if they ran the same configuration: NCCL has no macOS, and the baselines support different heterogeneous setups.

### Importing saved results

`import` accepts `tb_pingpong` CSV, `latency_sweep.py` raw CSV, `tbccl_collective_bench` and `tbccl_tensor_transfer_bench` CSV, `small_message_latency` JSONL, iperf3 JSON, `tools/benchmark_allreduce.py` reports (`tbccl-benchmark/1`) and the suite's own records. Imported records have provenance `imported`, the source file name, and `unknown` wherever the file does not say (hardware, topology, direction). Supply what you know:

```sh
python3 -m benchmarks.suite import results/linux-to-mac.csv --output-dir results/bench-suite-local \
    --set environment.topology.kind=direct-link --set environment.topology.interconnect=thunderbolt4-ip \
    --set environment.topology.physical=true \
    --set 'configuration.options={"direction":"linux-to-mac"}' \
    --set 'environment.hosts=[{"host_id":"host-a","ranks":[0],"os":"Linux","arch":"x86_64","cpu_model":"..."}]'
```

A malformed row rejects the file with its line number, unless `--skip-bad-rows` (skipped rows are listed in the manifest). Aggregate-only summaries are refused: import the raw files. Records with unknown hardware are never reported as equivalent to anything.

### Baselines

`tb_pingpong` is the transport reference (raw TCP, no TBCCL protocol). Single-stream `iperf3` (the `iperf3` benchmark; install iperf3 yourself, it is detected by `inventory` and `doctor` and is never a dependency) is the second raw-TCP reference. The baseline interface (`BASELINE_PROVIDERS` in `adapters/network.py`) is where TCP, Gloo, NCCL, Exo MlxRing and framework-native communication will be added; none of those is implemented; framework-native communication is benchmarked through the PyTorch and vLLM adapters below.

### Regression baselines

Stored results can serve as regression baselines: [Baselines and regression comparison](benchmark-baselines.md) describes the case definitions under `benchmarks/suite/baselines/`, which measurements may be compared for which purpose, the classification (`improvement`, `regression`, `indeterminate`, `incompatible`, `insufficient_evidence`, `invalid_measurement`) and why independent lifecycles, not requests, are the replication unit.

### Reproducing saved results

Every record names the configuration, the repository commit (and whether the tree had uncommitted changes), the executable's SHA-256 and version, its build type, its verified or declared build commit and the exact command lines (`artifacts.commands`, and `raw/*/commands.json` with the real values). Rebuild that commit, stamp it, run the same configuration and overlay, and compare with `compare`. The executable carries no build commit of its own: the record's `source_commit` is filled only from a verified stamp or a declaration (see Build provenance), and `repository_commit` is the controller checkout's HEAD when the suite ran.

## Generating reports

`python3 -m benchmarks.suite report RUN_ID` writes `report.md`, `summary.csv` and (when matplotlib is installed) static plots into the run directory; `--baseline RUN` adds baseline-versus-candidate figures and `--no-plots` skips the plots. Reports state operation, direction or orientation, memory backend, transport, metric, timing method, units, repetitions and build provenance, and they count (never hide) failed, invalid and skipped measurements.

Campaign-level tables and figures that combine several runs (tiers) of one campaign are produced by the report modules, which read only normalised results and never recompute a statistic from medians:

```sh
python3 -m benchmarks.suite.reporting.b3b RESULTS_DIR CAMPAIGN_ID [--out DIR] [--include-prelim]
python3 -m benchmarks.suite.reporting.b3c RESULTS_DIR CAMPAIGN_ID... [--out DIR] [--b3b-tables DIR] [--exclude TEXT] [--include-prelim]
python3 -m benchmarks.suite.reporting.b4 RESULTS_DIR CAMPAIGN_ID [--out DIR] [--exclude TEXT] [--include-prelim]
python3 -m benchmarks.suite.reporting.b5 RESULTS_DIR CAMPAIGN_ID [--out DIR] [--exclude TEXT] [--include-tier TEXT]
python3 -m benchmarks.suite.validation.tier_check RUN_DIR [--config CONFIG] [--overlay FILE] [--busy NAME]
```

Runs are selected by the campaign id in their label; `tier_check` audits one finished run (every record valid and verified, health and cleanup acceptable, the data interface carried the traffic, nothing left running) and exits 0 only when it is clean enough to continue.

## Framework and device benchmarks

### Device memory (CUDA and Metal)

The CUDA and Metal presets are `configs/b3c-staging-{linux,mac}.yaml` (local staging, one process, no network), `b3c-transfer.yaml`, `b3c-async.yaml`, `b3c-overlap.yaml`, `b3c-hetero-allreduce.yaml` and `b3c-validation.yaml`. Build the CUDA host with `-DTBCCL_ENABLE_CUDA=ON` and the Mac with `-DTBCCL_ENABLE_METAL=ON`, stamp both, and fill the placeholders in a git-ignored overlay; a backend the executable was not built with is reported as unsupported, never replaced by host memory. Give each machine its own memory backend with `memory_backend: {host-a: cuda-pinned, host-b: metal-shared}` so the backends stay with their machines under `orientations: [forward, reverse]`; a plain list is by rank number and would swap them. `async_overlap` is CPU-compute overlap only; `cuda_overlap` needs `options.compute_rounds` (pick it with the tool's `--calibrate`) and the CUDA rank as the source. `python3 -m benchmarks.suite.reporting.b3c RESULTS_DIR CAMPAIGN... --exclude TEXT --b3b-tables DIR` writes the tables and figures; a device configuration is compared only with the host-memory control of the same run, and the difference is an observed incremental path cost, not an isolated staging cost.

### PyTorch

`torch_collectives`, `torch_p2p`, `torch_async` and `torch_ddp` run PyTorch's `torch.distributed` over torch-tbccl (or Gloo, where both ranks can form a group) through the wrapper scripts of the torch-tbccl provider (`benchmarks/provider/` there; compatibility copy `torch_bench/`), which print one JSON line per rank (`TBCCL_TORCH_BENCH` / `TBCCL_TORCH_DDP`); the suite itself never imports torch and torch-tbccl is not modified. Per host, `executables.torch_python` names the interpreter of an environment with a matching torch and torch-tbccl (the same libtbccl as the native benchmarks it is compared with), and a remote host also needs `executables.torch_bench_dir`. Memory backends are torch device names (`host` = a CPU tensor, `cuda`, `mps`). Every point verifies its result bit for bit on the receiving rank, checks that both ranks report the same torch, torch-tbccl, libtbccl, ABI and wire versions and the configured devices, and proves the data path from each process's own TCP connections (a peer outside the two link addresses invalidates the run). A pairing the capability audit does not support is rejected before launch; nothing falls back to the CPU. `torch_ddp` takes `options.layers`, `width`, `batch`, `bucket_cap_mb`, `steps`, `tag` (and `profile` to add the traced loop) and requires `sizes` to equal the gradient bytes of the model. Results use two additive vocabulary entries, the throughput kinds `training_samples_per_second` and `training_steps_per_second` (units `samples/s`, `steps/s`); schema version 1 is unchanged. The presets are generated by `configs/generate_b4.py`; `python3 -m benchmarks.suite.reporting.b4 RESULTS_DIR CAMPAIGN --exclude TEXT` writes the tables and figures. Gloo does not work across the Linux and macOS PyTorch 2.13.0 builds (different rendezvous keys), so there is no cross-host Gloo preset.

### vLLM

`vllm_startup`, `vllm_correctness`, `vllm_latency`, `vllm_throughput`, `vllm_concurrency` and `vllm_pipeline` run **vLLM 0.31.0 pipeline-parallel inference over vllm-tbccl 0.2.0** (PP=2, TP=1, Qwen3-0.6B, bf16) and single-node references. The suite imports neither vLLM nor PyTorch; it starts the processes of an engine and a client that prints its measurements. One workload point is one **engine lifecycle**: the API node (pipeline stage 0, node rank 0), the headless node (stage 1), one passive sampler per host (CPU, memory, GPU, thermal state, the data interface's counters, and a census of the engine processes' TCP connections) and a stdlib-only streaming client (`vllm_bench/client.py`) that runs the point's ordered `options.phases` against the API on the API host. Ranks are pipeline stages; the `reverse` orientation swaps which machine runs stage 0 (and with it the API and the client), `rank_counts: [1]` with `rank_hosts: [host]` is a single-node reference. Stage devices are the `memory_backend` of each host: `cuda`, `metal`, `host` (that machine's CPU). Only `cuda`+`metal` in either order and `cuda`+`host` are accepted; anything else is rejected, not substituted.

Per host, `executables` names the environment: `vllm`, `vllm_python`, `vllm_model_dir`, `vllm_bench_dir` (the vllm-tbccl provider directory; required on a remote host unless the compatibility copy `vllm_bench/` was copied there), `vllm_workload` and `vllm_reference` (the API host's copies of the fixed prompt file and the single-rank reference; their SHA-256 are checked by the client) and, for a stage on another device, `name:device` overrides (`vllm:host`, `vllm_python:host` for the Mac CPU environment that has no vllm-metal). `options` carries `dtype` and `max_model_len` (required: nothing is picked silently), `gpu_memory_utilization` by host, `phases`, `primary` (the phase and metric the record's timing statistic is taken from) and `trace` (`tbccl`, `steps`, `full`: diagnostic runs only, see below). The fixed validated launch settings are written into every command: `--enforce-eager`, `--no-async-scheduling`, `--no-enable-prefix-caching`, `VLLM_USE_V2_MODEL_RUNNER=0`, `HF_HUB_OFFLINE=1` (nothing is downloaded).

**Client phases** (`kind`): `ready`, `warmup`, `rounds` (closed loop, `conc` simultaneous requests per round), `continuous` (closed loop, `conc` workers issuing their next request on completion), `open_loop` (Poisson arrivals at `rate`; offered and achieved rate are both reported), `reference` (greedy generations one at a time with the top-1/top-2 log-probability margin of every position), `check` (requests compared with the reference) and `cancel` (dropped streams, a drain check, then deterministic follow-ups). Closed-loop and open-loop results are labelled and never merged.

**Metric definitions** (`vllm_bench/metrics.py`, applied to the raw per-request timelines only): an *output event* is a server-sent event that carries at least one generated token; the token count of the request is the server-reported `usage.completion_tokens`, cross-checked against the event counts and the returned token ids, never the configured maximum. TTFT = first output event - submission; E2E = last output event - submission; TPOT = (last - first output event) / (tokens - tokens carried by the first event), undefined (never zero) when no token follows the first event; ITL = the intervals between successive events, per token only while every event carries one token (a multi-token event yields a *batched interval* that is counted but not turned into per-token timestamps). All durations come from one host's monotonic clock; timestamps of different hosts are never subtracted. Percentiles are linearly interpolated and withheld when there are too few observations (p90: 10, p95: 20, p99: 100).

**Correctness.** Output token ids are compared with a single-rank CUDA reference (same model, dtype and prompts). A divergence is *exact* (none), a *near tie* when the reference's own top-1/top-2 margin at the first divergent position is at most 0.251 (one bf16 spacing of a logit in [32, 64), plus float tolerance), and a failure above that: the row-order defect of vllm-tbccl 0.2.0rc1 produced margins of 5 to 10. The single-node CUDA control (`b5-tier0-control`) runs the same gates with no pipeline and shows the flips that batch composition alone causes. A failed gate means the point records no measurements.

**Transport proof.** A point is invalid unless (1) both workers logged `world_size=2 ... backend=tbccl`, (2) no log shows NCCL, Gloo or an MLX ring being initialised, (3) the Metal stage logged the TBCCL pipeline transport seam, (4) the vLLM version in every log is 0.31.0, (5) the sampler saw TCP connections between the two link addresses from the engine processes of each host and none to any other non-loopback address, and (6) the sampler data exists for both hosts. The passive sampler reports a lost engine process (a worker dead on the headless node, which vLLM's leader never notices) with exit status 7, and the runner fails the whole group on any member's nonzero exit; a bounded client timeout and the lease-supervised remote processes cover the rest.

Further tools: `options.client_host: other` runs the client on the other machine over the management LAN (a placement control; the API then binds all addresses and only its client connection is allowed outside the link); `options.trace` (`tbccl`, `steps`, `full`) turns on vllm-tbccl's own trace plus the suite's diagnostic hook `vllm_bench/trace_hook` (import-time wrappers around the CUDA/CPU worker's `execute_model` and the group's tensor-dict send/receive; no source of vLLM or vllm-tbccl is modified; traced runs are a separate campaign); `options.fault` (lifecycle tier only) kills one exact-title worker process once a request is running (`vllm_bench/fault.py`); `vllm_backend` is a model-free ping-pong of the pipeline boundary through vllm-tbccl's own backend at vLLM's GroupCoordinator level (`vllm_bench/pg_pingpong.py`, CUDA with CPU); `rank_hosts` in a workload selects the hosts of a single-node reference. A client request counts as successful only if the stream ends with `[DONE]` and a `length` or `stop` finish reason (a stream truncated by a dead engine is a failure), and a measured phase with any failed request ends the point unless it sets `allow_failures` (the lifecycle tier). Presets are generated by `configs/generate_b5.py` (`b5-tier0-reference`, `-tier0-control`, `-tier1-gates`, `-tier2-latency`, `-tier3-concurrency`, `-tier3b-mixed`, `-tier4-sweeps`, `-tier5b-placement`, `-tier6b-backend`, `-tier7-baselines`, `-tier8-lifecycle`, `-tier8-trace`, `-tier9-confirm`); `run --workload NAME` selects one point, and the report tool takes `--include-tier` so that instrumented, remote-client and confirmation runs are reported separately from the pooled tables. `python3 -m benchmarks.suite.reporting.b5 RESULTS_DIR CAMPAIGN` writes tables and figures from the raw timelines. Results use two additive throughput kinds, `inference_output_tokens_per_second` and `inference_requests_per_second` (units `tokens/s`, `requests/s`), and an `extra.inference` block; application metrics never use the transfer-throughput kinds. Schema version 1 is unchanged.

### Exo

`exo_correctness`, `exo_latency`, `exo_throughput`, `exo_transport_ab`, `exo_stage_profile`, `exo_large_model` and `exo_lifecycle` run **real Exo** (cluster discovery, master election, placement, runners, HTTP API and streaming) as a two-node pipeline, either with `MlxTbccl` (exo-tbccl over TBCCL) or with Exo's own `MlxRing`, which is the application-level baseline. The suite imports neither Exo nor MLX. One workload point is one **cluster lifecycle**: one `exo` node per host, one passive sampler per host and a stdlib client (`exo_bench/client.py`) on the first host that waits for the nodes, places one instance with Exo's normal placement (`Pipeline`, `min_nodes` 2; the suite never sets layer boundaries), reads back what was placed, waits for every runner, runs the ordered `options.phases`, deletes the instance and prints its evidence. A single node (`rank_counts: [1]`, `rank_hosts: [host]`, `options.backend: MlxRing`) is a single-device reference.

Per host, `executables` names the environment: `exo` (the node command), `exo_python` (the interpreter of the same environment: it runs the sampler and the client), `exo_bench_dir` (the exo-tbccl provider directory `benchmarks/provider/`; omit it to use this repository's compatibility copy `benchmarks/suite/exo_bench`, which on a remote host must be copied together with `vllm_bench/`), `exo_home` (an isolated `EXO_HOME`, relative to the user's home), `exo_dashboard_dir` (any directory with an `index.html`: only the HTTP API is used), `exo_lan_address` (a remote host's management address, so the client can read that node's own view of the cluster) and `exo_reference` (the reference file the correctness gates compare with). Install Exo on every node the way exo-tbccl's installation guide describes (the `tbccl-integration-v2` tag of exo-fork and the `tbccl` extra), keep the models in a local `models` directory of that `EXO_HOME`, and set nothing that downloads: every node runs with `EXO_OFFLINE=true` and `HF_HUB_OFFLINE=1`. Every node of a point gets a unique `EXO_ZENOH_NAMESPACE`, which also identifies the point's processes to the sampler (no process is ever matched by a name pattern). `options` carries `model_id`, `backend`, `prompts` (named prompts by sentence count or text), `phases`, `primary`, `memory_override_mb` by host (Exo's `OVERRIDE_MEMORY_MB`: Exo advertises host RAM, not GPU memory, and divides the layers by the advertised memory) and `tier`. The optional exo-tbccl fast paths (`EXO_TBCCL_RECV`, `EXO_TBCCL_ASYNC_SEND`, activity policies) are refused in a baseline point; a separate point must set `options.tier: fastpath`.

**Phases.** `warmup`, `rounds` and `continuous` are measured or untimed request loops; `reference` records a greedy single-device generation with the top-1/top-2 log-probability margin of every position; `check` compares requests with a reference; `cancel` drops streams after a few tokens (Exo cancels on disconnect) and proves with a follow-up check that the instance still answers correctly; `recreate` deletes the instance, places it again and checks the new placement. Exo exposes token text, not token ids, so correctness compares token texts: a difference is *exact* (none), a *near tie* when the reference's own margin at the first difference is below the declared tie margin (0.251, one bf16 spacing at the logit magnitudes involved), and a mismatch otherwise; a failed gate means the point records no measurements.

**Stream contract.** Exo sends one server-sent event per generated token (a token can decode to an empty string, so an event with an empty `content` is still a token), sets `finish_reason` and `usage` on the last event, and exposes no token ids. TTFT, TPOT, ITL and E2E are the definitions of the vLLM section applied to the arrival times of those events with `time.monotonic_ns()`; the token count of a request is the reported `usage.completion_tokens`, cross-checked against the event count (any difference, a missing finish reason, a missing `[DONE]` or an error event makes the request, and the point, fail its accounting). Prefill time is never inferred from TTFT.

**Placement and transport proof.** A point is invalid unless the placement read back from `/state` has the intended instance type (`MlxTbcclInstance` or `MlxRingInstance`), world size, model, contiguous layer ranges (and the layer counts of `options.expected_layers`, if given), uses only the data-link addresses, and is the same in both nodes' own views; the nodes agree on one master (read from their logs: Exo does not publish it); an `MlxTbccl` point also shows `TBCCL communicator ready` on both nodes and no ring group, and an `MlxRing` point no TBCCL communicator; and the sampler saw TCP connections between the two link addresses from the processes of the point's namespace. Exo chooses the stage order and the layer split per start, so every record carries `stage_hosts` and `stage_layers`, and the report module compares two lifecycles only when both are identical. The traffic of a pipeline cannot be predicted from the workload, so the adapter declares no traffic estimate and the runner's volume floor does not apply.

Diagnostic runs: `options.trace` sets `EXO_TBCCL_TRACE=1` and the point's processes load `exo_bench/trace_hook`, an import-time hook that attaches a file handler to exo-tbccl's logger and snapshots the adapter's `CopyStats` (copies made by exo-tbccl itself, not staging inside TBCCL's providers) without modifying Exo or exo-tbccl. Presets are generated by `configs/generate_b6.py`; `python3 -m benchmarks.suite.reporting.b6 RESULTS_DIR CAMPAIGN` writes tables from the raw timelines with the lifecycle as the replicated unit and refuses to compare backends across different placements.

## Extending the suite

1. Subclass `BenchmarkAdapter` in `adapters/` and fill in `Capabilities` (name, category, executable, operations, algorithms, memory backends, rank limits, native verification per operation).
2. `check_point` returns a reason when a point cannot run (a property of the point, not of the configuration); `validate_workload` returns configuration errors.
3. `describe` returns the payload semantics, timing method and completion semantics. Be exact: they decide what can be compared.
4. `plan` returns the argument array for each rank (never a shell string), which role is `persistent` (never exits by itself) and which rank waits for another's listener.
5. `parse` reads the executable's existing output and returns timing statistics with their provenance and percentile method, throughput with its definition, extras and the verification status. Raise `ParseError` on anything unexpected; do not invent a value the tool did not print.
6. Register it in `adapters/__init__.py`, add a saved real output under `tests/fixtures/`, and tests for parsing and the command line. Keep `measure(point, row)` separate from the file reading so importers can reuse it.

### Tests

```sh
python3 -m unittest discover -s benchmarks/suite/tests -t .       # unit tests, parsers, importers, runner failure injection, a short loopback run
TBCCL_BUILD_DIR=build-release python3 -m unittest discover -s benchmarks/suite/tests -t .
```

The loopback run is skipped when no executables are found and asserts no performance threshold. The tests are not part of CTest or the CMake build and need no framework. Performance numbers must never become unconditional CI gates: they depend on load, hardware state and topology.

## Known limitations

* **One machine pair, one link.** The suite has been exercised on one Linux laptop (NVIDIA RTX 3070 Ti Laptop GPU) and one Apple M4 Mac over a direct Thunderbolt 4 cable and over a LAN. Other hardware, other cables, other operating-system versions and other link types are untested. Results from different machines are not comparable by the suite and must not be presented as if they were.
* **TCP, not RDMA.** TBCCL's transport is TCP over IPv4 (IP over Thunderbolt on a direct cable). The suite measures that and nothing else.
* **Native-tool statistics only.** The native benchmarks print aggregates (a median, a mean, selected percentiles), so most records carry `reported` statistics. Tail percentiles are available only where a tool prints them or keeps raw samples; with 30 iterations p95 and p99 are not tail estimates.
* **Round trips are round trips.** `end_to_end` includes the ACK's return trip, a ping-pong is a round trip, and a submission time is not a completion time. The suite never halves a round trip into a one-way latency.
* **Cleanup after an abrupt host failure** or a network partition cannot be done from the controller. What bounds an orphan then is the lease (the supervisor kills the group when heartbeats stop or the connection closes, and its watchdog does the same if the supervisor dies) and the runtime cap; if the whole host is down or partitioned the processes are gone with it or are bounded by those timers, and the record says cleanup is unverified.
* **Descendants that clear their own environment** (and also leave the process group) are invisible to the token scan on every platform.
* **Remote benchmark output files are not copied back:** ranks on remote hosts are parsed from stdout (for `tb_pingpong` that is the rounded console line, which the record flags).
* **Probing and the independent cleanup check need the launcher to work a second time;** they are skipped, and reported as unverified, if it does not.
* **macOS** reports no Thunderbolt or PCIe error telemetry. A campaign on a Linux and macOS pair relies on the Linux kernel's AER and Thunderbolt events plus interface error counters on both hosts.
* **Gloo as a baseline.** The Linux and macOS PyTorch 2.13.0 Gloo builds did not rendezvous with each other on the tested pair, so the PyTorch adapter has no cross-host Gloo preset; NCCL has no macOS build. Where a baseline cannot run on both ends, the suite has none rather than a substitute.
* **Metal compute overlap** is not measured (no existing native tool has a Metal kernel); the CUDA overlap benchmark and a CPU-compute overlap benchmark exist.
* **Clock offsets between hosts are never estimated;** every latency comes from one process's monotonic clock inside the native tool or the suite's client.
* **Performance numbers are never CI gates.** They depend on load, thermal state, power management and topology; the tests assert behaviour, not speed.
* **Reports of historical campaigns are not part of this repository.** The [performance overview](performance-evaluation.md) summarises the reviewed results.
