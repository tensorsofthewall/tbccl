# Benchmark providers

A **provider** is a directory of framework-specific benchmark workloads that lives in the repository that owns the framework integration, plus a small entry point that tells the shared benchmark runner what the provider is and whether it can run here. The shared runner (`python3 -m benchmarks.suite`) stays framework-neutral: it imports no framework, loads no provider code into its own process, and starts a provider's workload scripts the way it starts every other benchmark, as supervised child processes.

| Provider | Owner | Directory (convention) | Workloads |
|---|---|---|---|
| native | this repository | built in | the executables built from this repository |
| `torch-tbccl` | the torch-tbccl repository | `benchmarks/provider/` | PyTorch collectives, point-to-point, asynchronous work, DDP |
| `vllm-tbccl` | the vllm-tbccl repository | `benchmarks/provider/` | vLLM pipeline inference |
| `exo-tbccl` | the exo-tbccl repository | `benchmarks/provider/` | Exo pipeline inference, `MlxRing` and `MlxTbccl` |

Adapter providers do not depend on each other and are tested in their own environments. Only this contract is shared.

## Three versions that move independently

| Version | Says | Value now |
|---|---|---|
| **Provider protocol** | how the runner talks to a provider | `1` (`providers.PROTOCOL_VERSION`; the runner speaks `[1]`) |
| **Result schema** | how a measurement record is represented | `tbccl-bench-result/1` (unchanged; provenance is added as an optional field) |
| **Methodology** | what a workload times and how | recorded in each record's `timing_method` and `completion_semantics` |

A protocol bump never changes a record, and a methodology change never needs a protocol bump. None of them is the TBCCL package version, the C ABI or the wire protocol, which this feature does not touch. A provider names the result schema versions it can produce (`supported_result_schema_versions`); the runner accepts a provider only if one of them is a version it reads.

## Operations

Protocol 1 has two operations that the provider implements and one that the runner owns.

**`provider.py describe`** prints exactly one JSON object on stdout and exits 0. It must be cheap and side-effect free: it must not import the framework, load weights, initialise CUDA or Metal, start Exo or vLLM, create a process group, download anything or change network configuration. Look packages up by metadata (`importlib.metadata`, `importlib.util.find_spec`). Logs go to stderr; anything else on stdout makes the description unusable. The runner limits the output to 256 KiB and the time to 30 seconds.

**`provider.py validate FILE`** reads a JSON workload configuration and prints `{"ok": bool, "errors": [...]}` (exit 0 if ok, 1 otherwise). The runner calls it in the preflight (not in a dry run): for every distinct planned workload of the provider it writes the configuration as a file **on each host that will run it** (through the helper for a remote host) and runs `validate` with the provider's own interpreter. A rejected configuration fails the preflight with the provider's reasons and nothing is launched; a provider that does not implement `validate` (usage exit 2) only produces a warning. The configuration holds the keys of the provider's `benchmark_configuration_schema` (workload, backend, payload bytes, warm-up, iterations, device pairing, model settings, tier and activity policy where they apply); it never changes a result or a measurement.

**Execution** is not a new interface. The runner builds the per-rank commands of a workload (the provider's scripts, run with the provider's interpreter), starts them under its own supervision and parses only the structured result lines they print (for example `TBCCL_TORCH_BENCH {json}`). The scripts take their configuration from arguments, a spec file or environment variables; they write machine-readable results as prefixed JSON lines on stdout (and raw evidence files into the directory the runner names), diagnostics on stderr, and they report no measurement after a correctness failure.

## The description

```json
{
  "provider_id": "torch-tbccl",
  "provider_protocol_version": 1,
  "provider_version": "0.2.0",
  "supported_workloads": ["collectives", "p2p", "async", "ddp"],
  "supported_platforms": ["linux", "darwin"],
  "supported_memory_kinds": ["host", "cuda", "mps"],
  "supported_device_pairings": ["cpu-cpu", "cpu-cuda", "cpu-mps", "cuda-mps"],
  "required_framework_versions": {"python": ">=3.13,<3.14", "torch": ">=2.13,<2.14"},
  "required_runtime_capabilities": ["torch.distributed process group 'tbccl'"],
  "optional_dependencies": ["psutil"],
  "required_model_configuration": {"required": false},
  "benchmark_configuration_schema": {"type": "object", "required": ["workload"], "properties": {}},
  "supported_result_schema_versions": ["tbccl-bench-result/1"],
  "entry_points": {"communication": "comm_bench.py", "ddp": "ddp_bench.py"},
  "entry_point_sha256": {"communication": "<sha256>", "ddp": "<sha256>"},
  "availability": {"status": "available", "reasons": [], "detected": {"torch": "2.13.0"}}
}
```

The JSON schema is `benchmarks/suite/schemas/provider-description-v1.json`. Beyond the schema the runner enforces:

* `provider_id` equals the id it was configured as;
* `provider_protocol_version` is one the runner speaks, and one of `supported_result_schema_versions` is a version it reads;
* `availability.status` is `available`, `environment_missing` or `unsupported`, and an unavailable provider lists its `reasons` (which package is missing, which platform is unsupported);
* `entry_points` are paths relative to the provider directory that do not leave it;
* `entry_point_sha256` (optional, recommended) names only known entry points and holds SHA-256 hex digests of the scripts the provider ships, so a result can say which scripts produced it;
* `detected` holds only what the provider could determine; a value it could not determine is `null`, never a guess. `detected.native_modules` is the SHA-256 and size of each compiled extension in the installed adapter distribution (read from the files, never loaded); the TBCCL library bundled in an adapter wheel is linked into it, so the hash identifies the binary that ran and a record does not have to assume it matches the TBCCL checkout the runner came from. It is `null` when the distribution is not installed.

## Configuring and discovering providers

A provider is **explicitly configured per host**; nothing is searched for on `PATH` or in the working directory.

```yaml
hosts:
  host-a:
    executables:
      torch_python: <PATH_TO_PYTHON_WITH_TORCH_TBCCL>
      torch_bench_dir: <PATH_TO_TORCH_TBCCL_CHECKOUT>/benchmarks/provider
```

The keys are `<name>_python` and `<name>_bench_dir` for `torch`, `vllm` and `exo`. The provider runs with its own interpreter, so each provider can live in its own virtual environment; do not put all frameworks into one environment.

`python3 -m benchmarks.suite providers [CONFIG]` lists the built-in native benchmarks and, per host, each provider's state: `available`, `environment missing`, `unsupported`, `not configured`, `describe failed` or `incompatible`, each with its reasons. A missing or broken provider never stops the native benchmarks or the other providers. Before a run starts, the preflight checks every framework provider the configuration uses on **every participating host** (`doctor CONFIG --probe` shows the same checks read-only): an unconfigured, missing, unsupported or incompatible provider, or a workload configuration the provider rejects, fails the preflight with its reason, and nothing is launched. A dry run only checks that the provider is configured. For a remote host the same call runs through the host's launcher and the suite's own helper; nothing is installed there.

A `run` records, for every framework workload, `environment.providers`: the provider id, protocol and provider version, the interpreter, the directory, the framework versions the provider detected, the hash of the compiled extension the adapter wheel ships and the hashes of its scripts. Versions it could not detect are absent, not invented. A library bundled in an adapter wheel is identified by what the running workload reports about itself at execution time (the workload scripts record the library, C ABI and wire protocol versions they actually loaded), not by the TBCCL checkout the runner was started from.

If a host configures no `<name>_bench_dir`, the runner falls back to the compatibility copy of the workload scripts that still ships in this repository (`benchmarks/suite/torch_bench`, `vllm_bench`, `exo_bench`) and records `status: bundled_compat` with the hashes of those scripts and no framework versions. The copies exist so that this repository works on its own until the providers are released, and they are removed once the adapter repositories ship them.

## Process ownership and security

* The runner owns process lifecycle, remote ownership, leases and heartbeats, timeouts, cancellation, cleanup, health checks and the evidence of failed runs. A provider describes what to launch; it does not start detached processes of its own.
* Provider code is executable code. Only a provider you configured is run; a description cannot add commands, and its entry points are relative to the provider directory. Review a provider before you configure it, as you would any script you run.
* `describe` runs with the working directory set to the provider directory, no shell, no standard input, and the same environment as other children minus every `TBCCL_*` variable.
* Result and log exports are sanitised by the runner's existing redaction; nothing is uploaded anywhere.

## Writing a provider

1. Put the workload scripts and `provider.py` in one directory of the framework's repository. Scripts print prefixed JSON lines and take everything from arguments or environment.
2. Make `describe` pass `tests/test_providers.py`-style checks: one JSON object, no framework import, a reason for every non-available state.
3. Verify results after the timed loop, and print no measurement when verification fails.
4. Add offline tests (a stand-in server or recorded outputs) and run them in CI without the framework, a model, a GPU or a second host.
5. Never import another adapter package.
6. Re-record the compatibility fixtures in this repository (`benchmarks/suite/tests/fixtures/providers/`) after a reviewed change to `provider.py`.
