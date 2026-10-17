# Framework benchmarks

The shared runner is framework-neutral: it imports no framework and works with none installed. Workloads that need PyTorch, vLLM or Exo are run by a **provider**, a directory of workload scripts that lives in the repository of the integration that owns the framework, together with a `provider.py` entry point that tells the runner what the provider is and whether this environment can run it.

```sh
python3 -m benchmarks.suite providers my-setup.local.yaml
```

```text
native       available
torch-tbccl  available            on host-a
vllm-tbccl   environment missing  on host-a  (Python package 'vllm' (module 'vllm') is not installed in this interpreter)
exo-tbccl    not configured       on host-a  (hosts.host-a.executables.exo_python, exo_bench_dir not set)
```

The command lists, for every host of your configuration, whether each provider is available, missing something (with what), unsupported on that platform, not configured, or broken. A provider that is not usable never stops the native benchmarks or the other providers. When you `run` a configuration, every provider it uses is checked on every participating host first; one that is missing or broken fails the preflight with its reason and nothing is launched (`doctor CONFIG --probe` shows the same checks without running anything). A provider is something you configure explicitly: the runner never searches `PATH` or the working directory for one, and it runs the provider's scripts with the provider's own interpreter, so each framework can live in its own environment.

A provider is configured per host with the interpreter of the framework environment and the provider directory:

```yaml
hosts:
  host-a:
    executables:
      torch_python: <PATH_TO_PYTHON_WITH_THE_FRAMEWORK>
      torch_bench_dir: <PATH_TO_THE_ADAPTER_CHECKOUT>/benchmarks/provider
```

(`vllm_python` / `vllm_bench_dir` and `exo_python` / `exo_bench_dir` for the other two.) For a multi-host run every participating host is checked.

Every framework result records which provider produced it: the provider and protocol versions, the interpreter, the framework versions the provider detected, the hash of the compiled extension the adapter wheel ships and the hashes of its scripts. See [benchmark providers](../development/benchmark-providers.md) for the protocol and for how to write one, and use the `pytorch`, `vllm` and `exo` [profiles](quickstart.md) as starting points.

Each adapter documents its own workloads, timing definitions, results and limits in its own documentation. The workloads are also described in the [benchmark suite guide](../development/benchmark-suite.md).

## Compatibility copy

Until the adapter repositories ship their providers, this repository still contains a copy of each framework's workload scripts (`benchmarks/suite/torch_bench`, `vllm_bench`, `exo_bench`). A host that configures only the interpreter (no `*_bench_dir`) uses that copy, and the result records it as `bundled_compat` with the hashes of those scripts and without framework versions. Prefer the provider.
