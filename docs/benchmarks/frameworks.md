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

Each adapter documents its own workloads, timing definitions, results and limits in its own documentation; this site owns the ecosystem overview and the native results.

## Obtaining and pairing a provider

A provider is the `benchmarks/provider` directory of an adapter repository. Take it from the adapter release you are measuring, by downloading the source archive of its tag or checking the tag out, and set `*_bench_dir` to that directory; the adapter's wheel, which is installed in the framework environment, is a separate thing.

| Provider | Source of the provider directory | Runs in |
|---|---|---|
| `torch-tbccl` | tag `v0.3.0` of `tensorsofthewall/torch-tbccl` | a PyTorch 2.13 environment with torch-tbccl installed |
| `vllm-tbccl` | tag `v0.3.0` of `tensorsofthewall/vllm-tbccl` | a vLLM 0.31.0 environment with vllm-tbccl installed |
| `exo-tbccl` | tag `v0.4.0` of `tensorsofthewall/exo-tbccl` | the Exo environment with exo-tbccl installed |

The runner and a provider do not need the same package version. They are paired by three things that are checked, not assumed: the **provider protocol** (the runner supports protocol 1 and `describe` reports the provider's), the **result schema** (`tbccl-bench-result/1`, listed by `describe`) and the **runtime** (the adapter's own C ABI and wire protocol, which are unchanged across these releases). The identity of what actually ran is recorded in every result: the provider's version, the framework versions, and the hash of the compiled extension in the installed wheel. An adapter wheel carries the TBCCL library it was built with, so the library that ran is the adapter's, not the runner's. The suite identifies it by the hash of the compiled extension and never reports the runner's version as the library's; the library version, C ABI and wire protocol of an installed adapter are shown by its own `info()` command (for example `python -m torch_tbccl.info`).

## Where each provider is documented

| Framework | Provider repository | Benchmark documentation |
|---|---|---|
| PyTorch over torch-tbccl (and Gloo as a CPU baseline) | torch-tbccl | [torch-tbccl benchmarks](https://torch-tbccl.tensorsofthewall.com/en/latest/benchmarks/index.html) |
| vLLM pipeline inference over vllm-tbccl | vllm-tbccl | [vllm-tbccl benchmarks](https://vllm-tbccl.tensorsofthewall.com/en/latest/benchmarks/index.html) |
| Exo `MlxRing` and `MlxTbccl` | exo-tbccl | [exo-tbccl benchmarks](https://exo-tbccl.tensorsofthewall.com/en/latest/benchmarks/index.html) |

The workloads are also described in the [benchmark suite guide](../development/benchmark-suite.md). A host that does not configure a framework's provider directory cannot plan that framework's workloads: the plan says which key is missing, and the native benchmarks are unaffected.
