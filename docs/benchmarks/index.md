# Benchmarks

TBCCL ships an optional benchmark suite that you run on **your** hardware. It drives the benchmark programs built from this repository, normalises their results into one schema, produces reports you can open offline, and compares two runs only when they measured the same thing. It is tooling around the runtime: it changes no C ABI, no wire protocol and no library behaviour.

```{toctree}
:maxdepth: 1

quickstart
your-hardware
native-results
methodology
reports
comparing
troubleshooting
frameworks
```

## What do you want to do?

| I want to... | Go to |
|---|---|
| see reference measurements with charts | [Native performance](native-results.md) |
| understand what is timed and what a number means | [Benchmark methodology](methodology.md) |
| run a benchmark in five minutes | [Quickstart](quickstart.md) |
| benchmark my own machines and link | [Running on your hardware](your-hardware.md) |
| generate a report from my run | [Report generation](reports.md) |
| compare two of my runs | [Comparing runs](comparing.md) |
| find out why a run failed | [Troubleshooting](troubleshooting.md) |
| benchmark PyTorch, vLLM or Exo | [Framework benchmarks](frameworks.md) |

## How the native and framework benchmarks fit together

One suite, one result schema and one report format cover everything. The native benchmarks are built from this repository. The PyTorch, vLLM and Exo workloads live in their own repositories as *providers* that the same runner drives, so a framework result is stored, reported and compared the same way as a native one. Each adapter site also documents how to run its workloads on their own, without the runner: [torch-tbccl](https://torch-tbccl.tensorsofthewall.com/en/latest/benchmarks/index.html), [vllm-tbccl](https://vllm-tbccl.tensorsofthewall.com/en/latest/benchmarks/index.html) and [exo-tbccl](https://exo-tbccl.tensorsofthewall.com/en/latest/benchmarks/index.html).

## What it can and cannot tell you

The reference measurements in [Native performance](native-results.md) come from one pair of machines joined by one cable. They show what the benchmarks produce and how large the effects were there. They are not a statement about your hardware, and you do not need them: the suite exists so you can measure your own systems and compare your own before-and-after runs. Do not compare your numbers with the reference numbers unless your hardware, software and workload match.

TBCCL's transport is TCP over IPv4. Whatever link joins the machines, the benchmarks measure TCP goodput and completion times through two operating-system network stacks, not the physical link rate, and there is no RDMA or direct device-to-device DMA in this transport.

Developer reference (configuration schema, adapters, the provider protocol, hardware safety rules): [benchmark suite guide](../development/benchmark-suite.md), [benchmark providers](../development/benchmark-providers.md), [baselines and regression comparison](../development/benchmark-baselines.md).
