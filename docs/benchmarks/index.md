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

| Question | Where |
|---|---|
| How do I run a benchmark in five minutes? | [Quickstart](quickstart.md) |
| How do I benchmark my own machines and link? | [Running on your hardware](your-hardware.md) |
| How does TBCCL perform, on what hardware, and what do the numbers mean? | [Native performance](native-results.md) |
| What exactly is timed, and what are the limits of a number? | [Benchmark methodology](methodology.md) |
| How do I get a report from my own run? | [Report generation](reports.md) |
| How do I compare two runs? | [Comparing runs](comparing.md) |
| Something failed. | [Troubleshooting](troubleshooting.md) |
| PyTorch, vLLM or Exo? | [Framework benchmarks](frameworks.md) |

## What it can and cannot tell you

The reference measurements in [Native performance](native-results.md) come from one pair of machines joined by one cable. They show what the benchmarks produce and how large the effects were there. They are not a statement about your hardware, and you do not need them: the suite exists so you can measure your own systems and compare your own before-and-after runs. Do not compare your numbers with the reference numbers unless your hardware, software and workload match.

TBCCL's transport is TCP over IPv4. Whatever link joins the machines, the benchmarks measure TCP goodput and completion times through two operating-system network stacks, not the physical link rate, and there is no RDMA or direct device-to-device DMA in this transport.

Developer reference (configuration schema, adapters, the provider protocol, hardware safety rules): [benchmark suite guide](../development/benchmark-suite.md), [benchmark providers](../development/benchmark-providers.md), [baselines and regression comparison](../development/benchmark-baselines.md).
