# TBCCL

TBCCL (Thunderbolt Collective Communication Library) is a C++17 runtime with a stable C ABI that moves tensors between machines that differ in operating system and memory type, for example a Linux host with an NVIDIA GPU and a Mac with Apple silicon, over TCP, including a direct Thunderbolt 4 link.

```{note}
**Release.** The current release is 0.6.0 (C ABI 1, wire protocol 4). It is the first release and is experimental.
```

## Which project do I need?

This site is the entry point for the whole TBCCL ecosystem. Four projects share one runtime:

| I want to... | Use | Documentation |
|---|---|---|
| call TBCCL directly from C or C++ | **tbccl** (this site) | [tbccl.tensorsofthewall.com](https://tbccl.tensorsofthewall.com/en/latest/) |
| run PyTorch `torch.distributed` (collectives, DDP) across machines | **torch-tbccl** | [torch-tbccl.tensorsofthewall.com](https://torch-tbccl.tensorsofthewall.com/en/latest/) |
| run a vLLM pipeline across a CUDA node and a Mac | **vllm-tbccl** | [vllm-tbccl.tensorsofthewall.com](https://vllm-tbccl.tensorsofthewall.com/en/latest/) |
| run an exo pipeline across a Mac and a Linux CUDA host | **exo-tbccl** | [exo-tbccl.tensorsofthewall.com](https://exo-tbccl.tensorsofthewall.com/en/latest/) |

```text
 PyTorch ---- torch-tbccl ----+
 vLLM ------- vllm-tbccl -----+---> libtbccl ---> TCP (Thunderbolt 4 or any IP network)
 exo --------- exo-tbccl -----+
```

torch-tbccl, vllm-tbccl and exo-tbccl are independent adapters. Each carries or links its own libtbccl, and none requires another adapter. The [ecosystem overview](project/ecosystem.md) explains how the projects fit together and the [compatibility matrix](reference/compatibility-matrix.md) lists exactly which versions work together.

**Get started.**

1. **One machine, five minutes:** the [demo](getting-started/examples.md) runs a real two-rank collective on loopback.
2. **Two machines over Thunderbolt:** [connect and verify the link](getting-started/two-machine-thunderbolt.md), with a [troubleshooting guide](guides/thunderbolt-troubleshooting.md).
3. **PyTorch `torch.distributed`:** install with the [install helper](getting-started/install-adapters.md), then the torch-tbccl [two-host all-reduce](https://torch-tbccl.tensorsofthewall.com/en/latest/getting-started/quickstart.html).
4. **Heterogeneous vLLM** (CUDA node and a Mac): the vllm-tbccl [install guide](https://vllm-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html).
5. **Exo:** the exo-tbccl [two-host guide](https://exo-tbccl.tensorsofthewall.com/en/latest/guides/two-host.html).

Then read what is [supported and validated](project/validated-configurations.md). Questions and feedback: [getting help](project/getting-help.md); there is [no telemetry](project/privacy.md).

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/overview
getting-started/install
getting-started/install-adapters
getting-started/two-machine-thunderbolt
getting-started/first-program
getting-started/examples
```

```{toctree}
:maxdepth: 2
:caption: User guide

guides/c-api-quickstart
guides/thunderbolt-link
guides/thunderbolt-troubleshooting
reference/platform-capabilities
reference/configuration
```

```{toctree}
:maxdepth: 2
:caption: Architecture

concepts/index
adr/index
```

```{toctree}
:maxdepth: 2
:caption: Reference

reference/index
```

```{toctree}
:maxdepth: 2
:caption: Troubleshooting

guides/troubleshooting
guides/diagnostics
```

```{toctree}
:maxdepth: 2
:caption: Development

development/index
```

```{toctree}
:maxdepth: 2
:caption: Project

project/index
validation/index
```
