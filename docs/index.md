# TBCCL

:::{container} tb-lead
TBCCL (Thunderbolt Collective Communication Library) lets machines that differ in operating system and accelerator work together. It moves tensors between hosts, for example a Linux machine with an NVIDIA GPU and a Mac with Apple silicon, over TCP/IP, including IP networking over a direct Thunderbolt 4 link. It is a C++17 runtime with a stable C ABI; PyTorch, vLLM and exo reach it through separate adapter packages.
:::

:::{container} tb-actions
[Get started](getting-started/overview.md) [Explore the ecosystem](#the-tbccl-ecosystem)
:::

The transport is TCP over IPv4. TBCCL does not use RDMA, and there is no direct GPU-to-GPU DMA: CUDA buffers are staged through pinned host memory and Metal shared memory is handed to the transport directly.

```{note}
**Release.** The current release is 0.6.1 (C ABI 1, wire protocol 4), a patch release of 0.6.0 that adds the public benchmark suite and no runtime change. TBCCL is experimental.
```

<nav class="tb-eco" aria-label="TBCCL ecosystem: the four documentation sites">
<ul>
<li class="tb-tbccl" aria-current="page"><span><span class="tb-eco-name">tbccl</span><span class="tb-eco-role">native runtime</span><span class="tb-eco-here">This site</span></span></li>
<li class="tb-torch"><a href="https://torch-tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">torch-tbccl</span><span class="tb-eco-role">PyTorch backend</span></a></li>
<li class="tb-vllm"><a href="https://vllm-tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">vllm-tbccl</span><span class="tb-eco-role">vLLM integration</span></a></li>
<li class="tb-exo"><a href="https://exo-tbccl.tensorsofthewall.com/en/latest/"><span class="tb-eco-name">exo-tbccl</span><span class="tb-eco-role">exo integration</span></a></li>
</ul>
</nav>

## The TBCCL ecosystem

Four projects, one runtime. **tbccl** is the native communication runtime. **torch-tbccl**, **vllm-tbccl** and **exo-tbccl** are separate packages: each connects one framework to the same runtime, carries its own copy of libtbccl, and works without the others.

::::{container} tb-cards
:::{container} tb-card tb-tbccl

<h3>tbccl</h3>

Native communication runtime. A C++17 library with a stable C ABI: point-to-point and collectives over TCP, with CUDA and Metal shared memory support. Use it to program against TBCCL from C or C++.

<span class="tb-meta">This site · [GitHub](https://github.com/tensorsofthewall/tbccl)</span>
:::

:::{container} tb-card tb-torch

<h3>torch-tbccl</h3>

PyTorch distributed backend. A `torch.distributed` backend named `"tbccl"`: collectives, point-to-point and DDP across a Linux CUDA host and a Mac.

[Documentation](https://torch-tbccl.tensorsofthewall.com/en/latest/) · [GitHub](https://github.com/tensorsofthewall/torch-tbccl)

<span class="tb-meta">Separate package, includes libtbccl</span>
:::

:::{container} tb-card tb-vllm

<h3>vllm-tbccl</h3>

Distributed inference integration. A vLLM platform plugin for two-stage pipeline-parallel serving across a CUDA node and a Mac.

[Documentation](https://vllm-tbccl.tensorsofthewall.com/en/latest/) · [GitHub](https://github.com/tensorsofthewall/vllm-tbccl)

<span class="tb-meta">Separate package, includes libtbccl</span>
:::

:::{container} tb-card tb-exo

<h3>exo-tbccl</h3>

Exo pipeline integration. An `MlxTbccl` pipeline data plane for exo: pipeline-parallel text generation across a Mac and a Linux CUDA host.

[Documentation](https://exo-tbccl.tensorsofthewall.com/en/latest/) · [GitHub](https://github.com/tensorsofthewall/exo-tbccl)

<span class="tb-meta">Separate package, includes libtbccl</span>
:::
::::

The [ecosystem overview](project/ecosystem.md) explains how the projects fit together and the [compatibility matrix](reference/compatibility-matrix.md) lists exactly which versions work together.

## How the pieces fit

<figure class="tb-figure">
<svg class="tb-wide" viewBox="0 0 760 380" role="img" aria-labelledby="tb-arch-t tb-arch-d" xmlns="http://www.w3.org/2000/svg">
<title id="tb-arch-t">TBCCL architecture</title>
<desc id="tb-arch-d">Four kinds of application sit on top: PyTorch programs, vLLM serving, exo clusters and C or C++ programs. PyTorch, vLLM and exo reach the TBCCL runtime through the separate adapter packages torch-tbccl, vllm-tbccl and exo-tbccl. C and C++ programs call the runtime directly. The runtime, libtbccl, sends data over TCP on IPv4, which can be any IP network including IP over a direct Thunderbolt 4 link. There is no RDMA and no direct GPU-to-GPU DMA.</desc>
<defs><marker id="tb-ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="tb-arrow" d="M0 0L10 5L0 10z"/></marker></defs>
<g class="tb-box"><rect x="10" y="10" width="170" height="52" rx="6"/><rect x="200" y="10" width="170" height="52" rx="6"/><rect x="390" y="10" width="170" height="52" rx="6"/><rect x="580" y="10" width="170" height="52" rx="6"/></g>
<text class="tb-label" x="95" y="40" text-anchor="middle">PyTorch programs</text><text class="tb-label" x="285" y="40" text-anchor="middle">vLLM serving</text><text class="tb-label" x="475" y="40" text-anchor="middle">exo clusters</text><text class="tb-label" x="665" y="40" text-anchor="middle">C and C++ programs</text>
<path class="tb-line" marker-end="url(#tb-ah)" d="M95 62V112"/><path class="tb-line" marker-end="url(#tb-ah)" d="M285 62V112"/><path class="tb-line" marker-end="url(#tb-ah)" d="M475 62V112"/><path class="tb-line" marker-end="url(#tb-ah)" d="M665 62V232"/>
<g class="tb-box"><rect x="10" y="114" width="170" height="66" rx="6"/><rect x="200" y="114" width="170" height="66" rx="6"/><rect x="390" y="114" width="170" height="66" rx="6"/></g>
<rect x="10" y="114" width="170" height="4" style="fill:var(--tb-accent-torch)"/><rect x="200" y="114" width="170" height="4" style="fill:var(--tb-accent-vllm)"/><rect x="390" y="114" width="170" height="4" style="fill:var(--tb-accent-exo)"/>
<text class="tb-label" x="95" y="144" text-anchor="middle">torch-tbccl</text><text class="tb-small" x="95" y="164" text-anchor="middle">PyTorch backend</text>
<text class="tb-label" x="285" y="144" text-anchor="middle">vllm-tbccl</text><text class="tb-small" x="285" y="164" text-anchor="middle">vLLM platform plugin</text>
<text class="tb-label" x="475" y="144" text-anchor="middle">exo-tbccl</text><text class="tb-small" x="475" y="164" text-anchor="middle">exo pipeline data plane</text>
<path class="tb-line" marker-end="url(#tb-ah)" d="M95 180V232"/><path class="tb-line" marker-end="url(#tb-ah)" d="M285 180V232"/><path class="tb-line" marker-end="url(#tb-ah)" d="M475 180V232"/>
<rect class="tb-box" x="10" y="234" width="740" height="62" rx="6"/><rect x="10" y="234" width="740" height="4" style="fill:var(--tb-accent-tbccl)"/>
<text class="tb-label" x="380" y="263" text-anchor="middle">tbccl runtime (libtbccl)</text><text class="tb-small" x="380" y="283" text-anchor="middle">C ABI v1 · point-to-point and collectives · CUDA staged through pinned host memory · Metal shared memory</text>
<path class="tb-line" marker-end="url(#tb-ah)" d="M380 296V332"/>
<rect class="tb-box" x="10" y="334" width="740" height="40" rx="6"/>
<text class="tb-label" x="380" y="359" text-anchor="middle">TCP over IPv4: any IP network, including IP over a direct Thunderbolt 4 link</text>
</svg>
<svg class="tb-narrow" viewBox="0 0 340 470" role="img" aria-labelledby="tb-arch-t2 tb-arch-d2" xmlns="http://www.w3.org/2000/svg">
<title id="tb-arch-t2">TBCCL architecture</title>
<desc id="tb-arch-d2">Applications are on top. PyTorch, vLLM and exo programs use the adapters torch-tbccl, vllm-tbccl and exo-tbccl, and C or C++ programs call the runtime directly. Below the adapters is the tbccl runtime, libtbccl, which sends data over TCP on IPv4, including IP over a direct Thunderbolt 4 link. There is no RDMA and no direct GPU-to-GPU DMA.</desc>
<defs><marker id="tb-ah2" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path class="tb-arrow" d="M0 0L10 5L0 10z"/></marker></defs>
<rect class="tb-box" x="5" y="5" width="330" height="62" rx="6"/>
<text class="tb-label" x="170" y="30" text-anchor="middle">Applications</text><text class="tb-small" x="170" y="50" text-anchor="middle">PyTorch · vLLM · exo · your C or C++ code</text>
<path class="tb-line" marker-end="url(#tb-ah2)" d="M60 67V100"/><path class="tb-line" marker-end="url(#tb-ah2)" d="M170 67V100"/><path class="tb-line" marker-end="url(#tb-ah2)" d="M280 67V100"/>
<g class="tb-box"><rect x="5" y="102" width="106" height="64" rx="6"/><rect x="117" y="102" width="106" height="64" rx="6"/><rect x="229" y="102" width="106" height="64" rx="6"/></g>
<rect x="5" y="102" width="106" height="4" style="fill:var(--tb-accent-torch)"/><rect x="117" y="102" width="106" height="4" style="fill:var(--tb-accent-vllm)"/><rect x="229" y="102" width="106" height="4" style="fill:var(--tb-accent-exo)"/>
<text class="tb-label" x="58" y="132" text-anchor="middle">torch-tbccl</text><text class="tb-small" x="58" y="152" text-anchor="middle">PyTorch</text>
<text class="tb-label" x="170" y="132" text-anchor="middle">vllm-tbccl</text><text class="tb-small" x="170" y="152" text-anchor="middle">vLLM</text>
<text class="tb-label" x="282" y="132" text-anchor="middle">exo-tbccl</text><text class="tb-small" x="282" y="152" text-anchor="middle">exo</text>
<path class="tb-line" marker-end="url(#tb-ah2)" d="M58 166V232"/><path class="tb-line" marker-end="url(#tb-ah2)" d="M170 166V232"/><path class="tb-line" marker-end="url(#tb-ah2)" d="M282 166V232"/>
<rect class="tb-box" x="5" y="234" width="330" height="92" rx="6"/><rect x="5" y="234" width="330" height="4" style="fill:var(--tb-accent-tbccl)"/>
<text class="tb-label" x="170" y="262" text-anchor="middle">tbccl runtime (libtbccl)</text>
<text class="tb-small" x="170" y="284" text-anchor="middle">C ABI v1 · point-to-point and collectives</text><text class="tb-small" x="170" y="302" text-anchor="middle">CUDA staged through pinned host memory</text><text class="tb-small" x="170" y="318" text-anchor="middle">Metal shared memory</text>
<path class="tb-line" marker-end="url(#tb-ah2)" d="M170 326V364"/>
<rect class="tb-box" x="5" y="366" width="330" height="92" rx="6"/>
<text class="tb-label" x="170" y="394" text-anchor="middle">TCP over IPv4</text><text class="tb-small" x="170" y="416" text-anchor="middle">Any IP network, including IP over</text><text class="tb-small" x="170" y="434" text-anchor="middle">a direct Thunderbolt 4 link</text><text class="tb-small" x="170" y="450" text-anchor="middle">No RDMA, no direct GPU DMA</text>
</svg>
<figcaption>Applications use TBCCL directly or through an adapter. Every path ends in the same runtime and the same TCP transport; the adapters are separate packages and do not depend on each other.</figcaption>
</figure>

## Get started

::::{container} tb-cards
:::{container} tb-card 

<h3>Run a local example</h3>

A real two-rank collective on loopback, on one machine, in about five minutes.

[Run the demo](getting-started/examples.md)
:::

:::{container} tb-card 

<h3>Connect two machines over Thunderbolt</h3>

Bring up the link and verify it before running anything across it.

[Two-machine setup](getting-started/two-machine-thunderbolt.md) · [Troubleshooting](guides/thunderbolt-troubleshooting.md)
:::

:::{container} tb-card 

<h3>Use PyTorch</h3>

Install torch-tbccl with the install helper, then run a two-host all-reduce.

[Install helper](getting-started/install-adapters.md) · [torch-tbccl quickstart](https://torch-tbccl.tensorsofthewall.com/en/latest/getting-started/quickstart.html)
:::

:::{container} tb-card 

<h3>Use vLLM</h3>

Serve a heterogeneous vLLM pipeline across a CUDA node and a Mac.

[vllm-tbccl install guide](https://vllm-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html)
:::

:::{container} tb-card 

<h3>Use Exo</h3>

Run an exo pipeline across a Mac and a Linux CUDA host.

[exo-tbccl two-host guide](https://exo-tbccl.tensorsofthewall.com/en/latest/guides/two-host.html)
:::
::::

## What works today

Only configurations that were actually exercised are listed as demonstrated. Everything else is either not supported or not claimed.

::::{container} tb-cols
:::{container}
**Demonstrated**

- Linux x86-64 (host and CUDA 13 buffers) and macOS arm64 (host and Metal shared buffers).
- A Linux CUDA host and a Mac joined by a direct Thunderbolt 4 link, in both rank orientations.
- World sizes 1 to 4: point-to-point, `broadcast`, `all_gather`, `barrier` and `all_reduce` with `Sum`.
- The three adapters on that hardware: PyTorch collectives between a CUDA host and a Mac, a two-stage vLLM pipeline and an exo pipeline (see each project's validated configurations).
:::
:::{container}
**Not supported or not claimed**

- RDMA or any transport other than TCP over IPv4; direct CUDA-to-Metal DMA does not exist.
- Reductions other than `Sum`; recovery after a communicator failure; authentication and encryption.
- Windows, 32-bit and ARM Linux.
- Not validated: world sizes 5 to 8, two Linux machines over a physical network, links other than Thunderbolt 4.
:::
::::

Authoritative pages: [validated configurations](project/validated-configurations.md), [known limitations](project/limitations.md), [platform capabilities](reference/platform-capabilities.md) and the [compatibility matrix](reference/compatibility-matrix.md).

## Measured performance

:::{container} tb-callout
On one pair of machines joined by one direct Thunderbolt 4 cable (a Linux laptop with an NVIDIA GPU and an Apple M4 Mac mini, measured with TBCCL 0.6.0), a 16 MiB transfer ran at about 15.5 Gbit/s from Linux to Mac and 18.3 Gbit/s the other way, matching a single iperf3 TCP stream (15.8 and 18.8 Gbit/s). The 64-byte round trip, 128 microseconds, equals raw TCP. These figures describe TCP goodput through two operating-system network stacks, not the Thunderbolt link rate, and they are not a statement about other hardware. [Native performance](benchmarks/native-results.md) has the method, the ranges and the caveats; the [benchmark suite](benchmarks/index.md) measures your own machines.
:::

Questions and feedback: [getting help](project/getting-help.md); there is [no telemetry](project/privacy.md).

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
:caption: Benchmarks

benchmarks/index
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
