# Compatibility matrix

This is the canonical statement of what works with what across the TBCCL ecosystem. The adapter documentation links here for core transport compatibility and adds its own, smaller matrix. The machine-readable sources are the `compatibility.json` files in each repository; the numbers below come from them.

## Release-candidate family

| Project | Version | Role | Distribution |
|---|---|---|---|
| tbccl | 0.6.0rc1 | runtime: C++ library, C ABI, TCP transport | native archives (Linux host, Linux CUDA 13, macOS arm64) |
| torch-tbccl | 0.2.0rc1 | PyTorch `torch.distributed` backend | platform wheels (CPython 3.13) |
| vllm-tbccl | 0.2.0rc1 | vLLM platform plugin | platform wheels and an sdist (CPython 3.13) |
| exo-tbccl | 0.3.0rc1 | exo pipeline data plane | platform wheels (CPython 3.13 or newer) |

All four are pre-release candidates attached to GitHub pre-releases. None is published on PyPI or TestPyPI. The final versions (0.6.0, 0.2.0, 0.2.0, 0.3.0) do not exist yet.

**Interface versions shared by the whole family:** C ABI **1**, wire protocol **4**. See [versioning](versioning.md) for what each version number means. Ranks that speak different wire protocols cannot communicate, and every adapter wheel bundles the libtbccl it was built with, so all ranks of one job must use adapters built against wire protocol 4.

## Platforms and devices

| | Linux x86-64 | macOS arm64 (Apple silicon) |
|---|---|---|
| libtbccl memory kinds | `Host`, `Cuda` | `Host`, `MetalShared` |
| CUDA | 13.x (archive built with CUDA 13.0, GPU architectures sm75 to sm120); validated on an NVIDIA RTX 3070 Ti Laptop GPU | not applicable |
| Metal / MLX | not applicable | Metal shared (unified) memory; MLX 0.32 for exo-tbccl |
| Validated machine | x86-64 Linux with an NVIDIA GPU | Apple M4 |
| Not built or tested | Windows, 32-bit, ARM Linux, AMD and Intel GPUs | Intel Macs |

## Software versions

| | tbccl | torch-tbccl | vllm-tbccl | exo-tbccl |
|---|---|---|---|---|
| Python | n/a (C/C++) | 3.13 only | 3.13 only | 3.13 or newer |
| PyTorch | n/a | 2.13.x (CUDA 13 build on Linux; macOS wheel on Mac) | 2.13.x | not used |
| vLLM | n/a | n/a | 0.30.0 (CUDA and Metal pairing, needs vllm-metal) and 0.31.0 (CUDA and CPU pairing) | n/a |
| exo | n/a | n/a | n/a | the exo integration with the `PipelineComm` seam and the runner byte exchange |
| MLX | n/a | n/a | via vllm-metal (Metal pairing) | 0.32 |
| TBCCL C ABI / wire | defines 1 / 4 | requires 1 / 4 | requires 1 / 4 (bundled) | requires C ABI 1; minimum package 0.6.0 |

## World sizes

| Project | Validated | Notes |
|---|---|---|
| tbccl | 1 to 4 ranks (up to 8 accepted) | Float16/BFloat16 reductions at world size 2 only |
| torch-tbccl | 1 to 4 (DDP 2 to 4); CUDA with MPS at 2 | collectives need at least 2 ranks |
| vllm-tbccl | 2 (pipeline parallel, `PP=2`, `TP=1`) | tensor parallelism above 1 not exercised |
| exo-tbccl | 2 stages on hardware; 2 to 4 synthetic stages on loopback | pipeline parallelism only |

## Supported workload types

| Workload | Project | State |
|---|---|---|
| point-to-point and `Sum` `all_reduce`, `broadcast`, `all_gather`, `barrier` from C/C++ | tbccl | validated |
| `torch.distributed` collectives, point-to-point, DDP training | torch-tbccl | validated; no FSDP, no recovery |
| two-stage pipeline-parallel LLM serving (CUDA with CPU, CUDA with Metal) | vllm-tbccl | validated with one small model |
| pipeline-parallel text generation (Mac Metal with Linux CUDA) | exo-tbccl | validated; not tensor parallelism, not image models |

## Where each statement is checked

`scripts/check_compatibility_manifest.py` (run in CI and by `make check-compat`) fails if `compatibility.json` disagrees with the C ABI header and the wire protocol constant. Each adapter has an equivalent check of its own manifest. The adapters' compatibility pages: [torch-tbccl](https://torch-tbccl.tensorsofthewall.com/en/latest/reference/compatibility.html), [vllm-tbccl](https://vllm-tbccl.tensorsofthewall.com/en/latest/reference/compatibility.html), [exo-tbccl](https://exo-tbccl.tensorsofthewall.com/en/latest/reference/compatibility.html).
