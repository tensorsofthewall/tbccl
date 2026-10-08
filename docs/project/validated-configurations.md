# Validated configurations

Configurations that were actually exercised, not a benchmark report. Detailed evidence is in the [validation reports](../validation/index.md). "Release-candidate artifacts" means the published 0.6.0rc1 archives were used, not a local build.

| Configuration | What was run | Result |
|---|---|---|
| Linux, loopback (several processes on one host) | the full test suite (CTest), world sizes 1 to 4, mixed point-to-point and collective traffic, failure and lifecycle tests, AddressSanitizer/UndefinedBehaviorSanitizer and ThreadSanitizer builds | passed (84 of 84 CTest cases with CUDA on Linux) |
| macOS, loopback | the full test suite | passed (75 of 75 CTest cases) |
| Linux CUDA host with macOS, direct Thunderbolt 4 link, host buffers | release-candidate artifacts, both rank orientations | passed in both orientations; link health unchanged |
| Linux CUDA with macOS Metal shared memory | release-candidate artifacts; CUDA staged through pinned host memory on one side and Metal shared memory on the other; both orientations | passed in both orientations; link health unchanged |
| Two Linux machines over a physical network | not part of the recorded validation | not claimed |

Adapter-level configurations (PyTorch DDP, vLLM pipeline parallelism, exo heterogeneous pipelines) are listed on the adapters' pages: [torch-tbccl](https://torch-tbccl.tensorsofthewall.com/en/latest/project/validated-configurations.html), [vllm-tbccl](https://vllm-tbccl.tensorsofthewall.com/en/latest/project/validated-configurations.html), [exo-tbccl](https://exo-tbccl.tensorsofthewall.com/en/latest/project/validated-configurations.html). The versions that were used together are in the [compatibility matrix](../reference/compatibility-matrix.md).
