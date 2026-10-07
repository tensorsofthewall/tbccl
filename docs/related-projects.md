# Related projects

TBCCL itself knows nothing about machine-learning frameworks. These separate projects use an installed TBCCL; each has its own documentation, version and release schedule.

| Project | Role | Documentation | Source | Planned release |
|---|---|---|---|---|
| torch-tbccl | A PyTorch `torch.distributed` backend over an installed TBCCL. | [torch-tbccl.tensorsofthewall.com](https://torch-tbccl.tensorsofthewall.com/) | [source](https://github.com/tensorsofthewall/torch-tbccl) | 0.2.0 (unreleased) |
| vllm-tbccl | A vLLM platform plugin that routes communication through torch-tbccl. | [vllm-tbccl.tensorsofthewall.com](https://vllm-tbccl.tensorsofthewall.com/) | [source](https://github.com/tensorsofthewall/vllm-tbccl) | 0.2.0 (unreleased) |
| exo-tbccl | A pipeline-parallel data plane for exo over the TBCCL C ABI. | [exo-tbccl.tensorsofthewall.com](https://exo-tbccl.tensorsofthewall.com/) | [source](https://github.com/tensorsofthewall/exo-tbccl) | 0.3.0 (unreleased) |

Each project's documentation is hosted on its own site and the version shown there is built from `main` (development documentation) until that project has a release; a release documentation version exists only after the first release is tagged. Every project's documentation can also be built from its repository with `make docs` (see {doc}`development/building-docs`). The planned releases are unreleased targets, recorded in each project's `compatibility.json`.
