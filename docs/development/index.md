# Development

This repository, **tbccl**, is the runtime. It owns the transport, the collectives, the C ABI, the wire protocol and the CUDA and Metal-shared memory providers. It must not depend on any framework.

| Belongs here | Does not belong here |
|---|---|
| transport, collective algorithms, bootstrap, wire protocol | PyTorch, vLLM, exo or MLX concepts |
| the C ABI and the C++ API | model, layer or placement logic |
| CUDA staging and Metal-shared handling | framework error types and lifecycle hooks |
| `tbccl-info`, release archives, the compatibility manifest | anything that only one adapter needs |

Start with [contributing](contributing.md), then [where should this change go?](where-should-this-change-go.md) and the [code map](code-map.md). The rules for the code base are in `AGENTS.md` at the repository root.

```{toctree}
:maxdepth: 1

contributing
where-should-this-change-go
code-map
testing
staged-transfer-cancellation
benchmark-methodology
benchmark-suite
benchmark-suite-inventory
benchmark-baselines
benchmark-providers
performance-evaluation
building-docs
release-process
maintenance
documentation-versions
```
