# Platform capabilities

What works where. "Validated" means exercised by the project's tests; see [Validation](../validation/index.md).

## Platforms

| Platform | Build | Memory kinds | Notes |
|---|---|---|---|
| Linux x86-64 | host-only, or with CUDA (`-DTBCCL_ENABLE_CUDA=ON`) | `Host`, `Cuda` | validated with an NVIDIA GPU |
| macOS arm64 | host-only, or with Metal (`-DTBCCL_ENABLE_METAL=ON`) | `Host`, `MetalShared` | validated on Apple silicon |

The C ABI header's layouts are pinned by static assertions for LP64 Linux and macOS (arm64 and x86-64). TCP over IPv4 is the only transport. Windows is neither built nor tested.

## Operations by world size

| Operation | World size 1 | World size 2 | World size 3 and above |
|---|---|---|---|
| `send` / `recv` | n/a | yes | yes |
| `broadcast`, `all_gather` (byte-generic) | local | yes | yes |
| `barrier` | local | yes | yes |
| `all_reduce` `Sum` | local | yes | yes (Float16 and BFloat16 excluded) |

World sizes 1 to 4 are validated; up to 8 are accepted. The two-rank collective path has no descriptor check, so mismatched counts or sizes between the two ranks are not detected there.

## Reduction datatypes

| Datatype | Host | Cuda | MetalShared | World size |
|---|---|---|---|---|
| Float32, Float64, Int32, Int64 | yes | yes | yes | any |
| Int8, UInt8 (sum modulo 256) | yes | yes | no | any |
| Float16, BFloat16 | yes | yes | no | 2 only |
| FP8, packed INT4/FP4, anything else | byte transport only | byte transport only | byte transport only | not reducible |

Only `Sum` is available through `Communicator::all_reduce` and the C `tbcclAllReduce` call accepts the valid (datatype, op) combinations of the runtime; others return `unsupported`. `Communicator::capabilities()` and `tbcclCommSupportsAllReduce` answer for a given rank, memory kind, datatype and op. See [Low-precision types and quantized payloads](../concepts/low-precision.md).

## Memory-kind requirements

| Kind | Requirement |
|---|---|
| `Host` | none |
| `MetalShared` | an `MTLBuffer` with shared storage; its CPU-visible `contents` pointer |
| `Cuda` | CUDA component built and `register_cuda_support()` called; out-of-place `all_reduce` is not supported |

## Not supported

Hidden rendezvous bootstrap, group start/end calls, user tags, communicator split, shrink or grow, fault recovery, FP8 or INT4 reductions, non-IPv4 endpoints, RDMA or other non-TCP transports. See {doc}`../adr/0008-nonblocking-submission-without-group-calls` for the group-call decision.
