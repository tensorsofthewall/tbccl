# Known limitations

What TBCCL does not do, what is supported but not validated, and the operational workarounds you may need. The same categories are used in the other TBCCL projects' limitation pages.

## Unsupported

| Limitation | Detail |
|---|---|
| Transport | TCP over IPv4 only. No RDMA, no other transport, no IPv6 endpoints |
| Direct CUDA to Metal DMA | does not exist. CUDA buffers are staged through pinned host memory; Metal shared (unified) memory is handed to the transport directly |
| Reductions | `Sum` only. No `Prod`, `Min`, `Max`, `Avg`. FP8 and packed INT4/FP4 move as bytes and cannot be reduced |
| Float16 and BFloat16 reductions | world size 2 only |
| Int8 and UInt8 reductions | not available for Metal shared memory |
| Out-of-place `all_reduce` | not supported for CUDA buffers (pass the same view for send and receive) |
| Metal private buffers | `MTLResourceStorageModePrivate` buffers are not CPU-addressable and are not supported; use shared storage |
| Recovery | after a communicator fails there is no recovery; create a new communicator |
| Communicator management | no split, shrink or grow; no group start/end calls; no message tags; no `ANY_SOURCE` |
| Bootstrap | no hidden rendezvous: the application exchanges every rank's endpoint |
| Platforms | Windows, 32-bit and ARM Linux are not built or tested |
| Security | no authentication and no encryption; run inside a trusted network or an encrypted tunnel ([security model](../concepts/security.md)) |

## Supported but not validated

| Item | State |
|---|---|
| World sizes 5 to 8 | accepted, not validated (1 to 4 are validated) |
| Two Linux hosts | the Linux-to-Linux path is exercised on loopback with several processes; a run between two Linux machines over a physical network is not part of the recorded validation |
| Transports other than Thunderbolt 4 | any IPv4 TCP network works in principle; only Thunderbolt 4 was measured |
| Thunderbolt controllers | only an Intel Thunderbolt 4 (Maple Ridge) controller on the Linux side was tested |
| GPUs | validated with one NVIDIA GPU architecture (an RTX 3070 Ti Laptop GPU); the CUDA 13 archive is built for sm75 to sm120 |

## Performance

- Two-rank collectives have no descriptor check, which keeps their latency low; the cost is that a count or size mismatch is not detected there.
- A live communicator pair holds more file descriptors and threads than before wire protocol 4 (two more descriptors and four more threads per pair).
- Small-message `all_reduce` latency on loopback is bimodal after the wire 4 change; it was characterized, not tuned.
- Throughput and latency depend on the host: GPU benchmark numbers taken under thermal throttling are invalid ([benchmark methodology](../development/benchmark-methodology.md)).

## Operational workarounds

| Situation | Workaround |
|---|---|
| macOS rank times out while the Linux side is fine | approve Local Network access and the per-application firewall prompt from a local session, and keep Mac ranks in a live SSH session or `tmux`. Never script firewall changes ([Thunderbolt link guide](../guides/thunderbolt-link.md)) |
| `pkg-config` link of a C program fails with C++ runtime symbols | add `-lstdc++` (`-lc++` on macOS), or link through CMake. With the CUDA archive also add the CUDA runtime |
| a peer may still be finishing its last collective | quiesce all ranks before aborting or destroying the communicator |
| mixed-version ranks | not possible by design: a wire protocol mismatch is rejected at the handshake |

## Not an upstream issue

TBCCL has no known limitation that comes from an upstream framework. Framework-specific limitations (PyTorch, vLLM, exo, MLX) are listed in the adapters' own pages: [torch-tbccl](https://torch-tbccl.tensorsofthewall.com/en/latest/project/limitations.html), [vllm-tbccl](https://vllm-tbccl.tensorsofthewall.com/en/latest/project/limitations.html), [exo-tbccl](https://exo-tbccl.tensorsofthewall.com/en/latest/project/limitations.html).
