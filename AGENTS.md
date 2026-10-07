# AGENTS.md

Technical guidance for contributors and coding agents working in this repository. User-facing documentation is in `README.md`; the public C++ API is in `docs/reference/cpp-api-overview.md` and the C ABI in `docs/reference/c-abi.md`. Contribution workflow is in `CONTRIBUTING.md`.

## Purpose

TBCCL is a C++17 collective-communication runtime with a stable C ABI. It moves tensors between heterogeneous machines (for example Linux/NVIDIA and a Mac) over TCP, including a direct Thunderbolt 4 link. It is a transport and collectives library only: it has no notion of models, placement or any ML framework.

## Layout

| Path | Role |
|---|---|
| `include/tbccl/` | Public C++ headers and the C header `tbccl.h` |
| `src/core/` | `Communicator`, connection management, wire protocol, workers |
| `src/transport/` | Transport abstraction and the TCP implementation |
| `src/collectives/` | Collective algorithms and the planner |
| `src/c_api/` | The C ABI v1 shim (`TBCCL::tbccl_c`) |
| `benchmarks/tensor/` | Optional CUDA and Metal device backends |
| `tests/`, `tools/`, `benchmarks/`, `examples/`, `scripts/` | Tests, diagnostics, benchmarks, examples, helper scripts |

Consumers (`torch-tbccl`, `vllm-tbccl`, `exo-tbccl`) live in separate repositories and depend on an installed TBCCL package. TBCCL never depends on them.

## Architecture boundaries

- The core stays framework-neutral. `include/tbccl/`, `src/core/`, `src/transport/`, `src/collectives/` and `src/c_api/` must not depend on CUDA, Metal, Objective-C or any ML framework. Device code is optional and lives behind `TBCCL_ENABLE_CUDA` / `TBCCL_ENABLE_METAL`.
- If a consumer seems to need a framework-specific feature, look for the missing generic primitive instead. Do not add framework-specific API.
- Consumers link an installed prefix, never the source tree.

## Critical invariants

- `BufferView` does not own caller memory. A `Work` keeps no copy; the caller must keep the memory valid until the `Work` is terminal. Do not add defensive copies to relax this contract.
- Submission never blocks. A post returns a `Work` immediately; staging and socket progress happen later on worker threads.
- Failures: a failure after admission fails the communicator, and the failure propagates communicator-wide. A rejected submission (bad argument, unregistered memory kind) does not. Abort is communicator-wide and there is no recovery. Errors carry an `ErrorCode` chosen where the failure happens; the C layer maps codes, never message text.
- Collective ordering must match across ranks: every rank issues the same collectives in the same order. Rank 0 plans each collective.
- Point-to-point is FIFO per (peer, direction).
- P2P and collective traffic are independent ordering domains: each peer pair has separate data connections and workers for the two, so the relative order of P2P and collective calls may differ between ranks.
- Version compatibility is explicit and separate: package version (`CMakeLists.txt`), C ABI version (`TBCCL_C_ABI_VERSION`, currently 1) and wire protocol version (`kWireProtocolVersion`, currently 4). Peers with different wire versions are rejected at the handshake; they do not interoperate.
- The C ABI is frozen at v1. Layout or constant changes are breaking. `tests/c_api/abi_symbols_v1.txt` pins the exported symbols and `scripts/check_c_external_consumer.sh` builds a pure-C consumer against an installed package. Any ABI change needs maintainer review.

## Build and test

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release   # add -DTBCCL_ENABLE_CUDA=ON on Linux/NVIDIA
cmake --build build-release -j
(cd build-release && ctest --output-on-failure)
```

- Run the full suite after any change to the core, transport, collectives or C API, on every platform the change touches.
- Concurrency and lifetime changes should also pass under AddressSanitizer/UndefinedBehaviorSanitizer and ThreadSanitizer. New concurrency gates need a negative control (the test fails when the old behavior is restored). Use explicit progress barriers rather than timing.
- Tests that need real CUDA, Metal or a Thunderbolt link are opt-in and are never required for ordinary changes. On macOS, run suites serially and use `TBCCL_TEST_WORLD_PACE_MS` to avoid ephemeral-port exhaustion.
- Benchmarks must not verify correctness inside a timed loop (`docs/development/benchmark-methodology.md`).
- Hardware safety: never mask hardware errors to make a test pass. Read `docs/guides/thunderbolt-link.md` before running anything on a real link.

## Code ownership expectations

Changes under `include/tbccl/`, `src/core/`, `src/transport/`, `src/collectives/`, `src/c_api/` and `tests/c_api/abi_symbols_v1.txt` need maintainer review. `benchmarks/tcp_pingpong.cpp` is a baseline benchmark and should not change unless the change is about it.

## Conventions

- Match the surrounding code. Add comments only for non-obvious constraints, and avoid referring to internal project history.
- Validate at real boundaries (network input, caller buffers); trust internal invariants.
- Add explicit paths when staging files. Build directories are not tracked.
