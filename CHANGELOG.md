# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. 0.6.0 is the first release; it is experimental.

## 0.7.0rc1

Release candidate (pre-release). First release candidate of 0.7.0, the first minor release after 0.6. Release candidates are for validation on real hardware; expect an rc2 if a blocker is found. The C ABI (version 1), the wire protocol (version 4) and the endpoint blob format (version 1) are unchanged, and ranks of 0.6.x and 0.7.0 interoperate. Every new tuning setting is opt-in; the only default behaviour that changed is the two-rank `all_gather` schedule described below.

### Installation

- Native archives built by CI and attached to the GitHub pre-release: `tbccl-0.7.0rc1-linux-x86_64.tar.gz`, `tbccl-0.7.0rc1-linux-x86_64-cuda13.tar.gz`, `tbccl-0.7.0rc1-macos-arm64.tar.gz`, each with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation.
- The adapters (torch-tbccl, vllm-tbccl, exo-tbccl) link TBCCL statically. They pick up the changes below only when the adapter wheel is built against 0.7.0; the 0.6.x adapter wheels keep their own TBCCL.

### Changed

- **Two-rank `all_gather` runs both directions at once** (default; no setting). With two ranks the send and the receive were sequential; they now use the collective worker's two existing duplex lanes together. Measured on a direct Thunderbolt 4 link between a Linux/NVIDIA host and an Apple-silicon Mac, completed latency fell by roughly 37% to 65% for host memory and 20% to 75% for CUDA/Metal memory from 64 B to 16 MiB per rank. No thread, connection or frame is added. A rank on the previous schedule interoperates with a rank on this one. With both ranks on one host and payloads of 4 MiB and above it was 14% to 24% slower in a pinned measurement (see Known limitations).

### Added

- **Opt-in pipelined two-rank `all_reduce`**: `TBCCL_ALLREDUCE_PIPELINE_BYTES=<n>` (unset or `0` keeps the existing schedule) splits the reduction into chunks so the two directions of the link are busy together. `131072` on both ranks is recommended for two-host Thunderbolt 4 jobs with payloads of 256 KiB and up; smaller payloads are unaffected. Measured completed latency fell by about 19% to 66% at 256 KiB to 16 MiB for host and CUDA/Metal memory. The setting is not negotiated: ranks may differ, but a pipelined non-root rank opposite a sequential root is correct and can be slower, so set it on both ranks. See the AllReduce pipeline page for when not to enable it.
- **Opt-in chunking of staged transfers**: `TBCCL_STAGED_CHUNK_BYTES=<n>` (at least 4096; unset or `0` keeps one chunk per transfer) overlaps device copies and network transfer for staged memory such as CUDA. Host and Metal-shared memory are not staged and are unaffected. An invalid value makes `Communicator::create` fail with `invalid_argument`.
- A verified Communicator performance probe (`communicator_probe`) used by the benchmark documentation, and the documentation checks that keep release statements consistent with the package version.

### Fixed

- **A chunked staged transfer could hang forever** when one of its stages failed (three or more chunks, pipeline depth above one, `TBCCL_STAGED_CHUNK_BYTES` set): the failing stage released its slot before recording the failure and nothing could wake a stage blocked acquiring a slot. Cancellation now wakes blocked stages, which fail with `Aborted`. Without the setting, transfers were not affected.

### Documentation

- New pages for the two-rank AllGather exchange, the AllReduce pipeline (with recommended use and its limits), staged-transfer cancellation and chunking; refreshed installation, ecosystem and onboarding pages; a validation page for this release.

### Compatibility

- C ABI 1, wire protocol 4, endpoint blob format 1. Platforms and world sizes are unchanged. The exported C symbols are unchanged.
- `StagingPool` grows from 152 to 160 bytes (C++ API in the static library; no ABI consequence). Adapters must be rebuilt against 0.7.0 to use the changes.

### Known limitations

- Single-host loopback with payloads of 4 MiB and above was 14% to 24% slower for `all_gather` in a pinned measurement because both directions then compete for the same memory bandwidth. It was 12% to 31% faster from 64 B to 1 MiB. No two-host regression was observed.
- All published measurements come from one Linux CUDA / Apple Metal machine pair. PCIe AER reporting was unavailable on the Linux host, so no statement about AER errors is made.
- ThreadSanitizer cannot run the peer-death scenario of `mixed_domain_ordering_test` (it forks a multi-threaded process, which TSan refuses); the other scenarios pass and TSan reports no race.

## 0.6.1

Patch release. It adds the public benchmark suite and its documentation. There is no change to the library (`src/`, `include/`), the C ABI (version 1), the wire protocol (version 4) or the endpoint blob format (version 1), and no change to any transport or collective algorithm. Nothing in this release is a performance improvement.

### Installation

- The same three native archives as 0.6.0, named `tbccl-0.6.1-...`, each with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. The archives contain the library, not the Python benchmark suite.
- The benchmark suite is obtained from the source archive of the tag `v0.6.1` (`https://github.com/tensorsofthewall/tbccl/archive/refs/tags/v0.6.1.tar.gz`) or a checkout of that tag. It needs Python 3.13 and PyYAML and builds the native benchmark programs with CMake. It needs no PyTorch, vLLM, Exo or MLX.

### Added

- A shared public benchmark suite (`python3 -m benchmarks.suite`): native point-to-point, latency, transfer, collective and device-memory benchmarks, ready-made profiles (`quick`, `native`, `device`, `pytorch`, `vllm`, `exo`, `custom`), health gates, remote supervision and cleanup, and a result schema (`tbccl-bench-result/1`).
- An external provider protocol (version 1): the PyTorch, vLLM and Exo workloads live in the repositories of torch-tbccl, vllm-tbccl and exo-tbccl and are run by the shared runner after `describe` and `validate` steps on every participating host.
- Reports in Markdown, HTML (offline), JSON and CSV; a sanitized export for sharing results.
- The PyTorch, vLLM and Exo workload scripts are not part of TBCCL: they live only in the torch-tbccl, vllm-tbccl and exo-tbccl repositories (tags `v0.2.1`, `v0.2.1`, `v0.3.1`) and are run through the provider protocol. A framework workload with no provider configured is refused by the preflight and names the missing setting; the native benchmarks are unaffected.
- Regression comparison of two runs of one configuration, with the compatibility checks and statistical classification, and a record of build identity (matching, different or unknown) for the TBCCL archive, the provider and the framework.

### Documentation

- Benchmark quickstart, hardware and network configuration, methodology and limitations, troubleshooting, and the measured reference results of the native benchmarks (with the software, hardware, direction, sample counts and limits of every figure).
- A validation page for this release.

### Compatibility

- C ABI 1, wire protocol 4, endpoint blob format 1. Platforms and world sizes are unchanged.
- The benchmark providers of the adapters are versioned by the provider protocol and the result schema, not by the TBCCL package version. An adapter's minimum TBCCL package is unchanged.

### Known limitations

- All published reference measurements come from one Linux CUDA / Apple Metal machine pair. PCIe AER reporting was unavailable on the Linux host, so no statement about AER errors is made.

## 0.6.0

First release. Promoted from release candidate 0.6.0rc1 with no change to the library, the C ABI (version 1) or the wire protocol (version 4); the differences are the version string, documentation examples and the documentation checks.

### Installation

- Native archives built by CI and attached to the GitHub release: `tbccl-0.6.0-linux-x86_64.tar.gz`, `tbccl-0.6.0-linux-x86_64-cuda13.tar.gz`, `tbccl-0.6.0-macos-arm64.tar.gz`, each with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. Extract and use `find_package(TBCCL CONFIG REQUIRED)`; `bin/tbccl-info` reports the version.
- Building from source remains supported (see the installation guide).

### Added

- Point-to-point `send` and `recv`, and the collectives `barrier`, `broadcast`, `all_gather` and `Sum` `all_reduce`, over TCP (IPv4), including a direct Thunderbolt 4 link.
- Nonblocking submission: every call returns a `Work` handle at once.
- Host, CUDA and Metal-shared buffers in one communicator, with validated world sizes 1 to 4.
- A stable C ABI (version 1) and CMake packages `TBCCL::tbccl` (C++) and `TBCCL::tbccl_c` (C).

### Changed

- Wire protocol 4 gives each peer pair a separate connection for collectives, so point-to-point and collective calls may be in flight together on one communicator.

### Fixed

- None.

### Compatibility

- C ABI 1, wire protocol 4. Wire protocol 4 does not interoperate with wire protocol 3; a mismatch is rejected at connection time with a `protocol_mismatch` error.
- Platforms: Linux x86-64 (host, CUDA) and macOS arm64 (host, Metal shared memory).

### Known limitations

- TCP over IPv4 is the only transport; TBCCL is not an RDMA implementation.
- No peer authentication, transport encryption or authorization: peers are assumed trusted. See the security model in the documentation.
- Float16 and BFloat16 reductions are validated at world size 2 only.
- A failure after an operation is admitted fails the communicator; there is no recovery.

## 0.6.0rc1 (release candidate)

First release candidate of the first planned release, 0.6.0. Release candidates are for validation on real hardware; expect an rc2 if a blocker is found.

### Installation

- Native archives built by CI and attached to the GitHub pre-release: `tbccl-0.6.0rc1-linux-x86_64.tar.gz`, `tbccl-0.6.0rc1-linux-x86_64-cuda13.tar.gz`, `tbccl-0.6.0rc1-macos-arm64.tar.gz`, each with an SPDX SBOM, `SHA256SUMS` and a build-provenance attestation. Extract and use `find_package(TBCCL CONFIG REQUIRED)`; `bin/tbccl-info` reports the version.
- Building from source remains supported (see the installation guide).

### Added

- Point-to-point `send` and `recv`, and the collectives `barrier`, `broadcast`, `all_gather` and `Sum` `all_reduce`, over TCP (IPv4), including a direct Thunderbolt 4 link.
- Nonblocking submission: every call returns a `Work` handle at once.
- Host, CUDA and Metal-shared buffers in one communicator, with validated world sizes 1 to 4.
- A stable C ABI (version 1) and CMake packages `TBCCL::tbccl` (C++) and `TBCCL::tbccl_c` (C).

### Changed

- Wire protocol 4 gives each peer pair a separate connection for collectives, so point-to-point and collective calls may be in flight together on one communicator.

### Fixed

- None.

### Compatibility

- C ABI 1, wire protocol 4. Wire protocol 4 does not interoperate with wire protocol 3; a mismatch is rejected at connection time with a `protocol_mismatch` error.
- Platforms: Linux x86-64 (host, CUDA) and macOS arm64 (host, Metal shared memory).

### Known limitations

- TCP over IPv4 is the only transport; TBCCL is not an RDMA implementation.
- No peer authentication, transport encryption or authorization: peers are assumed trusted. See the security model in the documentation.
- Float16 and BFloat16 reductions are validated at world size 2 only.
- A failure after an operation is admitted fails the communicator; there is no recovery.
