# Changelog

All notable user-facing changes are recorded here. The format follows Keep a Changelog, and the project follows Semantic Versioning once it has releases. tbccl has not had a final release. 0.6.0rc1 is a release candidate (pre-release), not production-ready.

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
