# Versioning and compatibility

TBCCL tracks four independent versions. They change for different reasons and must not be confused.

| What | Where it is defined | Current value | Changes when |
|---|---|---|---|
| Package version | `project(VERSION ...)` in `CMakeLists.txt`; `tbcclGetPackageVersion` | 0.5.1 (development) | a release is made |
| C ABI version | `TBCCL_C_ABI_VERSION` in `tbccl.h`; `tbcclGetAbiVersion` | 1 | the C ABI is broken (not planned) |
| Wire protocol version | `kWireProtocolVersion` in `include/tbccl/rank_directory.hpp` | 4 | what ranks exchange on the network changes incompatibly |
| Endpoint blob format | `tbcclEndpointBlob.format_version` | 1 | the bootstrap blob layout changes |

The machine-readable form of the first three rows is `compatibility.json` at the repository root; `scripts/check_compatibility_manifest.py` (run in CI and by `make check-compat`) fails if it disagrees with the sources. TBCCL owns this file; each adapter keeps its own manifest of the frameworks, Python versions and platforms it supports and refers to the C ABI and wire protocol versions defined here.

Adapter packages (torch-tbccl, vllm-tbccl, exo-tbccl) have their own package versions and their own framework compatibility ranges; they declare which C ABI and wire protocol versions they were built and validated against.

## Compatibility rules

- **C ABI.** Frozen at v1: constants are never renumbered, struct fields are never reordered or removed, exported functions are never removed or changed. Compatible growth only (append struct fields guarded by `struct_size`, add constants and functions). `tests/c_api/abi_symbols_v1.txt` pins the exported symbols. See [C ABI v1](c-abi.md).
- **Wire protocol.** Every connection starts with a handshake carrying the wire version; a peer with a different version is rejected cleanly on both sides (see below). A change that alters what peers exchange bumps the wire version and therefore makes mixed worlds impossible by design.
- **C++ API.** Source-compatible within a development line where possible, but it is not an ABI promise.
- **Package version.** Bumped on release. A wire protocol change is a minor-version change at least: wire 4 is planned for the 0.6.0 release, because wire 4 does not interoperate with wire 3.

## Wire 3 and wire 4 are incompatible

Wire protocol 4 adds a second data connection per peer pair (see [Ordering domains](../concepts/ordering-domains.md)). A rank speaking wire 3 never dials the second connection, and a wire 3 listener rejects the new connection role, so a mixed world cannot be built.

**Rejection behavior.** The accepting side checks the handshake's wire version first. A mismatch is answered with a rejection, so both sides fail promptly with an error of the form `protocol_mismatch: ... wire protocol ...` (the C ABI reports `TBCCL_PROTOCOL_MISMATCH`). It is bounded by the bootstrap timeout, no payload moves, and nothing hangs. Both directions are tested: wire 3 dialing wire 4, and wire 4 dialing wire 3.

If you see this error, check that every rank, and every adapter linked against TBCCL, was built against the same installed TBCCL package; rebuild the adapters (torch-tbccl, exo-tbccl) against the new prefix.

## Release status

The current tree is a development version (0.5.1). The next release is planned as 0.6.0. Until a release tag exists, documentation labelled "latest" describes the development branch; there is no published stable release.
