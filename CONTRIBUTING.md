# Contributing to TBCCL

Thank you for contributing. This document describes how to set up, test and submit changes. Technical rules for the code base are in `AGENTS.md`.

## Development setup

You need a C++17 compiler, CMake >= 3.20 and Python 3 (for some tests and tools). CUDA builds need the CUDA Toolkit (`-DTBCCL_ENABLE_CUDA=ON`); Metal builds need macOS (`-DTBCCL_ENABLE_METAL=ON`). A host-only build needs neither.

## Building and testing

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
(cd build-release && ctest --output-on-failure)
cmake --install build-release --prefix <prefix>    # consumers link the installed prefix
```

Run the full test suite for any change to the core, transport, collectives or C API, on every platform the change touches. Concurrency, lifetime and wire-level changes should also be run under AddressSanitizer/UndefinedBehaviorSanitizer and ThreadSanitizer, and new concurrency tests need a negative control. Tests that need a GPU or a Thunderbolt link are opt-in; say in the pull request what you ran.

## Contribution workflow

1. Open an issue for anything beyond a small fix, so the approach can be agreed first.
2. Fork or branch from `main` and keep each pull request focused on one change.
3. Make sure the build and tests pass locally and add tests for new behavior and for bug fixes.
4. Open a pull request using the template and describe what changed, why, and what you ran.
5. Maintainers review every pull request. Address review comments with follow-up commits; maintainers may ask you to squash before merging.

Project policy: changes to `main` land through pull requests, and `main` is never force-pushed. Repository settings may or may not enforce this; the policy applies either way.

## Commit messages

Use short, descriptive subjects in the form `area: summary` (imperative, no trailing period), with a body that explains why when it is not obvious. Internal tracking numbers are not required in subjects.

```text
transport: add collective data connection
protocol: validate connection roles
test: cover mixed-domain ordering
docs: document wire compatibility
ci: add documentation checks
```

## Pull request expectations

- The change builds and the relevant tests pass; state which platforms and hardware you tested on.
- Behavior changes come with tests; documentation is updated alongside code.
- No unrelated formatting or refactoring in the same pull request.
- Do not commit machine-specific paths, host names, credentials or model weights.

## Documentation

User-visible changes update the relevant documentation (`README.md` and `docs/`) in the same pull request. Document behavior that exists, not behavior you intend to add. Build the documentation with `make docs` before submitting; warnings fail the build and the same check runs in continuous integration.

## Compatibility requirements

- **C ABI:** frozen at v1 (`TBCCL_C_ABI_VERSION`). Layout or constant changes are breaking and need maintainer approval. Additions must be framework-neutral and backwards compatible; update `tests/c_api/abi_symbols_v1.txt` only deliberately.
- **Wire protocol:** `kWireProtocolVersion` (currently 4). A change to what peers exchange on the wire bumps it; peers with different versions are rejected at the handshake. Describe the compatibility effect in the pull request.
- **Package version:** semantic changes follow the versioning policy in the release documentation; do not bump versions in unrelated changes.
- **Public API:** changes to `include/tbccl/` need maintainer review and documentation updates (`docs/reference/cpp-api-overview.md`, `docs/reference/c-abi.md`).

## AI-assisted contributions

AI-assisted contributions are permitted. Contributors remain responsible for the correctness, licensing, testing, and review of their submissions. Material AI assistance should be disclosed according to the project's contribution guidelines: add trailers to the commit message, for example

```text
AI-Assisted-By: <tool or assistant>
AI-Assistance: documentation | tests | benchmark tooling | build automation | mechanical | implementation
```

and use `AI-Validated-By: <tool>` only when the tool itself ran and recorded the validation the commit reports. Do not list an AI tool as an author, co-author, signer or reviewer.

## Reporting problems

Open an issue with the version, platform, how to reproduce, and the observed and expected behavior. Please do not include credentials or private network details.
