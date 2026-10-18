# Building and testing

Copy-pasteable commands for every check. Run them from the repository root.

## Core build and tests

```sh
# host-only build
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
(cd build-release && ctest --output-on-failure)

# CUDA-enabled build (Linux with the CUDA Toolkit)
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DTBCCL_ENABLE_CUDA=ON
cmake --build build-cuda -j
(cd build-cuda && ctest --output-on-failure)

# install, then build the consumers against the INSTALLED prefix (never the source tree)
cmake --install build-release --prefix /tmp/tbccl-prefix
scripts/check_c_external_consumer.sh /tmp/tbccl-prefix       # pure-C consumer
scripts/check_doc_examples.sh /tmp/tbccl-prefix              # the install-page examples via CMake and pkg-config
/tmp/tbccl-prefix/bin/tbccl-info
```

On macOS add `-DTBCCL_ENABLE_METAL=ON` for the Metal device tests and run the suites serially.

## Documentation and manifest checks

```sh
make docs              # creates .venv-docs, installs the pinned requirements, builds docs/_build/html with warnings as errors
make docs-linkcheck    # checks local links and anchors (external URLs are not fetched)
make check-compat      # compatibility.json must agree with the C ABI header and the wire protocol constant
```

`DOCS_INTERSPHINX=1 make docs` additionally resolves cross-project references against the other projects' published `latest` inventories (needs the network).

## What needs what

You do not need a GPU, a Mac or Thunderbolt hardware to contribute. Pick the lowest tier that covers your change and say in the pull request which tiers you ran.

```{table}
:class: tb-stack tb-stack-tiers

| Tier | Needs | Commands (above) | Covers |
|---|---|---|---|
| Documentation only | Python 3 | `make docs`, `make docs-linkcheck`, `make check-compat`, `python3 tools/check_doc_commands.py`, `python3 tools/check_release_docs.py` | links, builds, commands named in the docs, version facts |
| Core, loopback | a C++17 toolchain and CMake | the host-only build and `ctest` | the transport, collectives and C API with every rank on `127.0.0.1` |
| CUDA | Linux, an NVIDIA GPU, the CUDA Toolkit | the CUDA-enabled build and `ctest` | device staging and CUDA buffers |
| Metal | an Apple-silicon Mac | `-DTBCCL_ENABLE_METAL=ON`, suites run serially | Metal-shared buffers |
| Physical link | two machines and a Thunderbolt cable | opt-in only; [Thunderbolt link guide](../guides/thunderbolt-link.md) and a health snapshot before and after | real-link behavior; never part of a generic run |
| Benchmarks | CPython 3.13, built benchmark executables | the benchmark suite tests below | the suite's own logic, never speed |
| Packaging | a clean directory | `scripts/package_native.sh` | the release archive and its consumers |
```

Changes to the runtime that touch only host-memory behavior can be fully validated on the loopback tier. A change that affects a device path should be run on the matching tier by someone who has the hardware; say so in the pull request instead of skipping it silently.

## What to run when

- Run the full suite after any change to the core, transport, collectives or C API, on every platform the change touches.
- **Sanitizers.** New tests, and any change to concurrency, lifetime or wire behavior, should pass under AddressSanitizer with UndefinedBehaviorSanitizer and under ThreadSanitizer, using separately configured build directories (for example `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"`). CUDA and Metal runtime correctness is validated on real hardware; sanitizer coverage of that device code is a documented limitation, not silently skipped.
- **Negative controls.** A new concurrency gate should fail when the behavior it protects is removed. Gate tests use explicit progress barriers (`set_progress_paused`) rather than timing.
- **Process per rank.** Tests that spawn real CUDA or MLX state run each rank in its own process.
- **macOS.** Ephemeral-port exhaustion is real: run suites serially and set `TBCCL_TEST_WORLD_PACE_MS`. Keep ranks inside a live SSH session or `tmux`.
- **C ABI.** `tests/c_api/` pins layouts and the exported symbols; update `abi_symbols_v1.txt` only deliberately.
- **Hardware.** Tests that need a GPU or a Thunderbolt link are opt-in. Follow the [Thunderbolt link guide](../guides/thunderbolt-link.md) for real-link sessions and take a health snapshot (`scripts/tb4_health_snapshot.sh`) before and after.

## Benchmark suite tests

The optional benchmark suite (`benchmarks/suite/`, see the [benchmark suite guide](benchmark-suite.md)) has its own unit tests. They need only CPython 3.13 and, for a short loopback run, built benchmark executables; they never touch a network interface or another machine, and they assert behaviour, not speed.

```sh
python3 -m unittest discover -s benchmarks/suite/tests -t .
TBCCL_BUILD_DIR=build-release python3 -m unittest discover -s benchmarks/suite/tests -t .   # also runs the loopback workflows
```

## Release archives

CI builds the release archives from the tagged commit; you can build and check one locally with the same script:

```sh
scripts/package_native.sh --out dist                       # host-only archive for this platform
scripts/package_native.sh --cuda --out dist                # Linux with CUDA
```

The script writes `dist/tbccl-<public-version>-<os>-<arch>[-cuda<N>].tar.gz` and a checksum, extracts it into a fresh directory and checks it: `tbccl-info` runs, the out-of-tree C++ and C consumers configure, build and run against the extracted prefix, and no build path leaks into any file. Maintainers verify published archives with `SHA256SUMS`, the SPDX SBOM and `gh attestation verify` as described in `RELEASE.md`.
