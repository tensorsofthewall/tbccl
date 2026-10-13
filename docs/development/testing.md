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

## What to run when

- Run the full suite after any change to the core, transport, collectives or C API, on every platform the change touches.
- **Sanitizers.** New tests, and any change to concurrency, lifetime or wire behavior, should pass under AddressSanitizer with UndefinedBehaviorSanitizer and under ThreadSanitizer, using separately configured build directories (for example `-DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"`). CUDA and Metal runtime correctness is validated on real hardware; sanitizer coverage of that device code is a documented limitation, not silently skipped.
- **Negative controls.** A new concurrency gate should fail when the behavior it protects is removed. Gate tests use explicit progress barriers (`set_progress_paused`) rather than timing.
- **Process per rank.** Tests that spawn real CUDA or MLX state run each rank in its own process.
- **macOS.** Ephemeral-port exhaustion is real: run suites serially and set `TBCCL_TEST_WORLD_PACE_MS`. Keep ranks inside a live SSH session or `tmux`.
- **C ABI.** `tests/c_api/` pins layouts and the exported symbols; update `abi_symbols_v1.txt` only deliberately.
- **Hardware.** Tests that need a GPU or a Thunderbolt link are opt-in. Follow the [Thunderbolt link guide](../guides/thunderbolt-link.md) for real-link sessions and take a health snapshot (`scripts/tb4_health_snapshot.sh`) before and after.

## Release archives

CI builds the release archives from the tagged commit; you can build and check one locally with the same script:

```sh
scripts/package_native.sh --out dist                       # host-only archive for this platform
scripts/package_native.sh --cuda --out dist                # Linux with CUDA
```

The script writes `dist/tbccl-<public-version>-<os>-<arch>[-cuda<N>].tar.gz` and a checksum, extracts it into a fresh directory and checks it: `tbccl-info` runs, the out-of-tree C++ and C consumers configure, build and run against the extracted prefix, and no build path leaks into any file. Maintainers verify published archives with `SHA256SUMS`, the SPDX SBOM and `gh attestation verify` as described in `RELEASE.md`.
