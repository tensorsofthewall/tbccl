# Testing

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j
(cd build-release && ctest --output-on-failure)
```

- Run the full suite after any change to the core, transport, collectives or C API, on every platform the change touches.
- **Sanitizers.** New tests, and any change to concurrency, lifetime or wire behavior, should pass under AddressSanitizer with UndefinedBehaviorSanitizer and under ThreadSanitizer, using separately configured build directories. CUDA and Metal runtime correctness is validated on real hardware; sanitizer coverage of that device code is a documented limitation, not silently skipped.
- **Negative controls.** A new concurrency gate should fail when the behavior it protects is removed. Gate tests use explicit progress barriers (`set_progress_paused`) rather than timing.
- **Process per rank.** Tests that spawn real CUDA or MLX state run each rank in its own process.
- **macOS.** Ephemeral-port exhaustion is real: run suites serially and set `TBCCL_TEST_WORLD_PACE_MS`. Keep ranks inside a live SSH session or `tmux`.
- **C ABI.** `tests/c_api/` pins layouts and the exported symbols; `scripts/check_c_external_consumer.sh` builds a pure-C consumer against an installed prefix.
- **Hardware.** Tests that need a GPU or a Thunderbolt link are opt-in. Follow the [Thunderbolt link guide](../guides/thunderbolt-link.md) for real-link sessions.
