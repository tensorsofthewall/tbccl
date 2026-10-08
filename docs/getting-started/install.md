# Installing TBCCL

TBCCL is built from source with CMake and consumed through an installed prefix. There is no published binary package.

```{note}
**Release candidate (pre-release) archives.** 0.6.0rc1 native archives are attached to the GitHub pre-release `v0.6.0rc1`. They are for validation, not production use. Download an archive and its `SHA256SUMS`, verify with `sha256sum -c`, extract it and point CMake at the extracted directory (`-DCMAKE_PREFIX_PATH=<dir>`). `bin/tbccl-info` prints the version (`0.6.0rc1`). Verify provenance with `gh attestation verify <archive> --repo tensorsofthewall/tbccl`. The final 0.6.0 install instructions will replace this note.
```

## Requirements

- A C++17 compiler, CMake 3.20 or newer, and Threads.
- For CUDA support: the CUDA Toolkit and an NVIDIA GPU (Linux).
- For Metal-shared buffers: macOS. On macOS over a non-interactive SSH session, `cmake` and `ctest` may not be on `PATH`; add the directory that holds them (for example `/opt/homebrew/bin`).

## Build, test and install

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release     # add -DTBCCL_ENABLE_CUDA=ON on Linux with CUDA
cmake --build build-release -j
(cd build-release && ctest --output-on-failure)
cmake --install build-release --prefix <prefix>
```

| Option | Effect |
|---|---|
| `-DTBCCL_ENABLE_CUDA=ON` | also builds and installs the CUDA memory provider (`TBCCL::tbccl_cuda`) |
| `-DTBCCL_ENABLE_METAL=ON` | Apple platforms only: builds the Metal device backends |

Consumers link the **installed** prefix, not the source tree. The package exports:

| Target | Use |
|---|---|
| `TBCCL::tbccl` | the C++ runtime |
| `TBCCL::tbccl_cuda` | optional CUDA provider (present when built with CUDA) |
| `TBCCL::tbccl_c` | the C ABI shim, for C-only consumers |

```cmake
find_package(TBCCL CONFIG REQUIRED)
target_link_libraries(app PRIVATE TBCCL::tbccl)       # or TBCCL::tbccl_c for C
```

Point CMake at the prefix with `-DCMAKE_PREFIX_PATH=<prefix>` (or `TBCCL_ROOT=<prefix>` for the framework adapters). `scripts/check_c_external_consumer.sh` builds and runs a pure-C consumer against an installed prefix and is a quick way to verify an install.

## Building the adapters

The framework adapters ([torch-tbccl](https://github.com/tensorsofthewall/torch-tbccl), [vllm-tbccl](https://github.com/tensorsofthewall/vllm-tbccl), [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl)) build against an installed prefix whose wire protocol version they support. After upgrading TBCCL, rebuild them against the new prefix ([Versioning](../reference/versioning.md)).
