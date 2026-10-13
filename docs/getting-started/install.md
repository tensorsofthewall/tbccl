# Installing TBCCL

There are two ways to get TBCCL, and you only need one:

- **A prebuilt release archive** (recommended): download, verify, extract, and point CMake or pkg-config at it.
- **Building from source**, when no archive matches your platform or you want to change the code.

If you only want to use PyTorch, vLLM or exo, you do not need this page: install the adapter wheel instead ([torch-tbccl](https://torch-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html), [vllm-tbccl](https://vllm-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html), [exo-tbccl](https://exo-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html)). The wheels contain libtbccl.

## Requirements

- A C++17 compiler (C11 for C-only consumers), CMake 3.20 or newer and Threads.
- Linux x86-64 or macOS arm64 (Apple silicon). Windows is not supported.
- For CUDA buffers: an NVIDIA GPU and driver on Linux, and the CUDA Toolkit when you build against the CUDA archive with CMake.
- For Metal-shared buffers: macOS. On macOS over a non-interactive SSH session, `cmake` and `ctest` may not be on `PATH`; add the directory that holds them (for example `/opt/homebrew/bin`).

## Path 1: a prebuilt release archive

Release archives are attached to the GitHub release of each version (currently the pre-release [`v0.6.0rc1`](https://github.com/tensorsofthewall/tbccl/releases/tag/v0.6.0rc1)). They are static builds with no runtime dependency beyond the system C++ runtime.

| Archive | Contents |
|---|---|
| `tbccl-0.6.0rc1-linux-x86_64-cuda13.tar.gz` | Linux x86-64 with the CUDA memory provider (built with CUDA 13.0; `Host` and `Cuda` buffers) |
| `tbccl-0.6.0rc1-linux-x86_64.tar.gz` | Linux x86-64, host buffers only |
| `tbccl-0.6.0rc1-macos-arm64.tar.gz` | macOS Apple silicon (`Host` and `MetalShared` buffers) |

Download an archive together with `SHA256SUMS`, verify it, and optionally verify its build provenance:

```sh
BASE=https://github.com/tensorsofthewall/tbccl/releases/download/v0.6.0rc1
curl -LO $BASE/tbccl-0.6.0rc1-linux-x86_64-cuda13.tar.gz -LO $BASE/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing                  # macOS: shasum -a 256 -c SHA256SUMS --ignore-missing
gh attestation verify tbccl-0.6.0rc1-linux-x86_64-cuda13.tar.gz --repo tensorsofthewall/tbccl    # optional; needs the GitHub CLI
tar -xzf tbccl-0.6.0rc1-linux-x86_64-cuda13.tar.gz
```

The extracted directory is an install prefix:

```text
bin/tbccl-info              version and capability tool
include/tbccl/              C and C++ headers (tbccl.h is the C ABI)
lib/libtbccl.a, libtbccl_c.a, libtbccl_cuda.a (CUDA archive only)
lib/cmake/TBCCL/            CMake package
lib/pkgconfig/tbccl.pc      pkg-config file
share/tbccl/BUILDINFO.json  commit, version, CUDA toolkit
```

Check that it works before anything else ([diagnostics](../guides/diagnostics.md)):

```sh
tbccl-0.6.0rc1-linux-x86_64-cuda13/bin/tbccl-info
```

### Use it from CMake

```cmake
cmake_minimum_required(VERSION 3.20)
project(app C CXX)
find_package(TBCCL CONFIG REQUIRED)

add_executable(version_check version_check.c)               # C consumer
target_link_libraries(version_check PRIVATE TBCCL::tbccl_c)

add_executable(two_rank_threads two_rank_threads.cpp)       # C++ consumer
target_link_libraries(two_rank_threads PRIVATE TBCCL::tbccl)
```

```sh
cmake -S . -B build -DCMAKE_PREFIX_PATH=$PWD/tbccl-0.6.0rc1-linux-x86_64-cuda13
cmake --build build && ./build/version_check
```

A tiny C program (`examples/version_check.c`):

```{literalinclude} ../../examples/version_check.c
:language: c
:start-at: "#include <stdint.h>"
```

A tiny C++ program that runs a two-rank `all_reduce` with two threads on loopback (`examples/two_rank_threads.cpp`):

```{literalinclude} ../../examples/two_rank_threads.cpp
:language: cpp
:start-at: "#include <tbccl/communicator.hpp>"
```

With the CUDA archive, CMake also finds the CUDA Toolkit and links the CUDA runtime for you. `scripts/check_doc_examples.sh <prefix>` builds and runs both programs through CMake and through pkg-config against an installed prefix; it was verified on Linux with the host-only and the CUDA archive.

### Use it from pkg-config

```sh
export PKG_CONFIG_PATH=$PWD/tbccl-0.6.0rc1-linux-x86_64/lib/pkgconfig      # the .pc file locates its own prefix
c++ -std=c++17 two_rank_threads.cpp $(pkg-config --cflags --libs tbccl) -pthread -o two_rank_threads
cc version_check.c $(pkg-config --cflags --libs tbccl) -lstdc++ -o version_check
```

Two things the `.pc` file does not do for you: a **C** program must add the C++ runtime itself (`-lstdc++` on Linux, `-lc++` on macOS) because the static libraries use C++ internally, and with the **CUDA archive** you must also add the CUDA runtime (for example `-L/usr/local/cuda/lib64 -lcudart`). CMake handles both automatically, so prefer it for the CUDA archive.

## Path 2: build from source

```sh
git clone https://github.com/tensorsofthewall/tbccl.git && cd tbccl
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release          # host-only build
cmake --build build-release -j
(cd build-release && ctest --output-on-failure)
cmake --install build-release --prefix <prefix>
```

| Option | Default | Effect |
|---|---|---|
| `-DTBCCL_ENABLE_CUDA=ON` | `OFF` | also builds and installs the CUDA memory provider (`TBCCL::tbccl_cuda`); Linux with the CUDA Toolkit |
| `-DTBCCL_ENABLE_METAL=ON` | `OFF` | Apple platforms only: builds the Metal device backends |
| `-DTBCCL_CUDA_STATIC_RUNTIME=ON` | `OFF` | link the CUDA component against the static CUDA runtime so a program or wheel that uses it has no `libcudart.so` dependency. The release archives are built this way; leave it off for a normal local build |
| `-DCMAKE_INSTALL_PREFIX=<prefix>` / `--prefix` | system default | where `cmake --install` puts the package |

A source build is versioned as the release candidate by default (`TBCCL_PUBLIC_VERSION`, e.g. `0.6.0rc1`); `tbccl-info` shows what you built. Consumers link the **installed** prefix, not the source tree. The package exports:

| Target | Use |
|---|---|
| `TBCCL::tbccl` | the C++ runtime |
| `TBCCL::tbccl_cuda` | optional CUDA provider (present when installed from a CUDA build) |
| `TBCCL::tbccl_c` | the C ABI shim, for C-only consumers |

## After installing or upgrading

Package versions, the C ABI version and the wire protocol version are separate ([versioning](../reference/versioning.md)). Ranks that were built against different wire protocol versions cannot connect. After installing a TBCCL with a different wire version, rebuild every program and adapter that links it, on every machine.
