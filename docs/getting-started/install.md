# Installing TBCCL

There are two ways to get TBCCL, and you only need one:

- **A prebuilt release archive** (recommended): download, verify, extract, and point CMake or pkg-config at it.
- **Building from source**, when no archive matches your platform or you want to change the code.

If you only want to use PyTorch, vLLM or exo, you do not need this page: install the adapter wheel instead ([torch-tbccl](https://torch-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html), [vllm-tbccl](https://vllm-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html), [exo-tbccl](https://exo-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html)). The wheels contain libtbccl. A [helper script](install-adapters.md) can pick, verify and install the right wheel for an environment.

## Which path do I need?

| You want to... | Do this |
|---|---|
| use PyTorch, vLLM or exo | install the **adapter wheel**, not this page: it contains libtbccl ([torch-tbccl](https://torch-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html), [vllm-tbccl](https://vllm-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html), [exo-tbccl](https://exo-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html)) |
| write C or C++ against TBCCL | [Path 1](#path-1-a-prebuilt-release-archive), then CMake or pkg-config |
| move CUDA buffers (Linux, NVIDIA) | the `linux-x86_64-cuda13` archive; the `linux-x86_64` archive carries host buffers only |
| move Metal shared buffers (Apple silicon) | the `macos-arm64` archive |
| change TBCCL, or no archive matches your platform | [Path 2](#path-2-build-from-source) |
| try it on one machine first | [Your first program](first-program.md): every example runs as two ranks on loopback, with no GPU and no second computer |
| connect two machines | install **the same version** on both, then [Two machines over Thunderbolt](two-machine-thunderbolt.md) |

A **release archive** (a built, installed prefix: library, headers, CMake and pkg-config packages, `tbccl-info`) is not a **source checkout** (the repository, with the examples, `scripts/` and the Python benchmark suite). The commands below that mention `examples/` or `scripts/` need the checkout; the archive needs neither.

## Requirements

- A C++17 compiler (C11 for C-only consumers), CMake 3.20 or newer and Threads.
- Linux x86-64 or macOS arm64 (Apple silicon). Windows is not supported.
- For CUDA buffers: an NVIDIA GPU and driver on Linux, and the CUDA Toolkit when you build against the CUDA archive with CMake.
- For Metal-shared buffers: macOS. On macOS over a non-interactive SSH session, `cmake` and `ctest` may not be on `PATH`; add the directory that holds them (for example `/opt/homebrew/bin`).

## Path 1: a prebuilt release archive

Release archives are attached to the GitHub release of each version (currently [`v0.7.0rc1`](https://github.com/tensorsofthewall/tbccl/releases/tag/v0.7.0rc1)). They are static builds with no runtime dependency beyond the system C++ runtime.

| Archive | Contents |
|---|---|
| `tbccl-0.7.0rc1-linux-x86_64-cuda13.tar.gz` | Linux x86-64 with the CUDA memory provider (built with CUDA 13.0; `Host` and `Cuda` buffers) |
| `tbccl-0.7.0rc1-linux-x86_64.tar.gz` | Linux x86-64, host buffers only |
| `tbccl-0.7.0rc1-macos-arm64.tar.gz` | macOS Apple silicon (`Host` and `MetalShared` buffers) |

The archives hold the library, headers, CMake and pkg-config packages and `tbccl-info`. The Python benchmark suite (the runner) is **not** in them: it is part of the source of each release (the source archive of the tag), as described in [run benchmarks](../benchmarks/quickstart.md). The runtime version, `0.7.0rc1`, is a different number from the C ABI (`1`) and the wire protocol (`4`); `tbccl-info` prints all three.

Pick the archive for your machine, then run the commands for your operating system in an empty directory. They need only `curl`, a checksum tool and `tar`, and nothing else from this repository.

**Linux x86-64** (host buffers only; use `linux-x86_64-cuda13` instead for CUDA buffers):

```sh
mkdir tbccl-quickstart && cd tbccl-quickstart
BASE=https://github.com/tensorsofthewall/tbccl/releases/download/v0.7.0rc1
curl -fsSLO $BASE/tbccl-0.7.0rc1-linux-x86_64.tar.gz -fsSLO $BASE/SHA256SUMS
sha256sum -c SHA256SUMS --ignore-missing
tar -xzf tbccl-0.7.0rc1-linux-x86_64.tar.gz
./tbccl-0.7.0rc1-linux-x86_64/bin/tbccl-info
```

**macOS arm64** (Apple silicon):

```sh
mkdir tbccl-quickstart && cd tbccl-quickstart
BASE=https://github.com/tensorsofthewall/tbccl/releases/download/v0.7.0rc1
curl -fsSLO $BASE/tbccl-0.7.0rc1-macos-arm64.tar.gz -fsSLO $BASE/SHA256SUMS
shasum -a 256 -c SHA256SUMS --ignore-missing
tar -xzf tbccl-0.7.0rc1-macos-arm64.tar.gz
./tbccl-0.7.0rc1-macos-arm64/bin/tbccl-info
```

`tbccl-info` prints the package version, C ABI and wire protocol, the platform and the memory providers (`host`, plus `cuda` for the CUDA archive, or `metal_shared` on macOS). Optionally verify the build provenance too: `gh attestation verify tbccl-0.7.0rc1-linux-x86_64.tar.gz --repo tensorsofthewall/tbccl` (needs the GitHub CLI). The extracted directory is an install prefix:

```text
bin/tbccl-info              version and capability tool
include/tbccl/              C and C++ headers (tbccl.h is the C ABI)
lib/libtbccl.a, libtbccl_c.a, libtbccl_cuda.a (CUDA archive only)
lib/cmake/TBCCL/            CMake package
lib/pkgconfig/tbccl.pc      pkg-config file
share/tbccl/BUILDINFO.json  commit, version, CUDA toolkit
```

See also the [diagnostics guide](../guides/diagnostics.md).

### A minimal CMake consumer

Still in `tbccl-quickstart`, create the two files below and build against the extracted prefix (the commands are the Linux ones; on macOS use the `tbccl-0.7.0rc1-macos-arm64` directory). The Linux flow was verified in a clean directory with both Linux archives.

```sh
cat > CMakeLists.txt <<'CMAKE'
cmake_minimum_required(VERSION 3.20)
project(tbccl_quickstart C)
find_package(TBCCL CONFIG REQUIRED)
add_executable(version_check version_check.c)
target_link_libraries(version_check PRIVATE TBCCL::tbccl_c)
CMAKE
cat > version_check.c <<'SOURCE'
#include <stdint.h>
#include <stdio.h>
#include <tbccl/tbccl.h>

int main(void)
{
    uint32_t abi = 0, major = 0, minor = 0, patch = 0;
    if (tbcclGetAbiVersion(&abi) != TBCCL_SUCCESS || tbcclGetPackageVersion(&major, &minor, &patch) != TBCCL_SUCCESS)
        return 1;
    printf("TBCCL %u.%u.%u, C ABI %u\n", major, minor, patch, abi);
    return 0;
}
SOURCE
cmake -S . -B build -DCMAKE_PREFIX_PATH=$PWD/tbccl-0.7.0rc1-linux-x86_64
cmake --build build && ./build/version_check          # TBCCL 0.7.0rc1, C ABI 1
```

`TBCCL::tbccl_c` is the C ABI target; a C++ program links `TBCCL::tbccl` instead. A C++ program that runs a two-rank `all_reduce` with two threads on loopback (`examples/two_rank_threads.cpp` in the source repository):

```{literalinclude} ../../examples/two_rank_threads.cpp
:language: cpp
:start-at: "#include <tbccl/communicator.hpp>"
```

With the CUDA archive, CMake also finds the CUDA Toolkit and links the CUDA runtime for you. `scripts/check_doc_examples.sh <prefix>` builds and runs both programs through CMake and through pkg-config against an installed prefix; it was verified on Linux with the host-only and the CUDA archive.

### Use it from pkg-config

```sh
export PKG_CONFIG_PATH=$PWD/tbccl-0.7.0rc1-linux-x86_64/lib/pkgconfig      # the .pc file locates its own prefix
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

A source build is versioned as the release by default (`TBCCL_PUBLIC_VERSION`, e.g. `0.7.0`; a candidate build uses an `rcN` suffix); `tbccl-info` shows what you built. Consumers link the **installed** prefix, not the source tree. The package exports:

| Target | Use |
|---|---|
| `TBCCL::tbccl` | the C++ runtime |
| `TBCCL::tbccl_cuda` | optional CUDA provider (present when installed from a CUDA build) |
| `TBCCL::tbccl_c` | the C ABI shim, for C-only consumers |

## After installing or upgrading

Package versions, the C ABI version and the wire protocol version are separate ([versioning](../reference/versioning.md)). Ranks that were built against different wire protocol versions cannot connect. After installing a TBCCL with a different wire version, rebuild every program and adapter that links it, on every machine.

## What success looks like

- `tbccl-info` prints the package version, `C ABI 1`, `wire protocol 4`, and the memory providers you expected (`host`, plus `cuda` or `metal_shared`).
- The CMake consumer prints `TBCCL 0.7.0rc1, C ABI 1`.

**This showed** that the headers, the library and the build system fit together on this machine. **It did not show** that two machines can reach each other, that a GPU path works end to end, or how fast anything is. **Next:** [Your first program](first-program.md) for a two-rank run on one machine, then [Two machines over Thunderbolt](two-machine-thunderbolt.md). **If it fails:** [Troubleshooting](../guides/troubleshooting.md) (installing and linking) and [diagnostics](../guides/diagnostics.md).
