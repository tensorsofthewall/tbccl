# Quickstart

This runs the native benchmarks on one machine, on loopback. It needs a C++ toolchain and CMake for the build, and Python 3.13 with PyYAML for the suite. No GPU, second machine, PyTorch, vLLM or Exo is needed.

## Get the suite

The Python benchmark suite is **not** inside the native release archives (`tbccl-<version>-<platform>.tar.gz`, which hold the library, headers, CMake packages and `tbccl-info`). It is part of the source of each release: download the source archive of the release tag, or check the tag out, and run it from that directory. Use the tag of the release you are measuring.

```sh
curl -fsSL https://github.com/tensorsofthewall/tbccl/archive/refs/tags/v0.6.1.tar.gz | tar -xz
cd tbccl-0.6.1
```

(`git clone --branch v0.6.1 https://github.com/tensorsofthewall/tbccl` gives the same files.) Nothing else is downloaded. The framework providers are obtained the same way from the matching tags of their repositories; see [framework benchmarks](frameworks.md).

## Run it

```sh
# 1. build the benchmark programs once
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release -j

# 2. check the environment (read-only)
pip install pyyaml                      # the shipped profiles are YAML; matplotlib is optional (plots)
python3 -m benchmarks.suite doctor

# 3. look at what exists, then at what a profile would run
python3 -m benchmarks.suite list
python3 -m benchmarks.suite profiles
python3 -m benchmarks.suite run benchmarks/suite/profiles/quick.yaml --dry-run

# 4. run it twice (a few seconds each), report on the first, compare the two
python3 -m benchmarks.suite run benchmarks/suite/profiles/quick.yaml --label first
python3 -m benchmarks.suite run benchmarks/suite/profiles/quick.yaml --label second
python3 -m benchmarks.suite report <first run id>
python3 -m benchmarks.suite compare <first run id> <second run id>
```

Results go to `results/bench-suite-local/` (git-ignored). `report` writes `report.md`, `summary.csv`, `report.json` and a self-contained `report.html` next to the run; open the HTML file in any browser, offline.

A few things to know before you read a number:

* **Loopback measures software, not a network.** Use it to compare builds and to check your installation. The latencies include only the kernel's loopback path.
* **Correctness is checked, not assumed.** Where a benchmark can verify its data it does so outside the timed loop, and a run that fails verification records no measurement.
* **Timings of a Debug build are flagged.** Build with `-DCMAKE_BUILD_TYPE=Release` for numbers that represent the library.
* **Nothing runs on a physical link unless you ask.** Loopback is the default. Two machines need a configuration that declares them and the `--allow-physical` flag; see [Running on your hardware](your-hardware.md).

## What the profiles are

`python3 -m benchmarks.suite profiles` lists seven ready-made configuration files.

| Profile | Runs | Needs |
|---|---|---|
| `quick` | every native benchmark on loopback, small counts | a TBCCL build |
| `native` | native TCP, latency, transfers and a collective between two hosts | a build on both hosts and a link; physical |
| `device` | CUDA-pinned and Metal-shared memory between two hosts | CUDA on one, Metal on the other; physical |
| `pytorch` | PyTorch over torch-tbccl on loopback | a PyTorch environment and the torch-tbccl provider |
| `vllm` | vLLM pipeline latency between two hosts | vLLM, vllm-tbccl, a local model; physical |
| `exo` | Exo `MlxRing` against `MlxTbccl` between two hosts | Exo with exo-tbccl, a local model; physical |
| `custom` | a commented template for your own workloads | |

A profile holds only `<PLACEHOLDER>` values and the names of benchmarks. Copy it, fill it in (or keep your real values in an overlay file), and `run --dry-run` first. The framework profiles use the [providers](frameworks.md) of the framework you configure; if it is missing the suite says so and the native benchmarks still run.
