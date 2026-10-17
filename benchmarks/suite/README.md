# TBCCL benchmark suite

Optional tooling that drives the benchmark executables TBCCL already builds (and, if you provide the environments, PyTorch and vLLM benchmarks), normalises their results into one schema and compares runs only when they are equivalent. It is not part of the libtbccl runtime: it changes no C ABI, wire protocol or adapter, and nothing runs on a physical link unless you opt in.

```sh
python3 -m benchmarks.suite --help
python3 -m benchmarks.suite run benchmarks/suite/configs/smoke.yaml --dry-run   # loopback; launches nothing
```

Documentation (usage, configuration, hardware safety, results and comparisons, extending the suite):

* [Benchmark suite guide](../../docs/development/benchmark-suite.md)
* [Benchmark inventory](../../docs/development/benchmark-suite-inventory.md)
* [Performance evaluation](../../docs/development/performance-evaluation.md)
* [Baselines and regression comparison](../../docs/development/benchmark-baselines.md) (`python3 -m benchmarks.suite.baselines --help`)
* [Benchmark providers](../../docs/development/benchmark-providers.md) (the protocol for framework workloads; `python3 -m benchmarks.suite providers --help`)
* [Running benchmarks on your hardware](../../docs/benchmarks/index.md) and the ready-made profiles in `profiles/` (`python3 -m benchmarks.suite profiles`)

Tracked configurations contain `<PLACEHOLDER>` values only. Real addresses, interpreters, paths and launchers belong in a git-ignored `*.local.yaml` overlay, and run output goes to the git-ignored `results/bench-suite-local/`.
