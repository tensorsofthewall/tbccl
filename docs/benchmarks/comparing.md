# Comparing runs

```sh
python3 -m benchmarks.suite compare <RUN_A> <RUN_B>                  # A is the baseline, B the candidate
python3 -m benchmarks.suite compare <RUN_A> <RUN_B> --json --output comparison.json
python3 -m benchmarks.suite compare <RUN_A> <RUN_B> --contextual     # also shows labelled, non-equivalent pairs for exploration
```

Run both configurations the same way, on the same hardware, with the same options, interleaved where the machine can drift (A, B, B, A), and with enough repetitions. `compare` pairs equivalent experiments by what they measured. For each pair it prints the baseline and candidate values, the relative difference, the repeat-to-repeat spread and a plain judgement: *exceeds the repeat-to-repeat spread*, *within repeat-to-repeat spread* or *too few repetitions*. Pairs that are not equivalent (different payload meaning, memory kinds, direction, topology, hardware, timing method or completion semantics, for example) are listed with the reason and are not compared.

Differences that could explain a change are shown beside it: a different commit, executable hash, library version, ABI or wire protocol, framework provider, adapter or framework version, the compiled extension an adapter wheel bundles (by hash) or the workload scripts (by hash). Every such fact is *matching* (both runs recorded it and it is the same), *different* (both recorded it and it differs) or *unknown* (only one run recorded it). Unknown is never read as a match: if a build was not recorded, the report says so and that a change cannot be attributed to a build. A report only states that two runs used identical builds when every recorded fact matches.

The differing facts are grouped into three components: **the TBCCL build** (executable hashes, library version, ABI, wire protocol), **the benchmark provider or adapter** (its version, its compiled extension and its workload scripts, which change together when you install a new adapter wheel) and **the framework** (PyTorch, vLLM, MLX, Exo and similar). The comparison then says what that allows you to conclude:

* **One component differs.** The report names it. If it is the change you meant to make, the measured difference can be attributed to it; if you did not change it, the software underneath your run changed and the comparison is not a test of your change.
* **Several components differ.** The report carries a warning that a performance difference cannot be attributed to one component alone. A deliberate comparison of an old and a new build is still allowed and still classified; the warning stays with it.
* **Facts are unknown.** They are listed and never counted as matching or as a change.

None of this changes the statistical classification of the measurements.

## Regression baselines

For regression tracking over time there is a baseline registry (`python3 -m benchmarks.suite.baselines --help`) with public workload definitions, comparability rules and an exact-test classifier. A comparison is classified as `improvement`, `regression`, `indeterminate`, `incompatible`, `insufficient_evidence` or `invalid_measurement`, with a tolerance derived from the baseline's own spread, and independent repetitions (or lifecycles for framework runs) are the replicates, never iterations or requests. See [baselines and regression comparison](../development/benchmark-baselines.md).

A comparison against numbers measured on different hardware is classified `incompatible`: the registry matches the hardware profile as well as the workload.
