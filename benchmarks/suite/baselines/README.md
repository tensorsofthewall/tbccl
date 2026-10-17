# Baseline registry

Case definitions (`definitions/`), the catalog schema (`schema.json`) and the offline regression tools of the benchmark suite. The method, the comparison purposes and the classification rules are in [docs/development/benchmark-baselines.md](../../../docs/development/benchmark-baselines.md).

| file | role |
|---|---|
| `schema.json` | schema of definition and snapshot catalogs (`tbccl-bench-baseline/1`) |
| `catalog.py` | loading, validation, privacy check of public definitions, provenance of source artifacts |
| `comparability.py` | which cases may be compared, for which purpose, and why not |
| `comparison.py` | exact tests, tolerance, classification, per-lifecycle aggregation, report text |
| `timelines.py` | TTFT, TPOT, ITL, E2E and throughput recomputed from per-request timeline files |
| `definitions/*.json` | public case definitions: no values, hosts, addresses or paths |

Measurement snapshots hold hardware-specific values and belong in `results/bench-suite-local/` (git-ignored).

```sh
python3 -m benchmarks.suite.baselines validate benchmarks/suite/baselines/definitions/*.json
python3 -m benchmarks.suite.baselines check-public benchmarks/suite/baselines/definitions/*.json
python3 -m benchmarks.suite.baselines compare BASELINE.json CANDIDATE.json --purpose regression
python3 -m benchmarks.suite.baselines provenance SNAPSHOT.json --results-root results/bench-suite-local
```
