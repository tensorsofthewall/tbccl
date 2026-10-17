# Benchmark baselines and regression comparison

This page describes the optional baseline registry of the [benchmark suite](benchmark-suite.md): how a measurement is defined so that two results can be compared honestly, and how a new result is judged against a stored one. It ships tools and definitions only. Measured values are hardware specific and stay in local, git-ignored storage.

```sh
python3 -m benchmarks.suite.baselines validate benchmarks/suite/baselines/definitions/*.json
python3 -m benchmarks.suite.baselines check-public benchmarks/suite/baselines/definitions/*.json
python3 -m benchmarks.suite.baselines compare baseline-snapshot.json candidate-snapshot.json --purpose regression
```

## Build identity in comparisons

A case's `software_versions` holds the build facts recorded for it. A comparison reports them in three states: facts both cases recorded that differ (`software_version_differences`, with both values), facts both recorded that match, and facts only one case recorded (`software_provenance.unknown`; unknown is never read as a match). A comparison between cases that recorded no build identity at all carries the label "build identity not recorded for at least one side"; the classification itself does not change, because a regression verdict is about the measurement, but it can no longer be attributed to a build. `benchmarks.suite.provenance_compare.identity_from_records(records)` extracts the facts from a run's records (executable hashes, TBCCL package, C ABI and wire versions per host, provider and framework versions, compiled-extension and script hashes), so a snapshot can fill `software_versions` from what was recorded instead of from memory.

## What a case is

A **case** names one measurement: the benchmark adapter, the metric and unit, how per-unit values are aggregated, and every dimension that defines the experiment (hardware profile, backend and transport, device pairing, memory kind, direction, payload, timing scope and completion semantics; for inference also model, quantization, stage order, layer split, workload, concurrency, sampling, runtime and client placement). Dimensions that do not apply to a case stay unset: native tensor cases carry no model metadata and single-device cases no rank layout.

`benchmarks/suite/baselines/definitions/` holds the public **definitions**: no values, no hosts, no addresses, no paths. A local **snapshot** adds the run ids, the independent observations, the value, the variation and the source artifacts (relative paths with SHA-256) of one set of campaigns. `schema.json` describes both; `check-public` fails on a definition that carries measured fields or anything that looks like an address, a private path or an e-mail address.

## The replication unit is the independent lifecycle

A snapshot case holds one observation per independent unit: a process pair (native and framework benchmarks, one per repetition) or an engine or cluster lifecycle (inference). Requests, iterations and tokens inside one lifecycle share its state and are not independent. A campaign with a thousand requests in one engine lifecycle has one unit. `aggregate_by_lifecycle` builds the observations (the median of each lifecycle's samples) and keeps the number of samples separately; the median of those medians is not the median of all requests pooled, and the comparison uses the former.

## What may be compared

`comparability.check` compares two cases for a stated **purpose**:

| purpose | what may differ | meaning of the result |
|---|---|---|
| `regression` | nothing except software versions (which are listed) | the same case measured again |
| `ab_transport` | the communication backend | an application-level difference of two integrations, not the overhead of a protocol |
| `ab_variant` | a declared variant (an experimental policy) | the variant against its default, not general |
| `ab_placement` | stage order and layer split | a placement contrast; device, API host, sampler location and traffic direction are confounded |
| `native_vs_framework` | the tool, the metric name and the datatype label, for p2p, collective or async pairs | an observed difference between two measurements, not an isolated adapter overhead |

Everything else must agree, and a dimension that defines the experiment and is not recorded on either side counts as a mismatch. Mismatches come with an explanation. The combinations that have caused wrong conclusions are named explicitly: a native round trip against a framework send plus acknowledgement (different timing scopes), a one-way latency obtained by halving a round trip (invalid), asynchronous submission against completion, different models or quantizations as a transport comparison, different placements as isolated transport overhead, traced against untraced runs, different models at different workloads as model-size scaling, and requests against lifecycles as replication units. A pair that cannot be compared has no ratio and no speedup; `contextual_view` shows both values side by side with the reasons.

## Classification

`comparison.compare(baseline, candidate, purpose)` returns one of:

| classification | meaning |
|---|---|
| `invalid_measurement` | a side is invalid or superseded, includes warm-up, has an undefined metric (time per output token of a one-token output), has non-finite values, or is a derived one-way latency |
| `incompatible` | the cases do not describe the same experiment for this purpose, or the units cannot be converted |
| `insufficient_evidence` | fewer independent units than required, a tail percentile from too few samples, requests used as replicates, or the smallest p-value the unit counts can attain is above the significance level |
| `regression`, `improvement` | the medians differ by more than the tolerance and the observations are separated by an exact test |
| `indeterminate` | neither a change nor the absence of one was shown; this is not a pass |

There is no fixed percentage threshold. The **tolerance** is the baseline's own repeat-to-repeat spread (largest relative range of the independent observations), or, for paired data, the spread of the within-pair relative differences, unless a case documents a benchmark-specific tolerance derived from repeated measurements. The **test** is an exact permutation (Wilcoxon rank-sum) test for independent units and an exact signed-rank test for paired units; the normal approximation is used only above 300,000 assignments. Observations that carry `pair_ids` (interleaved A/B blocks, the A,B,B,A designs of the campaigns) are compared as pairs and never collapsed into two unrelated samples. Candidates need at least three independent units by default (`min_independent_units` per case overrides this).

## Hardware observability

The suite's preflight reports whether kernel PCIe Advanced Error Reporting can show anything on a host, separately from link health: `observable`, `unavailable` (for example `pci=noaer` on the kernel command line, missing counters, or a non-Linux host) or `unknown`. With `pci=noaer` the AER counters stay at zero whatever the link does, so the suite never reports "zero AER errors"; it reports "AER unavailable; no detected link-state or interface-health regression under the available checks". The check is read-only: nothing changes boot parameters, AER, Thunderbolt, PCIe, firewall, power-management or network settings. CPU temperature and throttling are recorded passively and never gate a run.

## Tests

The tests (`benchmarks/suite/tests/test_baselines_*.py`, `test_aer_observability.py`, `test_report_robustness.py`) use synthetic observations only. They need no hardware, CUDA, Metal, Exo, vLLM or PyTorch, and no private report or raw result.
