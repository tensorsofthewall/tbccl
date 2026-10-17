import copy
import csv
import io
import json
import unittest

from benchmarks.suite import schema
from benchmarks.suite.reporting import compare, plots, summary
from benchmarks.suite.tests.common import SuiteTestCase


def rec(median=100.0, *, size=4096, rep=0, algo="ring", resolved=None, state="success", cpu="CPU-A", backends=("host", "host"), topology="loopback", commit="aaa", sha="1", op="all_gather",
        semantics="bytes per rank", method="m", dtype=None, p=0.95, run="r1", throughput=2.0, stat="median", pmethod="linear-interpolation-n-1"):
    timing = {"metric": "collective_duration", "unit": "us", "primary_statistic": stat, "statistics": {stat: median, "p95": median * 1.2}, "statistics_provenance": {stat: "reported", "p95": "reported"},
              "percentile_method": pmethod, "sample_count": 100, "samples": None, "per_rank": None}
    return schema.new_record(
        run_id=run, record_id=f"{run}/{op}-{size}/rep{rep}",
        benchmark={"name": "collective_bench", "category": "collective", "operation": op, "provenance": "native", "algorithm_requested": algo, "algorithm_resolved": resolved or algo,
                   "source_commit": commit, "executable_sha256": sha, "executable_version": "TBCCL 0.6.0", "build_type": "Release"},
        environment={"hosts": [{"host_id": "h", "ranks": [0, 1], "os": "Linux", "arch": "x86_64", "cpu_model": cpu, "gpu_model": None}], "memory_backend": list(backends),
                     "memory_domain": {"label": "cpu-cpu", "per_rank": list(backends)}, "topology": {"kind": topology, "interconnect": topology, "physical": False}, "transport": "tcp",
                     "network_interface": "lo", "mtu": 65536, "framework_versions": {}, "multi_host": False},
        configuration={"payload_bytes": size, "payload_semantics": semantics, "datatype": dtype, "rank_count": 2, "repetitions": 3, "repetition_index": rep, "timing_method": method,
                       "completion_semantics": "call returned", "options": {}},
        status=state, reason=None if state == "success" else "x",
        measurements={"timing": timing, "throughput": [{"kind": "collective_algorithmic", "value": throughput, "unit": "GB/s", "definition": "d"}], "extra": {}, "warnings": []} if state == "success" else None,
        verification={"status": "passed" if state == "success" else "not_run", "method": "m", "details": ""})


def reps(values, **kw):
    return [rec(v, rep=i, **kw) for i, v in enumerate(values)]


class Comparison(unittest.TestCase):
    def test_equivalent_runs_compare_with_relative_difference_and_spread(self):
        a, b = reps([100, 101, 99], run="a"), reps([80, 81, 79], run="b", commit="bbb", sha="2")
        r = compare.compare(a, b)
        self.assertEqual((len(r["comparisons"]), r["incompatible"]), (1, []))
        c = r["comparisons"][0]
        self.assertAlmostEqual(c["relative_latency_difference"], -0.2)
        self.assertAlmostEqual(c["baseline"], 100)
        self.assertEqual(c["judgement"], "exceeds the repeat-to-repeat spread")
        self.assertEqual(c["label"], "equivalent")
        self.assertNotEqual(c["versions_baseline"], c["versions_candidate"])
        md = compare.to_markdown(r)
        self.assertIn("-20.0%", md)
        self.assertIn("Version differences", md)
        self.assertNotIn("speedup", md.lower())

    def test_difference_inside_noise_is_not_claimed(self):
        r = compare.compare(reps([100, 130, 90], run="a"), reps([101, 128, 95], run="b"))
        self.assertIn("within repeat-to-repeat spread", r["comparisons"][0]["judgement"])

    def test_too_few_repetitions(self):
        r = compare.compare(reps([100], run="a"), reps([50], run="b"))
        self.assertIn("too few repetitions", r["comparisons"][0]["judgement"])

    def test_each_incompatibility_is_rejected_and_explained(self):
        base = reps([100, 100, 100], run="a")
        variants = {"hardware": dict(cpu="CPU-B"), "memory backend": dict(backends=("host", "cuda-pinned")), "topology": dict(topology="direct-link"),
                    "payload semantics": dict(semantics="total bytes"), "timing method": dict(method="other"), "resolved algorithm": dict(resolved="reference"),
                    "primary statistic": dict(stat="mean"), "percentile definition": dict(pmethod="floor-nearest-rank")}
        for name, kw in variants.items():
            r = compare.compare(base, reps([90, 90, 90], run="b", **kw))
            self.assertEqual(r["comparisons"], [], name)
            self.assertEqual(len(r["incompatible"]), 1, name)
            self.assertTrue(r["incompatible"][0]["differences"], name)
            self.assertIn("not equivalent", r["incompatible"][0]["note"])

    def test_contextual_comparison_is_labelled(self):
        r = compare.compare(reps([100] * 3, run="a"), reps([50] * 3, run="b", cpu="CPU-B"), contextual=True)
        c = r["comparisons"][0]
        self.assertTrue(c["label"].startswith("CONTEXTUAL"))
        self.assertTrue(any("hardware" in d for d in c["differences"]))
        self.assertIn("Contextual comparisons", compare.to_markdown(r))

    def test_different_payload_or_algorithm_never_pairs(self):
        r = compare.compare(reps([100] * 3, run="a"), reps([100] * 3, run="b", size=8192))
        self.assertEqual(r["comparisons"], [])
        self.assertEqual((len(r["only_a"]), len(r["only_b"])), (1, 1))
        r = compare.compare(reps([100] * 3, run="a"), reps([100] * 3, run="b", algo="auto", resolved="ring"))
        self.assertEqual(r["comparisons"], [])

    def test_unknown_hardware_is_never_treated_as_equivalent(self):
        def blank(records):
            for r in records:
                r["environment"]["hosts"] = []
            return records
        r = compare.compare(blank(reps([100] * 3, run="a")), blank(reps([100] * 3, run="b")))
        self.assertEqual(r["comparisons"], [])
        self.assertIn("hardware", " ".join(r["incompatible"][0]["differences"]))

    def test_failed_or_incomplete_measurements_never_enter_a_statistic(self):
        a = reps([100, 100, 100], run="a")
        b = [rec(10, rep=0, run="b", state="failure"), rec(10, rep=1, run="b", state="timeout")]
        r = compare.compare(a, b)
        self.assertEqual(r["comparisons"], [])
        self.assertIn("no successful measurement", r["incompatible"][0]["note"])
        b = [rec(80, rep=0, run="b"), rec(1, rep=1, run="b", state="failure"), rec(80, rep=2, run="b")]
        c = compare.compare(a, b)["comparisons"][0]
        self.assertEqual((c["candidate_ok"], c["candidate_total"]), (2, 3))
        self.assertAlmostEqual(c["candidate"], 80)


class Summary(unittest.TestCase):
    def records(self):
        return reps([100, 110, 90]) + [rec(0, rep=0, size=8192, state="failure")] + reps([50, 50, 50], size=65536)

    def test_rows_group_repetitions_and_list_failures_separately(self):
        rows = summary.summarize(self.records())
        by = {r["payload_bytes"]: r for r in rows}
        self.assertEqual((by[4096]["successful"], by[4096]["records"], by[4096]["value_median"]), (3, 3, 100))
        self.assertEqual((by[4096]["value_min"], by[4096]["value_max"]), (90, 110))
        self.assertIsNone(by[8192]["value_median"])
        self.assertEqual(by[8192]["successful"], 0)

    def test_markdown_and_csv(self):
        recs = self.records()
        rows = summary.summarize(recs)
        md = summary.to_markdown(rows, recs, "T", {"run_id": "x", "status": "partial", "repository": {"commit": "abc", "dirty": True}})
        self.assertIn("| collective_bench | all_gather |", md)
        self.assertIn("Points without a valid measurement", md)
        self.assertIn("failure", md)
        self.assertIn("uncommitted", md)
        parsed = list(csv.DictReader(io.StringIO(summary.to_csv(rows))))
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]["benchmark"], "collective_bench")
        self.assertEqual(summary.to_csv([]), "")

    def test_too_few_repetitions_are_flagged(self):
        rows = summary.summarize(reps([100, 101]))
        self.assertIn("spread is not meaningful", rows[0]["warnings"])


class Plots(SuiteTestCase):
    def test_plot_generation_when_matplotlib_is_available(self):
        if not plots.available():
            self.skipTest("matplotlib not installed")
        a, b = reps([100, 110, 90], run="a") + reps([200, 210, 190], run="a", size=65536), reps([80, 81, 79], run="b") + reps([150, 160, 140], run="b", size=65536)
        made = plots.make_all(b, self.dir, ["b/results.jsonl"], baseline=a, comparison=compare.compare(a, b))
        names = sorted(p.rsplit("/", 1)[1] for p in made)
        self.assertEqual(names, ["bandwidth-vs-size-collective_bench.png", "baseline-vs-candidate-collective_bench.png", "latency-vs-size-collective_bench.png",
                                 "repetition-variability-collective_bench.png"])
        for p in made:
            with open(p, "rb") as f:
                self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")

    def test_plots_skip_failed_records(self):
        if not plots.available():
            self.skipTest("matplotlib not installed")
        made = plots.make_all([rec(1, state="failure")], self.dir, ["x"])
        self.assertEqual(made, [])


if __name__ == "__main__":
    unittest.main()
