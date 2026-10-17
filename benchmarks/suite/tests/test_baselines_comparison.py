"""Regression classification, exact tests, pairing, units and report generation on synthetic observations. No hardware, no raw results."""
import copy
import math
import unittest

from benchmarks.suite.baselines import catalog, comparison


def snap(case_id, obs, **over):
    case = copy.deepcopy(catalog.case_by_id(case_id))
    case.update(observations=list(obs), independent_lifecycle_count=len(obs), sample_count=len(obs) * 10, value=sorted(obs)[len(obs) // 2] if len(obs) % 2 else
                (sorted(obs)[len(obs) // 2 - 1] + sorted(obs)[len(obs) // 2]) / 2, run_ids=["run"], source_artifacts=[{"path": "run/results.jsonl"}])
    case.update(over)
    return case


TPOT = "exo.tpot.p128-o32.c1.MlxTbccl.mac>linux"
STEADY = [15.40, 15.50, 15.60, 15.55, 15.45, 15.52]


class ExactTests(unittest.TestCase):
    def test_rank_sum_values_for_complete_separation(self):
        self.assertAlmostEqual(comparison.rank_sum_p([1, 2, 3], [7, 8, 9])[0], 0.1)
        self.assertAlmostEqual(comparison.rank_sum_p([1, 2, 3, 4, 5], [7, 8, 9, 10, 11])[0], 2 / 252)
        self.assertEqual(comparison.rank_sum_p([1, 2, 3], [7, 8, 9])[1], "exact permutation")

    def test_identical_samples_give_p_one(self):
        self.assertAlmostEqual(comparison.rank_sum_p([1, 2, 3, 4], [1, 2, 3, 4])[0], 1.0)

    def test_ties_are_handled(self):
        p, _ = comparison.rank_sum_p([5, 5, 5, 5], [5, 5, 5, 5])
        self.assertAlmostEqual(p, 1.0)

    def test_large_samples_use_the_normal_approximation(self):
        p, method = comparison.rank_sum_p(list(range(40)), list(range(100, 140)))
        self.assertEqual(method, "normal approximation")
        self.assertLess(p, 1e-6)

    def test_signed_rank_exact(self):
        p, method, n = comparison.signed_rank_p([1, 2, 3, 4, 5, 6])
        self.assertAlmostEqual(p, 2 / 64)
        self.assertEqual((method, n), ("exact signed-rank", 6))
        self.assertEqual(comparison.signed_rank_p([0, 0])[0], 1.0)

    def test_smallest_attainable_p(self):
        self.assertAlmostEqual(comparison.min_attainable_p(1, 4), 0.4)
        self.assertAlmostEqual(comparison.min_attainable_p(3, 3), 0.1)
        self.assertLess(comparison.min_attainable_p(5, 5), 0.01)
        self.assertAlmostEqual(comparison.min_attainable_p_paired(5), 2 / 32)

    def test_rank_sum_needs_observations(self):
        with self.assertRaises(ValueError):
            comparison.rank_sum_p([], [1])


class Classification(unittest.TestCase):
    def test_clear_slowdown_is_a_regression(self):
        r = comparison.compare(snap(TPOT, STEADY), snap(TPOT, [v * 1.5 for v in STEADY]))
        self.assertEqual(r["classification"], "regression")
        self.assertAlmostEqual(r["relative_difference"], 0.5, places=6)
        self.assertLessEqual(r["p_value"], 0.05)
        self.assertIsNone(r.get("speedup"))

    def test_clear_speedup_is_an_improvement(self):
        r = comparison.compare(snap(TPOT, STEADY), snap(TPOT, [v * 0.6 for v in STEADY]))
        self.assertEqual(r["classification"], "improvement")

    def test_direction_is_respected_for_higher_is_better(self):
        case = "vllm.throughput.p128-o32.c12.cuda>metal"
        base = [340, 345, 335, 342, 338, 341]
        self.assertEqual(comparison.compare(snap(case, base), snap(case, [v * 1.3 for v in base]))["classification"], "improvement")
        self.assertEqual(comparison.compare(snap(case, base), snap(case, [v * 0.7 for v in base]))["classification"], "regression")

    def test_noise_level_difference_is_indeterminate_not_a_pass(self):
        base = [15.0, 15.8, 15.2, 16.0, 15.5, 15.9]
        cand = [15.1, 15.7, 15.3, 15.9, 15.6, 15.8]
        r = comparison.compare(snap(TPOT, base), snap(TPOT, cand))
        self.assertEqual(r["classification"], "indeterminate")
        self.assertTrue(any("neither a pass" in x for x in r["reasons"]))

    def test_separated_but_below_the_noise_floor_is_indeterminate(self):
        base = [10.0, 10.5, 11.0, 11.5, 12.0, 12.5]  # relative range about 22%
        cand = [v * 1.05 for v in base]  # a consistent 5% shift, but the baseline itself varies by 22%
        r = comparison.compare(snap(TPOT, base), snap(TPOT, cand))
        self.assertEqual(r["classification"], "indeterminate")
        self.assertTrue(any("does not exceed the tolerance" in x or "not statistically separated" in x for x in r["reasons"]))

    def test_a_difference_of_medians_beyond_the_tolerance_but_not_separated_is_indeterminate(self):
        base = [10.0, 10.1, 10.0, 10.05, 10.1, 9.95]  # tight baseline: the derived tolerance is about 1.5%
        cand = [9.5, 10.6, 9.7, 10.9, 10.4, 10.3]  # median 10.35 (+3%), but the candidate overlaps the baseline
        r = comparison.compare(snap(TPOT, base), snap(TPOT, cand))
        self.assertGreater(abs(r["relative_difference"]), r["tolerance"]["relative"])
        self.assertGreater(r["p_value"], 0.05)
        self.assertEqual(r["classification"], "indeterminate")
        self.assertTrue(any("not statistically separated" in x for x in r["reasons"]))

    def test_explicit_benchmark_tolerance_overrides_the_derived_floor(self):
        base = [15.40, 15.41, 15.39, 15.40, 15.42, 15.40]
        cand = [v * 1.01 for v in base]
        self.assertEqual(comparison.compare(snap(TPOT, base), snap(TPOT, cand), tolerance=0.05)["classification"], "indeterminate")
        self.assertEqual(comparison.compare(snap(TPOT, base), snap(TPOT, cand), tolerance=0.005)["classification"], "regression")

    def test_a_single_candidate_lifecycle_is_insufficient(self):
        r = comparison.compare(snap(TPOT, STEADY), snap(TPOT, [30.0]))
        self.assertEqual(r["classification"], "insufficient_evidence")
        self.assertTrue(any("independent unit" in x for x in r["reasons"]))

    def test_one_baseline_lifecycle_is_insufficient_even_with_many_candidates(self):
        r = comparison.compare(snap(TPOT, [12.8]), snap(TPOT, STEADY))
        self.assertEqual(r["classification"], "insufficient_evidence")

    def test_three_against_three_cannot_reach_the_significance_level(self):
        r = comparison.compare(snap(TPOT, [10, 10.1, 10.2]), snap(TPOT, [20, 20.1, 20.2]))
        self.assertEqual(r["classification"], "insufficient_evidence")
        self.assertTrue(any("smallest attainable" in x for x in r["reasons"]))
        self.assertAlmostEqual(r["min_attainable_p"], 0.1)

    def test_requests_inside_one_lifecycle_are_not_independent_units(self):
        many = [15.0 + i * 0.01 for i in range(200)]
        base = snap(TPOT, many, replication_unit="request")
        r = comparison.compare(base, snap(TPOT, [v * 2 for v in many], replication_unit="request"))
        self.assertEqual(r["classification"], "insufficient_evidence")
        self.assertTrue(any("not independent" in x for x in r["reasons"]))

    def test_tail_percentile_needs_enough_samples(self):
        p99 = dict(aggregation="percentile:99")
        a, b = snap(TPOT, STEADY, **p99), snap(TPOT, [v * 2 for v in STEADY], **p99)
        a["sample_count"] = b["sample_count"] = 60
        self.assertEqual(comparison.compare(a, b)["classification"], "insufficient_evidence")
        a["sample_count"] = b["sample_count"] = 600
        self.assertEqual(comparison.compare(a, b)["classification"], "regression")

    def test_min_units_is_configurable_per_case(self):
        base = snap(TPOT, STEADY, min_independent_units=8)
        self.assertEqual(comparison.compare(base, snap(TPOT, [v * 2 for v in STEADY]))["classification"], "insufficient_evidence")


class Validity(unittest.TestCase):
    def test_invalid_or_superseded_measurements_are_never_compared(self):
        for status in ("invalid", "superseded"):
            bad = snap(TPOT, STEADY, validity={"status": status})
            for pair in ((bad, snap(TPOT, STEADY)), (snap(TPOT, STEADY), bad)):
                self.assertEqual(comparison.compare(*pair)["classification"], "invalid_measurement")

    def test_exploratory_measurements_are_compared_but_the_label_survives(self):
        r = comparison.compare(snap(TPOT, STEADY, validity={"status": "exploratory"}), snap(TPOT, [v * 1.5 for v in STEADY]))
        self.assertEqual(r["classification"], "regression")

    def test_one_token_tpot_is_undefined(self):
        r = comparison.compare(snap(TPOT, STEADY, output_tokens=1), snap(TPOT, STEADY))
        self.assertEqual(r["classification"], "invalid_measurement")
        self.assertTrue(any("one-token" in x for x in r["reasons"]))

    def test_warmup_included_is_invalid(self):
        r = comparison.compare(snap(TPOT, STEADY), snap(TPOT, STEADY, warmup_excluded=False))
        self.assertEqual(r["classification"], "invalid_measurement")

    def test_non_finite_or_negative_observations_are_invalid(self):
        for bad in ([float("nan")] + STEADY, [-1.0] + STEADY):
            self.assertEqual(comparison.compare(snap(TPOT, STEADY), snap(TPOT, bad))["classification"], "invalid_measurement")

    def test_halved_round_trip_is_not_a_measurement(self):
        case = snap("native.p2p-rtt.4KiB", [125.0, 126.0, 127.0, 126.5], timing_scope="derived_one_way_from_rtt")
        r = comparison.compare(case, case)
        self.assertEqual(r["classification"], "invalid_measurement")
        self.assertTrue(any("halving a round trip" in x for x in r["reasons"]))

    def test_incompatible_beats_statistics_and_never_yields_a_speedup(self):
        ring, tbccl = snap("exo.tpot.p128-o32.c1.MlxRing.mac>linux", STEADY), snap("exo.tpot.p128-o32.c1.MlxTbccl.linux>mac", [v * 1.5 for v in STEADY])
        r = comparison.compare(ring, tbccl, "ab_transport")
        self.assertEqual(r["classification"], "incompatible")
        self.assertNotIn("relative_difference", r)
        self.assertIsNone(r["speedup"])
        self.assertTrue(r["reasons"])


class Units(unittest.TestCase):
    def test_candidate_units_are_converted_to_the_baseline_unit(self):
        base = snap("native.p2p-rtt.4KiB", [125.0, 126.0, 127.0, 126.0, 125.5, 126.5])
        cand = snap("native.p2p-rtt.4KiB", [v * 2 / 1000 for v in (125.0, 126.0, 127.0, 126.0, 125.5, 126.5)], unit="ms")
        r = comparison.compare(base, cand)
        self.assertEqual(r["classification"], "regression")
        self.assertAlmostEqual(r["relative_difference"], 1.0, places=6)

    def test_decimal_and_binary_rate_units_are_distinct_but_convertible(self):
        case = "native.iperf3.tcp-stream.a-to-b"
        base = snap(case, [15.8, 15.9, 15.85, 15.82, 15.87, 15.81], unit="Gbit/s")
        same_in_gb = snap(case, [v / 8 for v in (15.8, 15.9, 15.85, 15.82, 15.87, 15.81)], unit="GB/s")
        r = comparison.compare(base, same_in_gb)
        self.assertEqual(r["classification"], "indeterminate")
        self.assertAlmostEqual(r["relative_difference"], 0.0, places=9)
        gib = snap(case, [v / 8 for v in (15.8, 15.9, 15.85, 15.82, 15.87, 15.81)], unit="GiB/s")
        self.assertGreater(comparison.compare(base, gib)["relative_difference"], 0.05)  # 1 GiB/s is 7.4% more than 1 GB/s: not interchangeable

    def test_unconvertible_units_are_incompatible(self):
        r = comparison.compare(snap(TPOT, STEADY), snap(TPOT, STEADY, unit="tokens/s"))
        self.assertEqual(r["classification"], "incompatible")
        self.assertIn("cannot be converted", " ".join(r["reasons"]))

    def test_conversion_helpers(self):
        self.assertAlmostEqual(comparison.sstats.convert(1.0, "GB/s", "Gbit/s"), 8.0)
        self.assertAlmostEqual(comparison.sstats.convert(1.0, "GiB/s", "GB/s"), 1.073741824)
        self.assertAlmostEqual(comparison.sstats.convert(2000.0, "us", "ms"), 2.0)


class Pairing(unittest.TestCase):
    def test_pairing_detects_a_consistent_shift_that_block_effects_would_hide(self):
        blocks = ["b1", "b2", "b3", "b4", "b5", "b6"]
        base = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]  # large block-to-block variation (thermal drift, load)
        cand = [v * 1.06 for v in base]
        unpaired = comparison.compare(snap(TPOT, base), snap(TPOT, cand))
        paired = comparison.compare(snap(TPOT, base, pair_ids=blocks), snap(TPOT, cand, pair_ids=blocks))
        self.assertEqual(unpaired["classification"], "indeterminate")
        self.assertEqual(paired["classification"], "regression")
        self.assertIn("paired", paired["test"])
        self.assertEqual(paired["pairs"], 6)
        self.assertEqual(paired["relative_difference_basis"], "median of within-pair relative differences")

    def test_pair_ids_are_matched_not_positional(self):
        blocks = ["b1", "b2", "b3", "b4", "b5", "b6"]
        base = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
        cand = [v * 1.06 for v in base]
        shuffled = list(zip(blocks, cand))[::-1]
        r = comparison.compare(snap(TPOT, base, pair_ids=blocks), snap(TPOT, [v for _, v in shuffled], pair_ids=[k for k, _ in shuffled]))
        self.assertEqual(r["classification"], "regression")

    def test_too_few_pairs_for_the_significance_level_is_insufficient(self):
        blocks = ["b1", "b2", "b3", "b4", "b5"]
        base = [10.0, 11.0, 12.0, 13.0, 14.0]
        r = comparison.compare(snap(TPOT, base, pair_ids=blocks), snap(TPOT, [v * 2 for v in base], pair_ids=blocks))
        self.assertEqual(r["classification"], "insufficient_evidence")  # 5 pairs: smallest two-sided p is 0.0625

    def test_disjoint_pair_ids_fall_back_to_unpaired_and_say_so(self):
        base = [10.0, 10.1, 10.2, 10.15, 10.05, 10.12]
        r = comparison.compare(snap(TPOT, base, pair_ids=list("abcdef")), snap(TPOT, [v * 1.5 for v in base], pair_ids=list("uvwxyz")))
        self.assertEqual(r["classification"], "regression")
        self.assertTrue(any("unpaired" in x for x in r["reasons"]))


class Aggregation(unittest.TestCase):
    def test_median_of_lifecycle_medians_is_not_the_pooled_median(self):
        lifecycles = {"l1": [10, 10, 10, 10, 10, 10, 10, 10, 10, 10], "l2": [20], "l3": [30]}
        agg = comparison.aggregate_by_lifecycle(lifecycles)
        self.assertEqual(agg["observations"], [10, 20, 30])
        self.assertEqual(comparison.sstats.median(agg["observations"]), 20)
        pooled = [v for vals in lifecycles.values() for v in vals]
        self.assertEqual(comparison.sstats.median(pooled), 10)
        self.assertEqual(agg["sample_count"], 12)
        self.assertEqual(len(agg["observations"]), 3)  # three independent units, not twelve

    def test_empty_lifecycles_are_not_counted_as_units(self):
        agg = comparison.aggregate_by_lifecycle({"a": [1.0, 2.0], "b": []})
        self.assertEqual(agg["unit_ids"], ["a"])

    def test_only_the_median_is_supported(self):
        with self.assertRaises(ValueError):
            comparison.aggregate_by_lifecycle({"a": [1.0]}, "mean")

    def test_tolerance_is_derived_from_repeated_measurements(self):
        tol = comparison.derive_tolerance([[10.0, 11.0, 10.5], [20.0, 20.2, 20.1]])
        self.assertAlmostEqual(tol["relative"], 1.0 / 10.5)
        self.assertIsNone(comparison.derive_tolerance([[1.0]])["relative"])


class Reporting(unittest.TestCase):
    def test_markdown_lists_classification_labels_versions_and_corrections(self):
        base = snap("exo.tpot.p128-o32.c1.MlxRing.mac>linux", STEADY, software_versions={"exo": "0.3.70"})
        cand = snap("exo.tpot.p128-o32.c1.MlxTbccl.mac>linux", [v * 1.5 for v in STEADY], software_versions={"exo": "0.3.71"},
                    corrections=[{"id": "c1", "statement": "an earlier statement was corrected"}])
        text = comparison.to_markdown([comparison.compare(base, cand, "ab_transport")])
        self.assertIn("regression", text)
        self.assertIn("not protocol overhead", text)
        self.assertIn("exo: 0.3.70 -> 0.3.71", text)
        self.assertIn("an earlier statement was corrected", text)

    def test_empty_report(self):
        self.assertIn("No comparisons were made", comparison.to_markdown([]))

    def test_contextual_view_never_computes_a_ratio(self):
        v = snap("vllm.tpot.p128-o32.c1.cuda>metal", STEADY)
        e = snap("exo.tpot.p128-o32.c1.MlxTbccl.mac>linux", STEADY)
        view = comparison.contextual_view(v, e)
        self.assertIsNone(view["ratio"])
        self.assertFalse(view["comparable"])
        self.assertIn("CONTEXTUAL", view["label"])
        self.assertTrue(view["explanations"])

    def test_software_version_differences_do_not_block_a_comparison(self):
        r = comparison.compare(snap(TPOT, STEADY, software_versions={"x": "1"}), snap(TPOT, [v * 1.5 for v in STEADY], software_versions={"x": "2"}))
        self.assertEqual(r["classification"], "regression")
        self.assertEqual(r["software_version_differences"], {"x": ["1", "2"]})


if __name__ == "__main__":
    unittest.main()
