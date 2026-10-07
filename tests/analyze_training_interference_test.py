#!/usr/bin/env python3
"""Tests for analyze_training_interference.py (the training-interference work)."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_training_interference as ati  # noqa: E402


class HistogramTests(unittest.TestCase):
    def test_bins_are_left_inclusive_right_exclusive(self):
        result = ati.histogram([199, 200, 224, 225, 999, 1000, 5000])
        self.assertEqual(result["0-200"], 1)   # 199
        self.assertEqual(result["200-225"], 2)  # 200, 224
        self.assertEqual(result["225-250"], 1)  # 225
        self.assertEqual(result["800-1000"], 1)  # 999
        self.assertEqual(result[">=1000"], 2)   # 1000, 5000

    def test_empty_samples_all_zero(self):
        result = ati.histogram([])
        self.assertTrue(all(v == 0 for v in result.values()))


class ModeFractionsTests(unittest.TestCase):
    def test_fractions_sum_to_one(self):
        samples = [250] * 60 + [400] * 30 + [600] * 5 + [1200] * 5
        result = ati.mode_fractions(samples)
        total = result["low"] + result["intermediate"] + result["high"] + result["tail"]
        self.assertAlmostEqual(total, 1.0)
        self.assertAlmostEqual(result["low"], 0.6)
        self.assertAlmostEqual(result["intermediate"], 0.3)
        self.assertAlmostEqual(result["high"], 0.05)
        self.assertAlmostEqual(result["tail"], 0.05)

    def test_empty_returns_none_not_zero(self):
        result = ati.mode_fractions([])
        self.assertIsNone(result["low"])
        self.assertEqual(result["n"], 0)


class SummaryStatsTests(unittest.TestCase):
    def test_percentiles_and_max(self):
        samples = list(range(1, 101))  # 1..100
        result = ati.summary_stats(samples)
        self.assertEqual(result["n"], 100)
        self.assertEqual(result["max_us"], 100)
        self.assertEqual(result["median_us"], 50.5)

    def test_empty_all_none(self):
        result = ati.summary_stats([])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["median_us"])


class EmpiricalCdfTests(unittest.TestCase):
    def test_monotonic_fractions(self):
        pairs = ati.empirical_cdf([30, 10, 20])
        self.assertEqual([v for v, _ in pairs], [10, 20, 30])
        self.assertAlmostEqual(pairs[-1][1], 1.0)


class WilsonIntervalTests(unittest.TestCase):
    def test_known_interval_half(self):
        result = ati.wilson_interval(50, 100)
        self.assertAlmostEqual(result["point"], 0.5)
        self.assertLess(result["low"], 0.5)
        self.assertGreater(result["high"], 0.5)

    def test_zero_successes_lower_bound_zero(self):
        result = ati.wilson_interval(0, 1200)
        self.assertAlmostEqual(result["low"], 0.0, places=9)
        self.assertGreater(result["high"], 0.0)
        self.assertLess(result["high"], 0.01)

    def test_zero_n_returns_none(self):
        result = ati.wilson_interval(0, 0)
        self.assertIsNone(result["point"])


class FisherExactTests(unittest.TestCase):
    def test_identical_proportions_high_p_value(self):
        # 10/100 vs 10/100 -- no difference, p should be near 1.
        table = [[10, 90], [10, 90]]
        p = ati.fisher_exact_two_sided(table)
        self.assertGreater(p, 0.9)

    def test_strong_difference_low_p_value(self):
        # 9/1000 vs 0/1200 -- the sustained-session reproduction work's
        # actual headline comparison shape.
        table = [[9, 991], [0, 1200]]
        p = ati.fisher_exact_two_sided(table)
        self.assertLess(p, 0.01)

    def test_symmetric_under_table_transpose(self):
        table_a = [[5, 95], [2, 98]]
        table_b = [[2, 98], [5, 95]]
        self.assertAlmostEqual(ati.fisher_exact_two_sided(table_a),
                                ati.fisher_exact_two_sided(table_b), places=9)


class EventRateComparisonTests(unittest.TestCase):
    def test_normal_case_both_nonzero(self):
        result = ati.event_rate_comparison(9, 1000, 3, 1200)
        self.assertAlmostEqual(result["on"]["rate"], 0.009)
        self.assertAlmostEqual(result["off"]["rate"], 0.0025)
        self.assertIsNotNone(result["risk_ratio_on_over_off"])
        self.assertFalse(result["risk_ratio_continuity_corrected"])

    def test_zero_off_events_uses_continuity_correction(self):
        result = ati.event_rate_comparison(9, 1000, 0, 1200)
        self.assertEqual(result["off"]["events"], 0)
        self.assertTrue(result["risk_ratio_continuity_corrected"])
        self.assertIsNotNone(result["risk_ratio_on_over_off"])
        self.assertAlmostEqual(result["off"]["wilson"]["low"], 0.0, places=9)

    def test_zero_both_events(self):
        result = ati.event_rate_comparison(0, 1000, 0, 1200)
        self.assertIsNone(result["risk_ratio_on_over_off"])
        self.assertIsNone(result["risk_ratio_continuity_corrected"])

    def test_zero_n_does_not_crash(self):
        result = ati.event_rate_comparison(0, 0, 0, 1200)
        self.assertIsNone(result["on"]["rate"])


class IrqCadenceTableTests(unittest.TestCase):
    def test_extracts_tx_rx_from_per_vector_baseline(self):
        baselines = {
            "training_on": {"177": {"median_us": 135.0, "p95_us": 260.0},
                             "178": {"median_us": 128.0, "p95_us": 140.0}},
            "training_off": {"177": {"median_us": 130.0, "p95_us": 150.0},
                              "178": {"median_us": 128.0, "p95_us": 135.0}},
        }
        rows = ati.irq_cadence_table(baselines)
        on_row = next(r for r in rows if r["condition"] == "training_on")
        self.assertEqual(on_row["tx_median_us"], 135.0)
        self.assertEqual(on_row["rx_median_us"], 128.0)

    def test_missing_baseline_handled(self):
        rows = ati.irq_cadence_table({"empty": None})
        self.assertIsNone(rows[0]["tx_median_us"])


class ConditionTableTests(unittest.TestCase):
    def test_builds_summary_row_per_condition(self):
        conditions = {"training_on": [250] * 90 + [1200] * 9 + [400] * 1,
                      "training_off": [250] * 100}
        rows = ati.condition_table(conditions)
        on_row = next(r for r in rows if r["condition"] == "training_on")
        off_row = next(r for r in rows if r["condition"] == "training_off")
        self.assertEqual(on_row["n"], 100)
        self.assertEqual(on_row["count_ge_1000us"], 9)
        self.assertEqual(off_row["count_ge_1000us"], 0)
        self.assertEqual(off_row["tail_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
