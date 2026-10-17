import unittest

from benchmarks.suite import statistics as st


class Percentiles(unittest.TestCase):
    def test_median_and_linear_interpolation(self):
        self.assertEqual(st.median([3, 1, 2]), 2)
        self.assertEqual(st.median([1, 2, 3, 4]), 2.5)
        self.assertAlmostEqual(st.percentile(range(1, 101), 0.95), 95.05)
        self.assertEqual(st.percentile([5], 0.99), 5)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            st.percentile([], 0.5)
        with self.assertRaises(ValueError):
            st.percentile([1, float("nan")], 0.5)
        with self.assertRaises(ValueError):
            st.percentile([1, 2], 1.5)

    def test_summary_fields(self):
        s = st.summarize(list(range(1, 601)))
        self.assertEqual((s["n"], s["min"], s["max"]), (600, 1, 600))
        self.assertAlmostEqual(s["median"], 300.5)
        self.assertEqual(s["warnings"], [])
        self.assertIn("stdev", s)


class TailSamples(unittest.TestCase):
    def test_insufficient_samples_warn(self):
        self.assertIsNotNone(st.tail_warning(50, 0.99))
        self.assertIsNotNone(st.tail_warning(30, 0.95))
        self.assertIsNone(st.tail_warning(100, 0.95))
        self.assertIsNone(st.tail_warning(500, 0.99))
        self.assertTrue(st.summarize([1, 2, 3])["warnings"])


class Differences(unittest.TestCase):
    def test_relative(self):
        self.assertAlmostEqual(st.relative_latency_difference(100, 80), -0.2)
        self.assertAlmostEqual(st.relative_throughput_difference(2.0, 3.0), 0.5)
        with self.assertRaises(ZeroDivisionError):
            st.relative_difference(0, 1)

    def test_repeat_variation(self):
        v = st.repeat_variation([10, 11, 12])
        self.assertEqual((v["repetitions"], v["median"], v["min"], v["max"]), (3, 11, 10, 12))
        self.assertAlmostEqual(v["range_relative"], 2 / 11)
        self.assertIn("cv", v)
        self.assertNotIn("stdev", st.repeat_variation([10]))


class AggregationRules(unittest.TestCase):
    def test_per_iteration_max_needs_aligned_ranks(self):
        self.assertEqual(st.per_iteration_max({0: [1, 5, 3], 1: [2, 4, 9]}), [2, 5, 9])
        with self.assertRaises(st.AggregationError):
            st.per_iteration_max({0: [1, 2], 1: [1]})

    def test_global_percentile_from_per_rank_aggregates_is_refused(self):
        with self.assertRaises(st.AggregationError):
            st.combine_aggregates({0: {"p95": 10}, 1: {"p95": 12}}, "p95")

    def test_pooling_raw_samples(self):
        self.assertEqual(sorted(st.pool_samples({0: [1, 2], 1: [3]})), [1, 2, 3])
        with self.assertRaises(st.AggregationError):
            st.pool_samples({0: []})


class Units(unittest.TestCase):
    def test_time(self):
        self.assertEqual(st.convert(0.002, "s", "us"), 2000)
        self.assertAlmostEqual(st.convert(1500, "us", "ms"), 1.5)

    def test_rates_keep_decimal_binary_and_bits_apart(self):
        self.assertAlmostEqual(st.convert(8.0, "Gbit/s", "GB/s"), 1.0)
        self.assertAlmostEqual(st.convert(1.0, "GiB/s", "GB/s"), 1.073741824)
        self.assertAlmostEqual(st.convert(2.5e9, "bit/s", "Gbit/s"), 2.5)
        with self.assertRaises(ValueError):
            st.convert(1, "us", "GB/s")


if __name__ == "__main__":
    unittest.main()
