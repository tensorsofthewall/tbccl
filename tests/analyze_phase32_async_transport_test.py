#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze_phase32_async_transport as analyze  # noqa: E402


class LoadSenderResultsTests(unittest.TestCase):
    def test_skips_non_sender_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a_sender.json").write_text(json.dumps({"role": "sender", "label": "x"}))
            (root / "b_receiver.json").write_text(json.dumps({"role": "receiver"}))
            results = analyze.load_sender_results(str(root / "*.json"))
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["label"], "x")

    def test_skips_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "bad.json").write_text("{not valid json")
            results = analyze.load_sender_results(str(root / "*.json"))
            self.assertEqual(results, [])


class ChunkSweepTableTests(unittest.TestCase):
    def test_renders_expected_columns(self):
        results = [{"label": "sweep_1m", "chunk_bytes": 1048576, "pipeline_depth": 1,
                    "gib_per_s": 1.5, "completion_median_us": 100.0}]
        table = analyze.chunk_sweep_table(results)
        self.assertIn("sweep_1m", table)
        self.assertIn("1048576", table)
        self.assertIn("1.500", table)


class AsyncOverheadTableTests(unittest.TestCase):
    def test_caller_block_reduction_computed(self):
        results = [{"label": "x", "bytes": 65536, "enqueue_median_us": 1.0,
                    "completion_median_us": 100.0}]
        table = analyze.async_overhead_table(results)
        # (1 - 1/100) * 100 = 99.0
        self.assertIn("99.00", table)

    def test_zero_completion_does_not_crash(self):
        results = [{"label": "x", "bytes": 0, "enqueue_median_us": 0, "completion_median_us": 0}]
        table = analyze.async_overhead_table(results)
        self.assertIn("0.00", table)


class OverlapTableTests(unittest.TestCase):
    def test_hidden_fraction_computed(self):
        rows = [{"bucket": "4MiB", "compute_us": 500, "comm_us": 300,
                 "serial_us": 800, "async_us": 550}]
        table = analyze.overlap_table(rows)
        # hidden = 800-550=250; hidden_pct = 250/300*100 = 83.33
        self.assertIn("83.33", table)

    def test_zero_comm_does_not_crash(self):
        rows = [{"bucket": "x", "compute_us": 100, "comm_us": 0,
                 "serial_us": 100, "async_us": 100}]
        table = analyze.overlap_table(rows)
        self.assertIn("0.00", table)


if __name__ == "__main__":
    unittest.main()
