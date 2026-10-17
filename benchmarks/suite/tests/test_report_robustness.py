"""Report generation on empty, partial and interrupted campaigns: a killed run leaves truncated files, which must be reported, not crash the whole report."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from benchmarks.suite import statistics as sstats
from benchmarks.suite.reporting import b5, b6
from benchmarks.suite.tests.test_b6_report import make_run


class Base(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def two_backends(self):
        make_run(self.root, "t4-r0", "MlxRing", tpot_ms=10.0, point="lat-1")
        make_run(self.root, "t4-r1", "MlxRing", tpot_ms=10.4, point="lat-3")
        make_run(self.root, "t4-t0", "MlxTbccl", tpot_ms=15.0, point="lat-2")
        make_run(self.root, "t4-t1", "MlxTbccl", tpot_ms=15.4, point="lat-4")


class EmptyCampaigns(Base):
    def test_a_campaign_with_no_runs_produces_empty_tables_and_exits_cleanly(self):
        out = self.root / "out"
        self.assertEqual(b6.main([str(self.root), "b6", "--out", str(out)]), 0)
        self.assertEqual(json.loads((out / "excluded_runs.json").read_text()), [])
        self.assertEqual(list((out / "tables").glob("*.csv")), [])  # nothing to tabulate: no empty tables are written

    def test_every_run_excluded_still_reports_them(self):
        make_run(self.root, "t4-r0", "MlxRing", state="failure")
        out = self.root / "out"
        self.assertEqual(b6.main([str(self.root), "b6", "--out", str(out)]), 0)
        excluded = json.loads((out / "excluded_runs.json").read_text())
        self.assertEqual(len(excluded), 1)

    def test_a_directory_without_results_is_skipped(self):
        (self.root / "20261017T000000Z-b6-t4-r0-0001").mkdir()
        self.assertEqual(b6.load(self.root, "b6"), [])


class InterruptedRuns(Base):
    def test_a_truncated_manifest_is_listed_as_unreadable_not_fatal(self):
        self.two_backends()
        run = make_run(self.root, "t4-t2", "MlxTbccl", point="lat-6")
        (run / "manifest.json").write_text('{"status": "succ')
        for module in (b6, b5):
            runs = module.load(self.root, "b6")
            names = {d.name: m for _, d, m, _ in runs}
            self.assertEqual(names[run.name]["status"], "unreadable")
            self.assertIn("manifest", names[run.name]["reason"])
        out = self.root / "out"
        self.assertEqual(b6.main([str(self.root), "b6", "--out", str(out)]), 0)
        excluded = {e["run"]: e for e in json.loads((out / "excluded_runs.json").read_text())}
        self.assertEqual(excluded[run.name]["status"], "unreadable")

    def test_a_truncated_last_request_line_loses_only_that_request(self):
        run = make_run(self.root, "t4-r0", "MlxRing", point="lat-1")
        target = next((run / "raw" / "lat-1" / "rep0" / "exo").glob("lat-r1-p128-o16.requests.jsonl"))
        text = target.read_text()
        target.write_text(text[: len(text) - 40])
        runs = b6.load(self.root, "b6")
        tier, run_dir, manifest, records = runs[0]
        rows = b6.read_requests(run_dir, records[0], "lat-r1-p128-o16")
        self.assertEqual(len(rows), 5)  # six were written; the sixth line is cut
        self.assertTrue(b6.cell_rows(runs))

    def test_a_corrupt_line_in_the_middle_is_still_an_error(self):
        run = make_run(self.root, "t4-r0", "MlxRing", point="lat-1")
        target = next((run / "raw" / "lat-1" / "rep0" / "exo").glob("lat-r1-p128-o16.requests.jsonl"))
        lines = target.read_text().splitlines()
        lines[2] = lines[2][:20]
        target.write_text("\n".join(lines) + "\n")
        _, run_dir, _, records = b6.load(self.root, "b6")[0]
        with self.assertRaises(ValueError):
            b6.read_requests(run_dir, records[0], "lat-r1-p128-o16")

    def test_a_truncated_sampler_file_keeps_the_complete_samples(self):
        run = make_run(self.root, "t4-r0", "MlxRing", point="lat-1")
        target = run / "raw" / "lat-1" / "rep0" / "exo" / "samples-host-a.jsonl"
        text = target.read_text()
        target.write_text(text[: len(text) - 25])
        _, run_dir, _, records = b6.load(self.root, "b6")[0]
        series = b6.sampler_series(run_dir, records[0])
        self.assertIn("host-a", series)
        self.assertTrue(all("wall_utc" in row for row in series["host-a"]))
        self.assertTrue(b6.resource_rows(b6.load(self.root, "b6")))


class Timestamps(unittest.TestCase):
    def test_utc_seconds(self):
        self.assertEqual(b6.utc_s("2026-10-17T08:14:10Z"), 1792224850)

    def test_fractional_seconds_and_offsets_are_accepted(self):
        self.assertEqual(b6.utc_s("2026-10-17T08:14:10.750Z"), 1792224850)
        self.assertEqual(b6.utc_s("2026-10-17T08:14:10+00:00"), 1792224850)

    def test_garbage_is_a_value_error(self):
        for text in ("", "yesterday", "2026-13-40T00:00:00Z"):
            with self.assertRaises(ValueError):
                b6.utc_s(text)


class SmallSamples(unittest.TestCase):
    def test_percentiles_on_one_and_two_samples(self):
        self.assertEqual(sstats.percentile([7.0], 0.99), 7.0)
        self.assertEqual(sstats.percentile([1.0, 3.0], 0.5), 2.0)
        s = sstats.summarize([5.0])
        self.assertEqual((s["n"], s["median"]), (1, 5.0))
        self.assertNotIn("stdev", s)
        self.assertEqual(len(s["warnings"]), 2)  # p95 and p99 both come with a sample-size warning

    def test_repeat_variation_of_one_repetition_has_no_spread_claim(self):
        v = sstats.repeat_variation([3.0])
        self.assertEqual(v["repetitions"], 1)
        self.assertEqual(v["range_relative"], 0.0)
        self.assertNotIn("stdev", v)

    def test_no_samples_is_an_error_not_a_zero(self):
        with self.assertRaises(ValueError):
            sstats.summarize([])
        with self.assertRaises(ValueError):
            sstats.repeat_variation([])


if __name__ == "__main__":
    unittest.main()
