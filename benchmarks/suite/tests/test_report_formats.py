"""User-generated reports: Markdown, CSV, JSON and a self-contained offline HTML page from a run's own records (no historical data needed)."""
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

from benchmarks.suite import schema
from benchmarks.suite.reporting import summary
from benchmarks.suite.tests.common import REPO, SuiteTestCase
from benchmarks.suite.tests.test_compare_report import rec, reps

PROVIDER = {"provider_id": "torch-tbccl", "host": "local", "protocol_version": 1, "provider_version": "0.2.0", "status": "available", "interpreter": "/venv/bin/python", "directory": "/p",
            "framework_versions": {"torch": "2.13.0", "python": "3.13.0"}, "native_modules": {"torch_tbccl/_C.cpython-313.so": {"sha256": "cd" * 32, "bytes": 7}}, "entry_point_sha256": {"communication": "ab" * 32}, "result_schema_versions": ["tbccl-bench-result/1"]}


def with_provider(records, entry=PROVIDER):
    for r in records:
        r["environment"]["providers"] = [dict(entry)]
    return records


def write_run(directory, records, manifest=None):
    directory.mkdir(parents=True, exist_ok=True)
    schema.write_jsonl(directory / "results.jsonl", records)
    (directory / "manifest.json").write_text(json.dumps(manifest or {"run_id": directory.name, "status": "success", "suite_version": "0.1.0", "repository": {"commit": "abc123", "dirty": False}}))
    return directory


class Formats(SuiteTestCase):
    def records(self):
        return with_provider(reps([100, 101, 99], run="r1")) + [rec(0, rep=3, run="r1", state="failure", size=65536)]

    def test_the_four_reports_are_written_by_default_from_the_run_alone(self):
        run = write_run(self.dir / "r1", self.records())
        out = subprocess.run([sys.executable, "-m", "benchmarks.suite", "report", str(run), "--no-plots", "--output", str(self.dir / "o")], capture_output=True, text=True, cwd=str(REPO))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(sorted(p.name for p in (self.dir / "o").iterdir()), ["report.html", "report.json", "report.md", "summary.csv"])

    def test_format_selects_files_and_rejects_unknown_ones(self):
        run = write_run(self.dir / "r1", self.records())
        ok = subprocess.run([sys.executable, "-m", "benchmarks.suite", "report", str(run), "--no-plots", "--format", "json", "--output", str(self.dir / "j")], capture_output=True, text=True, cwd=str(REPO))
        self.assertEqual((ok.returncode, [p.name for p in (self.dir / "j").iterdir()]), (0, ["report.json"]))
        bad = subprocess.run([sys.executable, "-m", "benchmarks.suite", "report", str(run), "--format", "pdf"], capture_output=True, text=True, cwd=str(REPO))
        self.assertEqual(bad.returncode, 2)
        self.assertIn("unknown report format", bad.stderr)

    def test_json_lists_results_providers_and_the_invalid_points_apart(self):
        recs = self.records()
        doc = json.loads(summary.to_json(summary.summarize(recs), recs, "t", {"run_id": "r1", "status": "success"}))
        self.assertEqual(doc["report_version"], "tbccl-bench-report/1")
        self.assertEqual([r["payload_bytes"] for r in doc["results"] if r["successful"]], [4096])
        self.assertEqual(doc["record_states"], {"success": 3, "failure": 1})
        self.assertEqual(doc["points_without_valid_measurement"][0]["state"], "failure")
        self.assertEqual(doc["providers"][0]["provider_id"], "torch-tbccl")

    def test_the_html_page_is_self_contained_and_offline(self):
        recs = self.records()
        page = summary.to_html(summary.summarize(recs), recs, "T", {"run_id": "r1", "status": "success", "suite_version": "0.1.0", "repository": {"commit": "abc", "dirty": True}})
        self.assertNotRegex(page, r"(?i)<script|<link|<iframe|<object|@import|src=\"https?:|href=\"https?:|url\(")
        self.assertIn("default-src 'none'", page)
        self.assertIn("uncommitted changes", page)
        self.assertIn("never averaged into a result", page)
        self.assertEqual(page.count("<html"), 1)

    def test_html_escapes_everything_that_came_from_a_result(self):
        recs = with_provider(reps([100, 101, 99], run="r1"), {**PROVIDER, "provider_id": "p<img src=x>", "framework_versions": {"torch": "<b>1</b>"}})
        recs.append(rec(0, rep=4, run="r1", state="failure", size=1))
        recs[-1]["status"]["reason"] = "</td><script>alert(1)</script>"
        page = summary.to_html(summary.summarize(recs), recs, "<T>", {"run_id": "<r>", "status": "success"})
        self.assertNotIn("<script>alert(1)", page)
        self.assertNotIn("<img src=x>", page)
        self.assertNotIn("<b>1</b>", page)
        self.assertIn("&lt;script&gt;", page)

    def test_images_are_embedded_as_data_uris_and_nothing_else_is_fetched(self):
        recs = self.records()
        page = summary.to_html(summary.summarize(recs), recs, "T", None, [("latency vs size", b"\x89PNG\r\n")])
        self.assertRegex(page, r'src="data:image/png;base64,[A-Za-z0-9+/=]+"')
        self.assertEqual(re.findall(r'(?:src|href)="(?!data:)[^"]+"', page), [])

    def test_the_framework_backend_is_a_column_so_gloo_and_tbccl_rows_are_distinguishable(self):
        a, b = reps([10, 11, 12], run="r1"), reps([500, 510, 490], run="r1")
        for r in a:
            r["benchmark"]["backend"] = "pytorch-tbccl"
            r["configuration"]["options"] = {"backend": "tbccl"}
        for r in b:
            r["benchmark"]["backend"] = "pytorch-gloo"
            r["configuration"]["options"] = {"backend": "gloo"}
        rows = summary.summarize(a + b)
        self.assertEqual(sorted(r["backend"] for r in rows), ["pytorch-gloo", "pytorch-tbccl"])
        md = summary.to_markdown(rows, a + b)
        self.assertIn("pytorch-gloo", md)
        self.assertIn("pytorch-tbccl", md)

    def test_the_two_rank_orientations_are_separate_rows_with_an_orientation_column(self):
        fwd, rev = reps([100, 101, 99], run="r1"), reps([130, 131, 129], run="r1")
        for r in rev:
            r["configuration"]["orientation"] = "reverse"
            r["configuration"]["options"] = {"orientation": "reverse"}
        rows = summary.summarize(fwd + rev)
        self.assertEqual(sorted(r["orientation"] for r in rows), ["forward", "reverse"])
        self.assertIn("| forward |", summary.to_markdown(rows, fwd + rev))
        self.assertIn("<th>orientation</th>", summary.to_html(rows, fwd + rev))

    def test_the_compiled_extension_hash_is_shown_and_unavailable_when_not_reported(self):
        recs = with_provider(reps([100, 101, 99], run="r1"))
        md = summary.to_markdown(summary.summarize(recs), recs)
        self.assertIn("_C cdcdcdcdcd", md)
        self.assertIn("_C cdcdcdcdcd", summary.to_html(summary.summarize(recs), recs))
        none = with_provider(reps([100, 101, 99], run="r1"), {**PROVIDER, "native_modules": None})
        self.assertIn("| unavailable |", summary.to_markdown(summary.summarize(none), none))

    def test_a_bundled_compatibility_run_says_versions_were_not_collected(self):
        recs = with_provider(reps([100, 101, 99], run="r1"), {**PROVIDER, "status": "bundled_compat", "provider_version": None, "framework_versions": {}, "entry_point_sha256": {"comm_bench.py": "cd" * 32}})
        md = summary.to_markdown(summary.summarize(recs), recs)
        self.assertIn("bundled_compat", md)
        self.assertIn("unavailable", md)
        self.assertIn("framework versions were not collected", md)

    def test_hosts_and_cleanup_are_in_every_format(self):
        recs = reps([100, 101, 99], run="r1")
        for r in recs:
            r["environment"]["multi_host"] = True
            r["environment"]["hosts"][0]["build"] = {"build_type": "Release", "compiler": "cc", "status": "stamped", "build_commit": "abcdef1234567890", "executable_sha256": "ab" * 32,
                                                      "tbccl_package_version": "0.6.0", "c_abi": 1, "wire_protocol": 4}
            r["artifacts"]["ranks"] = [{"rank": 0, "host_id": "h", "supervised": True, "cleanup_confirmed": True}]
        rows = summary.summarize(recs)
        doc = json.loads(summary.to_json(rows, recs))
        self.assertEqual((doc["hosts"][0]["host"], doc["hosts"][0]["tbccl_version"], doc["cleanup"]["state"]), ("h", "0.6.0", "confirmed"))
        for text in (summary.to_markdown(rows, recs), summary.to_html(rows, recs)):
            self.assertIn("Cleanup was confirmed", text)
            self.assertIn("stamped (commit abcdef1234)", text)
        recs[0]["artifacts"]["ranks"][0].update(cleanup_confirmed=False, cleanup_note="a leftover <b>process</b>")
        page = summary.to_html(rows, recs)
        self.assertIn("Cleanup NOT confirmed", page)
        self.assertNotIn("<b>process</b>", page)
        self.assertEqual(json.loads(summary.to_json(rows, recs))["cleanup"]["state"], "not_confirmed")

    def test_a_run_from_a_source_export_says_the_commit_is_unavailable_and_never_invents_one(self):
        recs = reps([100, 101, 99], run="r1")
        manifest = {"run_id": "r1", "status": "success", "suite_version": "0.1.0", "repository": {"commit": None, "dirty": None, "branch": None}}
        for text in (summary.to_markdown(summary.summarize(recs), recs, "t", manifest), summary.to_html(summary.summarize(recs), recs, "t", manifest)):
            self.assertIn("unavailable (the source tree has no Git metadata)", text)
            self.assertNotIn("None", text.replace("None of", ""))
        with_commit = summary.to_markdown(summary.summarize(recs), recs, "t", {**manifest, "repository": {"commit": "abc123", "dirty": True}})
        self.assertIn("`abc123` (uncommitted changes)", with_commit)

    def test_a_native_only_run_has_no_provider_section(self):
        recs = reps([100, 101, 99], run="r1")
        self.assertNotIn("Framework providers", summary.to_markdown(summary.summarize(recs), recs))
        self.assertNotIn("Framework providers", summary.to_html(summary.summarize(recs), recs))


if __name__ == "__main__":
    unittest.main()
