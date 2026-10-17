"""The baselines command line on synthetic snapshot files: validation, privacy check, comparison and provenance."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.suite.baselines import BASELINE_SCHEMA_VERSION, __main__ as cli, catalog

STEADY = [15.40, 15.50, 15.60, 15.55, 15.45, 15.52]
CASE = "exo.tpot.p128-o32.c1.MlxTbccl.mac>linux"


def snapshot(observations, case_id=CASE, **over):
    case = copy.deepcopy(catalog.case_by_id(case_id))
    case.update(observations=list(observations), independent_lifecycle_count=len(observations), sample_count=10 * len(observations), run_ids=["r"],
                value=sorted(observations)[len(observations) // 2 - 1] / 2 + sorted(observations)[len(observations) // 2] / 2, source_artifacts=[{"path": "r/results.jsonl"}])
    case.update(over)
    return {"schema_version": BASELINE_SCHEMA_VERSION, "kind": "snapshot", "catalog_id": "synthetic", "cases": [case]}


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(list(argv))
    return code, out.getvalue()


class Cli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def write(self, name, doc):
        path = self.dir / name
        path.write_text(json.dumps(doc))
        return str(path)

    def test_validate_accepts_the_shipped_definitions_and_rejects_a_bad_file(self):
        code, out = run("validate", *[str(p) for p in sorted(catalog.DEFINITIONS_DIR.glob("*.json"))])
        self.assertEqual(code, 0)
        self.assertIn("ok (definitions", out)
        bad = self.write("bad.json", {"schema_version": "nope", "kind": "snapshot", "catalog_id": "x", "cases": []})
        code, out = run("validate", bad)
        self.assertEqual(code, 1)
        self.assertIn("INVALID", out)

    def test_check_public_rejects_snapshots_and_private_values(self):
        code, out = run("check-public", *[str(p) for p in sorted(catalog.DEFINITIONS_DIR.glob("*.json"))])
        self.assertEqual((code, out.strip()), (0, "public catalogs are clean"))
        code, out = run("check-public", self.write("snap.json", snapshot(STEADY)))
        self.assertEqual(code, 1)
        self.assertIn("only definitions may be published", out)
        leaky = {"schema_version": BASELINE_SCHEMA_VERSION, "kind": "definitions", "catalog_id": "x", "cases": [copy.deepcopy(catalog.case_by_id(CASE))]}
        leaky["cases"][0]["notes"] = "see 192.0.2.55"
        code, out = run("check-public", self.write("leaky.json", leaky))
        self.assertEqual(code, 1)
        self.assertIn("IPv4 address", out)
        self.assertNotIn("192.0.2.55", out)

    def test_compare_reports_a_regression_and_writes_json(self):
        base, cand = self.write("base.json", snapshot(STEADY)), self.write("cand.json", snapshot([v * 1.5 for v in STEADY]))
        code, out = run("compare", base, cand, "--json", str(self.dir / "r.json"))
        self.assertEqual(code, 0)
        self.assertIn("regression", out)
        self.assertEqual(json.loads((self.dir / "r.json").read_text())[0]["classification"], "regression")

    def test_compare_flags_a_candidate_case_without_a_baseline(self):
        other = "exo.tpot.p128-o32.c1.MlxRing.mac>linux"
        code, out = run("compare", self.write("base.json", snapshot(STEADY)), self.write("cand.json", snapshot(STEADY, case_id=other)))
        self.assertIn("incompatible", out)
        self.assertIn("no baseline case", out)

    def test_compare_rejects_an_unsupported_catalog_version(self):
        stale = snapshot(STEADY)
        stale["schema_version"] = "tbccl-bench-baseline/0"
        with self.assertRaises(Exception) as ctx:
            run("compare", self.write("base.json", stale), self.write("cand.json", snapshot(STEADY)))
        self.assertIn("tbccl-bench-baseline/0", str(ctx.exception))

    def test_compare_purpose_and_case_filter(self):
        ring = "exo.tpot.p128-o32.c1.MlxRing.mac>linux"
        base = snapshot(STEADY, case_id=ring)
        cand = snapshot([v * 1.5 for v in STEADY], case_id=ring)
        cand["cases"][0]["backend"] = "MlxTbccl"
        code, out = run("compare", self.write("b.json", base), self.write("c.json", cand), "--purpose", "ab_transport", "--case", ring)
        self.assertIn("regression", out)
        self.assertIn("not protocol overhead", out)

    def test_partial_campaign_with_only_some_cases_compares_those(self):
        a, b = snapshot(STEADY), snapshot(STEADY, case_id="exo.tpot.p128-o32.c1.MlxRing.mac>linux")
        both = copy.deepcopy(a)
        both["cases"] += b["cases"]
        code, out = run("compare", self.write("base.json", both), self.write("cand.json", a))
        self.assertEqual(code, 0)
        self.assertEqual(out.count("| exo."), 1)

    def test_provenance_command(self):
        (self.dir / "r").mkdir()
        artifact = self.dir / "r" / "results.jsonl"
        artifact.write_text("{}\n")
        snap = snapshot(STEADY)
        snap["cases"][0]["source_artifacts"] = [{"path": "r/results.jsonl", "sha256": catalog.sha256_file(artifact)}]
        path = self.write("s.json", snap)
        self.assertEqual(run("provenance", path, "--results-root", str(self.dir))[0], 0)
        artifact.write_text("changed\n")
        code, out = run("provenance", path, "--results-root", str(self.dir))
        self.assertEqual(code, 1)
        self.assertIn("no longer matches", out)


if __name__ == "__main__":
    unittest.main()
