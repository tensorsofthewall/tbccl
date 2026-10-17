import json
import os
import subprocess
import sys
import unittest

from benchmarks.suite import inventory
from benchmarks.suite.tests.common import FIXTURES, LOOPBACK_IF, REPO, SuiteTestCase, requires_yaml


def suite(*args, env=None, cwd=REPO, timeout=300):
    e = {**os.environ, **(env or {})}
    return subprocess.run([sys.executable, "-m", "benchmarks.suite", *args], cwd=cwd, capture_output=True, text=True, env=e, timeout=timeout)


class Help(unittest.TestCase):
    def test_every_command_has_help(self):
        top = suite("--help")
        self.assertEqual(top.returncode, 0)
        for cmd in ("list", "inventory", "doctor", "run", "import", "compare", "report"):
            self.assertIn(cmd, top.stdout)
            h = suite(cmd, "--help")
            self.assertEqual(h.returncode, 0, cmd)
            self.assertIn("usage:", h.stdout)
        self.assertIn("--dry-run", suite("run", "--help").stdout)
        for flag in ("--output-dir", "--label"):
            self.assertIn(flag, suite("run", "--help").stdout)

    def test_unknown_command_fails(self):
        self.assertNotEqual(suite("bogus").returncode, 0)

    def test_list_and_inventory(self):
        out = suite("list", "--json")
        names = {r["name"]: r for r in json.loads(out.stdout)}
        self.assertTrue(names["collective_bench"]["integrated"])
        self.assertTrue(names["async_transfer"]["integrated"])
        self.assertFalse(names["async_raw_transport"]["integrated"])
        self.assertIn("ring-pipelined", names["collective_bench"]["algorithms"])
        inv = json.loads(suite("inventory", "--json").stdout)
        self.assertIn("tbccl_collective_bench", inv["executables"])


class DryRun(SuiteTestCase):
    @requires_yaml
    def test_dry_run_prints_matrix_and_commands_and_launches_nothing(self):
        from benchmarks.suite.tests.test_provenance_health import fake_native_build
        fake_native_build(self.dir / "build")
        out = suite("run", "benchmarks/suite/configs/smoke.yaml", "--dry-run", "--output-dir", str(self.dir / "o"), "--benchmark", "collective_bench", env={"TBCCL_BUILD_DIR": str(self.dir / "build")})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Nothing was launched", out.stdout)
        self.assertIn("SKIPPED: 1 elements are not divisible by 2 ranks", out.stdout)
        self.assertIn("--collective all-reduce --algorithm ring", out.stdout.replace("\n", " ") if False else out.stdout)
        self.assertFalse((self.dir / "o").exists())

    def test_invalid_config_is_reported_with_exit_2(self):
        (self.dir / "bad.json").write_text(json.dumps({"schema": "tbccl-bench-config/1", "hosts": {}, "ranks": [], "workloads": []}))
        out = suite("run", str(self.dir / "bad.json"), "--dry-run")
        self.assertEqual(out.returncode, 2)
        self.assertIn("invalid configuration", out.stderr + out.stdout)

    @requires_yaml
    def test_physical_example_dry_runs_but_cannot_run(self):
        ex = "benchmarks/suite/configs/tb4-core.example.yaml"
        dry = suite("run", ex, "--dry-run")
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertIn("PHYSICAL", dry.stdout)
        self.assertIn("unresolved placeholders", dry.stdout)
        real = suite("run", ex, "--output-dir", str(self.dir / "o"))
        self.assertEqual(real.returncode, 2)
        self.assertIn("--allow-physical", real.stderr)
        self.assertFalse((self.dir / "o").exists())
        allowed = suite("run", ex, "--output-dir", str(self.dir / "o"), "--allow-physical")
        self.assertNotEqual(allowed.returncode, 0)  # placeholders are unresolved: preflight fails before anything starts
        self.assertIn("unresolved placeholders", allowed.stdout)


class ImportCompareReport(SuiteTestCase):
    def test_import_compare_report_round_trip(self):
        o = str(self.dir / "o")
        for label in ("a", "b"):
            r = suite("import", str(FIXTURES / "collective_all_reduce_w2.csv"), "--output-dir", o, "--label", label, "--set", "environment.topology.kind=loopback",
                      "--set", 'environment.hosts=[{"host_id":"h","ranks":[0,1],"os":"Linux","arch":"x86_64","cpu_model":"X"}]')
            self.assertEqual(r.returncode, 0, r.stderr)
        runs = sorted(p.name for p in (self.dir / "o").iterdir())
        self.assertEqual(len(runs), 2)
        cmp_out = suite("compare", runs[0], runs[1], "--output-dir", o)
        self.assertEqual(cmp_out.returncode, 0, cmp_out.stderr)
        self.assertIn("Comparison:", cmp_out.stdout)
        rep = suite("report", runs[0], "--output-dir", o, "--no-plots")
        self.assertEqual(rep.returncode, 0, rep.stderr)
        text = (self.dir / "o" / runs[0] / "report.md").read_text()
        self.assertIn("Imported historical results", text)
        self.assertTrue((self.dir / "o" / runs[0] / "summary.csv").exists())

    def test_comparing_imports_without_known_hardware_is_refused(self):
        o = str(self.dir / "o")
        for label in ("a", "b"):
            suite("import", str(FIXTURES / "collective_all_reduce_w2.csv"), "--output-dir", o, "--label", label)
        runs = sorted(p.name for p in (self.dir / "o").iterdir())
        out = suite("compare", runs[0], runs[1], "--output-dir", o)
        self.assertEqual(out.returncode, 1)
        self.assertIn("hardware", out.stdout)
        self.assertIn("nothing to compare", out.stdout)

    def test_import_failure_exit_code_and_message(self):
        (self.dir / "x.csv").write_text("a,b\n1,2\n")
        r = suite("import", str(self.dir / "x.csv"), "--output-dir", str(self.dir / "o"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("import failed", r.stderr)

    def test_compare_unknown_run(self):
        r = suite("compare", "nope", "nada", "--output-dir", str(self.dir))
        self.assertEqual(r.returncode, 2)
        self.assertIn("not found", r.stderr)


@unittest.skipUnless(inventory.find_executable("tbccl_collective_bench", ["build-release", "build-bench", "build"]) or os.environ.get("TBCCL_BUILD_DIR"),
                     "no built TBCCL benchmark executables (set TBCCL_BUILD_DIR)")
class LoopbackSmoke(SuiteTestCase):
    """A minimal real run of the native executables on loopback: small, fast, no performance threshold."""

    def test_minimal_loopback_run_end_to_end(self):
        cfg = {"schema": "tbccl-bench-config/1", "name": "ci-smoke", "defaults": {"warmup": 2, "iterations": 10, "repetitions": 2, "timeout_s": 60},
               "topology": {"kind": "loopback", "interconnect": "loopback", "transport": "tcp", "physical": False},
               "hosts": {"local": {"address": "127.0.0.1", "interface": LOOPBACK_IF}},
               "ranks": [{"rank": i, "host": "local"} for i in range(2)],
               "workloads": [{"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [64]},
                             {"benchmark": "small_message_latency", "operation": "p2p", "sizes": [64]},
                             {"benchmark": "tensor_transfer", "operation": "latency_floor", "sizes": [64]},
                             {"benchmark": "collective_bench", "operation": "all_gather", "algorithms": ["ring"], "sizes": [64]},
                             {"benchmark": "collective_bench", "operation": "all_reduce", "algorithms": ["auto"], "sizes": [64], "datatype": "float32"}]}
        (self.dir / "c.json").write_text(json.dumps(cfg))
        o = self.dir / "o"
        r = suite("run", str(self.dir / "c.json"), "--output-dir", str(o), "--label", "ci")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        run_dir = next(o.iterdir())
        from benchmarks.suite import schema
        records = schema.read_records(run_dir / "results.jsonl")
        self.assertEqual(len(records), 10)
        self.assertEqual({x["status"]["state"] for x in records}, {"success"})
        for x in records:
            self.assertEqual(schema.validate_record(x), [])
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "success")
        self.assertTrue(any((run_dir / p).exists() for p in records[0]["artifacts"]["stdout"]))
        auto = next(x for x in records if x["benchmark"]["algorithm_requested"] == "auto")
        self.assertIn(auto["benchmark"]["algorithm_resolved"], ("reference", "ring"))
        rep = suite("report", run_dir.name, "--output-dir", str(o), "--no-plots")
        self.assertEqual(rep.returncode, 0, rep.stderr)


if __name__ == "__main__":
    unittest.main()
