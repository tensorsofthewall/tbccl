#!/usr/bin/env python3
"""tools/benchmark_allreduce.py: report structure and statistics (always), and a real two-rank loopback run when an installed torch-tbccl can be imported."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parent.parent / "tools" / "benchmark_allreduce.py"
spec = importlib.util.spec_from_file_location("benchmark_allreduce", TOOL)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class Report(unittest.TestCase):
    def test_summary_statistics(self):
        s = bench.summarize([0.002, 0.001, 0.003], 4_000_000)
        self.assertEqual(s["seconds_per_op_median"], 0.002)
        self.assertEqual((s["seconds_per_op_min"], s["seconds_per_op_max"], s["repeats"]), (0.001, 0.003, 3))
        self.assertAlmostEqual(s["algbw_gbps_median"], 2.0)

    def test_report_records_what_a_reader_needs(self):
        env = bench.environment("gloo", "127.0.0.1", 2)
        report = bench.build_report(env, {"sizes": "1024"}, [], "label")
        self.assertEqual(report["schema"], "tbccl-benchmark/1")
        for key in ("python", "platform", "backend", "world_size", "master_addr_is_loopback"):
            self.assertIn(key, report["environment"])
        self.assertTrue(report["environment"]["master_addr_is_loopback"])
        self.assertIn("not RDMA", report["note"])
        self.assertFalse(bench.environment("gloo", "192.168.3.1", 2)["master_addr_is_loopback"])
        json.dumps(report)

    def test_a_non_loopback_rank_needs_a_master_address(self):
        r = subprocess.run([sys.executable, str(TOOL), "--rank", "0"], capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("--master-addr", r.stderr)


def torch_tbccl_importable():
    r = subprocess.run([sys.executable, "-c", "import torch, torch_tbccl"], capture_output=True)
    return r.returncode == 0


@unittest.skipUnless(torch_tbccl_importable(), "needs an environment with torch and torch-tbccl installed")
class LoopbackRun(unittest.TestCase):
    def test_two_ranks_write_a_verified_result(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "r.json"
            r = subprocess.run([sys.executable, str(TOOL), "--sizes", "1024,65536", "--warmup", "1", "--iters", "3", "--repeats", "2", "--quiet", "--out", str(out), "--launch-timeout", "120"],
                               capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            rep = json.loads(out.read_text())
            self.assertEqual([x["bytes"] for x in rep["results"]], [1024, 65536])
            self.assertTrue(all(x["verified"] for x in rep["results"]))
            self.assertEqual(rep["environment"]["backend"], "tbccl")


if __name__ == "__main__":
    unittest.main()
