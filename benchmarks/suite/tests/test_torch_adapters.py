"""The PyTorch adapters: registration without importing torch, capability rejection, command construction, parsing of the wrapper's output contract (saved from real runs)."""
import copy
import json
import subprocess
import sys
import unittest

from benchmarks.suite import config as cfg, schema
from benchmarks.suite.adapters import REGISTRY
from benchmarks.suite.adapters.base import Outcome, ParseError, Point
from benchmarks.suite.adapters.torch import stats
from benchmarks.suite.tests.common import BASE, FIXTURES, SuiteTestCase


def point(bench, op, size, backends=("host", "host"), options=None, source=None):
    return Point(workload="t", benchmark=bench, operation=op, algorithm=None, payload_bytes=size, datatype=None, reduce_op=None, rank_count=2,
                 ranks=[{"rank": i, "host": "local", "memory_backend": b} for i, b in enumerate(backends)], options=options or {}, warmup=10, iterations=50, repetitions=1, timeout_s=60,
                 source_rank=source)


class Ctx:
    class _Host:
        def __init__(self, interface="lo", local=True):
            self.id, self.interface, self.is_local = "local", interface, local
            self.executables = {"torch_bench_dir": "/provider"}

    def __init__(self):
        self.config = type("C", (), {"hosts": {"local": self._Host()}})()
        self.repo_root = "/repo"

    def rank_address(self, spec):
        return f"192.0.2.{spec['rank'] + 1}"

    def tool(self, name, host):
        return "/venv/bin/python"


def lines(prefix, name):
    return [l for l in (FIXTURES / name).read_text().splitlines() if l.startswith(prefix)]


def outcome(tmp, texts):
    paths = {}
    for r, text in texts.items():
        path = tmp / f"rank{r}.out"
        path.write_text(text)
        paths[r] = str(path)
    return Outcome({0: 0, 1: 0}, set(), False, None, paths, {}, 0.1)


def patched(name, rank, **changes):
    line = lines("TBCCL_TORCH", name)[0]
    prefix, body = line.split(" ", 1)
    d = json.loads(body)
    d.update(changes)
    return prefix + " " + json.dumps(d)


class Registration(unittest.TestCase):
    def test_adapters_are_registered_and_importing_the_suite_does_not_import_torch(self):
        for name in ("torch_collectives", "torch_p2p", "torch_async", "torch_ddp"):
            self.assertTrue(REGISTRY[name].caps.integrated)
        code = "import sys; import benchmarks.suite.adapters; print('torch' in sys.modules)"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(__import__("pathlib").Path(__file__).resolve().parents[3]))
        self.assertEqual(out.stdout.strip(), "False", out.stderr)

    def test_schema_accepts_the_training_throughput_kinds(self):
        s = schema.load_schema()
        t = s["properties"]["measurements"]["properties"]["throughput"]["items"]["properties"]
        self.assertIn("training_samples_per_second", t["kind"]["enum"])
        self.assertIn("samples/s", t["unit"]["enum"])
        self.assertIn("GB/s", t["unit"]["enum"])  # nothing was removed


class Capabilities(SuiteTestCase):
    def test_gloo_is_cpu_only_and_unsupported_pairings_are_rejected_not_substituted(self):
        a = REGISTRY["torch_collectives"]
        p = point("torch_collectives", "all_reduce", 4096, ("cuda", "host"), {"backend": "gloo"})
        self.assertIn("CPU tensors only", a.check_point(p, None))
        self.assertEqual(p.skip_state, "unsupported")
        self.assertIsNone(a.check_point(point("torch_collectives", "all_reduce", 4096, ("cuda", "mps")), None))
        self.assertIn("multiple of 4", a.check_point(point("torch_collectives", "all_reduce", 10), None))
        self.assertIn("not a torch device", a.check_point(point("torch_collectives", "all_reduce", 4096, ("cuda-pinned", "host")), None))
        self.assertIn("unknown backend", a.check_point(point("torch_collectives", "all_reduce", 4096, options={"backend": "nccl"}), None))

    def test_unknown_operations_and_dtypes_fail_configuration_validation(self):
        raw = copy.deepcopy(BASE)
        raw["workloads"] = [{"name": "w", "benchmark": "torch_collectives", "operations": ["reduce"], "sizes": [4096]}]
        with self.assertRaises(cfg.ConfigError):
            cfg.build_config(raw)
        raw["workloads"] = [{"name": "w", "benchmark": "torch_collectives", "operation": "all_reduce", "sizes": [4096], "options": {"dtype": "bfloat16"}}]
        with self.assertRaises(cfg.ConfigError) as ctx:
            cfg.build_config(raw)
        self.assertIn("float32", str(ctx.exception))

    def test_ddp_sizes_must_equal_the_gradient_bytes_of_the_model(self):
        errors = REGISTRY["torch_ddp"].validate_workload({"sizes": [1], "options": {"layers": 4, "width": 512, "batch": 64, "tag": "s"}}, ["train"])
        self.assertIn("4202496", " ".join(errors))
        ok = REGISTRY["torch_ddp"].validate_workload({"sizes": [4202496], "options": {"layers": 4, "width": 512, "batch": 64, "tag": "s"}}, ["train"])
        self.assertEqual(ok, [])
        self.assertIn("options.tag", " ".join(REGISTRY["torch_ddp"].validate_workload({"sizes": [4202496], "options": {"layers": 4, "width": 512, "batch": 64}}, ["train"])))

    def test_point_ids_keep_backend_and_tag_apart(self):
        a = point("torch_ddp", "train", 4202496, options={"tag": "bs64"})
        b = point("torch_ddp", "train", 4202496, options={"tag": "bs256"})
        c = point("torch_collectives", "all_reduce", 4096, options={"backend": "gloo"})
        d = point("torch_collectives", "all_reduce", 4096)
        self.assertNotEqual(a.point_id, b.point_id)
        self.assertNotEqual(c.point_id, d.point_id)


class Planning(SuiteTestCase):
    def test_a_host_without_a_provider_directory_is_refused_with_a_pointer_to_the_provider(self):
        ctx = Ctx()
        ctx.config.hosts["local"].executables = {}
        with self.assertRaises(ParseError) as raised:
            REGISTRY["torch_p2p"].plan(point("torch_p2p", "p2p_ack", 4096), ctx, [31000, 31001], self.dir)
        self.assertIn("torch_bench_dir", str(raised.exception))
        self.assertIn("torch-tbccl provider", str(raised.exception))

    def test_command_lines_and_environment(self):
        p = point("torch_collectives", "broadcast", 65536, ("cuda", "mps"), source=1)
        g = REGISTRY["torch_collectives"].plan(p, Ctx(), [31000, 31001], self.dir)
        a0, a1 = g.invocations
        self.assertEqual(a0.argv[:2], ["/venv/bin/python", "/provider/comm_bench.py"])
        self.assertEqual(a0.argv[a0.argv.index("--devices") + 1], "cuda,mps")
        self.assertEqual(a0.argv[a0.argv.index("--source") + 1], "1")
        self.assertEqual(a0.argv[a0.argv.index("--expect-addresses") + 1], "192.0.2.1,192.0.2.2")
        self.assertEqual((a0.env["RANK"], a1.env["RANK"], a0.env["MASTER_ADDR"], a1.env["MASTER_ADDR"], a1.env["MASTER_PORT"]), ("0", "1", "192.0.2.1", "192.0.2.1", "31000"))
        self.assertEqual(a1.env["TBCCL_LOCAL_ENDPOINT"], "192.0.2.2:0")
        self.assertNotIn("--gloo-address", a1.argv)
        self.assertEqual((a0.listens_on, a1.wait_for_port), (31000, 31000))

    def test_gloo_binds_each_rank_to_its_own_link_address_and_never_sets_a_tbccl_endpoint(self):
        g = REGISTRY["torch_collectives"].plan(point("torch_collectives", "all_reduce", 4096, options={"backend": "gloo"}), Ctx(), [31000, 31001], self.dir)
        for rank, inv in enumerate(g.invocations):
            self.assertEqual(inv.argv[inv.argv.index("--gloo-address") + 1], f"192.0.2.{rank + 1}")
        self.assertNotIn("TBCCL_LOCAL_ENDPOINT", g.invocations[1].env)

    def test_profile_turns_the_adapter_trace_on_only_for_tbccl(self):
        p = point("torch_ddp", "train", 4202496, options={"layers": 4, "width": 512, "batch": 64, "tag": "s", "profile": True})
        g = REGISTRY["torch_ddp"].plan(p, Ctx(), [31000, 31001], self.dir)
        self.assertEqual(g.invocations[0].env["TORCH_TBCCL_TRACE"], "1")
        self.assertIn("--profile", g.invocations[0].argv)
        plain = REGISTRY["torch_ddp"].plan(point("torch_ddp", "train", 4202496, options={"layers": 4, "width": 512, "batch": 64, "tag": "s"}), Ctx(), [31000, 31001], self.dir)
        self.assertNotIn("TORCH_TBCCL_TRACE", plain.invocations[0].env)
        self.assertNotIn("--profile", plain.invocations[0].argv)


class Parsing(SuiteTestCase):
    def comm(self, over=None):
        texts = {r: (FIXTURES / f"torch_comm_allreduce_rank{r}.txt").read_text() for r in (0, 1)}
        for r, changes in (over or {}).items():
            texts[r] = patched(f"torch_comm_allreduce_rank{r}.txt", r, **changes) + "\n"
        return outcome(self.dir, texts)

    def parse(self, o, backends=("host", "host"), **kw):
        return REGISTRY["torch_collectives"].parse(point("torch_collectives", "all_reduce", 4096, backends, **kw), None, o)

    def test_a_good_pair_of_outputs(self):
        r = self.parse(self.comm())
        t = r.measurements["timing"]
        d0 = json.loads(lines("TBCCL_TORCH_BENCH", "torch_comm_allreduce_rank0.txt")[0].split(" ", 1)[1])
        d1 = json.loads(lines("TBCCL_TORCH_BENCH", "torch_comm_allreduce_rank1.txt")[0].split(" ", 1)[1])
        self.assertEqual(t["samples"], [max(a, b) for a, b in zip(d0["samples_us"]["completion_us"], d1["samples_us"]["completion_us"])])
        self.assertEqual(len(t["per_rank"]), 2)
        self.assertEqual(r.verification["status"], "passed")
        self.assertIn("device_ready_median_us", r.measurements["extra"])
        self.assertEqual(r.measurements["throughput"][0]["unit"], "GB/s")

    def test_missing_or_duplicate_rank_output_is_invalid(self):
        o = self.comm()
        (self.dir / "rank1.out").write_text("nothing here\n")
        with self.assertRaises(ParseError):
            self.parse(o)
        (self.dir / "rank1.out").write_text((FIXTURES / "torch_comm_allreduce_rank1.txt").read_text() * 2)
        with self.assertRaises(ParseError):
            self.parse(o)

    def test_failed_verification_version_mismatch_device_and_shape_are_rejected(self):
        with self.assertRaises(ParseError):
            self.parse(self.comm({0: {"verify_ok": False}}))
        env = json.loads(lines("TBCCL_TORCH_BENCH", "torch_comm_allreduce_rank1.txt")[0].split(" ", 1)[1])["environment"]
        with self.assertRaises(ParseError) as ctx:
            self.parse(self.comm({1: {"environment": {**env, "libtbccl": "0.5.1"}}}))
        self.assertIn("version mismatch", str(ctx.exception))
        with self.assertRaises(ParseError) as ctx:
            self.parse(self.comm(), backends=("cuda", "host"))
        self.assertIn("device", str(ctx.exception))
        with self.assertRaises(ParseError):
            self.parse(self.comm({0: {"shape": [7]}}))

    def test_traffic_outside_the_data_link_is_rejected(self):
        o = self.comm({0: {"expect_addresses": "192.0.2.1,192.0.2.2", "connections": [["192.0.2.1:5", "192.0.2.2:6"], ["192.0.2.2:7", "192.0.2.9:8"]]},
                         1: {"expect_addresses": "192.0.2.1,192.0.2.2", "connections": [["192.0.2.2:6", "192.0.2.1:5"]]}})
        with self.assertRaises(ParseError) as ctx:
            self.parse(o)
        self.assertIn("outside the data link", str(ctx.exception))
        ok = self.comm({0: {"expect_addresses": "192.0.2.1,192.0.2.2", "connections": [["192.0.2.1:5", "192.0.2.2:6"], ["[::ffff:192.0.2.1]:9", "[::ffff:192.0.2.2]:10"]]},
                          1: {"expect_addresses": "192.0.2.1,192.0.2.2", "connections": [["192.0.2.2:6", "192.0.2.1:5"]]}})
        self.assertEqual(self.parse(ok).verification["status"], "passed")
        unknown = self.comm({0: {"connections": None}})
        with self.assertRaises(ParseError):
            self.parse(unknown)

    def test_async_keeps_submission_wait_and_callback_separate(self):
        o = outcome(self.dir, {r: (FIXTURES / f"torch_async_rank{r}.txt").read_text() for r in (0, 1)})
        r = REGISTRY["torch_async"].parse(point("torch_async", "all_reduce_async", 4096), None, o)
        names = set(r.measurements["extra"])
        self.assertTrue({"submit_median_us_rank0", "wait_median_us_rank0", "callbacks_seen_rank0"} <= names)
        self.assertEqual(r.measurements["timing"]["metric"], "pytorch_async_all_reduce_lifetime_max_across_ranks")
        self.assertIn("submission time is not a communication time", " ".join(r.measurements["warnings"]))

    def ddp(self, over=None):
        texts = {r: (FIXTURES / f"torch_ddp_rank{r}.txt").read_text() for r in (0, 1)}
        for r, changes in (over or {}).items():
            texts[r] = patched(f"torch_ddp_rank{r}.txt", r, **changes) + "\n"
        return outcome(self.dir, texts)

    def test_ddp_record_has_training_throughput_and_inferred_exposure(self):
        p = point("torch_ddp", "train", 4202496, ("cuda", "host"), {"layers": 4, "width": 512, "batch": 64, "tag": "s"})
        r = REGISTRY["torch_ddp"].parse(p, None, self.ddp())
        self.assertEqual({t["kind"] for t in r.measurements["throughput"]}, {"training_samples_per_second", "training_steps_per_second"})
        e = r.measurements["extra"]
        self.assertIn("inference", e["rank0_exposed_communication_inferred_us"]["note"])
        self.assertIn("compute_imbalance_ratio", e)
        self.assertEqual(r.measurements["timing"]["metric"], "ddp_step_time_slowest_rank")

    def test_ddp_correctness_failure_and_wrong_model_are_rejected(self):
        p = point("torch_ddp", "train", 4202496, ("cuda", "host"), {"layers": 4, "width": 512, "batch": 64, "tag": "s"})
        good = json.loads(lines("TBCCL_TORCH_DDP", "torch_ddp_rank0.txt")[0].split(" ", 1)[1])["correctness"]
        with self.assertRaises(ParseError):
            REGISTRY["torch_ddp"].parse(p, None, self.ddp({0: {"correctness": {**good, "passed": False}}}))
        with self.assertRaises(ParseError):
            REGISTRY["torch_ddp"].parse(p, None, self.ddp({1: {"failed": "correctness"}}))
        with self.assertRaises(ParseError):
            REGISTRY["torch_ddp"].parse(point("torch_ddp", "train", 4202496, ("cuda", "host"), {"layers": 4, "width": 512, "batch": 128, "tag": "s"}), None, self.ddp())

    def test_stats_use_linear_interpolation(self):
        s = stats([1, 2, 3, 4])
        self.assertEqual((s["median"], s["min"], s["max"]), (2.5, 1.0, 4.0))
        self.assertAlmostEqual(s["p95"], 3.85)
        with self.assertRaises(ParseError):
            stats([])


if __name__ == "__main__":
    unittest.main()
