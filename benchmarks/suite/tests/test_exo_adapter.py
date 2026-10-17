"""The Exo adapter: registration without importing Exo or MLX, configuration and stage checks, command construction, and parsing of the client / node-log / sampler output contracts
(synthetic fixtures in the real formats)."""
import base64
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.suite import schema
from benchmarks.suite.adapters import REGISTRY
from benchmarks.suite.adapters.base import Outcome, ParseError, Point
from benchmarks.suite.stream import exo_metrics as metrics

NAMES = ("exo_correctness", "exo_latency", "exo_throughput", "exo_transport_ab", "exo_stage_profile", "exo_large_model", "exo_lifecycle")
LIP, MIP = "192.0.2.1", "192.0.2.2"
PROMPTS = {"short": {"items": 1, "approx_tokens": 24}, "medium": {"items": 6, "approx_tokens": 140}}
PHASES = [{"name": "warm", "kind": "warmup", "prompt": "short", "max_tokens": 4, "requests": 2}, {"name": "m", "kind": "rounds", "prompt": "short", "max_tokens": 8, "conc": 1, "requests": 6}]
OPTIONS = {"model_id": "mlx-community/Qwen3-0.6B-8bit", "backend": "MlxTbccl", "prompts": PROMPTS, "phases": PHASES, "primary": {"phase": "m", "metric": "tpot"},
           "memory_override_mb": {"host-a": 5200, "host-b": 10800}}


class Host:
    def __init__(self, hid, local, executables, interface="link0"):
        self.id, self.is_local, self.executables, self.interface, self.interfaces, self.os = hid, local, executables, interface, {"link": "x"}, None
        self.env = {}
        self.launcher = {"kind": "local"} if local else {"kind": "prefix", "argv": ["ssh", "mac"], "shell_join": True}


class Ctx:
    def __init__(self):
        exe = lambda h: {"exo": f"/{h}/bin/exo", "exo_python": f"/{h}/bin/python", "exo_bench_dir": f"/{h}/bench", "exo_home": ".exo_b6", "exo_lan_address": "192.0.2.1" if h == "a" else "192.0.2.2"}
        self.config = type("C", (), {"hosts": {"host-a": Host("host-a", True, exe("a")), "host-b": Host("host-b", False, exe("b"), "bridge0")}})()
        self.repo_root = "/repo"

    def rank_address(self, spec):
        return LIP if spec["host"] == "host-a" else MIP


def point(devices=("cuda", "metal"), hosts=("host-a", "host-b"), options=None, name="exo_latency", op="single_request"):
    ranks = [{"rank": i, "host": h, "memory_backend": d, "interface": "link"} for i, (h, d) in enumerate(zip(hosts, devices))]
    return Point(workload="w", benchmark=name, operation=op, algorithm=None, payload_bytes=1, datatype=None, reduce_op=None, rank_count=len(ranks), ranks=ranks,
                 options=dict(options or OPTIONS), warmup=0, iterations=1, repetitions=1, timeout_s=900)


class Registration(unittest.TestCase):
    def test_registered_with_their_own_categories_and_the_suite_does_not_import_exo_or_mlx(self):
        for name in NAMES:
            self.assertTrue(REGISTRY[name].caps.integrated)
            self.assertEqual(REGISTRY[name].caps.category, name)
        code = "import sys; import benchmarks.suite.adapters, benchmarks.suite.cli; print([m for m in ('exo', 'mlx', 'exo_tbccl', 'torch', 'psutil') if m in sys.modules])"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[3]))
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)

    def test_schema_knows_the_exo_categories_and_keeps_its_version(self):
        s = schema.load_schema()
        for category in NAMES + ("vllm_pipeline", "application"):
            self.assertIn(category, s["properties"]["benchmark"]["properties"]["category"]["enum"])
        self.assertEqual(schema.SCHEMA_VERSION, "tbccl-bench-result/1")


class Configuration(unittest.TestCase):
    def test_valid_options_pass_and_missing_ones_are_named(self):
        a = REGISTRY["exo_latency"]
        self.assertEqual(a.validate_workload({"options": OPTIONS}, []), [])
        errors = a.validate_workload({"options": {"phases": PHASES}}, [])
        for key in ("model_id", "backend", "prompts"):
            self.assertTrue(any(key in e for e in errors), key)

    def test_unknown_backend_and_phase_problems(self):
        a = REGISTRY["exo_latency"]
        self.assertTrue(any("backend" in e for e in a.validate_workload({"options": {**OPTIONS, "backend": "Nccl"}}, [])))
        self.assertTrue(any("unique" in e for e in a.validate_workload({"options": {**OPTIONS, "phases": PHASES + [dict(PHASES[0])]}}, [])))
        self.assertTrue(any("not defined" in e for e in a.validate_workload({"options": {**OPTIONS, "phases": [dict(PHASES[1], prompt="nope")]}}, [])))
        self.assertTrue(any("max_tokens" in e for e in a.validate_workload({"options": {**OPTIONS, "phases": [{"name": "x", "kind": "rounds", "prompt": "short"}]}}, [])))

    def test_gate_phases_need_their_inputs(self):
        a = REGISTRY["exo_correctness"]
        gate = {"name": "g", "kind": "check", "cases": [{"prompt": "short", "max_tokens": 4}]}
        self.assertEqual(a.validate_workload({"options": {**OPTIONS, "phases": [gate, {"name": "r", "kind": "recreate"}]}}, []), [])
        self.assertTrue(any("cases" in e for e in a.validate_workload({"options": {**OPTIONS, "phases": [{"name": "g", "kind": "check"}]}}, [])))
        self.assertTrue(any("followup" in e for e in a.validate_workload({"options": {**OPTIONS, "phases": [{"name": "c", "kind": "cancel", "prompt": "short", "max_tokens": 9}]}}, [])))
        self.assertTrue(any("not defined" in e for e in a.validate_workload({"options": {**OPTIONS, "phases": [{"name": "g", "kind": "check", "cases": [{"prompt": "nope", "max_tokens": 4}]}]}}, [])))
        self.assertTrue(any("kind" in e for e in a.validate_workload({"options": {**OPTIONS, "phases": [{"name": "g", "kind": "magic"}]}}, [])))

    def test_fast_paths_are_rejected_in_the_baseline_and_allowed_in_their_own_tier(self):
        a = REGISTRY["exo_latency"]
        for var in ("EXO_TBCCL_RECV", "EXO_TBCCL_ASYNC_SEND", "EXO_TBCCL_ACTIVITY_MODE"):
            self.assertTrue(any(var in e for e in a.validate_workload({"options": {**OPTIONS, "env": {var: "1"}}}, [])), var)
            self.assertEqual(a.validate_workload({"options": {**OPTIONS, "env": {var: "1"}, "tier": "fastpath"}}, []), [])

    def test_there_is_no_traffic_estimate_because_exo_traffic_is_not_predictable_from_the_workload(self):
        self.assertIsNone(REGISTRY["exo_correctness"].estimate_bytes(point()))

    def test_stage_assignment_checks(self):
        a, ctx = REGISTRY["exo_latency"], Ctx()
        self.assertIsNone(a.check_point(point(), ctx))
        self.assertIsNone(a.check_point(point(("metal", "cuda")), ctx))
        p = point(("cuda", "cuda"))
        self.assertIn("not a validated", a.check_point(p, ctx))
        self.assertEqual(p.skip_state, "unsupported")
        self.assertIn("two different hosts", a.check_point(point(hosts=("host-a", "host-a")), ctx))
        p = point(("cuda",), hosts=("host-a",))
        self.assertIn("two-node", a.check_point(p, ctx))
        self.assertIsNone(a.check_point(point(("cuda",), hosts=("host-a",), options={**OPTIONS, "backend": "MlxRing"}), ctx))


class Planning(unittest.TestCase):
    def plan(self, **kw):
        p = point(**kw)
        return p, REGISTRY["exo_latency"].plan(p, Ctx(), [36001, 36002], Path("/tmp/x"))

    def test_one_node_per_host_with_isolated_offline_environment_and_a_shared_namespace(self):
        p, g = self.plan()
        nodes = [i for i in g.invocations if i.role == "exo-node"]
        self.assertEqual([n.host_id for n in nodes], ["host-a", "host-b"])
        self.assertEqual(nodes[0].argv, ["/a/bin/exo"])
        self.assertTrue(all(n.persistent for n in nodes))
        ns = {n.env["EXO_ZENOH_NAMESPACE"] for n in nodes}
        self.assertEqual(len(ns), 1)
        self.assertTrue(ns.pop().startswith("b6-"))
        for n in nodes:
            self.assertEqual(n.env["EXO_OFFLINE"], "true")
            self.assertEqual(n.env["HF_HUB_OFFLINE"], "1")
            self.assertEqual(n.env["EXO_HOME"], ".exo_b6")
            self.assertFalse([k for k in n.env if k.startswith("EXO_TBCCL")], n.env)
        self.assertEqual(nodes[0].env["OVERRIDE_MEMORY_MB"], "5200")
        self.assertEqual(nodes[1].env["OVERRIDE_MEMORY_MB"], "10800")

    def test_namespaces_differ_between_groups(self):
        a = self.plan()[1].invocations[0].env["EXO_ZENOH_NAMESPACE"]
        b = self.plan()[1].invocations[0].env["EXO_ZENOH_NAMESPACE"]
        self.assertNotEqual(a, b)

    def test_samplers_select_by_namespace_and_the_client_runs_on_the_first_host(self):
        p, g = self.plan()
        samplers = [i for i in g.invocations if i.role.startswith("sampler@")]
        self.assertEqual({s.host_id for s in samplers}, {"host-a", "host-b"})
        ns = g.invocations[0].env["EXO_ZENOH_NAMESPACE"]
        for s in samplers:
            self.assertEqual(s.argv[s.argv.index("--namespace") + 1], ns)
            self.assertIn("--liveness", s.argv)
            self.assertTrue(s.stop_first)
        a = next(s for s in samplers if s.host_id == "host-a")
        self.assertIn("--gpu", a.argv)
        self.assertEqual(set(a.argv[a.argv.index("--expect-addresses") + 1].split(",")), {LIP, MIP})
        client = g.invocations[-1]
        self.assertEqual((client.role, client.host_id, client.persistent), ("client", "host-a", False))
        spec = json.loads(base64.b64decode(client.argv[client.argv.index("--spec-b64") + 1]))
        self.assertEqual((spec["backend"], spec["world_size"], spec["node_count"]), ("MlxTbccl", 2, 2))
        self.assertEqual({x["host"] for x in spec["peers"]}, {"127.0.0.1", "192.0.2.2"})
        self.assertEqual(spec["phases"], PHASES)

    def test_a_relative_reference_file_resolves_against_the_repository(self):
        p, g = self.plan(options={**OPTIONS, "reference_file": "refs/x.json"})
        spec = json.loads(base64.b64decode(g.invocations[-1].argv[g.invocations[-1].argv.index("--spec-b64") + 1]))
        self.assertEqual(spec["reference"], "/repo/refs/x.json")

    def test_a_remote_node_without_a_lan_address_is_an_error(self):
        ctx = Ctx()
        del ctx.config.hosts["host-b"].executables["exo_lan_address"]
        with self.assertRaises(ParseError):
            REGISTRY["exo_latency"].plan(point(), ctx, [1, 2], Path("/tmp/x"))

    def test_single_node_reference_has_no_peers_or_link_sampling(self):
        p, g = self.plan(devices=("cuda",), hosts=("host-a",), options={**OPTIONS, "backend": "MlxRing"})
        sampler = next(i for i in g.invocations if i.role.startswith("sampler@"))
        self.assertNotIn("--link-interface", sampler.argv)
        spec = json.loads(base64.b64decode(g.invocations[-1].argv[g.invocations[-1].argv.index("--spec-b64") + 1]))
        self.assertEqual((spec["peers"], spec["world_size"]), ([], 1))

    def test_trace_sets_the_adapter_trace_variable_and_the_hook_only_when_asked(self):
        self.assertFalse(any("EXO_TBCCL_TRACE" in i.env or "PYTHONPATH" in i.env for i in self.plan()[1].invocations))
        env = self.plan(options={**OPTIONS, "trace": True})[1].invocations[0].env
        self.assertEqual(env["EXO_TBCCL_TRACE"], "1")
        self.assertEqual(env["PYTHONPATH"], "/a/bench/trace_hook")
        self.assertTrue(env["TBCCL_B6_TRACE_DIR"].startswith("/tmp/trace-"))
        b = self.plan(options={**OPTIONS, "trace": True})[1].invocations[1].env
        self.assertEqual(b["PYTHONPATH"], "/b/bench/trace_hook")  # each host loads its own copy of the hook


# ---- parsing ----
LAYOUT = {"instance_id": "i1", "kind": "MlxTbcclInstance", "model_id": OPTIONS["model_id"], "world_size": 2,
          "stages": [{"runner": "r0", "node": "n0", "device_rank": 0, "start_layer": 0, "end_layer": 7, "n_layers": 7, "total_layers": 28, "world_size": 2, "model_id": OPTIONS["model_id"]},
                     {"runner": "r1", "node": "n1", "device_rank": 1, "start_layer": 7, "end_layer": 28, "n_layers": 21, "total_layers": 28, "world_size": 2, "model_id": OPTIONS["model_id"]}]}


def record(index, n_tokens=8, ttft_ms=100, itl_ms=10, phase="m"):
    tl = metrics.ExoTimeline({"phase": phase, "index": index, "max_tokens": n_tokens}, 0)
    t = ttft_ms
    for k in range(1, n_tokens + 1):
        ev = {"choices": [{"index": 0, "delta": {"role": "assistant", "content": "x"}}]}
        if k == n_tokens:
            ev["choices"][0]["finish_reason"] = "length"
            ev["usage"] = {"prompt_tokens": 20, "completion_tokens": n_tokens}
        tl.on_event("data", ev, int(t * 1e6))
        t += itl_ms
    tl.on_event("done", None, int(t * 1e6))
    tl.finish("ok", int(t * 1e6))
    return tl.to_record()


def client_output(layout=LAYOUT, error=None, done=True, agreement=(), n=6):
    out = ["TBCCL_EXO_BENCH " + json.dumps({"event": "placement", "layout": layout, "problems": []})]
    out.append("TBCCL_EXO_BENCH " + json.dumps({"event": "ready", "load_s": 12.5, "runner_timeline": [], "node_ids": {"host-a": "n0", "host-b": "n1"}, "agreement_problems": list(agreement)}))
    recs = [record(i, ttft_ms=100 + i) for i in range(n)]
    out += ["TBCCL_EXO_RAW " + json.dumps({"phase": "m", "request": r}) for r in recs]
    out.append("TBCCL_EXO_STATE " + json.dumps({"step": "ready", "state": {}}))
    out.append("TBCCL_EXO_BENCH " + json.dumps({"event": "phase", "phase": "m", "kind": "rounds", "distributions": metrics.request_distributions(recs), "throughput": metrics.phase_throughput(recs)}))
    if error:
        out.append("TBCCL_EXO_BENCH " + json.dumps({"event": "error", "error": error}))
    if done:
        out.append("TBCCL_EXO_BENCH " + json.dumps({"event": "done", "status": "error" if error else "ok", "steps": {}}))
    return "\n".join(out) + "\n"


def node_log(host_rank, tbccl=True, master=None):
    lines = [f"INFO Starting initialization for rank {host_rank}", f"INFO Group size: 2, group rank: {host_rank}"]
    if tbccl:
        lines += [f"INFO rank {host_rank}/2 TBCCL bootstrap on 192.0.2.{host_rank + 1}", f"INFO rank {host_rank}/2 TBCCL communicator ready (instance i1)"]
    else:
        lines.append(f"INFO Rank {host_rank} mlx distributed initialization complete")
    lines.append("INFO Node elected Master - promoting self" if (master if master is not None else host_rank == 0) else "INFO Node n0 elected master")
    return "\n".join(lines) + "\n"


def sampler_output(host, link_connections=2, samples=4, gpu=False):
    out = ["TBCCL_EXO_SAMPLE " + json.dumps({"event": "start", "role": host})]
    for i in range(samples):
        s = {"role": host, "wall_utc": "2026-10-17T00:00:0%dZ" % i, "cpu_percent": 30.0 + i, "mem_used_percent": 50.0,
             "processes": [{"pid": 1, "role": "exo-node", "rss_mib": 900 + i, "cpu_percent": 90.0, "threads": 30}, {"pid": 2, "role": "runner", "rss_mib": 1500, "cpu_percent": 100.0, "threads": 12}],
             "link": {"rx_bytes": 5 * i, "tx_bytes": 7 * i, "rx_errors": 0, "tx_errors": 0, "rx_dropped": 0, "tx_dropped": 0}, "connections": {"link": link_connections, "loopback": 1, "other": 0, "listen": 3}}
        if gpu:
            s["gpu"] = {"temperature_c": "55", "utilization_gpu_percent": "40", "memory_used_mib": "4000"}
        out.append("TBCCL_EXO_SAMPLE " + json.dumps(s))
    return "\n".join(out) + "\n"


def write_outcome(tmp, g, **o):
    texts = {0: o.get("log_a", node_log(0)), 1: o.get("log_b", node_log(1))}
    for s in (i for i in g.invocations if i.role.startswith("sampler@")):
        texts[s.rank] = (o.get("samplers") or {}).get(s.host_id) or sampler_output(s.host_id, gpu=s.host_id == "host-a")
    client = next(i for i in g.invocations if i.role == "client")
    texts[client.rank] = o.get("client", client_output())
    stdout, stderr = {}, {}
    for r, text in texts.items():
        (tmp / f"rank{r}.stdout").write_text(text)
        (tmp / f"rank{r}.stderr").write_text("")
        stdout[r], stderr[r] = str(tmp / f"rank{r}.stdout"), str(tmp / f"rank{r}.stderr")
    return Outcome({r: 0 for r in texts}, set(), False, None, stdout, stderr, 10.0)


class Parsing(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.p = point()
        self.g = REGISTRY["exo_latency"].plan(self.p, Ctx(), [36001, 36002], self.tmp)

    def parse(self, p=None, **o):
        return REGISTRY["exo_latency"].parse(p or self.p, self.g, write_outcome(self.tmp, self.g, **o))

    def test_a_good_run_yields_timing_throughput_placement_and_evidence(self):
        parsed = self.parse()
        tm = parsed.measurements["timing"]
        self.assertEqual(tm["metric"], "exo_tpot_client_observed")
        self.assertEqual(tm["sample_count"], 6)
        self.assertAlmostEqual(tm["statistics"]["median"], 10_000.0, delta=1)
        kinds = {t["kind"] for t in parsed.measurements["throughput"]}
        self.assertEqual(kinds, {"inference_output_tokens_per_second", "inference_requests_per_second"})
        inf = parsed.measurements["extra"]["inference"]
        self.assertEqual([s["n_layers"] for s in inf["placement"]["stages"]], [7, 21])
        self.assertEqual(inf["transport_evidence"]["masters"], {"n0": "n0", "n1": "n0"})
        self.assertEqual(inf["transport_evidence"]["samples"]["host-a"]["gpu"]["utilization_percent"]["max"], 40.0)
        self.assertEqual(parsed.verification["status"], "passed")
        self.assertEqual(parsed.configuration_extra["inference"]["stage_layers"], [7, 21])
        self.assertEqual((parsed.configuration_extra["inference"]["variant"], parsed.configuration_extra["inference"]["fast_path_env"]), ("default", {}))
        self.assertEqual(parsed.configuration_extra["inference"]["stage_hosts"], ["host-a", "host-b"])  # the stage order is Exo's choice and is recorded as a covariate
        self.assertTrue((self.tmp / "exo" / "m.requests.jsonl").exists())
        self.assertTrue((self.tmp / "exo" / "cluster-state.jsonl").exists())

    def test_wrong_instance_type_is_rejected(self):
        with self.assertRaisesRegex(ParseError, "unexpected backend"):
            self.parse(client=client_output(layout=dict(LAYOUT, kind="MlxRingInstance")))

    def test_tbccl_run_without_a_communicator_line_is_not_proven(self):
        with self.assertRaisesRegex(ParseError, "TBCCL communicator ready"):
            self.parse(log_a=node_log(0, tbccl=False))

    def test_tbccl_run_that_also_built_a_ring_group_is_a_fallback(self):
        with self.assertRaisesRegex(ParseError, "also initialised an MLX ring"):
            self.parse(log_b=node_log(1) + "INFO Rank 1 mlx distributed initialization complete\n")

    def test_ring_run_must_not_use_tbccl(self):
        p = point(options={**OPTIONS, "backend": "MlxRing"})
        g = REGISTRY["exo_latency"].plan(p, Ctx(), [1, 2], self.tmp)
        out = write_outcome(self.tmp, g, client=client_output(layout=dict(LAYOUT, kind="MlxRingInstance")), log_a=node_log(0, tbccl=True), log_b=node_log(1, tbccl=False))
        with self.assertRaisesRegex(ParseError, "unexpected backend"):
            REGISTRY["exo_latency"].parse(p, g, out)
        out = write_outcome(self.tmp, g, client=client_output(layout=dict(LAYOUT, kind="MlxRingInstance")), log_a=node_log(0, tbccl=False), log_b=node_log(1, tbccl=False))
        self.assertEqual(REGISTRY["exo_latency"].parse(p, g, out).verification["status"], "passed")

    def test_two_masters_are_rejected(self):
        with self.assertRaisesRegex(ParseError, "master"):
            self.parse(log_b=node_log(1, master=True))

    def test_no_link_traffic_means_the_data_path_is_not_proven(self):
        with self.assertRaisesRegex(ParseError, "no TCP connection between the two link addresses"):
            self.parse(samplers={"host-a": sampler_output("host-a", link_connections=0, gpu=True), "host-b": sampler_output("host-b", link_connections=0)})

    def test_missing_sampler_data(self):
        with self.assertRaisesRegex(ParseError, "no sampler data"):
            self.parse(samplers={"host-a": sampler_output("host-a", samples=0, gpu=True)})

    def test_nodes_that_disagree_fail_the_point(self):
        with self.assertRaisesRegex(ParseError, "disagree"):
            self.parse(client=client_output(agreement=["the nodes see different runner states"]))

    def test_client_error_or_missing_done_marker(self):
        with self.assertRaisesRegex(ParseError, "lifecycle failed"):
            self.parse(client=client_output(error="placement does not match the experiment"))
        with self.assertRaisesRegex(ParseError, "no done marker"):
            self.parse(client=client_output(done=False))

    def test_traced_points_collect_the_local_trace_directory_next_to_the_raw_records(self):
        work = self.tmp / "work"
        ctx = Ctx()
        ctx.config.hosts["host-a"].executables["exo_work_dir"] = str(work)
        ctx.config.hosts["host-b"].launcher = {"kind": "prefix", "argv": ["false"], "shell_join": True}  # the remote collection must not reach for a real network
        p = point(options={**OPTIONS, "trace": True})
        g = REGISTRY["exo_latency"].plan(p, ctx, [1, 2], self.tmp)
        trace_dir = Path(g.invocations[0].env["TBCCL_B6_TRACE_DIR"])
        trace_dir.mkdir(parents=True)
        (trace_dir / "copystats-1.json").write_text("{}")
        parsed = REGISTRY["exo_latency"].parse(p, g, write_outcome(self.tmp, g))
        self.assertEqual(parsed.measurements["extra"]["inference"]["trace"]["files"]["host-a"], ["trace/host-a/copystats-1.json"])
        self.assertTrue((self.tmp / "trace" / "host-a" / "copystats-1.json").exists())
        self.assertFalse(trace_dir.exists())  # the host's trace directory is removed after collection

    def test_a_fast_path_point_records_its_variant_and_the_exact_setting(self):
        p = point(options={**OPTIONS, "tier": "fastpath", "variant": "recv-reuse", "env": {"EXO_TBCCL_RECV": "reuse"}})
        g = REGISTRY["exo_latency"].plan(p, Ctx(), [1, 2], self.tmp)
        self.assertEqual(g.invocations[0].env["EXO_TBCCL_RECV"], "reuse")
        parsed = REGISTRY["exo_latency"].parse(p, g, write_outcome(self.tmp, g))
        self.assertEqual(parsed.configuration_extra["inference"]["variant"], "recv-reuse")
        self.assertEqual(parsed.configuration_extra["inference"]["fast_path_env"], {"EXO_TBCCL_RECV": "reuse"})
        self.assertEqual(parsed.configuration_extra["inference"]["tier"], "fastpath")

    def test_a_failed_correctness_gate_fails_the_point_even_with_good_timing(self):
        out = client_output().replace('"event": "phase", "phase": "m"', '"event": "phase", "check": {"passed": false, "outcome": {"mismatch": 1}}, "phase": "m"')
        with self.assertRaisesRegex(ParseError, "correctness gate failed"):
            self.parse(client=out)

    def test_a_reference_is_saved_next_to_the_raw_records(self):
        ref = "TBCCL_EXO_REFERENCE " + json.dumps({"name": "ref", "reference": {"cases": {"short:8": {"tokens": ["a"]}}}}) + "\n"
        self.parse(client=client_output() + ref)
        self.assertEqual(json.loads((self.tmp / "exo" / "ref.reference.json").read_text())["cases"]["short:8"]["tokens"], ["a"])

    def test_accounting_problems_fail_verification_and_are_listed(self):
        bad = client_output().replace('"completion_tokens": 8', '"completion_tokens": 9')
        parsed = self.parse(client=bad)
        self.assertEqual(parsed.verification["status"], "failed")
        self.assertTrue(parsed.measurements["extra"]["inference"]["request_accounting_problems"])


if __name__ == "__main__":
    unittest.main()
