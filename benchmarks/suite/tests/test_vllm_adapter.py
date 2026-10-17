"""The vLLM adapter: registration without importing vLLM, configuration and stage-assignment checks, command construction for every orientation, and parsing of the client / server /
sampler output contracts (synthetic fixtures in the real formats)."""
import base64
import json
import subprocess
import sys
import unittest
from pathlib import Path

from benchmarks.suite import config as cfg, schema
from benchmarks.suite.adapters import REGISTRY
from benchmarks.suite.adapters.base import Outcome, ParseError, Point
from benchmarks.suite.tests.common import make_config
from benchmarks.suite.stream import metrics

NAMES = ("vllm_startup", "vllm_correctness", "vllm_latency", "vllm_throughput", "vllm_concurrency", "vllm_pipeline")
LIP, MIP = "192.0.2.1", "192.0.2.2"
PHASES = [{"name": "gate", "kind": "check", "cases": ["p16"], "conc": 1, "tokens": 8}, {"name": "m", "kind": "rounds", "prompt": "p128", "tokens": 8, "conc": 2, "rounds": 2}]
OPTIONS = {"dtype": "bfloat16", "max_model_len": 4096, "gpu_memory_utilization": {"host-a": 0.5, "host-b": 0.3}, "phases": PHASES, "primary": {"phase": "m", "metric": "tpot"}}


class Host:
    def __init__(self, hid, local, executables, interface="link0"):
        self.id, self.is_local, self.executables, self.interface, self.interfaces, self.os = hid, local, executables, interface, {"link": "x"}, None
        self.env = {}
        self.launcher = {"kind": "local"} if local else {"kind": "prefix", "argv": ["ssh", "-o", "BatchMode=yes", "mac"], "shell_join": True}


class Ctx:
    def __init__(self):
        exe = lambda h: {"vllm": f"/{h}/bin/vllm", "vllm_python": f"/{h}/bin/python", "vllm_model_dir": f"/{h}/model", "vllm_workload": f"/{h}/work/workload.json",
                         "vllm_reference": f"/{h}/work/reference.json", "vllm_bench_dir": f"/{h}/bench", "vllm_work_dir": f"/{h}/work"}
        self.config = type("C", (), {"hosts": {"host-a": Host("host-a", True, exe("a")), "host-b": Host("host-b", False, exe("b"), "bridge0")}})()
        self.repo_root = "/repo"

    def rank_address(self, spec):
        return LIP if spec["host"] == "host-a" else MIP


def point(devices=("cuda", "metal"), hosts=("host-a", "host-b"), options=None, name="vllm_latency", op="single_request"):
    ranks = [{"rank": i, "host": h, "memory_backend": d, "interface": "link"} for i, (h, d) in enumerate(zip(hosts, devices))]
    return Point(workload="w", benchmark=name, operation=op, algorithm=None, payload_bytes=1, datatype=None, reduce_op=None, rank_count=len(ranks), ranks=ranks,
                 options=dict(options or OPTIONS), warmup=0, iterations=1, repetitions=1, timeout_s=900)


class Registration(unittest.TestCase):
    def test_adapters_are_registered_with_their_own_categories_and_the_suite_does_not_import_vllm(self):
        for name in NAMES:
            self.assertTrue(REGISTRY[name].caps.integrated)
            self.assertEqual(REGISTRY[name].caps.category, name)
        code = "import sys; import benchmarks.suite.adapters, benchmarks.suite.cli; print([m for m in ('vllm', 'torch', 'psutil', 'vllm_tbccl') if m in sys.modules])"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[3]))
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)

    def test_schema_accepts_the_inference_kinds_and_categories_without_removing_anything(self):
        s = schema.load_schema()
        t = s["properties"]["measurements"]["properties"]["throughput"]["items"]["properties"]
        for kind in ("inference_output_tokens_per_second", "inference_requests_per_second", "training_samples_per_second", "tcp_application_payload"):
            self.assertIn(kind, t["kind"]["enum"])
        for unit in ("tokens/s", "requests/s", "samples/s", "GB/s"):
            self.assertIn(unit, t["unit"]["enum"])
        for category in ("vllm_startup", "vllm_pipeline", "collective", "application"):
            self.assertIn(category, s["properties"]["benchmark"]["properties"]["category"]["enum"])
        self.assertEqual(schema.SCHEMA_VERSION, "tbccl-bench-result/1")


class Configuration(unittest.TestCase):
    def test_a_missing_dtype_or_context_length_is_an_error_not_a_default(self):
        a = REGISTRY["vllm_latency"]
        self.assertEqual(a.validate_workload({"options": OPTIONS}, ["single_request"]), [])
        errors = a.validate_workload({"options": {"phases": PHASES}}, ["single_request"])
        self.assertTrue(any("dtype" in e for e in errors) and any("max_model_len" in e for e in errors))
        self.assertTrue(any("phases is required" in e for e in a.validate_workload({"options": {"dtype": "bfloat16", "max_model_len": 1}}, [])))

    def test_duplicate_phase_names_and_unknown_primary_phase_are_rejected(self):
        a = REGISTRY["vllm_latency"]
        errors = a.validate_workload({"options": {**OPTIONS, "phases": PHASES + [dict(PHASES[0])]}}, [])
        self.assertTrue(any("unique" in e for e in errors))
        errors = a.validate_workload({"options": {**OPTIONS, "primary": {"phase": "nope", "metric": "ttft"}}}, [])
        self.assertTrue(any("primary" in e for e in errors))
        self.assertTrue(any("dtype must be" in e for e in a.validate_workload({"options": {**OPTIONS, "dtype": "int8"}}, [])))

    def test_only_validated_stage_assignments_run_and_nothing_is_substituted(self):
        a = REGISTRY["vllm_latency"]
        for devices in (("cuda", "metal"), ("metal", "cuda"), ("cuda", "host")):
            self.assertIsNone(a.check_point(point(devices), None), devices)
        for devices in (("host", "metal"), ("metal", "host"), ("host", "cuda"), ("cuda", "cuda"), ("metal", "metal")):
            p = point(devices)
            self.assertIn("not a validated", a.check_point(p, None) or "x", devices)
            self.assertIn(p.skip_state, ("unsupported", "skipped"))
        p = point(("cuda", "metal"), hosts=("host-a", "host-a"))
        self.assertIn("two different hosts", a.check_point(p, None))
        self.assertIsNone(a.check_point(point(("metal",), hosts=("host-b",)), None))  # a single-node reference
        self.assertIn("unknown single-node device", a.check_point(point(("tpu",), hosts=("host-b",)), None))

    def test_the_context_limit_is_not_raised_silently(self):
        a = REGISTRY["vllm_latency"]
        long_phase = [{"name": "m", "kind": "rounds", "prompt": "p2000", "tokens": 64, "conc": 1, "rounds": 1}]
        p = point(options={**OPTIONS, "max_model_len": 1024, "phases": long_phase, "primary": {"phase": "m", "metric": "ttft"}})
        self.assertIn("exceed options.max_model_len", a.check_point(p, None))
        p = point(options={**OPTIONS, "max_model_len": 4096, "phases": long_phase, "primary": {"phase": "m", "metric": "ttft"}})
        self.assertIsNone(a.check_point(p, None))

    def test_volume_estimate_is_for_pipelines_only(self):
        a = REGISTRY["vllm_latency"]
        self.assertGreater(a.estimate_bytes(point()), 0)
        self.assertIsNone(a.estimate_bytes(point(("cuda",), hosts=("host-a",))))
        self.assertEqual(a.data_direction(point()), {"mode": "directional", "source_rank": 0})


class Planning(unittest.TestCase):
    def plan(self, **kw):
        p = point(**kw)
        return p, REGISTRY["vllm_latency"].plan(p, Ctx(), [36001, 36002], Path("/tmp/x"))

    def test_cuda_first_metal_second_commands_environment_and_placement(self):
        p, g = self.plan()
        servers = {i.rank: i for i in g.invocations if i.role in ("api-node", "headless-node")}
        api, head = servers[0], servers[1]
        self.assertEqual((api.host_id, head.host_id), ("host-a", "host-b"))
        for argv in (api.argv, head.argv):
            for flag in ("--enforce-eager", "--no-async-scheduling", "--no-enable-prefix-caching", "--pipeline-parallel-size", "--distributed-executor-backend"):
                self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index("--dtype") + 1], "bfloat16")
            self.assertEqual(argv[argv.index("--max-model-len") + 1], "4096")
            self.assertEqual(argv[argv.index("--master-addr") + 1], LIP)  # the API node's link address, for both
            self.assertEqual(argv[argv.index("--master-port") + 1], "36002")
            self.assertEqual(argv[argv.index("--tensor-parallel-size") + 1], "1")
        self.assertEqual(api.argv[api.argv.index("--node-rank") + 1], "0")
        self.assertEqual(api.argv[api.argv.index("--port") + 1], "36001")
        self.assertEqual(api.argv[api.argv.index("--host") + 1], LIP)
        self.assertNotIn("--headless", api.argv)
        self.assertIn("--headless", head.argv)
        self.assertEqual(head.argv[head.argv.index("--node-rank") + 1], "1")
        self.assertNotIn("--port", head.argv)
        self.assertEqual(api.env["VLLM_USE_V2_MODEL_RUNNER"], "0")
        self.assertEqual(api.env["TBCCL_LOCAL_ENDPOINT"], f"{LIP}:0")
        self.assertEqual(head.env["TBCCL_LOCAL_ENDPOINT"], f"{MIP}:0")
        self.assertEqual(head.env["VLLM_TBCCL_BACKEND"], "metal")
        self.assertNotIn("VLLM_TBCCL_BACKEND", api.env)
        self.assertEqual(api.env["VLLM_TBCCL_ENABLE"], "1")
        self.assertEqual(api.env["HF_HUB_OFFLINE"], "1")  # no automatic download
        self.assertEqual(api.argv[api.argv.index("--gpu-memory-utilization") + 1], "0.5")
        self.assertEqual(head.argv[head.argv.index("--gpu-memory-utilization") + 1], "0.3")

    def test_metal_first_orientation_moves_the_api_node_and_the_client_to_the_mac(self):
        p, g = self.plan(devices=("metal", "cuda"), hosts=("host-b", "host-a"))
        api = next(i for i in g.invocations if i.role == "api-node")
        head = next(i for i in g.invocations if i.role == "headless-node")
        client = next(i for i in g.invocations if i.role == "client")
        self.assertEqual((api.host_id, head.host_id, client.host_id), ("host-b", "host-a", "host-b"))
        self.assertEqual(api.argv[api.argv.index("--host") + 1], MIP)
        self.assertEqual(api.argv[api.argv.index("--master-addr") + 1], MIP)
        self.assertEqual(api.env["VLLM_TBCCL_BACKEND"], "metal")
        spec = json.loads(base64.b64decode(client.argv[client.argv.index("--spec-b64") + 1]))
        self.assertEqual(spec["api"], {"host": MIP, "port": 36001})
        self.assertEqual(spec["workload"], "/b/work/workload.json")  # the client reads the API host's own copies

    def test_cuda_cpu_pairing_uses_the_torch_backend_on_the_cpu_stage(self):
        p, g = self.plan(devices=("cuda", "host"))
        head = next(i for i in g.invocations if i.role == "headless-node")
        self.assertEqual(head.env["VLLM_TBCCL_BACKEND"], "torch")
        self.assertNotIn("--gpu-memory-utilization", head.argv)
        self.assertIn("VLLM_CPU_KVCACHE_SPACE", head.env)

    def test_helper_invocations_follow_the_ranks_and_samplers_stop_first(self):
        p, g = self.plan()
        self.assertEqual([i.rank for i in g.invocations], list(range(len(g.invocations))))
        roles = [i.role for i in g.invocations]
        self.assertEqual(roles[:2], ["api-node", "headless-node"])
        self.assertEqual(roles[-1], "client")
        samplers = [i for i in g.invocations if i.role.startswith("sampler@")]
        self.assertEqual({s.host_id for s in samplers}, {"host-a", "host-b"})
        self.assertTrue(all(s.stop_first and s.persistent and "--liveness" in s.argv for s in samplers))
        self.assertTrue(all(i.persistent for i in g.invocations if i.role in ("api-node", "headless-node")))
        self.assertFalse(g.invocations[-1].persistent)
        a = next(s for s in samplers if s.host_id == "host-a")
        self.assertIn("--gpu", a.argv)
        self.assertEqual(a.argv[a.argv.index("--link-interface") + 1], "link0")
        self.assertEqual(set(a.argv[a.argv.index("--expect-addresses") + 1].split(",")), {LIP, MIP})

    def test_single_node_reference_has_no_pipeline_flags_and_stays_on_loopback(self):
        p = point(("metal",), hosts=("host-b",))
        g = REGISTRY["vllm_pipeline"].plan(p, Ctx(), [36001], Path("/tmp/x"))
        api = g.invocations[0]
        self.assertNotIn("--nnodes", api.argv)
        self.assertNotIn("--pipeline-parallel-size", api.argv)
        self.assertEqual(api.argv[api.argv.index("--host") + 1], "127.0.0.1")
        self.assertNotIn("VLLM_TBCCL_ENABLE", api.env)
        client = g.invocations[-1]
        spec = json.loads(base64.b64decode(client.argv[client.argv.index("--spec-b64") + 1]))
        self.assertEqual(spec["api"]["host"], "127.0.0.1")
        self.assertEqual(spec["phases"][0]["kind"], "ready")  # readiness is the client's first phase
        self.assertEqual(spec["phases"][1:], PHASES)

    def test_client_on_the_other_machine_uses_the_management_lan_and_the_api_binds_all_addresses(self):
        ctx = Ctx()
        for h in ctx.config.hosts.values():
            h.executables["vllm_lan_address"] = "192.0.2.1" if h.id == "host-a" else "192.0.2.2"
        p = point(options={**OPTIONS, "client_host": "other"})
        g = REGISTRY["vllm_latency"].plan(p, ctx, [36001, 36002], Path("/tmp/x"))
        api = next(i for i in g.invocations if i.role == "api-node")
        client = next(i for i in g.invocations if i.role == "client")
        self.assertEqual(api.argv[api.argv.index("--host") + 1], "0.0.0.0")
        self.assertEqual(client.host_id, "host-b")
        spec = json.loads(base64.b64decode(client.argv[client.argv.index("--spec-b64") + 1]))
        self.assertEqual(spec["api"]["host"], "192.0.2.1")
        self.assertEqual(spec["client_role"], "remote-lan")
        self.assertEqual(spec["workload"], "/b/work/workload.json")  # the client machine's own copy

    def test_trace_options_are_off_by_default_and_explicit_when_asked(self):
        p, g = self.plan()
        self.assertTrue(all("VLLM_TBCCL_TRACE" not in i.env for i in g.invocations))
        p, g = self.plan(options={**OPTIONS, "trace": "steps"})
        api = g.invocations[0]
        self.assertEqual(api.env["VLLM_TBCCL_TRACE"], "1")
        self.assertEqual(api.env["VLLM_TBCCL_TRACE_STEPS"], "1")
        self.assertTrue(api.env["PYTHONPATH"].endswith("/bench/trace_hook"))  # the diagnostic timing hook, only in traced runs
        self.assertEqual(api.env["TBCCL_B5_TRACE_DIR"], api.env["VLLM_TBCCL_TRACE_FILE"].rsplit("/", 1)[0])
        self.assertTrue(api.env["TBCCL_B5_TRACE_DIR"].startswith("/a/work/trace-"))
        head = g.invocations[1]
        self.assertNotEqual(head.env["TBCCL_B5_TRACE_DIR"], api.env["TBCCL_B5_TRACE_DIR"])  # a directory per host work area
        self.assertTrue(head.env["TBCCL_B5_TRACE_DIR"].startswith("/b/work/trace-"))


# ---- parsing ----
def sse_raw(phase, index, n_tokens, ttft_ms=100, itl_ms=10, prompt=128):
    tl = metrics.Timeline({"phase": phase, "round": 0, "slot": 0, "request_index": index, "prompt": "p128", "prompt_tokens_planned": prompt, "max_tokens": n_tokens}, 0)
    t = ttft_ms
    for k in range(1, n_tokens + 1):
        tl.on_event("data", {"choices": [{"token_ids": [k], "text": "x"}], "usage": {"prompt_tokens": prompt, "completion_tokens": k}}, int(t * 1e6))
        t += itl_ms
    tl.on_event("done", None, int(t * 1e6))
    tl.finish("ok", int(t * 1e6))
    return tl.to_record()


def client_output(phases_ok=True, gate_ok=True, n=6):
    out = ["TBCCL_VLLM_BENCH " + json.dumps({"phase": "ready", "kind": "ready", "ready": True, "waited_s": 70.0})]
    out.append("TBCCL_VLLM_BENCH " + json.dumps({"phase": "gate", "kind": "check", "check": {"passed": gate_ok, "outcome": {"exact": 1}}, "distributions": None}))
    recs = [sse_raw("m", i, 8, ttft_ms=100 + i, itl_ms=10 + i % 2) for i in range(n)]
    out += ["TBCCL_VLLM_RAW " + json.dumps({"phase": "m", "request": r}) for r in recs]
    out.append("TBCCL_VLLM_SCRAPE " + json.dumps({"phase": "m", "t_ns": 1, "values": {"vllm:num_requests_running{}": 2.0}}))
    out.append("TBCCL_VLLM_BENCH " + json.dumps({"phase": "m", "kind": "rounds", "loop": "closed-rounds", "concurrency": 2, "requests": n, "problems": [],
                                                 "throughput": metrics.phase_throughput(recs), "distributions": metrics.request_distributions(recs)}))
    out.append("TBCCL_VLLM_BENCH " + json.dumps({"phase": "end", "kind": "end", "gate_failed": not gate_ok}))
    return "\n".join(out) + "\n"


def server_log(rank, device, backend="tbccl", version="0.31.0", extra=""):
    lines = [f"(EngineCore pid=1) INFO 10-17 00:00:00 [core.py:129] Initializing a V1 LLM engine (v{version}) with config" if rank == 0 else f"(MainProcess pid=1) INFO [serve.py:221] Launching vLLM (v{version}) headless multiproc executor",
             f"(Worker pid=2) INFO 10-17 00:00:09 [parallel_state.py:1825] world_size=2 rank={rank} local_rank=0 distributed_init_method=tcp://{LIP}:29611 backend={backend}"]
    if device == "metal":
        lines.append(f"(Worker pid=2) INFO [pipeline.py:175] Pipeline transport vllm_tbccl.backends.metal.TBCCLMetalPipelineTransport: rank={rank}/2")
    return "\n".join(lines) + "\n" + extra


def sampler_output(host, link_connections=2, other=0, rx=0, tx=10_000_000, samples=4, gpu=False, lost=False):
    out = ["TBCCL_VLLM_SAMPLE " + json.dumps({"event": "start", "role": host})]
    for i in range(samples):
        s = {"role": host, "t_mono_s": float(i), "wall_utc": "2026-10-17T00:00:0%dZ" % i, "cpu_percent": 30.0 + i, "mem_used_percent": 50.0, "processes": [{"pid": 1, "role": "Worker", "rss_mib": 900 + i, "cpu_percent": 90.0}],
             "link": {"rx_bytes": rx * i, "tx_bytes": tx * i, "rx_errors": 0, "tx_errors": 0, "rx_dropped": 0, "tx_dropped": 0}, "connections": {"link": link_connections, "loopback": 1, "other": other, "listen": 3},
             "connection_pairs": ([[LIP, MIP, "Worker"]] if link_connections else []) + ([[LIP, "192.0.2.9", "vllm-serve:other"]] if other else [])}
        if gpu:
            s["gpu"] = {"temperature_c": "55", "utilization_gpu_percent": "40", "memory_used_mib": "4000", "clock_sm_mhz": "1500", "throttle_reasons": "0x0000000000000001"}
            s["thermal"] = {"zones_c": {"x86_pkg_temp": 80.0}, "package_throttle_count": 5 + i, "cpu_freq_khz_min": 800000}
        out.append("TBCCL_VLLM_SAMPLE " + json.dumps(s))
    if lost:
        out.append("TBCCL_VLLM_SAMPLE " + json.dumps({"event": "engine_lost", "role": host}))
    return "\n".join(out) + "\n"


def write_outcome(tmp, p, g, **o):
    files = {}
    texts = {0: o.get("api_log", server_log(0, p.ranks[0]["memory_backend"])), 1: o.get("head_log", server_log(1, p.ranks[1]["memory_backend"]))}
    samplers = [i for i in g.invocations if i.role.startswith("sampler@")]
    for i, s in enumerate(samplers):
        texts[s.rank] = o.get("samplers", {}).get(s.host_id) or sampler_output(s.host_id, gpu=s.host_id == "host-a", rx=0 if s.host_id == "host-a" else 10_000_000, tx=10_000_000 if s.host_id == "host-a" else 0)
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
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.p = point()
        self.g = REGISTRY["vllm_latency"].plan(self.p, Ctx(), [36001, 36002], self.tmp)

    def parse(self, **o):
        return REGISTRY["vllm_latency"].parse(self.p, self.g, write_outcome(self.tmp, self.p, self.g, **o))

    def test_a_good_run_yields_timing_throughput_and_the_inference_block(self):
        parsed = self.parse()
        m = parsed.measurements
        t = m["timing"]
        self.assertEqual(t["metric"], "vllm_tpot_client_observed")
        self.assertEqual(t["sample_count"], 6)
        self.assertEqual(t["primary_statistic"], "median")
        self.assertAlmostEqual(t["statistics"]["median"], 10_500, delta=600)  # microseconds
        kinds = {x["kind"]: x for x in m["throughput"]}
        self.assertEqual(kinds["inference_output_tokens_per_second"]["unit"], "tokens/s")
        self.assertEqual(kinds["inference_requests_per_second"]["unit"], "requests/s")
        for x in m["throughput"]:
            self.assertNotIn("tcp_application_payload", x["kind"])  # application metrics never masquerade as transfer throughput
        inf = m["extra"]["inference"]
        self.assertEqual(inf["placement"]["api_host"], "host-a")
        self.assertEqual([s["device"] for s in inf["placement"]["stages"]], ["cuda", "metal"])
        self.assertEqual(inf["raw_files"]["m"]["requests"], "vllm/m.requests.jsonl")  # relative to the repetition directory
        self.assertEqual(len((self.tmp / inf["raw_files"]["m"]["requests"]).read_text().splitlines()), 6)
        self.assertTrue((self.tmp / inf["raw_files"]["m"]["server_metrics"]).exists())
        self.assertEqual(parsed.verification["status"], "passed")
        record = schema.new_record(run_id="r", record_id="r/p/rep0", benchmark={"name": "vllm_latency", "adapter": "vllm_latency", "category": "vllm_latency", "source_repository": "x", "operation": "single_request", "provenance": "native"},
                                   environment=schema.empty_environment(), configuration={"payload_bytes": 1, "payload_semantics": "n/a", "rank_count": 2, "repetitions": 1, "repetition_index": 0,
                                                                                         "timing_method": "t", "completion_semantics": "c"},
                                   measurements=m, verification=parsed.verification)
        self.assertEqual(schema.validate_record(record), [])

    def test_the_transport_is_proven_from_logs_connections_and_counters(self):
        ev = self.parse().measurements["extra"]["inference"]["transport_evidence"]
        self.assertEqual(ev["hosts"]["host-a"]["distributed_init"], [{"world_size": 2, "rank": 0, "backend": "tbccl"}])
        self.assertTrue(ev["hosts"]["host-b"]["pipeline_transport_class"].endswith("TBCCLMetalPipelineTransport"))
        self.assertEqual(ev["samples"]["host-a"]["link_delta"]["tx_bytes"], 30_000_000)
        self.assertEqual(ev["samples"]["host-a"]["gpu"]["temperature_c_max"], 55.0)
        self.assertEqual(ev["samples"]["host-a"]["cpu_thermal"]["package_throttle_count_last"] - ev["samples"]["host-a"]["cpu_thermal"]["package_throttle_count_first"], 3)

    def test_a_backend_other_than_tbccl_invalidates_the_run(self):
        for backend in ("nccl", "gloo"):
            with self.assertRaises(ParseError) as cm:
                self.parse(api_log=server_log(0, "cuda", backend=backend))
            self.assertIn("fallback" if backend else "", str(cm.exception))

    def test_fallback_markers_in_a_log_invalidate_the_run(self):
        with self.assertRaises(ParseError) as cm:
            self.parse(head_log=server_log(1, "metal", extra="(Worker) INFO PyNcclCommunicator initialised\n"))
        self.assertIn("fallback", str(cm.exception))

    def test_a_missing_backend_line_or_seam_is_not_assumed(self):
        with self.assertRaises(ParseError) as cm:
            self.parse(api_log="(EngineCore pid=1) INFO Initializing a V1 LLM engine (v0.31.0)\n")
        self.assertIn("not proven", str(cm.exception))
        with self.assertRaises(ParseError) as cm:
            self.parse(head_log=server_log(1, "cuda"))  # a Metal stage that never reported the TBCCL seam
        self.assertIn("seam", str(cm.exception))

    def test_unsupported_vllm_versions_are_rejected(self):
        with self.assertRaises(ParseError) as cm:
            self.parse(api_log=server_log(0, "cuda", version="0.30.0"))
        self.assertIn("0.31.0", str(cm.exception))
        with self.assertRaises(ParseError):
            self.parse(api_log="(Worker) INFO world_size=2 rank=0 local_rank=0 distributed_init_method=tcp://x backend=tbccl\n")

    def test_no_link_connection_or_a_connection_outside_the_link_invalidates_the_run(self):
        with self.assertRaises(ParseError) as cm:
            self.parse(samplers={"host-a": sampler_output("host-a", link_connections=0, gpu=True)})
        self.assertIn("no TCP connection between the two link addresses", str(cm.exception))
        with self.assertRaises(ParseError) as cm:
            self.parse(samplers={"host-b": sampler_output("host-b", other=1)})
        self.assertIn("outside the link", str(cm.exception))

    def test_a_remote_clients_lan_connection_to_the_api_is_allowed_only_for_the_api_process(self):
        self.p.options["client_host"] = "other"
        rows = sampler_output("host-a", other=1, gpu=True)  # 'vllm-serve:other'
        self.assertEqual(self.parse(samplers={"host-a": rows}).measurements["extra"]["inference"]["transport_evidence"]["samples"]["host-a"]["other_pairs"], ["vllm-serve:other"])
        with self.assertRaises(ParseError):
            self.parse(samplers={"host-a": rows.replace("vllm-serve:other", "Worker:other")})

    def test_missing_sampler_data_means_the_data_path_is_unproven(self):
        with self.assertRaises(ParseError):
            self.parse(samplers={"host-a": "TBCCL_VLLM_SAMPLE " + json.dumps({"event": "start"}) + "\n"})

    def test_a_failed_gate_or_unfinished_client_never_yields_measurements(self):
        with self.assertRaises(ParseError) as cm:
            self.parse(client=client_output(gate_ok=False))
        self.assertIn("gate", str(cm.exception))
        with self.assertRaises(ParseError):
            self.parse(client=client_output().replace('"kind": "end"', '"kind": "xx"'))
        with self.assertRaises(ParseError):
            self.parse(client=client_output().replace('"ready": true', '"ready": false'))

    def test_an_undefined_primary_metric_is_an_error_not_zero(self):
        one_token = client_output().replace('"tokens": 8', '"tokens": 1')
        # all requests carry 8 tokens in the raw lines, so build one-token requests explicitly
        recs = [sse_raw("m", i, 1) for i in range(3)]
        out = ["TBCCL_VLLM_BENCH " + json.dumps({"phase": "ready", "kind": "ready", "ready": True})]
        out += ["TBCCL_VLLM_RAW " + json.dumps({"phase": "m", "request": r}) for r in recs]
        out.append("TBCCL_VLLM_BENCH " + json.dumps({"phase": "m", "kind": "rounds", "distributions": metrics.request_distributions(recs), "throughput": metrics.phase_throughput(recs), "problems": []}))
        out.append("TBCCL_VLLM_BENCH " + json.dumps({"phase": "end", "kind": "end"}))
        with self.assertRaises(ParseError) as cm:
            self.parse(client="\n".join(out) + "\n")
        self.assertIn("undefined", str(cm.exception))
        self.p.options["primary"] = {"phase": "m", "metric": "ttft"}  # TTFT is defined for one-token requests
        self.assertEqual(self.parse(client="\n".join(out) + "\n").measurements["timing"]["metric"], "vllm_ttft_client_observed")

    def test_the_sampler_engine_loss_is_reported(self):
        parsed = self.parse(samplers={"host-b": sampler_output("host-b", lost=True)})
        self.assertTrue(parsed.measurements["extra"]["inference"]["transport_evidence"]["samples"]["host-b"]["engine_lost"])


class TraceCollection(unittest.TestCase):
    def test_local_trace_files_are_copied_next_to_the_run_and_the_work_directory_is_removed(self):
        import shutil
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        src = tmp / "work" / "trace-x"
        src.mkdir(parents=True)
        (src / "hook.1.jsonl").write_text('{"kind": "step"}\n')
        (src / "tbccl.2.json").write_text("[]")
        a = REGISTRY["vllm_latency"]
        a._trace_sources = {"host-a": (None, True, str(src))}
        out = a._collect_traces(tmp / "rep" / "trace")
        self.assertEqual(out["files"]["host-a"], ["trace/host-a/hook.1.jsonl", "trace/host-a/tbccl.2.json"])
        self.assertTrue((tmp / "rep" / "trace" / "host-a" / "hook.1.jsonl").exists())
        self.assertFalse(src.exists())
        a._trace_sources = {"host-a": (None, True, str(tmp / "missing"))}
        self.assertEqual(a._collect_traces(tmp / "rep2" / "trace")["files"]["host-a"], [])  # nothing to collect is not an error


class SavedOutputs(unittest.TestCase):
    """Real-format outputs saved from a CUDA-first, Metal-second engine (addresses, paths and user names replaced): the contracts the adapter parses are the ones the tools print."""
    FIX = Path(__file__).resolve().parent / "fixtures" / "vllm"

    def test_a_saved_cuda_metal_run_parses_and_proves_the_transport(self):
        import shutil
        import tempfile
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        p = point(options={**OPTIONS, "primary": {"phase": "gate-seq", "metric": "e2e"}})
        g = REGISTRY["vllm_latency"].plan(p, Ctx(), [36001, 36002], tmp)
        roles = {i.role: i.rank for i in g.invocations}
        files = {roles["api-node"]: "pp_cuda_metal_api.log", roles["headless-node"]: "pp_cuda_metal_headless.log", roles["sampler@host-a"]: "pp_sampler_host_a.out",
                 roles["sampler@host-b"]: "pp_sampler_host_b.out", roles["client"]: "pp_client.out"}
        stdout, stderr = {}, {}
        for rank, name in files.items():
            (tmp / f"rank{rank}.stdout").write_text((self.FIX / name).read_text())
            (tmp / f"rank{rank}.stderr").write_text("")
            stdout[rank], stderr[rank] = str(tmp / f"rank{rank}.stdout"), str(tmp / f"rank{rank}.stderr")
        parsed = REGISTRY["vllm_latency"].parse(p, g, Outcome({r: 0 for r in files}, set(), False, None, stdout, stderr, 1.0))
        ev = parsed.measurements["extra"]["inference"]["transport_evidence"]
        self.assertEqual(ev["hosts"]["host-a"]["vllm_versions"], ["0.31.0"])
        self.assertEqual(ev["hosts"]["host-b"]["vllm_versions"], ["0.31.0"])
        self.assertTrue(all(h["distributed_init"][0]["backend"] == "tbccl" for h in ev["hosts"].values()))
        self.assertGreaterEqual(ev["samples"]["host-a"]["max_link_connections"], 1)
        self.assertEqual(ev["samples"]["host-a"]["other_connections_max"], 0)
        t = parsed.measurements["timing"]
        self.assertEqual(t["sample_count"], 5)
        self.assertTrue(all(x > 0 for x in t["samples"]))
        raw = (tmp / "vllm" / "gate-seq.requests.jsonl").read_text().splitlines()
        self.assertEqual(len(raw), 5)
        first = json.loads(raw[0])
        self.assertEqual(first["schema"], "tbccl-vllm-request/1")
        d = metrics.derive(first)
        self.assertEqual(d["token_source"], "usage")
        self.assertEqual(d["output_tokens"], 32)
        self.assertEqual(len(d["itl_s"]), 31)  # one event per token in vLLM 0.31.0's stream: 32 tokens, 31 intervals
        self.assertEqual(d["problems"], [])


def pg_line(rank, device, tokens=(1, 12), verified=True, backend="tbccl", communicator="TBCCLDeviceCommunicator", peers=(f"{LIP}>{MIP}",), ops=None):
    sizes = []
    for t in tokens:
        if rank == 0:
            samples = [300.0 + i for i in range(40)]
            sizes.append({"tokens": t, "bytes_per_direction": 2 * t * 1024 * 2, "rtt_us": {"p50": 320.0}, "samples_us": samples, "echo_verified": verified})
        else:
            sizes.append({"tokens": t, "peer_received_verified": verified})
    return "TBCCL_VLLM_PG " + json.dumps({"rank": rank, "platform": "TbcclCudaPlatform" if device == "cuda" else "TbcclCpuPlatform", "device": "cuda:0" if device == "cuda" else "cpu", "communicator": communicator,
                                          "torch": "2.13.0", "dist_backend": backend, "hidden": 1024, "sizes": sizes, "tbccl_ops": ops if ops is not None else {"send": 40, "recv": 40}, "tbccl_bytes": 1, "connections": list(peers)}) + "\n"


class BackendPingPong(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        self.a = REGISTRY["vllm_backend"]
        self.p = point(("cuda", "host"), name="vllm_backend", op="pingpong", options={"dtype": "bfloat16", "max_model_len": 4096, "tokens": [1, 12], "primary_tokens": 1, "phases": [{"name": "u", "kind": "warmup"}]})
        self.g = self.a.plan(self.p, Ctx(), [36001], self.tmp)

    def parse(self, lines):
        stdout = {}
        for rank, text in lines.items():
            (self.tmp / f"r{rank}.out").write_text(text)
            stdout[rank] = str(self.tmp / f"r{rank}.out")
        return self.a.parse(self.p, self.g, Outcome({0: 0, 1: 0}, set(), False, None, stdout, {}, 1.0))

    def test_commands_use_each_hosts_interpreter_and_the_vllm_tbccl_platform_variables(self):
        r0, r1 = self.g.invocations
        self.assertEqual(r0.argv[r0.argv.index("--init") + 1], f"tcp://{LIP}:36001")
        self.assertEqual((r0.env["VLLM_TBCCL_PLATFORM"], r1.env["VLLM_TBCCL_PLATFORM"]), ("cuda", "cpu"))
        self.assertEqual(r1.env["VLLM_TBCCL_BACKEND"], "torch")
        self.assertEqual((r0.listens_on, r1.wait_for_port), (36001, 36001))
        self.assertEqual(r1.env["TBCCL_LOCAL_ENDPOINT"], f"{MIP}:0")

    def test_a_good_run_reports_round_trips_per_size_and_verifies_the_echo(self):
        parsed = self.parse({0: pg_line(0, "cuda"), 1: pg_line(1, "host")})
        self.assertEqual(parsed.measurements["timing"]["metric"], "vllm_boundary_round_trip_1tok")
        self.assertEqual(parsed.measurements["timing"]["sample_count"], 40)
        self.assertEqual(set(parsed.measurements["extra"]["inference"]["boundary"]), {"1", "12"})
        self.assertEqual(parsed.verification["status"], "passed")

    def test_wrong_backend_unverified_payload_missing_traffic_or_foreign_peers_invalidate(self):
        for lines, text in ((({0: pg_line(0, "cuda", backend="nccl"), 1: pg_line(1, "host")}), "tbccl backend"), ({0: pg_line(0, "cuda", verified=False), 1: pg_line(1, "host")}, "unchanged"),
                            ({0: pg_line(0, "cuda", ops={}), 1: pg_line(1, "host")}, "ProcessGroupTBCCL"), ({0: pg_line(0, "cuda", peers=(f"{LIP}>192.0.2.7",)), 1: pg_line(1, "host")}, "outside the data link"),
                            ({0: pg_line(0, "cuda", peers=()), 1: pg_line(1, "host")}, "cannot be verified"), ({0: pg_line(0, "host"), 1: pg_line(1, "host")}, "ran on")):
            with self.assertRaises(ParseError) as cm:
                self.parse(lines)
            self.assertIn(text, str(cm.exception))

    def test_dual_stack_notation_of_the_link_addresses_is_not_a_foreign_peer(self):
        peers = (f"::ffff:{LIP}>::ffff:{MIP}", f"{LIP}>::{MIP}")
        self.assertIn("timing", self.parse({0: pg_line(0, "cuda", peers=peers), 1: pg_line(1, "host", peers=peers)}).measurements)

    def test_only_cuda_with_cpu_is_covered_and_metal_is_unsupported(self):
        q = point(("cuda", "metal"), name="vllm_backend", op="pingpong", options=self.p.options)
        self.assertIn("CPU stage only", self.a.check_point(q, None))
        self.assertEqual(q.skip_state, "unsupported")
        self.assertIsNone(self.a.check_point(self.p, None))


class Workloads(unittest.TestCase):
    def test_presets_expand_and_unsupported_assignments_are_visible(self):
        base = Path(__file__).resolve().parents[1] / "configs"
        for preset in sorted(base.glob("b5-*.yaml")):
            raw = cfg.load_config(str(preset), overlays=[], auto_overlay=False) if False else None
            self.assertIsNone(raw)
            self.assertIn("vllm_", preset.read_text())


if __name__ == "__main__":
    unittest.main()
