"""Integration of Exo pipeline-parallel inference (MlxTbccl through exo-tbccl, MlxRing as the application-level baseline) into the benchmark suite.

The suite never imports Exo, MLX or exo-tbccl. One workload point is one cluster lifecycle: the adapter starts one Exo node per host (rank r = the r-th configured host; Exo elects the
master and chooses the pipeline stage order itself, so the stage assignment is read back, never assumed), one passive sampler per host and a client that waits for the nodes, places one
instance with Exo's normal placement, checks what was placed, waits for the runners, runs the phases and deletes the instance (``exo_bench/client.py``). The runner owns every process:
argument arrays only, a lease/heartbeat on remote hosts, bounded termination of everything started, no name matching. Every node gets a unique ``EXO_ZENOH_NAMESPACE`` (the sampler
selects Exo's processes by it), an isolated ``EXO_HOME``, ``EXO_OFFLINE=true`` and no optional exo-tbccl fast path.

The client prints ``TBCCL_EXO_BENCH`` summaries, the raw evidence (``TBCCL_EXO_RAW`` per request timeline, ``TBCCL_EXO_STATE`` per cluster-state capture) and the adapter parses only those
contracts. Application-level results use their own throughput kinds (``inference_*``) and an ``extra.inference`` block; they are never filed as transfer throughput and never compared with
the direct two-host driver or with vLLM.
"""
from __future__ import annotations

import base64
import json
import re
import uuid
from pathlib import Path

from .base import BenchmarkAdapter, Capabilities, Group, Invocation, ParseError, Parsed, timing
from .core import PERCENTILE_LINEAR
from .vllm import VllmPipeline, _lines, _read, _summary

PREFIXES = {"bench": "TBCCL_EXO_BENCH", "raw": "TBCCL_EXO_RAW", "state": "TBCCL_EXO_STATE", "sample": "TBCCL_EXO_SAMPLE", "reference": "TBCCL_EXO_REFERENCE"}
BACKENDS = ("MlxRing", "MlxTbccl")
STAGE_DEVICES = ("cuda", "metal", "host")
PAIRINGS = {("cuda", "metal"), ("metal", "cuda")}
SUPPORTED_EXO_TBCCL = "0.3.0"
PHASE_KINDS = {"warmup", "rounds", "continuous", "reference", "check", "cancel", "recreate"}
FORBIDDEN_FASTPATH_ENV = ("EXO_TBCCL_RECV", "EXO_TBCCL_ASYNC_SEND", "EXO_TBCCL_ACTIVITY_MODE", "EXO_TBCCL_MANAGED", "EXO_TBCCL_ACTIVITY_THREAD")
EXO_PORTS = (52413, 52414, 52415)

SCENARIOS = {
    "exo_correctness": ("exo_correctness", ("gates",), "greedy token streams against a single-device reference, repeated requests, instance recreation, cancellation"),
    "exo_latency": ("exo_latency", ("single_request",), "one request at a time: TTFT, TPOT, ITL, E2E"),
    "exo_throughput": ("exo_throughput", ("closed_loop",), "output-token and request throughput at a fixed closed-loop concurrency"),
    "exo_transport_ab": ("exo_transport_ab", ("ab",), "MlxRing against MlxTbccl with the same model, placement and workload, interleaved"),
    "exo_stage_profile": ("exo_stage_profile", ("trace",), "per-stage diagnostic timing and resource cost (traced runs are never mixed with the throughput campaign)"),
    "exo_large_model": ("exo_large_model", ("large",), "a model that needs the combined device memory"),
    "exo_lifecycle": ("exo_lifecycle", ("startup",), "cluster start, placement, model load, first request, instance deletion and shutdown"),
}


def masters_from_logs(logs):
    from ..exo_bench import validation
    return validation.masters_by_node(logs)


class ExoPipeline(BenchmarkAdapter):
    """One adapter implementation behind several benchmark names (see SCENARIOS); the names only select the category and operations reported."""

    def __init__(self, name):
        category, operations, description = SCENARIOS[name]
        self.caps = Capabilities(
            name=name, category=category, description=f"Exo {description} (exo-tbccl {SUPPORTED_EXO_TBCCL}, exo-fork tbccl-integration-v2)", executable="exo", operations=operations,
            memory_backends=STAGE_DEVICES, min_ranks=1, max_ranks=2, executable_in_build=False, supports_warmup=False, supports_iterations=False, verification={op: "native" for op in operations},
            source_files=("benchmarks/suite/exo_bench/client.py", "benchmarks/suite/exo_bench/sampler.py", "benchmarks/suite/exo_bench/metrics.py", "benchmarks/suite/exo_bench/validation.py"),
            notes="Framework benchmark, not a native TBCCL benchmark. payload_bytes is a placeholder (1): the workload is defined by options.prompts and options.phases. Ranks are hosts (one Exo "
                  "node each); Exo chooses the stage order, and the placement that happened is recorded. Needs per-host executables: exo, exo_python, exo_bench_dir (and exo_lan_address on a "
                  "remote host so the client can read that node's own view).")

    # ---- configuration ----
    def validate_workload(self, workload, operations):
        errors = []
        o = workload.get("options") or {}
        for key in ("model_id", "backend", "prompts", "phases"):
            if not o.get(key):
                errors.append(f"options.{key} is required")
        if o.get("backend") not in (None, *BACKENDS):
            errors.append(f"options.backend must be one of {BACKENDS}")
        names = [ph.get("name") for ph in o.get("phases") or []]
        for i, ph in enumerate(o.get("phases") or []):
            if "kind" not in ph or "name" not in ph:
                errors.append(f"options.phases[{i}] needs 'name' and 'kind'")
            elif ph["kind"] not in PHASE_KINDS:
                errors.append(f"options.phases[{i}].kind '{ph['kind']}' is not one of {sorted(PHASE_KINDS)}")
            elif ph["kind"] in ("warmup", "rounds", "continuous", "cancel") and ("max_tokens" not in ph or "prompt" not in ph):
                errors.append(f"options.phases[{i}] needs 'prompt' and 'max_tokens' (the benchmark never picks a workload silently)")
            elif ph["kind"] in ("reference", "check") and not ph.get("cases"):
                errors.append(f"options.phases[{i}] needs 'cases' (a list of {{prompt, max_tokens}})")
            elif ph["kind"] == "cancel" and not (ph.get("followup") or {}).get("cases"):
                errors.append(f"options.phases[{i}] needs a 'followup' check with 'cases'")
            else:
                used = [ph["prompt"]] if isinstance(ph.get("prompt"), str) else list(ph.get("prompt") or [])
                used += [c["prompt"] for c in ph.get("cases") or []] + [c["prompt"] for c in (ph.get("followup") or {}).get("cases") or []]
                for p in used:
                    if p not in (o.get("prompts") or {}):
                        errors.append(f"options.phases[{i}] uses prompt '{p}' that is not defined in options.prompts")
        if len(set(names)) != len(names):
            errors.append("options.phases names must be unique")
        env = {**(o.get("env") or {}), **(o.get("server_env") or {})}
        enabled = sorted(k for k in env if k in FORBIDDEN_FASTPATH_ENV and o.get("tier") != "fastpath")
        if enabled:
            errors.append(f"{enabled} are optional exo-tbccl fast paths and stay off in the baseline; set options.tier to 'fastpath' for a separate experimental tier")
        return errors

    def check_point(self, point, ctx):
        devices = tuple(r["memory_backend"] for r in point.ranks)
        o = point.options
        if point.rank_count == 2 and devices not in PAIRINGS:
            point.skip_state = "unsupported"
            return f"stage assignment {devices} is not a validated Exo pairing (validated: CUDA with Metal in either order)"
        if point.rank_count == 2 and point.ranks[0]["host"] == point.ranks[1]["host"]:
            return "the two Exo nodes must run on two different hosts"
        if point.rank_count == 1 and o.get("backend") == "MlxTbccl":
            point.skip_state = "unsupported"
            return "MlxTbccl needs a two-node pipeline"
        if point.rank_count == 1 and o.get("backend") != "MlxRing":
            return "a single-node reference uses the MlxRing instance type (min_nodes 1)"
        return None

    def estimate_bytes(self, point):
        """None on purpose. The runner would use an estimate as a floor for its data-interface check, but the traffic of a pipeline is not predictable from the workload: Exo stops at the end-of-sequence
        token before the configured limit, reuses a cached prompt prefix, and its own discovery and control traffic shares the link. The data path is proven by the adapter instead: the unique
        namespace's processes must hold TCP connections between the two link addresses, a TBCCL run must report its communicators on both nodes, and the placement must use the link addresses."""
        return None

    def data_direction(self, point):
        return {"mode": "directional", "source_rank": 0} if point.rank_count == 2 else {"mode": "bidirectional", "source_rank": None}

    def describe(self, point):
        return {"payload_semantics": "not applicable: the payload size is a placeholder; the workload is defined by prompt and output token counts in options.phases",
                "backend": f"exo-{point.options.get('backend')}" if point.rank_count == 2 else "exo-single-node",
                "timing_method": "client-side time.monotonic_ns() on the API host: submission time and the arrival time of every token event of the server-sent stream (Exo streams one event per token); "
                                 "TTFT, TPOT, ITL and E2E are derived from the raw timelines; timestamps of different hosts are never subtracted",
                "completion_semantics": "TTFT = first token event; E2E = last token event; the request's token count is the server-reported usage, never the configured maximum"}

    # ---- planning ----
    def _tool(self, ctx, name, host_id):
        host = ctx.config.hosts[host_id]
        if name in host.executables:
            return host.executables[name]
        raise ParseError(f"host '{host_id}' needs executables.{name} (the Exo environment is configured per host)")

    def _opt(self, ctx, name, host_id, default=None):
        return ctx.config.hosts[host_id].executables.get(name, default)

    def _node_env(self, point, ctx, spec, namespace):
        o = point.options
        env = {"EXO_HOME": self._opt(ctx, "exo_home", spec["host"], ".exo_b6"), "EXO_OFFLINE": "true", "HF_HUB_OFFLINE": "1", "EXO_ZENOH_NAMESPACE": namespace}
        dash = self._opt(ctx, "exo_dashboard_dir", spec["host"])
        if dash:
            env["EXO_DASHBOARD_DIR"] = dash
        override = (o.get("memory_override_mb") or {}).get(spec["host"])
        if override is not None:
            env["OVERRIDE_MEMORY_MB"] = str(override)
        if o.get("trace"):
            work = self._opt(ctx, "exo_work_dir", spec["host"], "/tmp")
            env.update({"EXO_TBCCL_TRACE": "1", "TBCCL_B6_TRACE_DIR": f"{work}/trace-{point.point_id}", "PYTHONPATH": str(Path(self._tool(ctx, "exo_bench_dir", spec["host"])) / "trace_hook")})
        env.update({str(k): str(v) for k, v in {**(o.get("env") or {}), **(o.get("server_env") or {})}.items()})
        return env

    def _spec(self, point, ctx, leader, namespace):
        o = point.options
        peers = []
        for rank, spec in enumerate(point.ranks):
            if spec["host"] == leader["host"]:
                peers.append({"name": spec["host"], "host": "127.0.0.1", "port": EXO_PORTS[2]})
            else:
                lan = self._opt(ctx, "exo_lan_address", spec["host"])
                if not lan:
                    raise ParseError(f"remote host '{spec['host']}' needs executables.exo_lan_address (the client reads that node's own cluster view)")
                peers.append({"name": spec["host"], "host": lan, "port": EXO_PORTS[2]})
        return {"api": {"host": "127.0.0.1", "port": EXO_PORTS[2]}, "model": o["model_id"], "backend": o["backend"], "world_size": point.rank_count, "node_count": point.rank_count,
                "expected_layers": o.get("expected_layers"), "prompts": o["prompts"], "phases": [dict(ph) for ph in o["phases"]], "peers": peers if point.rank_count > 1 else [],
                "request_timeout_s": o.get("request_timeout_s", 300), "load_timeout_s": o.get("load_timeout_s", 900), "discovery_timeout_s": o.get("discovery_timeout_s", 120),
                "tag": o.get("tag"), "namespace": namespace,
                "link_addresses": [ctx.rank_address(s) for s in point.ranks] if point.rank_count == 2 else None,
                "reference": (str(Path(ctx.repo_root) / o["reference_file"]) if o.get("reference_file") else self._opt(ctx, "exo_reference", leader["host"])) if o.get("use_reference", True) else None, "tie_margin": o.get("tie_margin", 0.251)}

    def plan(self, point, ctx, ports, rep_dir):
        o = point.options
        specs = point.ranks
        leader = specs[0]
        namespace = f"b6-{uuid.uuid4().hex[:12]}"
        self._trace_sources = {}
        if o.get("trace"):
            for spec in specs:
                host = ctx.config.hosts[spec["host"]]
                work = self._opt(ctx, "exo_work_dir", spec["host"], "/tmp")
                self._trace_sources[spec["host"]] = (None if host.is_local else list(host.launcher.get("argv", [])), bool(host.launcher.get("shell_join", True)), f"{work}/trace-{point.point_id}")
        invs = []
        for rank, spec in enumerate(specs):  # ranks first: Invocation.rank indexes point.ranks for the real ranks
            invs.append(Invocation(rank, "exo-node", spec["host"], [self._tool(ctx, "exo", spec["host"])], env=self._node_env(point, ctx, spec, namespace), persistent=True))
        next_rank = len(invs)
        expect = ",".join(ctx.rank_address(s) for s in specs) if point.rank_count == 2 else ""
        for spec in specs:
            host = ctx.config.hosts[spec["host"]]
            iface = spec.get("interface")
            iface = host.interface if (iface in host.interfaces or not iface) else iface
            argv = [self._tool(ctx, "exo_python", spec["host"]), str(Path(self._tool(ctx, "exo_bench_dir", spec["host"])) / "sampler.py"), "--namespace", namespace, "--role", spec["host"],
                    "--interval", str(o.get("sample_interval_s", 1.0)), "--expect-addresses", expect, "--liveness"]
            if iface and point.rank_count == 2:
                argv += ["--link-interface", iface]
            if spec["memory_backend"] == "cuda":
                argv += ["--gpu"]
            if host.os is None or host.os.lower().startswith("linux"):
                argv += ["--thermal"]
            invs.append(Invocation(next_rank, f"sampler@{spec['host']}", spec["host"], argv, persistent=True, stop_first=True))
            next_rank += 1
        encoded = base64.b64encode(json.dumps(self._spec(point, ctx, leader, namespace)).encode()).decode()
        invs.append(Invocation(next_rank, "client", leader["host"], [self._tool(ctx, "exo_python", leader["host"]), str(Path(self._tool(ctx, "exo_bench_dir", leader["host"])) / "client.py"),
                                                                    "--spec-b64", encoded], persistent=False))
        return Group(invs, ports=list(ports), notes=[f"client and API on {leader['host']}", f"namespace {namespace}", f"stages (hosts, before Exo's placement): {[s['memory_backend'] for s in specs]}"])

    # ---- parsing ----
    def parse(self, point, group, outcome):
        client = next(inv for inv in group.invocations if inv.role == "client")
        bench = _lines(outcome.stdout[client.rank], PREFIXES["bench"])
        if not bench or bench[-1].get("event") != "done":
            raise ParseError("the client did not finish: no done marker")
        errors = [b["error"] for b in bench if b.get("event") == "error"]
        if errors or bench[-1].get("status") != "ok":
            raise ParseError(f"the Exo lifecycle failed: {errors or bench[-1]}")
        placement = next((b for b in bench if b.get("event") == "placement"), None)
        ready = next((b for b in bench if b.get("event") == "ready"), None)
        if not placement or not ready:
            raise ParseError("the client reported no placement or no readiness")
        phases = {b["phase"]: b for b in bench if b.get("event") == "phase"}
        failed = [n for n, b in phases.items() if (b.get("check") or {}).get("passed") is False or (b.get("cancel") or {}).get("passed") is False]
        if failed:
            raise ParseError(f"correctness gate failed in phase(s) {failed}")
        logs = {point.ranks[r]["host"]: _read(outcome.stdout[r]) + "\n" + _read(outcome.stderr[r]) for r in range(point.rank_count) if r in outcome.stdout}
        node_host = {v: k for k, v in (ready.get("node_ids") or {}).items()}
        for stage in placement["layout"]["stages"]:  # Exo chooses the stage order per start: record which host ran which stage
            stage["host"] = node_host.get(stage["node"])
        evidence = self._evidence(point, group, placement["layout"], ready, logs, outcome)
        raw = self._extract_raw(outcome.stdout[client.rank], Path(outcome.stdout[client.rank]).parent / "exo")
        raw["files"]["samples"] = self._write_samples(group, outcome, Path(outcome.stdout[client.rank]).parent / "exo")
        primary = point.options.get("primary") or self._default_primary(phases)
        parsed = self._measure(point, phases, evidence, raw, primary, placement["layout"], ready)
        if point.options.get("trace"):
            parsed.measurements["extra"]["inference"]["trace"] = VllmPipeline._collect_traces(self, Path(outcome.stdout[client.rank]).parent / "trace")
        return parsed

    def _default_primary(self, phases):
        for name, b in phases.items():
            if b.get("kind") in ("rounds", "continuous"):
                return {"phase": name, "metric": "e2e"}
        for name, b in phases.items():
            if b.get("kind") in ("check", "reference"):  # a correctness-only point: the first gate's requests give the latency record
                return {"phase": name, "metric": "e2e"}
        raise ParseError("no measured phase in the client output")

    def _extract_raw(self, stdout_path, out_dir):
        out_dir.mkdir(parents=True, exist_ok=True)
        files, per_phase = {}, {}
        for item in _lines(stdout_path, PREFIXES["raw"]):
            per_phase.setdefault(item["phase"], []).append(item["request"])
        for phase, recs in per_phase.items():
            p = out_dir / f"{phase}.requests.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in recs))
            files.setdefault(phase, {})["requests"] = f"{out_dir.name}/{p.name}"
        for item in _lines(stdout_path, PREFIXES["reference"]):
            p = out_dir / f"{item['name']}.reference.json"
            p.write_text(json.dumps(item["reference"]))
            files.setdefault(item["name"], {})["reference"] = f"{out_dir.name}/{p.name}"
        states = _lines(stdout_path, PREFIXES["state"])
        if states:
            p = out_dir / "cluster-state.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in states))
            files["cluster_state"] = f"{out_dir.name}/{p.name}"
        return {"files": files, "records": per_phase}

    def _write_samples(self, group, outcome, out_dir):
        """The per-host sampler series next to the raw records (reports compute CPU seconds and link bytes over the measured window from them)."""
        files = {}
        for inv in group.invocations:
            if not inv.role.startswith("sampler@"):
                continue
            host = inv.role.split("@", 1)[1]
            rows = _lines(outcome.stdout[inv.rank], PREFIXES["sample"])
            p = out_dir / f"samples-{host}.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in rows))
            files[host] = f"{out_dir.name}/{p.name}"
        return files

    def _evidence(self, point, group, layout, ready, logs, outcome):
        """Prove which backend carried the pipeline and that the cluster was what the experiment says."""
        from ..exo_bench import validation
        backend = point.options["backend"]
        want = validation.KINDS[backend]
        if layout["kind"] != want:
            raise ParseError(f"unexpected backend: Exo placed {layout['kind']}, the experiment asked for {want}")
        if ready.get("agreement_problems"):
            raise ParseError(f"the nodes disagree on the cluster state: {ready['agreement_problems']}")
        ev = {"backend": backend, "layout": layout, "node_ids": ready.get("node_ids"), "hosts": {}, "findings": []}
        for host, text in logs.items():
            tbccl_ready = re.findall(r"rank (\d+)/(\d+) TBCCL communicator ready", text)
            ring_done = re.findall(r"Rank (\d+) mlx distributed initialization complete", text)
            groups = re.findall(r"Group size: (\d+), group rank: (\d+)", text)
            ev["hosts"][host] = {"tbccl_communicator_ready": [{"rank": int(r), "world": int(w)} for r, w in tbccl_ready], "ring_initialized": [int(r) for r in ring_done],
                                 "mlx_group": [{"size": int(s), "rank": int(r)} for s, r in groups]}
        if point.rank_count == 2:
            if backend == "MlxTbccl":
                for host, e in ev["hosts"].items():
                    if not e["tbccl_communicator_ready"]:
                        raise ParseError(f"transport not proven: no 'TBCCL communicator ready' line in the log of {host}")
                    if e["ring_initialized"]:
                        raise ParseError(f"transport fallback: {host} also initialised an MLX ring group")
            else:
                for host, e in ev["hosts"].items():
                    if e["tbccl_communicator_ready"]:
                        raise ParseError(f"unexpected backend: {host} initialised a TBCCL communicator in a MlxRing run")
            masters = validation.masters_by_node({n: logs.get(h, "") for n, h in self._node_hosts(point, ready).items()})
            problems = validation.check_masters(masters)
            if problems:
                raise ParseError(f"cluster master: {problems}")
            ev["masters"] = masters
        ev["samples"] = self._samples(group, outcome)
        if point.rank_count == 2:
            for host, s in ev["samples"].items():
                if s["samples"] == 0:
                    raise ParseError(f"no sampler data from {host}: the data path cannot be verified")
                if backend == "MlxTbccl" and s["max_link_connections"] < 1:
                    raise ParseError(f"transport not proven: no TCP connection between the two link addresses seen from the Exo processes on {host}")
        return ev

    def _node_hosts(self, point, ready):
        ids = ready.get("node_ids") or {}
        return {ids[spec["host"]]: spec["host"] for spec in point.ranks if spec["host"] in ids}

    def _samples(self, group, outcome):
        sampler_invs = {inv.rank: inv for inv in group.invocations if inv.role.startswith("sampler@")}
        out = {}
        for rank, inv in sampler_invs.items():
            host = inv.role.split("@", 1)[1]
            rows = [r for r in _lines(outcome.stdout[rank], PREFIXES["sample"]) if "wall_utc" in r]
            link0 = next((r["link"] for r in rows if r.get("link")), None)
            link1 = next((r["link"] for r in reversed(rows) if r.get("link")), None)
            delta = {k: link1[k] - link0[k] for k in link0} if link0 and link1 else None

            def series(key, rows_=rows):
                return [float(r[key]) for r in rows_ if r.get(key) is not None]

            gpu_rows = [r["gpu"] for r in rows if r.get("gpu")]
            gpu_util = []
            for g in gpu_rows:
                try:
                    gpu_util.append(float(g["utilization_gpu_percent"]))
                except (KeyError, TypeError, ValueError):
                    continue
            out[host] = {"samples": len(rows), "link_delta": delta, "max_link_connections": max((r["connections"]["link"] for r in rows if r.get("connections")), default=0),
                         "cpu_percent": _summary(series("cpu_percent")), "mem_used_percent_max": max(series("mem_used_percent"), default=None),
                         "process_rss_mib_max": max((p["rss_mib"] for r in rows for p in r.get("processes", [])), default=None),
                         "process_cpu_percent_sum": _summary([sum(p["cpu_percent"] for p in r.get("processes", [])) for r in rows if r.get("processes")]),
                         "process_threads_max": max((sum(p["threads"] for p in r.get("processes", [])) for r in rows), default=None),
                         "gpu": ({"utilization_percent": _summary(gpu_util), "memory_used_mib_max": max((float(g["memory_used_mib"]) for g in gpu_rows if g.get("memory_used_mib") is not None), default=None),
                                  "temperature_c_max": max((float(g["temperature_c"]) for g in gpu_rows if g.get("temperature_c") is not None), default=None),
                                  "throttle_reasons_seen": sorted({g.get("throttle_reasons") for g in gpu_rows if g.get("throttle_reasons")})} if gpu_rows else None),
                         "node_lost": any(r.get("event") == "engine_lost" for r in _lines(outcome.stdout[rank], PREFIXES["sample"]))}
        return out

    def _measure(self, point, phases, evidence, raw, primary, layout, ready):
        from ..exo_bench import metrics
        pname = primary["phase"]
        recs = raw["records"].get(pname)
        if not recs:
            raise ParseError(f"no raw timelines for the primary phase '{pname}'")
        ok = [r for r in recs if r["status"] == "ok"]
        if not ok:
            raise ParseError(f"no successful request in the primary phase '{pname}'")
        metric = primary.get("metric", "e2e")
        key = {"ttft": "ttft_s", "tpot": "tpot_s", "e2e": "e2e_s", "itl": "itl_s"}[metric]
        values = []
        for r in ok:
            v = metrics.derive(r)[key]
            if key == "itl_s":
                values.extend(v)
            elif v is not None:
                values.append(v)
        if not values:
            raise ParseError(f"the primary metric {metric} is undefined for every request of '{pname}'")
        us = [v * 1e6 for v in values]
        s = metrics.summarize(us)
        stats = {k: s[k] for k in ("min", "max", "mean") if s[k] is not None}
        stats.update({"median": s["p50"], **{k: s[k] for k in ("p90", "p95", "p99") if s[k] is not None}})
        tm = timing(f"exo_{metric}_client_observed", stats, "computed", "median", percentile_method=PERCENTILE_LINEAR, sample_count=len(us), samples=us)
        thr = metrics.phase_throughput(recs)
        throughput = []
        if thr["rates"]:
            throughput = [{"kind": "inference_output_tokens_per_second", "value": thr["rates"]["output_tokens_per_s"], "unit": "tokens/s",
                           "definition": f"actual generated tokens of successful requests in phase '{pname}' / the phase wall time (first submission to last end)"},
                          {"kind": "inference_requests_per_second", "value": thr["rates"]["requests_per_s"], "unit": "requests/s",
                           "definition": f"successful requests in phase '{pname}' / the phase wall time"}]
        accounting = [p for r in ok for p in metrics.accounting(r)["problems"]]
        extra = {"inference": {"primary": primary, "phases": {n: b for n, b in phases.items()}, "transport_evidence": evidence, "raw_files": raw["files"],
                               "placement": layout, "load_s": ready.get("load_s"), "runner_timeline": ready.get("runner_timeline"),
                               "request_accounting_problems": accounting[:20]}}
        warnings = ["client-observed latencies include the HTTP/SSE path on the API host; they are not native TBCCL transfer latencies",
                    "Exo exposes no token ids: tokens are counted from the stream's token events and cross-checked against the reported usage"]
        warnings += [f"request accounting: {p}" for p in accounting[:5]]
        gated = [n for n, b in phases.items() if b.get("check") or b.get("cancel")]
        ver = {"status": "passed" if not accounting else "failed", "method": "the placement, the instance type and the runner readiness read back from /state and checked against the experiment; every stream checked for "
               "finish_reason, usage and token-count agreement", "details": f"{len(ok)} successful requests in '{pname}', {len(accounting)} accounting problems; correctness gates passed: {gated}"}
        return Parsed({"timing": tm, "throughput": throughput, "extra": extra, "warnings": warnings}, ver,
                      configuration_extra={"inference": {"model_id": point.options["model_id"], "backend": point.options["backend"], "pipeline": point.rank_count == 2,
                                                         "stage_layers": [s["n_layers"] for s in layout["stages"]], "stage_hosts": [s.get("host") for s in layout["stages"]], "primary": primary,
                                                         "variant": point.options.get("variant", "default"), "tier": point.options.get("tier", "baseline"),
                                                         "fast_path_env": {k: str(v) for k, v in {**(point.options.get("env") or {}), **(point.options.get("server_env") or {})}.items() if k.startswith("EXO_TBCCL_")}}})
