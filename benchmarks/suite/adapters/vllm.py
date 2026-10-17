"""Integration of vLLM pipeline-parallel inference over vllm-tbccl (and single-node vLLM references) into the benchmark suite.

The suite never imports vLLM, PyTorch or vllm-tbccl. One workload point is one engine lifecycle: the adapter starts the API node (pipeline stage 0, node rank 0), the headless
node (stage 1) when the point is a two-node pipeline, one passive sampler per host (resources, link counters, TCP connection census, engine liveness) and a client that runs the
point's ordered phases against the API (``vllm_bench/client.py``). The runner owns every process: argument arrays only, one session per process, a lease/heartbeat on remote
hosts, and bounded termination of everything it started on both hosts. vLLM's leader never notices a dead headless worker, so liveness is supervised from outside: the
sampler of a host exits with a failure status when an engine process it saw disappears, and the runner fails the whole group on any nonzero exit.

The client prints ``TBCCL_VLLM_BENCH`` summaries and the raw evidence (``TBCCL_VLLM_RAW`` per request timeline, ``TBCCL_VLLM_SCRAPE`` per server-metrics sample); the adapter parses only
those contracts. Rank numbers of a point are the pipeline stages (rank 0 = node rank 0 = the API node); the ``reverse`` orientation swaps which host runs which stage and, with it, where
the API and the client run. Further invocations (samplers, the client) are numbered after the ranks.

Application-level results are never filed as transfer throughput: they use their own throughput kinds (``inference_*``) and an ``extra.inference`` block.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
from pathlib import Path

from .base import BenchmarkAdapter, Capabilities, Group, Invocation, ParseError, Parsed, timing
from .core import PERCENTILE_LINEAR

PREFIXES = {"bench": "TBCCL_VLLM_BENCH", "raw": "TBCCL_VLLM_RAW", "scrape": "TBCCL_VLLM_SCRAPE", "reference": "TBCCL_VLLM_REFERENCE", "sample": "TBCCL_VLLM_SAMPLE"}
STAGE_DEVICES = ("cuda", "metal", "host")  # host = the CPU of that machine
PAIRINGS = {("cuda", "metal"), ("metal", "cuda"), ("cuda", "host"), ("host", "cuda")}
SINGLE = {"cuda", "metal", "host"}
SUPPORTED_VLLM = "0.31.0"
SUPPORTED_VLLM_TBCCL = "0.2.0"
ACTIVATION_BYTES_PER_TOKEN_MIN = 1024 * 2  # Qwen3-0.6B hidden size x bf16: the least one boundary tensor can weigh per token
FORBIDDEN_FALLBACKS = (r"backend=(nccl|gloo|mlx)", r"PyNcclCommunicator", r"vLLM is using nccl", r"NCCL version", r"mlx[^\n]*ring", r"ring[^\n]*backend")

SCENARIOS = {
    "vllm_startup": ("vllm_startup", ("startup",), "engine lifecycle: start, first request, shutdown"),
    "vllm_correctness": ("vllm_correctness", ("gates",), "output tokens against a single-rank reference: sequential, concurrent, mixed, staggered, repeated, cancellation"),
    "vllm_latency": ("vllm_latency", ("single_request",), "one request at a time: TTFT, TPOT, ITL, E2E"),
    "vllm_throughput": ("vllm_throughput", ("closed_loop", "open_loop", "mixed"), "request and token throughput at a fixed concurrency or a fixed arrival rate"),
    "vllm_concurrency": ("vllm_concurrency", ("sweep",), "latency and throughput against closed-loop concurrency"),
    "vllm_pipeline": ("vllm_pipeline", ("order", "baseline", "diagnostic"), "pipeline stage order, single-node references and stage-level diagnostics"),
    "vllm_backend": ("vllm_pipeline", ("pingpong",), "pipeline-boundary ping-pong through vllm-tbccl's own transport at vLLM's GroupCoordinator level, without a model"),
}


def _lines(path, prefix):
    out = []
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return out
    for line in text.splitlines():
        if line.startswith(prefix + " "):
            try:
                out.append(json.loads(line[len(prefix) + 1:]))
            except ValueError:
                raise ParseError(f"malformed {prefix} line: {line[:80]!r}")
    return out


def _read(path):
    try:
        return Path(path).read_text(errors="replace")
    except OSError:
        return ""


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


class VllmPipeline(BenchmarkAdapter):
    """One adapter implementation behind several benchmark names (see SCENARIOS); the names only select the category and operations reported."""

    def __init__(self, name):
        category, operations, description = SCENARIOS[name]
        self.caps = Capabilities(
            name=name, category=category, description=f"vLLM {description} (vllm-tbccl {SUPPORTED_VLLM_TBCCL}, vLLM {SUPPORTED_VLLM}, Qwen3-0.6B)", executable="vllm", operations=operations,
            memory_backends=STAGE_DEVICES, min_ranks=1, max_ranks=2, executable_in_build=False, supports_warmup=False, supports_iterations=False, verification={op: "native" for op in operations},
            source_files=("benchmarks/suite/vllm_bench/client.py", "benchmarks/suite/vllm_bench/sampler.py", "benchmarks/suite/vllm_bench/metrics.py"),
            notes="Framework benchmark, not a native TBCCL benchmark. payload_bytes is a placeholder (1): the workload is defined by options.phases (prompt tokens, output tokens, concurrency). "
                  "Ranks are pipeline stages; rank_counts [1] runs a single-node reference with the stage-0 host. Needs per-host executables: vllm, vllm_python, vllm_bench_dir, vllm_model_dir "
                  "(and vllm_workload, vllm_reference on the API host).")

    # ---- configuration ----
    def validate_workload(self, workload, operations):
        errors = []
        o = workload.get("options") or {}
        if not o.get("phases"):
            errors.append("options.phases is required (the ordered client phases)")
        for key in ("dtype", "max_model_len"):
            if key not in o:
                errors.append(f"options.{key} is required (the benchmark never picks a dtype or a context length silently)")
        if o.get("dtype") not in (None, "bfloat16", "float16", "float32"):
            errors.append("options.dtype must be bfloat16, float16 or float32")
        for i, ph in enumerate(o.get("phases") or []):
            if "kind" not in ph or "name" not in ph:
                errors.append(f"options.phases[{i}] needs 'name' and 'kind'")
        names = [ph.get("name") for ph in o.get("phases") or []]
        if len(set(names)) != len(names):
            errors.append("options.phases names must be unique")
        if o.get("primary") and o["primary"].get("phase") not in names:
            errors.append("options.primary.phase must name one of the phases")
        return errors

    def check_point(self, point, ctx):
        devices = tuple(r["memory_backend"] for r in point.ranks)
        if point.rank_count == 2 and devices not in PAIRINGS:
            point.skip_state = "unsupported"
            return f"stage assignment {devices} is not a validated vllm-tbccl pairing (validated: CUDA with Metal in either order, CUDA with CPU; vllm-tbccl 0.2.0)"
        if point.rank_count == 1 and devices[0] not in SINGLE:
            point.skip_state = "unsupported"
            return f"unknown single-node device '{devices[0]}'"
        if point.rank_count == 2 and point.ranks[0]["host"] == point.ranks[1]["host"]:
            return "the two pipeline stages must run on two different hosts"
        if point.rank_count == 2 and devices in {("host", "cuda")}:
            point.skip_state = "unsupported"
            return "CPU as stage 0 with CUDA as stage 1 is not a validated stage assignment"
        o = point.options
        ml = int(o.get("max_model_len", 0))
        worst = 0
        for ph in o.get("phases") or []:
            for item in ([{"prompt": ph.get("prompt"), "tokens": ph.get("tokens")}] if "prompt" in ph else []) + list(ph.get("mix") or []):
                n = _prompt_len(item.get("prompt"), o)
                if n is not None and item.get("tokens"):
                    worst = max(worst, n + int(item["tokens"]))
        if ml and worst > ml:
            return f"prompt plus output tokens ({worst}) exceed options.max_model_len ({ml}); the context is not raised silently"
        return None

    def estimate_bytes(self, point):
        """Boundary activation bytes the point moves (both directions summed), a coarse upper bound for the volume budget: every prompt and output token crosses the boundary as
        a hidden state plus a residual."""
        if point.rank_count < 2:
            return None
        tokens = 0
        for ph in point.options.get("phases") or []:
            slots = _slots(ph)
            if ph.get("kind") == "rounds":
                count = int(ph.get("rounds", 1)) * max(len(slots), 1)
            elif ph.get("kind") in ("continuous", "open_loop", "warmup"):
                count = int(ph.get("requests", 1))
            else:
                count = max(len(ph.get("cases") or []) * int(ph.get("repeat", 1)), int(ph.get("conc", 1)), int(ph.get("streams", 0)), 1)
            per = [(_prompt_len(p, point.options) or 128) + int(t or 32) for p, t in slots] or [128 + int(ph.get("tokens") or 32)]
            tokens += count * (sum(per) / len(per))
        return int(tokens * ACTIVATION_BYTES_PER_TOKEN_MIN * 2)

    def data_direction(self, point):
        return {"mode": "directional", "source_rank": 0} if point.rank_count == 2 else {"mode": "bidirectional", "source_rank": None}

    def describe(self, point):
        return {"payload_semantics": "not applicable: the payload size is a placeholder; the workload is defined by prompt and output token counts in options.phases",
                "backend": "vllm-tbccl" if point.rank_count == 2 else "vllm-single-node",
                "timing_method": "client-side time.monotonic_ns() on the API host: submission time and the arrival time of every output-carrying server-sent event; per-request TTFT, TPOT, ITL and E2E are "
                                 "derived from the raw timelines (vllm_bench/metrics.py); timestamps of different hosts are never subtracted",
                "completion_semantics": "TTFT = first output event; E2E = last output event; the request's token count is the server-reported usage, never the configured maximum"}

    # ---- planning ----
    def _tool(self, ctx, name, host_id, device=None):
        """A per-host executable; ``name:device`` (for example ``vllm:host``) overrides ``name`` for a stage on that device (the Mac CPU environment has no vllm-metal)."""
        host = ctx.config.hosts[host_id]
        if device and f"{name}:{device}" in host.executables:
            return host.executables[f"{name}:{device}"]
        if name in host.executables:
            return host.executables[name]
        if host.is_local:
            raise ParseError(f"host '{host_id}' needs executables.{name} (the vLLM environment is configured per host)")
        raise ParseError(f"remote host '{host_id}' needs executables.{name}")

    def _bench_script(self, ctx, host_id, script):
        host = ctx.config.hosts[host_id]
        base = host.executables.get("vllm_bench_dir")
        if not base:
            if host.is_local:
                base = str(Path(ctx.repo_root) / "benchmarks" / "suite" / "vllm_bench")
            else:
                raise ParseError(f"remote host '{host.id}' needs executables.vllm_bench_dir")
        return str(Path(base) / script)

    def _server_env(self, point, ctx, rank, spec, ports):
        o = point.options
        host = ctx.config.hosts[spec["host"]]
        device = spec["memory_backend"]
        ip = ctx.rank_address(spec)
        env = {"VLLM_USE_V2_MODEL_RUNNER": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "DO_NOT_TRACK": "1"}
        if point.rank_count == 2:
            env.update({"VLLM_TBCCL_ENABLE": "1", "TBCCL_LOCAL_ENDPOINT": f"{ip}:0", "VLLM_HOST_IP": ip})
            if device == "metal":
                env["VLLM_TBCCL_BACKEND"] = "metal"
            elif device == "host":
                env["VLLM_TBCCL_BACKEND"] = "torch"  # the CPU platform is owned by vllm-tbccl even where vllm-metal is installed
        if device == "host":
            env["VLLM_CPU_KVCACHE_SPACE"] = str(o.get("cpu_kvcache_gib", 4))
        trace = o.get("trace")
        if trace:
            work = host.executables.get("vllm_work_dir") or "/tmp"
            tdir = f"{work}/trace-{point.point_id}"
            env.update({"VLLM_TBCCL_TRACE": "1", "VLLM_TBCCL_TRACE_FILE": f"{tdir}/tbccl", "VLLM_TBCCL_TRACE_PERIOD_S": "2", "TBCCL_B5_TRACE_DIR": tdir,
                        "PYTHONPATH": self._bench_script(ctx, spec["host"], "trace_hook")})
            if trace in ("steps", "full"):
                env["VLLM_TBCCL_TRACE_STEPS"] = "1"
            if trace == "full":
                env["VLLM_TBCCL_CHECKSUM"] = "1"
        env.update({str(k): str(v) for k, v in (o.get("server_env") or {}).items()})
        return env

    def _server_argv(self, point, ctx, rank, spec, ports, leader_ip):
        o = point.options
        device = spec["memory_backend"]
        model = o["model_dir_override"] if o.get("model_dir_override") else self._tool(ctx, "vllm_model_dir", spec["host"], device)  # the override is for the failed-initialisation test
        argv = [self._tool(ctx, "vllm", spec["host"], device), "serve", model, "--served-model-name", o.get("served_model_name", "m"), "--dtype", o["dtype"],
                "--enforce-eager", "--no-async-scheduling", "--no-enable-prefix-caching", "--max-model-len", str(o["max_model_len"])]
        util = o.get("gpu_memory_utilization")
        if isinstance(util, dict):
            util = util.get(spec["host"])
        if device != "host" and util is not None:
            argv += ["--gpu-memory-utilization", str(util)]
        if point.rank_count == 2:
            argv += ["--pipeline-parallel-size", "2", "--tensor-parallel-size", "1", "--nnodes", "2", "--master-addr", leader_ip, "--master-port", str(ports[1]),
                     "--distributed-executor-backend", "mp", "--node-rank", str(rank)]
            if rank == 1:
                argv += ["--headless"]
        if rank == 0:
            bind = "0.0.0.0" if o.get("client_host") == "other" else (leader_ip if point.rank_count == 2 else "127.0.0.1")  # a client on the other machine reaches the API over the management LAN
            argv += ["--host", bind, "--port", str(ports[0])]
        argv += [str(x) for x in (o.get("extra_args") or [])]
        return argv

    def _spec(self, point, ctx, ports, api_host, api_ip, client_host=None):
        """The client's phases and inputs. ``client_host`` is the machine the client runs on (the API host by default); its workload and reference copies are used."""
        o = point.options
        client_host = client_host or api_host
        client_cfg = ctx.config.hosts[client_host].executables
        wl_path = self._tool(ctx, "vllm_workload", client_host)
        ref = client_cfg.get("vllm_reference")
        spec = {"api": {"host": api_ip, "port": ports[0]}, "model": o.get("served_model_name", "m"), "workload": wl_path, "reference": ref if o.get("use_reference", True) else None,
                "request_timeout_s": o.get("request_timeout_s", 180), "tie_margin": o.get("tie_margin", 0.251),
                "client_role": "api-local" if client_host == api_host else "remote-lan",
                "workload_sha256": client_cfg.get("vllm_workload_sha256") or o.get("workload_sha256"), "reference_sha256": client_cfg.get("vllm_reference_sha256") or o.get("reference_sha256"),
                "phases": [dict(ph) for ph in o["phases"]]}
        ready = {"name": "ready", "kind": "ready", "timeout_s": o.get("startup_timeout_s", 600)}
        spec["phases"] = [ready] + spec["phases"]
        return spec

    def plan(self, point, ctx, ports, rep_dir):
        o = point.options
        specs = point.ranks
        leader = specs[0]
        leader_ip = ctx.rank_address(leader)
        invs = []
        for rank, spec in enumerate(specs):  # ranks first: Invocation.rank indexes point.ranks for the real ranks
            invs.append(Invocation(rank, "api-node" if rank == 0 else "headless-node", spec["host"], self._server_argv(point, ctx, rank, spec, ports, leader_ip),
                                   env=self._server_env(point, ctx, rank, spec, ports), persistent=True))  # readiness is proven by the client (the API takes minutes), not by a runner port wait
        self._trace_sources = {}
        if o.get("trace"):
            for spec in specs:
                host = ctx.config.hosts[spec["host"]]
                work = host.executables.get("vllm_work_dir") or "/tmp"
                self._trace_sources[spec["host"]] = (None if host.is_local else list(host.launcher.get("argv", [])), bool(host.launcher.get("shell_join", True)), f"{work}/trace-{point.point_id}")
        next_rank = len(invs)
        expect = ",".join(ctx.rank_address(s) for s in specs) if point.rank_count == 2 else ""
        for spec in specs:
            host = ctx.config.hosts[spec["host"]]
            iface = spec.get("interface")
            iface = host.interface if (iface in host.interfaces or not iface) else iface
            argv = [self._tool(ctx, "vllm_python", spec["host"], spec["memory_backend"]), self._bench_script(ctx, spec["host"], "sampler.py"), "--role", spec["host"], "--interval", str(o.get("sample_interval_s", 1.0)),
                    "--expect-addresses", expect, "--liveness"]
            if iface and point.rank_count == 2:
                argv += ["--link-interface", iface]
            if spec["memory_backend"] == "cuda":
                argv += ["--gpu"]
            if host.os is None or host.os.lower().startswith("linux"):
                argv += ["--thermal"]
            invs.append(Invocation(next_rank, f"sampler@{spec['host']}", spec["host"], argv, persistent=True, stop_first=True))
            next_rank += 1
        api_ip = leader_ip if point.rank_count == 2 else "127.0.0.1"
        fault = o.get("fault")
        if fault:  # lifecycle tier only: one exact-title kill on the stated stage's host once a request is running
            target = specs[int(fault.get("stage", 1))]
            invs.append(Invocation(next_rank, f"fault@{target['host']}", target["host"],
                                   [self._tool(ctx, "vllm_python", target["host"], target["memory_backend"]), self._bench_script(ctx, target["host"], "fault.py"), "--api", f"{api_ip}:{ports[0]}",
                                    "--title", fault.get("title", f"VLLM::Worker_PP{int(fault.get('stage', 1))}"), "--delay-s", str(fault.get("delay_s", 3)), "--signal", fault.get("signal", "KILL")]))
            next_rank += 1
        client = specs[-1] if (o.get("client_host") == "other" and point.rank_count == 2) else leader
        if client is not leader:  # the client runs on the other machine and reaches the API over the management LAN, never over the pipeline link
            lan = self._tool(ctx, "vllm_lan_address", leader["host"])
            api_ip = lan
        spec_json = self._spec(point, ctx, ports, leader["host"], api_ip, client["host"])
        encoded = base64.b64encode(json.dumps(spec_json).encode()).decode()
        invs.append(Invocation(next_rank, "client", client["host"],
                               [self._tool(ctx, "vllm_python", client["host"], client["memory_backend"]), self._bench_script(ctx, client["host"], "client.py"), "--spec-b64", encoded], persistent=False))
        return Group(invs, ports=list(ports), notes=[f"api on {leader['host']}", f"client on {client['host']}", f"stages: {[s['memory_backend'] for s in specs]}"])

    # ---- parsing ----
    def _split(self, point, group, outcome):
        n = point.rank_count
        server = {r: _read(outcome.stdout[r]) + "\n" + _read(outcome.stderr[r]) for r in range(n) if r in outcome.stdout}
        samplers = {inv.rank: inv for inv in group.invocations if inv.role.startswith("sampler@")}
        client_inv = next(inv for inv in group.invocations if inv.role == "client")
        return server, samplers, client_inv

    def parse(self, point, group, outcome):
        server_logs, sampler_invs, client = self._split(point, group, outcome)
        bench = _lines(outcome.stdout[client.rank], PREFIXES["bench"])
        if not bench or bench[-1].get("kind") != "end":
            raise ParseError("the client did not finish: no end marker")
        phases = {b["phase"]: b for b in bench if b.get("kind") not in ("scrape", "end")}
        ready = next((b for b in bench if b.get("kind") == "ready"), None)
        if not ready or not ready.get("ready"):
            raise ParseError("the API never became ready")
        failed = [b["phase"] for b in bench if (b.get("check") or {}).get("passed") is False or (b.get("cancel") or {}).get("passed") is False]
        if failed:
            raise ParseError(f"correctness gate failed in phase(s) {failed}")
        evidence = self._transport_evidence(point, server_logs, sampler_invs, outcome)
        raw = self._extract_raw(outcome.stdout[client.rank], Path(outcome.stdout[client.rank]).parent / "vllm")  # files are named relative to the repetition directory
        primary = point.options.get("primary") or self._default_primary(phases)
        parsed = self._measure(point, phases, evidence, raw, primary, outcome)
        if point.options.get("trace"):
            parsed.measurements["extra"]["inference"]["trace"] = self._collect_traces(Path(outcome.stdout[client.rank]).parent / "trace")
        return parsed

    def _collect_traces(self, out_dir):
        """Copy the per-process trace files of every host next to the run (a remote host over its management launcher), then remove the host's trace directory (it holds only these files)."""
        import io
        import shlex
        import shutil
        import subprocess
        import tarfile
        files = {}
        for host, (prefix, shell_join, directory) in getattr(self, "_trace_sources", {}).items():
            dest = out_dir / host
            dest.mkdir(parents=True, exist_ok=True)
            try:
                if prefix is None:
                    src = Path(directory)
                    for f in sorted(src.glob("*")) if src.exists() else []:
                        shutil.copy2(f, dest / f.name)
                    shutil.rmtree(src, ignore_errors=True)
                else:
                    cmd = ["sh", "-c", f"tar -C {shlex.quote(directory)} -cf - . && rm -rf {shlex.quote(directory)}"]
                    out = subprocess.run(prefix + ([shlex.join(cmd)] if shell_join else cmd), capture_output=True, timeout=120)
                    if out.returncode == 0 and out.stdout:
                        with tarfile.open(fileobj=io.BytesIO(out.stdout)) as tf:
                            tf.extractall(dest, filter="data")
            except (OSError, subprocess.SubprocessError, tarfile.TarError) as error:
                files[host] = {"error": str(error)}
                continue
            files[host] = sorted(f"trace/{host}/{f.name}" for f in dest.glob("*") if f.is_file())
        return {"files": files}

    def _default_primary(self, phases):
        for name, b in phases.items():
            if b.get("kind") in ("rounds", "continuous", "open_loop"):
                return {"phase": name, "metric": "e2e"}
        for name, b in phases.items():
            if b.get("kind") == "check":  # a correctness-only point: the first check phase's requests give the latency record
                return {"phase": name, "metric": "e2e"}
        raise ParseError("no measured phase in the client output")

    def _extract_raw(self, stdout_path, out_dir):
        """Write the raw per-request timelines, server-metrics samples and any reference to files next to the rank output; returns {phase: {...}}."""
        out_dir.mkdir(parents=True, exist_ok=True)
        files, per_phase = {}, {}
        rel = lambda p: f"{out_dir.name}/{p.name}"  # noqa: E731
        for item in _lines(stdout_path, PREFIXES["raw"]):
            per_phase.setdefault(item["phase"], []).append(item["request"])
        for phase, recs in per_phase.items():
            p = out_dir / f"{phase}.requests.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in recs))
            files.setdefault(phase, {})["requests"] = rel(p)
        scr = {}
        for item in _lines(stdout_path, PREFIXES["scrape"]):
            scr.setdefault(item["phase"], []).append({k: v for k, v in item.items() if k != "phase"})
        for phase, samples in scr.items():
            p = out_dir / f"{phase}.server-metrics.jsonl"
            p.write_text("".join(json.dumps(r) + "\n" for r in samples))
            files.setdefault(phase, {})["server_metrics"] = rel(p)
        for item in _lines(stdout_path, PREFIXES["reference"]):
            p = out_dir / f"{item['name']}.reference.json"
            p.write_text(json.dumps(item["reference"]))
            files.setdefault(item["name"], {})["reference"] = rel(p)
        return {"files": files, "records": per_phase}

    def _transport_evidence(self, point, logs, sampler_invs, outcome):
        """Prove TBCCL carried the pipeline: backend lines of both workers, no fallback marker, TCP connections between the two link addresses, traffic on the data interface."""
        ev = {"pipeline": point.rank_count == 2, "hosts": {}, "findings": []}
        for rank, text in logs.items():
            host = point.ranks[rank]["host"]
            versions = sorted(set(re.findall(r"Initializing a V1 LLM engine \(v([0-9.]+[^)]*)\)", text) + re.findall(r"Launching vLLM \(v([0-9.]+[^)]*)\)", text)
                                  + re.findall(r"\bversion ([0-9]+\.[0-9]+\.[0-9]+)\s", text)))
            backend = re.findall(r"world_size=(\d+) rank=(\d+) local_rank=\d+ distributed_init_method=\S+ backend=(\w+)", text)
            fallbacks = sorted({m for pat in FORBIDDEN_FALLBACKS for m in re.findall(pat, text, re.I) if m})
            seam = re.findall(r"Pipeline transport (\S+): rank=(\d)/(\d)", text)
            ev["hosts"][host] = {"stage": rank, "device": point.ranks[rank]["memory_backend"], "vllm_versions": versions, "distributed_init": [{"world_size": int(w), "rank": int(r), "backend": b} for w, r, b in backend],
                                 "pipeline_transport_class": seam[0][0] if seam else None, "fallback_markers": fallbacks}
        for host, e in ev["hosts"].items():
            if not e["vllm_versions"]:
                raise ParseError(f"the vLLM version of {host} could not be read from its log")
            if any(not v.startswith(SUPPORTED_VLLM) for v in e["vllm_versions"]):
                raise ParseError(f"vLLM {e['vllm_versions']} on {host}: only {SUPPORTED_VLLM} is validated with vllm-tbccl {SUPPORTED_VLLM_TBCCL}")
        if point.rank_count == 2:
            for host, e in ev["hosts"].items():
                if any(d["backend"] != "tbccl" for d in e["distributed_init"] if d["world_size"] == 2):
                    raise ParseError(f"transport fallback: {host} initialised a two-rank group with a backend other than tbccl")
                if not any(d["backend"] == "tbccl" and d["world_size"] == 2 for d in e["distributed_init"]):
                    raise ParseError(f"transport not proven: no 'world_size=2 ... backend=tbccl' line in the log of {host}")
                if e["fallback_markers"]:
                    raise ParseError(f"transport fallback marker(s) in the log of {host}: {e['fallback_markers']}")
                if e["device"] == "metal" and not (e["pipeline_transport_class"] or "").endswith("TBCCLMetalPipelineTransport"):
                    raise ParseError(f"the Metal stage on {host} did not report the TBCCL pipeline transport seam")
        samples = self._samples(sampler_invs, outcome)
        ev["samples"] = samples
        if point.rank_count == 2:
            for host, s in samples.items():
                if s["samples"] == 0:
                    raise ParseError(f"no sampler data from {host}: the data path cannot be verified")
                if s["max_link_connections"] < 1:
                    raise ParseError(f"transport not proven: no TCP connection between the two link addresses seen from the vLLM processes on {host}")
                allowed = {"vllm-serve:other", "APIServer:other"} if point.options.get("client_host") == "other" else set()  # a remote client's LAN connection to the API
                bad = [x for x in s["other_pairs"] if x not in allowed]
                if bad:
                    raise ParseError(f"transport not proven: vLLM processes on {host} held TCP connections outside the link and loopback: {bad}")
        return ev

    def _samples(self, sampler_invs, outcome):
        out = {}
        for rank, inv in sampler_invs.items():
            host = inv.role.split("@", 1)[1]
            rows = [r for r in _lines(outcome.stdout[rank], PREFIXES["sample"]) if "wall_utc" in r]
            link0 = next((r["link"] for r in rows if r.get("link")), None)
            link1 = next((r["link"] for r in reversed(rows) if r.get("link")), None)
            delta = {k: link1[k] - link0[k] for k in link0} if link0 and link1 else None
            pairs = {tuple(p) for r in rows for p in r.get("connection_pairs", [])}
            other = sorted({f"{p[2]}" for p in pairs if p[2].endswith(":other")})
            gpu_rows = [r["gpu"] for r in rows if r.get("gpu")]

            def series(key, rows_=rows):
                return [float(r[key]) for r in rows_ if r.get(key) is not None]

            def gpu_series(key):
                vals = []
                for g in gpu_rows:
                    try:
                        vals.append(float(g[key]))
                    except (KeyError, TypeError, ValueError):
                        continue
                return vals

            thermal = [r["thermal"] for r in rows if r.get("thermal")]
            out[host] = {"samples": len(rows), "link_delta": delta, "max_link_connections": max((r["connections"]["link"] for r in rows if r.get("connections")), default=0),
                         "other_connections_max": max((r["connections"]["other"] for r in rows if r.get("connections")), default=0), "other_pairs": other,
                         "cpu_percent": _summary(series("cpu_percent")), "mem_used_percent_max": max(series("mem_used_percent"), default=None),
                         "process_rss_mib_max": max((p["rss_mib"] for r in rows for p in r.get("processes", [])), default=None),
                         "gpu": ({"temperature_c_max": max(gpu_series("temperature_c"), default=None), "utilization_percent": _summary(gpu_series("utilization_gpu_percent")),
                                  "memory_used_mib_max": max(gpu_series("memory_used_mib"), default=None), "clock_sm_mhz_min": min(gpu_series("clock_sm_mhz"), default=None),
                                  "throttle_reasons_seen": sorted({g.get("throttle_reasons") for g in gpu_rows if g.get("throttle_reasons")})} if gpu_rows else None),
                         "cpu_thermal": ({"x86_pkg_temp_max": max((t["zones_c"].get("x86_pkg_temp") for t in thermal if t.get("zones_c", {}).get("x86_pkg_temp") is not None), default=None),
                                          "package_throttle_count_first": thermal[0].get("package_throttle_count"), "package_throttle_count_last": thermal[-1].get("package_throttle_count"),
                                          "cpu_freq_khz_min": min((t["cpu_freq_khz_min"] for t in thermal if t.get("cpu_freq_khz_min")), default=None)} if thermal else None),
                         "engine_lost": any(r.get("event") == "engine_lost" for r in _lines(outcome.stdout[rank], PREFIXES["sample"]))}
        return out

    def _measure(self, point, phases, evidence, raw, primary, outcome):
        from ..vllm_bench import metrics
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
            d = metrics.derive(r)
            v = d[key]
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
        tm = timing(f"vllm_{metric}_client_observed", stats, "computed", "median", percentile_method=PERCENTILE_LINEAR, sample_count=len(us), samples=us)
        thr = metrics.phase_throughput(recs)
        throughput = []
        if thr["rates"]:
            throughput = [{"kind": "inference_output_tokens_per_second", "value": thr["rates"]["output_tokens_per_s"], "unit": "tokens/s",
                           "definition": f"actual generated tokens of successful requests in phase '{pname}' / the phase wall time (first submission to last end)"},
                          {"kind": "inference_requests_per_second", "value": thr["rates"]["requests_per_s"], "unit": "requests/s",
                           "definition": f"successful requests in phase '{pname}' / the phase wall time"}]
        dists = {name: b.get("distributions") for name, b in phases.items() if b.get("distributions")}
        extra = {"inference": {"primary": primary, "phases": {n: _public(b) for n, b in phases.items()}, "distributions_s": dists, "transport_evidence": evidence,
                               "raw_files": raw["files"], "placement": {"api_host": point.ranks[0]["host"], "client_host": (point.ranks[-1]["host"] if point.options.get("client_host") == "other" and point.rank_count == 2 else point.ranks[0]["host"]), "stages": [{"stage": i, "host": r["host"], "device": r["memory_backend"]}
                                                                                                                                                       for i, r in enumerate(point.ranks)]}}}
        warnings = ["client-observed latencies include the HTTP/SSE path on the API host; they are not native TBCCL transfer latencies",
                    "the primary statistic pools the raw observations of the primary phase; percentiles with too few observations are withheld in the distributions"]
        warnings += [f"{n}: {p}" for n, b in phases.items() for p in b.get("problems", [])]
        gated = [n for n, b in phases.items() if b.get("check") or b.get("cancel")]
        ver = {"status": "passed" if gated else "not_run", "method": "greedy token ids compared with a single-rank reference; near-tie divergences classified by the reference's own top-1/top-2 margin",
               "details": f"gate phases passed: {gated}" if gated else "no correctness phase in this point"}
        return Parsed({"timing": tm, "throughput": throughput, "extra": extra, "warnings": warnings}, ver,
                      configuration_extra={"inference": {"dtype": point.options["dtype"], "max_model_len": point.options["max_model_len"], "pipeline": point.rank_count == 2,
                                                         "stage_devices": [r["memory_backend"] for r in point.ranks], "scheduler": {"async_scheduling": False, "prefix_caching": False, "enforce_eager": True},
                                                         "primary": primary}})


class VllmBackend(VllmPipeline):
    """The transport of a vLLM pipeline boundary on its own: ``pg_pingpong.py`` on both ranks (rank 0 on the first configured host), the CUDA and CPU stages only (the Metal boundary
    uses a different transport class and codec). It uses vLLM's GroupCoordinator and vllm-tbccl's bundled backend, so it measures what the engine's boundary exchange costs without any
    compute, scheduler or API; it is not the native TBCCL benchmark and is never merged with it."""
    PAIRS = {("cuda", "host"), ("host", "cuda")}

    def validate_workload(self, workload, operations):
        o = workload.get("options") or {}
        errors = [] if o.get("tokens") else ["options.tokens (a list of token counts) is required"]
        return errors

    def check_point(self, point, ctx):
        devices = tuple(r["memory_backend"] for r in point.ranks)
        if point.rank_count != 2 or devices not in self.PAIRS:
            point.skip_state = "unsupported"
            return f"the boundary ping-pong covers a CUDA stage with a CPU stage only, not {devices}"
        if point.ranks[0]["host"] == point.ranks[1]["host"]:
            return "the two ranks must run on two different hosts"
        return None

    def estimate_bytes(self, point):
        o = point.options
        return int(sum(2 * int(t) * 1024 * 2 * 2 * (int(o.get("warmup", 50)) + int(o.get("iterations", 300))) for t in o["tokens"]))

    def data_direction(self, point):
        return {"mode": "bidirectional", "source_rank": None}

    def describe(self, point):
        return {"payload_semantics": "bf16 hidden_states + residual of [tokens, 1024] per direction (the Qwen3-0.6B stage boundary), sent with GroupCoordinator.send_tensor_dict",
                "backend": "vllm-tbccl-groupcoordinator",
                "timing_method": "rank 0 perf_counter_ns around send_tensor_dict plus recv_tensor_dict of the echoed dict; raw round trips kept; the round trip of the whole exchange (several point-to-point messages) is the measurement, not a one-way time",
                "completion_semantics": "the echoed tensor dict was received and (first iteration and last) compared with the sent values"}

    def plan(self, point, ctx, ports, rep_dir):
        o = point.options
        leader_ip = ctx.rank_address(point.ranks[0])
        invs = []
        for rank, spec in enumerate(point.ranks):
            device = spec["memory_backend"]
            env = {"VLLM_USE_V2_MODEL_RUNNER": "0", "HF_HUB_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1", "VLLM_TBCCL_ENABLE": "1", "VLLM_HOST_IP": ctx.rank_address(spec),
                   "TBCCL_LOCAL_ENDPOINT": f"{ctx.rank_address(spec)}:0", "VLLM_TBCCL_PLATFORM": "cuda" if device == "cuda" else "cpu"}
            if device == "host":
                env["VLLM_TBCCL_BACKEND"] = "torch"
            argv = [self._tool(ctx, "vllm_python", spec["host"], device), self._bench_script(ctx, spec["host"], "pg_pingpong.py"), "--rank", str(rank), "--init", f"tcp://{leader_ip}:{ports[0]}",
                    "--tokens", ",".join(str(t) for t in o["tokens"]), "--warmup", str(o.get("warmup", 50)), "--iterations", str(o.get("iterations", 300))]
            invs.append(Invocation(rank, "rank0" if rank == 0 else "rank1", spec["host"], argv, env=env, wait_for_port=ports[0] if rank == 1 else None, listens_on=ports[0] if rank == 0 else None))
        return Group(invs, ports=list(ports))

    def parse(self, point, group, outcome):
        got = {}
        for rank in (0, 1):
            lines = _lines(outcome.stdout[rank], "TBCCL_VLLM_PG")
            if len(lines) != 1:
                raise ParseError(f"rank {rank} printed {len(lines)} TBCCL_VLLM_PG lines, expected exactly one")
            got[rank] = lines[0]
        addrs = {a for inv in group.invocations for a in (inv.env.get("VLLM_HOST_IP"),) if a}
        for rank in (0, 1):
            d = got[rank]
            want = {"cuda": "cuda", "host": "cpu"}[point.ranks[rank]["memory_backend"]]
            if not d["device"].startswith(want):
                raise ParseError(f"rank {rank} ran on '{d['device']}', the configuration asked for '{want}'")
            if d["communicator"] != "TBCCLDeviceCommunicator" or d["dist_backend"] != "tbccl":
                raise ParseError(f"rank {rank} did not use the tbccl backend ({d['communicator']}, {d['dist_backend']})")
            if not (d["tbccl_ops"].get("send") or d["tbccl_ops"].get("recv")):
                raise ParseError(f"rank {rank}: no tensor moved through ProcessGroupTBCCL")
            conns = d.get("connections")
            if not conns:
                raise ParseError(f"rank {rank} could not list its TCP connections; the data path cannot be verified")
            peers = {_ipv4(c.split(">")[1]) for c in conns}
            outside = sorted(h for h in peers if not h.startswith("127.") and h not in addrs)
            if outside:
                raise ParseError(f"rank {rank} has TCP peers outside the data link: {outside}")
        sizes = got[0]["sizes"]
        if not all(s["echo_verified"] for s in sizes) or not all(s["peer_received_verified"] for s in got[1]["sizes"]):
            raise ParseError("a tensor dict did not arrive unchanged")
        from ..vllm_bench import metrics
        primary = next((s for s in sizes if s["tokens"] == int(point.options.get("primary_tokens", sizes[0]["tokens"]))), sizes[0])
        s = metrics.summarize(primary["samples_us"])
        stats = {k: s[k] for k in ("min", "max", "mean") if s[k] is not None}
        stats.update({"median": s["p50"], **{k: s[k] for k in ("p90", "p95", "p99") if s[k] is not None}})
        tm = timing(f"vllm_boundary_round_trip_{primary['tokens']}tok", stats, "computed", "median", percentile_method=PERCENTILE_LINEAR, sample_count=len(primary["samples_us"]), samples=primary["samples_us"])
        extra = {"inference": {"boundary": {str(x["tokens"]): {"bytes_per_direction": x["bytes_per_direction"], "rtt_us": x["rtt_us"]} for x in sizes},
                               "devices": [got[0]["device"], got[1]["device"]], "platforms": [got[0]["platform"], got[1]["platform"]], "torch": [got[0]["torch"], got[1]["torch"]],
                               "tbccl_ops": [got[0]["tbccl_ops"], got[1]["tbccl_ops"]], "connections": [got[0]["connections"], got[1]["connections"]]}}
        warnings = ["a round trip includes both directions' control message, tensors and receive handling; it is not a native TBCCL transfer time"]
        ver = {"status": "passed", "method": "the echoed tensor dict compared element by element on rank 0, and the first received dict on rank 1", "details": f"{len(sizes)} sizes verified"}
        return Parsed({"timing": tm, "extra": extra, "throughput": [], "warnings": warnings}, ver, configuration_extra={"inference": {"tokens": [x["tokens"] for x in sizes], "hidden": got[0]["hidden"],
                                                                                                                                 "stage_devices": [r["memory_backend"] for r in point.ranks]}})


def _ipv4(address):
    """Dual-stack sockets report an IPv4 peer as ::ffff:a.b.c.d (macOS: ::a.b.c.d)."""
    if address.startswith("::ffff:"):
        return address[7:]
    if address.startswith("::") and address[2:].count(".") == 3:
        return address[2:]
    return address


def _summary(values):
    if not values:
        return None
    v = sorted(values)
    return {"n": len(v), "median": v[len(v) // 2], "max": v[-1], "mean": sum(v) / len(v)}


def _public(phase):
    return {k: v for k, v in phase.items() if k != "client"}


def _prompt_len(name, options):
    if name is None:
        return None
    m = re.fullmatch(r"p(\d+)", str(name))
    return int(m.group(1)) if m else None


def _slots(phase):
    if phase.get("mix"):
        return [(m["prompt"], m["tokens"]) for m in phase["mix"]]
    return [(phase.get("prompt"), phase.get("tokens"))] * int(phase.get("conc", 1)) if phase.get("prompt") else []
