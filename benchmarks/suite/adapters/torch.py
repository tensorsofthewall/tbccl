"""Integrations for PyTorch communication over torch-tbccl (and Gloo as a baseline): collectives, point-to-point, asynchronous Work handles and DistributedDataParallel.

The suite never imports PyTorch. Each point starts one wrapper process per rank (``torch_bench/comm_bench.py`` or ``ddp_bench.py``) with the torch environment's own
interpreter, configured per host as ``executables.torch_python`` (and ``executables.torch_bench_dir`` on a remote host). The wrappers print one JSON line per rank
(``TBCCL_TORCH_BENCH`` / ``TBCCL_TORCH_DDP``); the adapter parses only that contract, never console text.

Memory backends are torch device names: ``host`` (a CPU tensor), ``cuda`` and ``mps``. A pairing the capability audit does not support is rejected before anything runs; nothing is
moved to the CPU to make an unsupported operation work.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import BenchmarkAdapter, Capabilities, Group, Invocation, ParseError, Parsed, timing
from .core import PERCENTILE_LINEAR, peers

COMM_PREFIX, DDP_PREFIX = "TBCCL_TORCH_BENCH", "TBCCL_TORCH_DDP"
DEVICE_NAMES = {"host": "cpu", "cuda": "cuda", "mps": "mps"}
BACKENDS = ("tbccl", "gloo")
SIZE_FACTOR = {"all_reduce": 2, "all_reduce_async": 2, "all_gather": 2, "broadcast": 1, "p2p_ack": 1, "isend_irecv_ack": 1, "p2p_rtt": 2}


def _lines(path, prefix):
    out = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(prefix + " "):
            try:
                out.append(json.loads(line[len(prefix) + 1:]))
            except ValueError:
                raise ParseError(f"malformed {prefix} line: {line[:80]!r}")
    return out


def stats(values):
    v = sorted(float(x) for x in values)
    n = len(v)
    if n == 0:
        raise ParseError("no samples")

    def q(f):
        pos = f * (n - 1)
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        return v[lo] + (v[hi] - v[lo]) * (pos - lo)

    return {"median": q(0.5), "p25": q(0.25), "p75": q(0.75), "p95": q(0.95), "p99": q(0.99), "min": v[0], "max": v[-1], "mean": sum(v) / n}


def _versions(env):
    return {"torch": str(env.get("torch", "")).split("+")[0], "torch_tbccl": env.get("torch_tbccl"), "libtbccl": env.get("libtbccl"), "wire_protocol": env.get("wire_protocol"),
            "c_abi": env.get("c_abi"), "backend": env.get("backend")}


def _host_of(address):
    a = address.rsplit(":", 1)[0].strip("[]")
    return a[len("::ffff:"):] if a.startswith("::ffff:") else a


class _TorchBase(BenchmarkAdapter):
    script = "comm_bench.py"
    prefix = COMM_PREFIX

    def _device(self, spec):
        return DEVICE_NAMES[spec["memory_backend"]]

    def check_point(self, point, ctx):
        backend = point.options.get("backend", "tbccl")
        if backend not in BACKENDS:
            return f"unknown backend '{backend}' (tbccl or gloo)"
        for spec in point.ranks:
            if spec["memory_backend"] not in DEVICE_NAMES:
                return f"memory backend '{spec['memory_backend']}' is not a torch device (host, cuda, mps)"
        if backend == "gloo" and any(b != "host" for b in point.backends):
            point.skip_state = "unsupported"
            return "Gloo is run on CPU tensors only (no CUDA or MPS pairing is claimed)"
        if point.payload_bytes < 4 or point.payload_bytes % 4:
            return "payloads are float32 tensors: a positive multiple of 4 bytes"
        return None

    def validate_workload(self, workload, operations):
        errors = []
        opts = workload.get("options") or {}
        if opts.get("backend", "tbccl") not in BACKENDS:
            errors.append(f"options.backend must be one of {BACKENDS}")
        if opts.get("dtype", "float32") != "float32":
            errors.append("only float32 is benchmarked")
        return errors

    def _env(self, point, ctx, ports, rank, spec):
        backend = point.options.get("backend", "tbccl")
        host = ctx.config.hosts[spec["host"]]
        env = {"RANK": str(rank), "WORLD_SIZE": "2", "MASTER_ADDR": ctx.rank_address(point.ranks[0]), "MASTER_PORT": str(ports[0])}
        if backend == "tbccl":
            env["TBCCL_LOCAL_ENDPOINT"] = f"{ctx.rank_address(spec)}:0"
            if point.options.get("profile"):
                env["TORCH_TBCCL_TRACE"] = "1"
        return env

    def _script(self, ctx, spec):
        host = ctx.config.hosts[spec["host"]]
        python = ctx.tool("torch_python", spec["host"])
        if "torch_bench_dir" in host.executables:
            base = host.executables["torch_bench_dir"]
        elif host.is_local:
            base = str(Path(ctx.repo_root) / "benchmarks" / "suite" / "torch_bench")
        else:
            raise ParseError(f"remote host '{host.id}' needs executables.torch_bench_dir")
        return python, str(Path(base) / self.script)

    def _common_argv(self, point, ctx):
        addrs = ",".join(ctx.rank_address(s) for s in point.ranks)
        return ["--backend", point.options.get("backend", "tbccl"), "--devices", ",".join(self._device(s) for s in point.ranks), "--expect-addresses", addrs]

    def _rank_argv(self, point, ctx, rank):
        """Per-rank arguments: Gloo binds its transport to this rank's own data-link address (never an interface name, which may resolve to an IPv6 link-local address)."""
        return ["--gloo-address", ctx.rank_address(point.ranks[rank])] if point.options.get("backend", "tbccl") == "gloo" else []

    def _group(self, point, ctx, ports, extra_argv, roles=("rank0", "rank1")):
        invs = []
        for rank, spec in enumerate(point.ranks):
            python, script = self._script(ctx, spec)
            argv = [python, script] + extra_argv(rank) + self._rank_argv(point, ctx, rank)
            invs.append(Invocation(rank, roles[rank], spec["host"], argv, env=self._env(point, ctx, ports, rank, spec), wait_for_port=ports[0] if rank == 1 else None,
                                   listens_on=ports[0] if rank == 0 else None))
        return Group(invs, ports=list(ports))

    def _outputs(self, point, outcome):
        got = {}
        for rank in (0, 1):
            try:
                lines = _lines(outcome.stdout[rank], self.prefix)
            except (KeyError, OSError):
                raise ParseError(f"rank {rank} produced no output file")
            if len(lines) != 1:
                raise ParseError(f"rank {rank} printed {len(lines)} {self.prefix} line(s), expected exactly one")
            got[rank] = lines[0]
            if got[rank].get("rank") != rank:
                raise ParseError(f"rank {rank}'s output says it is rank {got[rank].get('rank')}")
        return got

    def _check_environment(self, point, got):
        envs = [_versions(got[r].get("environment") or {}) for r in (0, 1)]
        if point.options.get("backend", "tbccl") == "tbccl":
            for key in ("torch", "torch_tbccl", "libtbccl", "wire_protocol", "c_abi"):
                if envs[0][key] is None or envs[0][key] != envs[1][key]:
                    raise ParseError(f"version mismatch between ranks: {key} = {envs[0][key]!r} vs {envs[1][key]!r}")
        elif envs[0]["torch"] != envs[1]["torch"]:
            raise ParseError(f"torch versions differ between ranks: {envs[0]['torch']!r} vs {envs[1]['torch']!r}")
        for r in (0, 1):
            want = self._device(point.ranks[r])
            have = str(got[r].get("device", "")).split(":")[0]
            if have != want:
                raise ParseError(f"rank {r} ran on device '{have}', the configuration asked for '{want}'")
        return envs

    def _check_link(self, point, got):
        """Prove the traffic used the configured data link: every non-loopback peer address of either process is one of the two link addresses, and each process has at least one."""
        for r in (0, 1):
            conns = got[r].get("connections")
            if conns is None:
                raise ParseError(f"rank {r} could not list its TCP connections; the data path cannot be verified")
            allowed = set((got[r].get("expect_addresses") or "").split(","))
            peers_seen = {_host_of(c[1]) for c in conns}
            outside = sorted(h for h in peers_seen if not (h.startswith("127.") or h in ("::1", "localhost") or h in allowed))
            if outside:
                raise ParseError(f"rank {r} has TCP peers outside the data link: {outside}")
            if not any(h in allowed and not h.startswith("127.") for h in peers_seen) and not all(h.startswith("127.") for h in allowed):
                raise ParseError(f"rank {r} has no connection to a data-link address (saw {sorted(peers_seen)})")
        return {"verified": True, "peers": {r: sorted({_host_of(c[1]) for c in got[r]["connections"]}) for r in (0, 1)}}


class TorchCollectives(_TorchBase):
    OPS = {"all_reduce": ("pytorch_all_reduce_time_max_across_ranks", "completion_us"), "broadcast": ("pytorch_broadcast_time_max_across_ranks", "completion_us"),
           "all_gather": ("pytorch_all_gather_time_max_across_ranks", "completion_us")}
    caps = Capabilities(
        name="torch_collectives", category="collective", description="PyTorch collectives (blocking) over torch-tbccl or Gloo: all_reduce SUM, broadcast, all_gather; float32, 2 ranks",
        executable="python", operations=tuple(OPS), memory_backends=("host", "cuda", "mps"), min_ranks=2, max_ranks=2, executable_in_build=False, source_selectable=True,
        verification={op: "native" for op in OPS}, source_files=("benchmarks/suite/torch_bench/comm_bench.py",),
        notes="Framework benchmark, not a native TBCCL benchmark. Blocking dist.* calls; time = the call returning, per iteration, maximum over the two ranks; device_ready adds a measurement-only device synchronize.")

    def estimate_bytes(self, point):
        return point.payload_bytes * (point.iterations + point.warmup + 1) * SIZE_FACTOR[point.operation]

    def data_direction(self, point):
        if point.operation == "broadcast":
            return {"mode": "directional", "source_rank": point.source_rank or 0}
        return {"mode": "bidirectional", "source_rank": None}

    def describe(self, point):
        sem = {"all_reduce": "tensor bytes per rank (SUM, in place)", "broadcast": "tensor bytes broadcast from the source rank", "all_gather": "bytes contributed by each rank (output is 2x)"}[point.operation]
        return {"payload_semantics": sem, "backend": f"pytorch-{point.options.get('backend', 'tbccl')}",
                "timing_method": "perf_counter_ns around the dist call per iteration on each rank (barrier and source refill outside); the record's value is the per-iteration maximum over the two ranks; "
                                 "raw samples kept; percentiles by linear interpolation",
                "completion_semantics": "the blocking call returned on a rank; device_ready additionally includes a measurement-only torch.cuda/mps synchronize"}

    def plan(self, point, ctx, ports, rep_dir):
        def argv(rank):
            return (["--op", point.operation, "--bytes", str(point.payload_bytes), "--source", str(point.source_rank or 0), "--warmup", str(point.warmup), "--iterations", str(point.iterations),
                     "--timeout", str(min(point.timeout_s, 300))] + self._common_argv(point, ctx))
        return self._group(point, ctx, ports, argv)

    def parse(self, point, group, outcome):
        got = self._outputs(point, outcome)
        envs = self._check_environment(point, got)
        link = self._check_link(point, got)
        return self.measure(point, got, envs, link)

    def _primary_samples(self, point, got):
        key = self.OPS[point.operation][1]
        per = [got[r]["samples_us"].get(key) for r in (0, 1)]
        if not all(per) or len(per[0]) != len(per[1]):
            raise ParseError("the ranks reported different numbers of samples")
        return key, [max(a, b) for a, b in zip(*per)]

    def measure(self, point, got, envs, link):
        for r in (0, 1):
            if not got[r].get("verify_ok"):
                raise ParseError(f"rank {r} reports the result failed verification: {got[r].get('verify_detail')}")
            if got[r].get("bytes") != point.payload_bytes or got[r].get("op") != point.operation or got[r].get("dtype") != "float32" or got[r].get("shape") != [point.payload_bytes // 4]:
                raise ParseError(f"rank {r} ran a different operation, size, dtype or shape than requested")
        key, mx = self._primary_samples(point, got)
        metric = self.OPS[point.operation][0]
        tm = timing(metric, stats(mx), "computed", "median", percentile_method=PERCENTILE_LINEAR, sample_count=len(mx), samples=mx,
                    per_rank=[{"rank": r, "statistics": stats(got[r]["samples_us"][key]), "sample_count": len(got[r]["samples_us"][key]), "samples_us": got[r]["samples_us"],
                               "cpu_cores": got[r]["cpu_seconds"] / got[r]["wall_seconds"] if got[r].get("wall_seconds") else None, "device": got[r]["device"]} for r in (0, 1)])
        ready = [max(a, b) for a, b in zip(got[0]["samples_us"]["device_ready_us"], got[1]["samples_us"]["device_ready_us"])]
        extra = {"device_ready_median_us": {"value": stats(ready)["median"], "unit": "us", "note": "completion plus a measurement-only device synchronize (maximum over ranks)"},
                 "dev_sync_median_us_rank0": {"value": stats(got[0]["samples_us"]["dev_sync_us"])["median"], "unit": "us", "note": "the measurement synchronize on rank 0"},
                 "dev_sync_median_us_rank1": {"value": stats(got[1]["samples_us"]["dev_sync_us"])["median"], "unit": "us"},
                 "link_peers_rank0": {"value": ",".join(link["peers"][0]), "unit": "addresses"}, "link_peers_rank1": {"value": ",".join(link["peers"][1]), "unit": "addresses"}}
        throughput = [{"kind": "application_payload", "value": point.payload_bytes / tm["statistics"]["median"] / 1e3, "unit": "GB/s",
                       "definition": "tensor bytes / median per-iteration maximum completion time over the two ranks; decimal GB/s; one tensor per call"}]
        warnings = [f"{len(mx)} iterations: p95/p99 are not meaningful tail estimates"] if len(mx) <= 100 else []
        ver = {"status": "passed", "method": "native wrapper: one untimed round after the loop compared bit for bit with the expected pattern on each rank, device preserved", "details": "both ranks verify_ok"}
        return Parsed({"timing": tm, "throughput": throughput, "extra": extra, "warnings": warnings}, ver, configuration_extra={"torch_environment": envs[0], "datatype_checked": "float32", "tensor_shape": got[0]["shape"]})


class TorchP2P(_TorchBase):
    OPS = {"p2p_ack": ("pytorch_send_recv_completion_confirmed_time", "completion_confirmed_us"), "p2p_rtt": ("pytorch_ping_pong_round_trip_time", "round_trip_us")}
    caps = Capabilities(
        name="torch_p2p", category="p2p", description="PyTorch send/recv over torch-tbccl or Gloo: completion-confirmed one-way transfer (1-byte CPU ACK) and ping-pong round trip; float32",
        executable="python", operations=tuple(OPS), memory_backends=("host", "cuda", "mps"), min_ranks=2, max_ranks=2, executable_in_build=False, source_selectable=True,
        verification={op: "native" for op in OPS}, source_files=("benchmarks/suite/torch_bench/comm_bench.py",),
        notes="p2p_ack matches tbccl_tensor_transfer_bench end-to-end semantics (the source's time includes the ACK's return trip); p2p_rtt matches small_message_latency p2p.")

    def estimate_bytes(self, point):
        return point.payload_bytes * (point.iterations + point.warmup + 1) * SIZE_FACTOR[point.operation]

    def data_direction(self, point):
        if point.operation == "p2p_rtt":
            return {"mode": "bidirectional", "source_rank": point.source_rank or 0}
        return {"mode": "directional", "source_rank": point.source_rank or 0}

    def describe(self, point):
        if point.operation == "p2p_rtt":
            return {"payload_semantics": "bytes sent each way; the source sends, the other rank echoes", "backend": f"pytorch-{point.options.get('backend', 'tbccl')}",
                    "timing_method": "perf_counter_ns on the source around dist.send + dist.recv of the echo; raw samples kept; linear-interpolation percentiles",
                    "completion_semantics": "the echoed tensor was received on the source (device_ready adds a measurement synchronize)"}
        return {"payload_semantics": "tensor bytes sent from the source rank to the other rank", "backend": f"pytorch-{point.options.get('backend', 'tbccl')}",
                "timing_method": "perf_counter_ns on the source from before dist.send until the 1-byte CPU ACK from the destination was received; the source tensor is refilled and the device "
                                 "synchronized before the timer; raw samples kept; linear-interpolation percentiles",
                "completion_semantics": "the destination's recv() returned and its ACK reached the source (includes the ACK's return trip; not a one-way latency)"}

    def plan(self, point, ctx, ports, rep_dir):
        def argv(rank):
            return (["--op", point.operation, "--bytes", str(point.payload_bytes), "--source", str(point.source_rank or 0), "--warmup", str(point.warmup), "--iterations", str(point.iterations),
                     "--timeout", str(min(point.timeout_s, 300))] + self._common_argv(point, ctx))
        return self._group(point, ctx, ports, argv, roles=("source", "destination") if (point.source_rank or 0) == 0 else ("destination", "source"))

    def parse(self, point, group, outcome):
        got = self._outputs(point, outcome)
        envs = self._check_environment(point, got)
        link = self._check_link(point, got)
        return self.measure(point, got, envs, link)

    def measure(self, point, got, envs, link):
        src = point.source_rank or 0
        for r in (0, 1):
            if not got[r].get("verify_ok"):
                raise ParseError(f"rank {r} reports the result failed verification: {got[r].get('verify_detail')}")
            if got[r].get("bytes") != point.payload_bytes or got[r].get("op") != point.operation or got[r].get("source") != src:
                raise ParseError(f"rank {r} ran a different operation, size or source than requested")
        metric, key = self.OPS[point.operation]
        s = got[src]["samples_us"].get(key)
        if not s:
            raise ParseError("the source rank reported no samples")
        tm = timing(metric, stats(s), "computed", "median", percentile_method=PERCENTILE_LINEAR, sample_count=len(s), samples=s,
                    per_rank=[{"rank": r, "statistics": {"median": stats(got[r]["samples_us"][key])["median"]} if got[r]["samples_us"].get(key) else {"median": None}, "sample_count": a_len(got[r]["samples_us"]),
                               "samples_us": got[r]["samples_us"], "cpu_cores": got[r]["cpu_seconds"] / got[r]["wall_seconds"] if got[r].get("wall_seconds") else None, "device": got[r]["device"]} for r in (0, 1)])
        extra = {}
        for name in ("send_us", "dev_sync_us", "device_ready_us"):
            if got[src]["samples_us"].get(name):
                extra[f"source_{name[:-3]}_median_us"] = {"value": stats(got[src]["samples_us"][name])["median"], "unit": "us"}
        dst = 1 - src
        for name in ("recv_us", "dev_sync_us"):
            if got[dst]["samples_us"].get(name):
                extra[f"destination_{name[:-3]}_median_us"] = {"value": stats(got[dst]["samples_us"][name])["median"], "unit": "us", "note": "destination clock; waiting time included, not a transfer time"}
        extra["link_peers_rank0"] = {"value": ",".join(link["peers"][0]), "unit": "addresses"}
        extra["link_peers_rank1"] = {"value": ",".join(link["peers"][1]), "unit": "addresses"}
        device = any(b != "host" for b in point.backends)
        factor = 2 if point.operation == "p2p_rtt" else 1
        throughput = [{"kind": "device_to_device_effective" if device else "application_payload", "value": factor * point.payload_bytes / tm["statistics"]["median"] / 1e3, "unit": "GB/s",
                       "definition": f"{'2 x ' if factor == 2 else ''}tensor bytes / median {'round-trip' if factor == 2 else 'completion-confirmed'} time on the source (includes the {'echo' if factor == 2 else 'ACK return trip'}); decimal GB/s"}]
        warnings = [f"{len(s)} iterations: p95/p99 are not meaningful tail estimates"] if len(s) <= 100 else []
        ver = {"status": "passed", "method": "native wrapper: one untimed transfer after the loop compared bit for bit on the receiving rank, device preserved", "details": "both ranks verify_ok"}
        return Parsed({"timing": tm, "throughput": throughput, "extra": extra, "warnings": warnings}, ver, configuration_extra={"torch_environment": envs[0], "datatype_checked": "float32", "tensor_shape": got[0]["shape"]})


def a_len(samples):
    return max((len(v) for v in samples.values()), default=0)


class TorchAsync(_TorchBase):
    OPS = {"all_reduce_async": ("pytorch_async_all_reduce_lifetime_max_across_ranks", "lifetime_us"), "isend_irecv_ack": ("pytorch_isend_irecv_completion_confirmed_time", "completion_confirmed_us")}
    caps = Capabilities(
        name="torch_async", category="overlap", description="PyTorch asynchronous communication: all_reduce(async_op=True) and isend/irecv with Work.wait(); submission, wait and completion kept apart",
        executable="python", operations=tuple(OPS), memory_backends=("host", "cuda", "mps"), min_ranks=2, max_ranks=2, executable_in_build=False, source_selectable=True,
        verification={op: "native" for op in OPS}, source_files=("benchmarks/suite/torch_bench/comm_bench.py",),
        notes="Submission time is the call returning with a Work handle, not completion. all_reduce_async lifetime = submit start until Work.wait() returned; the Work's completion callback timestamp is recorded when the backend supports futures.")

    def estimate_bytes(self, point):
        return point.payload_bytes * (point.iterations + point.warmup + 1) * SIZE_FACTOR[point.operation]

    def data_direction(self, point):
        if point.operation == "isend_irecv_ack":
            return {"mode": "directional", "source_rank": point.source_rank or 0}
        return {"mode": "bidirectional", "source_rank": None}

    def describe(self, point):
        if point.operation == "all_reduce_async":
            return {"payload_semantics": "tensor bytes per rank (SUM)", "backend": f"pytorch-{point.options.get('backend', 'tbccl')}",
                    "timing_method": "perf_counter_ns: submit = dist.all_reduce(async_op=True) returned; wait = time inside Work.wait(); lifetime = submit start until wait returned (maximum over ranks); "
                                     "callback = a done-callback on the Work's future stamped by the backend's completion worker; raw samples kept",
                    "completion_semantics": "Work.wait() returned on the rank; a returned handle (submission) is never counted as completion"}
        return {"payload_semantics": "tensor bytes sent from the source rank to the other rank", "backend": f"pytorch-{point.options.get('backend', 'tbccl')}",
                "timing_method": "perf_counter_ns on the source from before dist.isend until the 1-byte CPU ACK was received; submit = isend returned, wait = time in Work.wait(); raw samples kept",
                "completion_semantics": "the destination's irecv Work completed and its ACK reached the source (includes the ACK's return trip)"}

    def plan(self, point, ctx, ports, rep_dir):
        def argv(rank):
            return (["--op", point.operation, "--bytes", str(point.payload_bytes), "--source", str(point.source_rank or 0), "--warmup", str(point.warmup), "--iterations", str(point.iterations),
                     "--timeout", str(min(point.timeout_s, 300))] + self._common_argv(point, ctx))
        return self._group(point, ctx, ports, argv)

    def parse(self, point, group, outcome):
        got = self._outputs(point, outcome)
        envs = self._check_environment(point, got)
        link = self._check_link(point, got)
        return self.measure(point, got, envs, link)

    def measure(self, point, got, envs, link):
        for r in (0, 1):
            if not got[r].get("verify_ok"):
                raise ParseError(f"rank {r} reports the result failed verification: {got[r].get('verify_detail')}")
            if got[r].get("bytes") != point.payload_bytes or got[r].get("op") != point.operation:
                raise ParseError(f"rank {r} ran a different operation or size than requested")
        metric, key = self.OPS[point.operation]
        src = point.source_rank or 0
        extra = {}
        if point.operation == "all_reduce_async":
            per = [got[r]["samples_us"].get(key) for r in (0, 1)]
            if not all(per) or len(per[0]) != len(per[1]):
                raise ParseError("the ranks reported different numbers of samples")
            s = [max(a, b) for a, b in zip(*per)]
            for name in ("submit_us", "wait_us", "device_ready_us", "dev_sync_us"):
                for r in (0, 1):
                    if got[r]["samples_us"].get(name):
                        extra[f"{name[:-3]}_median_us_rank{r}"] = {"value": stats(got[r]["samples_us"][name])["median"], "unit": "us"}
            for r in (0, 1):
                cb = got[r]["samples_us"].get("callback_us")
                extra[f"callbacks_seen_rank{r}"] = {"value": got[r]["flags"]["callback_seen"], "unit": f"of {got[r]['iterations']} iterations"}
                if cb:
                    extra[f"callback_median_us_rank{r}"] = {"value": stats(cb)["median"], "unit": "us", "note": "a done-callback on the Work's future; stamped by the backend's completion worker"}
                extra[f"complete_at_submit_rank{r}"] = {"value": got[r]["flags"]["done_at_submit"], "unit": f"of {got[r]['iterations']} iterations"}
        else:
            s = got[src]["samples_us"].get(key)
            if not s:
                raise ParseError("the source rank reported no samples")
            for name in ("submit_us", "wait_us", "send_lifetime_us"):
                if got[src]["samples_us"].get(name):
                    extra[f"source_{name[:-3]}_median_us"] = {"value": stats(got[src]["samples_us"][name])["median"], "unit": "us"}
            dst = 1 - src
            for name in ("recv_us", "submit_us", "wait_us", "dev_sync_us"):
                if got[dst]["samples_us"].get(name):
                    extra[f"destination_{name[:-3]}_median_us"] = {"value": stats(got[dst]["samples_us"][name])["median"], "unit": "us", "note": "destination clock; waiting included"}
        tm = timing(metric, stats(s), "computed", "median", percentile_method=PERCENTILE_LINEAR, sample_count=len(s), samples=s,
                    per_rank=[{"rank": r, "statistics": {"median": stats(got[r]["samples_us"][key])["median"]} if got[r]["samples_us"].get(key) else {"median": None}, "sample_count": a_len(got[r]["samples_us"]),
                               "samples_us": got[r]["samples_us"], "cpu_cores": got[r]["cpu_seconds"] / got[r]["wall_seconds"] if got[r].get("wall_seconds") else None, "device": got[r]["device"]} for r in (0, 1)])
        extra["link_peers_rank0"] = {"value": ",".join(link["peers"][0]), "unit": "addresses"}
        extra["link_peers_rank1"] = {"value": ",".join(link["peers"][1]), "unit": "addresses"}
        device = any(b != "host" for b in point.backends)
        throughput = [{"kind": "device_to_device_effective" if device else "application_payload", "value": point.payload_bytes / tm["statistics"]["median"] / 1e3, "unit": "GB/s",
                       "definition": "tensor bytes / median " + ("lifetime (submit to Work.wait() return, max over ranks)" if point.operation == "all_reduce_async" else "completion-confirmed time on the source (includes the ACK return trip)") + "; decimal GB/s"}]
        warnings = ["the submission time is not a communication time"] + ([f"{len(s)} iterations: p95/p99 are not meaningful tail estimates"] if len(s) <= 100 else [])
        ver = {"status": "passed", "method": "native wrapper: one untimed operation after the loop compared bit for bit, device preserved", "details": "both ranks verify_ok"}
        return Parsed({"timing": tm, "throughput": throughput, "extra": extra, "warnings": warnings}, ver, configuration_extra={"torch_environment": _versions(got[0]["environment"]), "datatype_checked": "float32", "tensor_shape": got[0]["shape"]})


class TorchDDP(_TorchBase):
    script = "ddp_bench.py"
    prefix = DDP_PREFIX
    caps = Capabilities(
        name="torch_ddp", category="application", description="Two-rank DistributedDataParallel training of a synthetic deep MLP over torch-tbccl or Gloo: step time, phases, exposed communication, samples/s",
        executable="python", operations=("train",), memory_backends=("host", "cuda", "mps"), min_ranks=2, max_ranks=2, executable_in_build=False,
        verification={"train": "native"}, source_files=("benchmarks/suite/torch_bench/ddp_bench.py",),
        notes="payload_bytes is the gradient bytes per step (layers*(width^2+width)*4) and must equal sizes[0]. Options: layers, width, batch (per rank), bucket_cap_mb, steps, profile, tag. "
              "Correctness (CPU reference, finite values, cross-rank agreement) is checked before the timed phases and is never timed.")

    def estimate_bytes(self, point):
        steps = int(point.options.get("steps", point.iterations))
        profile = int(point.options.get("profile_steps", 20)) if point.options.get("profile") else 0
        return point.payload_bytes * (2 * (steps + point.warmup) + int(point.options.get("correctness_steps", 3)) + profile) * 2  # free + phased loops communicate; no_sync does not

    def data_direction(self, point):
        return {"mode": "bidirectional", "source_rank": None}

    def validate_workload(self, workload, operations):
        errors = super().validate_workload(workload, operations)
        o = workload.get("options") or {}
        for k in ("layers", "width", "batch", "tag"):
            if k not in o:
                errors.append(f"torch_ddp needs options.{k}")
        if all(k in o for k in ("layers", "width")):
            grad = int(o["layers"]) * (int(o["width"]) ** 2 + int(o["width"])) * 4
            sizes = workload.get("sizes")
            if sizes != [grad]:
                errors.append(f"sizes must be [{grad}] (the gradient bytes per step of {o['layers']} x Linear({o['width']}, {o['width']}))")
        return errors

    def check_point(self, point, ctx):
        reason = super().check_point(point, ctx)
        if reason:
            return reason
        return None

    def describe(self, point):
        return {"payload_semantics": "gradient bytes per step (float32 parameters of the model; one all-reduce per DDP bucket)", "backend": f"pytorch-ddp-{point.options.get('backend', 'tbccl')}",
                "timing_method": "per step on each rank's own perf_counter_ns, device synchronized once at the end of the step (free loop); the record's value is the per-step maximum over the two ranks "
                                 "(step i is delimited by the same gradient all-reduces on both ranks, so the slower rank is the critical path); phases come from a second loop with synchronization after "
                                 "forward, backward and optimizer; a third loop under no_sync() gives each rank's compute-only time",
                "completion_semantics": "the optimizer step finished and the device is synchronized on the rank"}

    def plan(self, point, ctx, ports, rep_dir):
        o = point.options

        def argv(rank):
            a = ["--layers", str(o["layers"]), "--width", str(o["width"]), "--batch", str(o["batch"]), "--bucket-cap-mb", str(o.get("bucket_cap_mb", 25)), "--steps", str(o.get("steps", point.iterations)),
                 "--warmup", str(point.warmup), "--correctness-steps", str(o.get("correctness_steps", 3)), "--timeout", str(min(point.timeout_s, 600))] + self._common_argv(point, ctx)
            if o.get("profile"):
                a += ["--profile", "--profile-steps", str(o.get("profile_steps", 20))]
            return a

        return self._group(point, ctx, ports, argv)

    def parse(self, point, group, outcome):
        got = self._outputs(point, outcome)
        envs = self._check_environment(point, got)
        link = self._check_link(point, got)
        return self.measure(point, got, envs, link)

    def measure(self, point, got, envs, link):
        o = point.options
        for r in (0, 1):
            d = got[r]
            if d.get("failed"):
                raise ParseError(f"rank {r}: {d['failed']} failed: {json.dumps(d.get('correctness'))[:300]}")
            if not (d.get("correctness") or {}).get("passed"):
                raise ParseError(f"rank {r} reports the correctness checks failed")
            if (d.get("layers"), d.get("width"), d.get("batch")) != (int(o["layers"]), int(o["width"]), int(o["batch"])) or d.get("grad_bytes_per_step") != point.payload_bytes:
                raise ParseError(f"rank {r} trained a different model or batch than requested")
        a_, b_ = got[0]["samples_us"], got[1]["samples_us"]
        if len(a_["step_us"]) != len(b_["step_us"]):
            raise ParseError("the ranks reported different numbers of steps")
        step = [max(x, y) for x, y in zip(a_["step_us"], b_["step_us"])]
        st = stats(step)
        tm = timing("ddp_step_time_slowest_rank", st, "computed", "median", percentile_method=PERCENTILE_LINEAR, sample_count=len(step), samples=step,
                    per_rank=[{"rank": r, "statistics": stats(got[r]["samples_us"]["step_us"]), "sample_count": len(got[r]["samples_us"]["step_us"]), "samples_us": got[r]["samples_us"],
                               "cpu_cores": got[r]["cpu_seconds"] / got[r]["wall_seconds"] if got[r].get("wall_seconds") else None, "device": got[r]["device"]} for r in (0, 1)])
        gb = got[0]["global_batch"]
        throughput = [{"kind": "training_samples_per_second", "value": gb / (st["median"] / 1e6), "unit": "samples/s",
                       "definition": f"global batch ({gb} samples = 2 ranks x {got[0]['batch']}) / median per-step maximum over ranks (the free loop)"},
                      {"kind": "training_steps_per_second", "value": 1e6 / st["median"], "unit": "steps/s", "definition": "1 / median per-step maximum over ranks (the free loop)"}]
        extra = {"model": {"value": got[0]["model"], "unit": "id"}, "parameters": {"value": got[0]["parameters"], "unit": "count"},
                 "grad_bytes_per_step": {"value": got[0]["grad_bytes_per_step"], "unit": "B", "note": "float32 gradients all-reduced once per step (the communication payload; one DDP all-reduce per bucket)"},
                 "bucket_cap_mb": {"value": got[0]["bucket_cap_mb"], "unit": "MB"}, "batch_per_rank": {"value": got[0]["batch"], "unit": "samples"},
                 "ddp_bucket_sizes": {"value": str(got[0]["ddp_logging"].get("rebuilt_bucket_sizes") or got[0]["ddp_logging"].get("bucket_sizes")), "unit": "bytes per bucket (rank 0 logging data)"},
                 "link_peers_rank0": {"value": ",".join(link["peers"][0]), "unit": "addresses"}, "link_peers_rank1": {"value": ",".join(link["peers"][1]), "unit": "addresses"}}
        med = {}
        for r in (0, 1):
            s = got[r]["samples_us"]
            for name in ("forward_us", "backward_us", "optimizer_us", "phased_step_us", "nosync_forward_us", "nosync_backward_us", "nosync_optimizer_us", "nosync_step_us"):
                med[(r, name)] = stats(s[name])["median"]
                extra[f"rank{r}_{name[:-3]}_median_us"] = {"value": med[(r, name)], "unit": "us"}
            extra[f"rank{r}_exposed_communication_inferred_us"] = {"value": max(0.0, med[(r, "backward_us")] - med[(r, "nosync_backward_us")]), "unit": "us",
                                                                   "note": "median backward with DDP minus median backward under no_sync() in the same process: an inference from two loops, not a measured interval"}
            extra[f"rank{r}_peak_memory_bytes"] = {"value": got[r]["peak_memory_bytes"], "unit": "B", "note": "cuda: max_memory_allocated; mps: driver_allocated_memory; cpu: process peak RSS"}
            extra[f"rank{r}_cpu_cores"] = {"value": got[r]["cpu_seconds"] / got[r]["wall_seconds"], "unit": "busy cores (process CPU time / wall time, free loop)"}
            extra[f"rank{r}_device"] = {"value": got[r]["device"], "unit": "device"}
        lo, hi = sorted((med[(0, "nosync_step_us")], med[(1, "nosync_step_us")]))
        extra["compute_imbalance_ratio"] = {"value": hi / lo if lo else None, "unit": "ratio", "note": "slower rank's compute-only step (no_sync) over the faster rank's"}
        extra["slowest_compute_rank"] = {"value": 0 if med[(0, "nosync_step_us")] >= med[(1, "nosync_step_us")] else 1, "unit": "rank"}
        for r in (0, 1):
            pr = got[r].get("profile")
            if pr:
                extra[f"rank{r}_profile"] = {"value": json.dumps({k: v for k, v in pr.items() if k not in ("gpu_utilization_percent", "gpu_temperature_c")}, separators=(",", ":")), "unit": "json",
                                             "note": "traced steps (a separate loop; hooks and tracing add overhead): application-visible events of this process only"}
                if pr.get("gpu_utilization_percent"):
                    extra[f"rank{r}_gpu_utilization_median_percent"] = {"value": stats(pr["gpu_utilization_percent"])["median"], "unit": "%", "note": f"nvidia-smi sampled every 0.5 s during the profile loop only ({len(pr['gpu_utilization_percent'])} samples)"}
        c = got[0]["correctness"]
        extra["correctness_max_grad_rel_err"] = {"value": max(got[r]["correctness"]["max_grad_rel_err"] for r in (0, 1)), "unit": "relative (max|diff| / max|reference|)"}
        extra["correctness_max_param_rel_err"] = {"value": max(got[r]["correctness"]["max_param_rel_err"] for r in (0, 1)), "unit": "relative"}
        extra["correctness_cross_rank_max_abs_diff"] = {"value": c["cross_rank_max_abs_diff"], "unit": "absolute (float32 parameters)"}
        warnings = ["critical path = per-step maximum over ranks of each rank's own step duration (steps are delimited by the gradient all-reduces on both ranks); clocks of different hosts are never compared",
                    "exposed communication is inferred from two loops of the same process"]
        ver = {"status": "passed", "method": "native wrapper: before timing, steps compared with a CPU reference (gradients, parameters, finite loss and gradients, constructor sync) and cross-rank parameter agreement",
               "details": f"max grad rel err {extra['correctness_max_grad_rel_err']['value']:.2e}, max param rel err {extra['correctness_max_param_rel_err']['value']:.2e}, cross-rank {c['cross_rank_max_abs_diff']:.2e}"}
        return Parsed({"timing": tm, "throughput": throughput, "extra": extra, "warnings": warnings}, ver,
                      configuration_extra={"torch_environment": envs[0], "model": {"id": got[0]["model"], "layers": got[0]["layers"], "width": got[0]["width"], "batch_per_rank": got[0]["batch"],
                                                                                    "bucket_cap_mb": got[0]["bucket_cap_mb"], "seed": got[0]["seed"], "lr": got[0]["lr"], "dtype": "float32", "optimizer": "SGD"},
                                           "steps": got[0]["steps"]})
