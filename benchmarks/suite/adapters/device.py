"""Integrations for the device-memory benchmarks: local staging, the chunked async transfer, CPU and CUDA compute overlap and the heterogeneous all-reduce.

Every adapter calls the existing executable with its existing command line and parses its existing output; no timing or communication lives here.
Where a tool prints only a few statistics the record keeps exactly those, and the adapter says what each figure is and is not.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .base import BenchmarkAdapter, Capabilities, Group, Invocation, ParseError, Parsed, timing
from .core import PERCENTILE_LINEAR, _number, peers, provider_for

GIB_TO_GB = 1.073741824  # 1 GiB = 1.073741824 GB (decimal); the only place a binary figure is converted
ASYNC_BACKENDS = ("host", "cuda-pageable", "cuda-pinned", "cuda-chunked", "metal-shared", "metal-private-staged")
STAGING_BACKENDS = ("host", "cuda-pageable", "cuda-pinned", "metal-shared", "metal-private-staged")


def _json_object(path):
    text = Path(path).read_text()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ParseError("the output holds no JSON object")
    try:
        return json.loads(text[start:end + 1])
    except ValueError as error:
        raise ParseError(f"malformed JSON output: {error}")


def _flag(obj, key):
    if key not in obj:
        raise ParseError(f"the output has no '{key}' field")
    return obj[key] is True


def _two_rank_invocations(adapter, point, ctx, ports, argv_for, roles=("sender", "receiver")):
    peer_arg = peers(point, ctx, ports)
    invs = []
    for rank, spec in enumerate(point.ranks):
        role = roles[0] if rank == (point.source_rank or 0) else roles[1]
        invs.append(Invocation(rank, role, spec["host"], [ctx.executable(adapter.caps.executable, spec["host"])] + argv_for(rank, spec, peer_arg),
                               wait_for_port=ports[0] if rank == 1 else None, listens_on=ports[0] if rank == 0 else None))
    return Group(invs, ports=list(ports))


def _check_providers(point, ctx):
    for spec in point.ranks:
        have = ctx.providers(spec["host"]) if ctx else None
        if have is not None and provider_for(spec["memory_backend"]) not in have:
            point.skip_state = "unsupported"
            return f"memory backend '{spec['memory_backend']}' is not available in the executable's build on host '{spec['host']}' (providers: {', '.join(sorted(have)) or 'none'})"
    return None


class TensorStaging(BenchmarkAdapter):
    OPS = {"staging_d2h": ("d2h", "device_to_host_staging_time"), "staging_h2d": ("h2d", "host_to_device_staging_time")}
    caps = Capabilities(
        name="tensor_staging", category="memory_backend", description="Local device staging only (single process, no network): device-to-host and host-to-device staging time",
        executable="tbccl_tensor_transfer_bench", operations=tuple(OPS), memory_backends=STAGING_BACKENDS, min_ranks=1, max_ranks=1,
        verification={"staging_d2h": "native", "staging_h2d": "native"}, source_files=("benchmarks/tensor_transfer_bench.cpp",),
        notes="tbccl_tensor_transfer_bench --mode staging-only. One process prints a d2h row and an h2d row per size; this adapter keeps one of them per point. Allocation is outside the timed region.")

    def data_direction(self, point):
        return {"mode": "local", "source_rank": None}

    def estimate_bytes(self, point):
        return 0

    def check_point(self, point, ctx):
        return _check_providers(point, ctx)

    def describe(self, point):
        stage, _ = self.OPS[point.operation]
        what = "device memory to the host staging buffer" if stage == "d2h" else "the host staging buffer to device memory"
        return {"payload_semantics": f"bytes staged from {what} inside one process; no network traffic", "backend": "tbccl-tensor-backend",
                "timing_method": "steady_clock around one stage_" + ("device_to_host" if stage == "d2h" else "host_to_device") + "() call per iteration; allocation, source generation and verification are outside the timer; "
                                 "percentiles by linear interpolation",
                "completion_semantics": "the backend's staging call has returned (a device backend synchronizes inside that call; it is the backend's own definition of complete)"}

    def plan(self, point, ctx, ports, rep_dir):
        spec = point.ranks[0]
        argv = [ctx.executable(self.caps.executable, spec["host"]), "--mode", "staging-only", "--local-backend", spec["memory_backend"], "--sizes", str(point.payload_bytes),
                "--warmup", str(point.warmup), "--iterations", str(point.iterations)]
        return Group([Invocation(0, "local", spec["host"], argv)], ports=list(ports))

    def parse(self, point, group, outcome):
        stage, metric = self.OPS[point.operation]
        lines = [line for line in Path(outcome.stdout[0]).read_text().splitlines() if line and not line.startswith("[")]
        rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
        row = next((r for r in rows if r.get("mode") == "staging-only" and r.get("stage") == stage and r.get("payload_bytes", "").isdigit() and int(r["payload_bytes"]) == point.payload_bytes), None)
        if row is None:
            raise ParseError("the tool printed no staging row for the requested stage and payload")
        stats = {"median": _number(row, "median_us"), "p95": _number(row, "p95_us"), "p99": _number(row, "p99_us"), "min": _number(row, "min_us"), "max": _number(row, "max_us")}
        n = int(_number(row, "iterations"))
        if n < 1:
            raise ParseError("the staging row reports no samples")
        tm = timing(metric, stats, "reported", "median", percentile_method=PERCENTILE_LINEAR, sample_count=n)
        throughput = [{"kind": "application_payload", "value": _number(row, "effective_GBps"), "unit": "GB/s", "definition": "payload bytes / median staging time; decimal GB/s; local staging only"}]
        warnings = []
        if stats["median"] < 0.1:
            throughput = []
            warnings.append("median staging time is below 0.1 us, about the cost of reading the clock (a host or shared-memory backend stages by no copy); no throughput is derived from it")
        extra = {"allocation_count": {"value": _number(row, "allocation_count"), "unit": "count"}, "staging_capacity_bytes": {"value": _number(row, "staging_capacity_bytes"), "unit": "bytes"}}
        if n < 100:
            warnings.append(f"{n} iterations: p95/p99 are not meaningful tail estimates")
        ver = {"status": "passed", "method": "native: source and destination verified once, untimed, on the last iteration", "details": "exit status 0"}
        return Parsed({"timing": tm, "throughput": throughput, "extra": extra, "warnings": warnings}, ver)


class AsyncTransfer(BenchmarkAdapter):
    caps = Capabilities(
        name="async_transfer", category="p2p", description="Chunked asynchronous tensor transfer: submission (enqueue) time and ACK-confirmed completion time",
        executable="tbccl_async_transfer_bench", operations=("transfer",), memory_backends=ASYNC_BACKENDS, min_ranks=2, max_ranks=2, source_selectable=True,
        verification={"transfer": "native"}, source_files=("benchmarks/async_transfer_bench.cpp",),
        notes="Options: chunk_bytes (0 = one chunk), depth (pipeline depth), metal_async_path (adapter|direct, metal-shared only), verify. The source prints enqueue median and completion statistics with raw completion samples.")

    def source(self, point):
        return point.source_rank or 0

    def data_direction(self, point):
        return {"mode": "directional", "source_rank": self.source(point)}

    def estimate_bytes(self, point):
        return point.payload_bytes * (point.iterations + point.warmup + 1)

    def check_point(self, point, ctx):
        return _check_providers(point, ctx)

    def describe(self, point):
        return {"payload_semantics": "tensor bytes moved from the source rank's backend to the destination rank's backend in chunks over the async worker", "backend": "tbccl-async-substrate",
                "timing_method": "enqueue time = time for TensorCommWorker::enqueue() to return; completion time = enqueue start until the destination's 1-byte ACK is received by the source "
                                 "(the work's wait() plus the ACK round trip); source generation is outside both; percentiles by the tool's own method; raw completion samples kept",
                "completion_semantics": "the destination has committed the tensor and its ACK has been observed by the source; enqueue time is submission only and does not mean the transfer finished"}

    def plan(self, point, ctx, ports, rep_dir):
        opts = point.options

        def argv_for(rank, spec, peer_arg):
            argv = ["--rank", str(rank), "--peers", peer_arg, "--source-rank", str(self.source(point)), "--bytes", str(point.payload_bytes), "--chunk-bytes", str(opts.get("chunk_bytes", 0)),
                    "--depth", str(opts.get("depth", 2)), "--warmup", str(point.warmup), "--iterations", str(point.iterations), "--backend", spec["memory_backend"], "--label", "suite"]
            if spec["memory_backend"] == "metal-shared" and opts.get("metal_async_path"):
                argv += ["--metal-async-path", str(opts["metal_async_path"])]
            return argv

        return _two_rank_invocations(self, point, ctx, ports, argv_for)

    def parse(self, point, group, outcome):
        obj = _json_object(outcome.stdout[self.source(point)])
        if obj.get("role") != "sender":
            raise ParseError("the source rank did not report as the sender")
        samples = obj.get("completion_samples_us")
        if not isinstance(samples, list) or not samples or not all(isinstance(v, (int, float)) and v >= 0 for v in samples):
            raise ParseError("the sender reported no valid completion samples")
        stats = {"median": _number(obj, "completion_median_us"), "p95": _number(obj, "completion_p95_us"), "p99": _number(obj, "completion_p99_us"), "max": _number(obj, "completion_max_us"),
                 "mean": _number(obj, "completion_mean_us"), "min": float(min(samples))}
        if stats["median"] <= 0:
            raise ParseError("the sender reports a non-positive median completion time")
        tm = timing("completion_confirmed_time", stats, {k: ("computed" if k == "min" else "reported") for k in stats}, "median", percentile_method="tool-defined (see the tool's percentile())",
                    sample_count=len(samples), samples=[float(v) for v in samples])
        gib = _number(obj, "gib_per_s")
        device = any(b != "host" for b in point.backends)
        throughput = [{"kind": "device_to_device_effective" if device else "application_payload", "value": gib * GIB_TO_GB, "unit": "GB/s",
                       "definition": "payload bytes / MEAN completion-confirmed time (includes the ACK return trip); the tool prints GiB/s and the suite converts x1.073741824 to decimal GB/s"}]
        extra = {"enqueue_median_us": {"value": _number(obj, "enqueue_median_us"), "unit": "us", "note": "submission only: time for enqueue() to return; not a transfer time; the tool prints no other enqueue statistic"},
                 "gib_per_s_native": {"value": gib, "unit": "GiB/s", "note": "as printed by the tool: payload / mean completion time, binary GiB"},
                 "chunk_bytes": {"value": _number(obj, "chunk_bytes"), "unit": "bytes"}, "pipeline_depth": {"value": _number(obj, "pipeline_depth"), "unit": "count"}}
        warnings = [f"{len(samples)} samples: p95/p99 are not meaningful tail estimates"] if len(samples) < 100 else []
        if not _flag(obj, "verify_ok"):
            raise ParseError("the sender reports verify_ok false")
        ver = {"status": "passed", "method": "native: one untimed full verification round after the timed loop on the destination (exit status 1 on mismatch)", "details": "exit status 0 on every rank"}
        return Parsed({"timing": tm, "throughput": throughput, "extra": extra, "warnings": warnings}, ver)


def _hidden(sender):
    """The tool's own overlap figures, checked for physical meaning instead of clipped: hidden_fraction must lie in [0, 1] and hidden_us cannot exceed the smaller phase."""
    notes = []
    frac = sender["hidden_fraction"]
    if frac < 0 or frac > 1:
        notes.append(f"hidden_fraction {frac:.3f} is outside [0, 1]: the single-shot phases are too noisy for a meaningful overlap figure; it is recorded but not valid")
    return notes


class AsyncOverlap(BenchmarkAdapter):
    caps = Capabilities(
        name="async_overlap", category="overlap", description="CPU compute / communication overlap (a CPU hash loop on the sender stands in for compute; not GPU compute)",
        executable="tbccl_async_overlap_bench", operations=("overlap",), min_ranks=2, max_ranks=2, source_selectable=True, supports_warmup=False, supports_iterations=False,
        verification={}, source_files=("benchmarks/async_overlap_bench.cpp",),
        notes="Host memory only. One run measures compute-only, comm-only, serial and overlapped totals once each (no warmup, no repetition inside the tool, no data verification). Options: bucket_count, compute_rounds.")

    def source(self, point):
        return point.source_rank or 0

    def data_direction(self, point):
        return {"mode": "directional", "source_rank": self.source(point)}

    def estimate_bytes(self, point):
        return point.payload_bytes * int(point.options.get("bucket_count", 4)) * 3

    def validate_workload(self, workload, operations):
        return [] if (workload.get("options") or {}).get("compute_rounds") else ["async_overlap needs options.compute_rounds (the CPU work per bucket)"]

    def describe(self, point):
        return {"payload_semantics": "bytes per bucket; bucket_count buckets are produced by CPU hashing on the sender and sent over the async worker", "backend": "tbccl-async-substrate",
                "timing_method": "four single-shot phases timed with steady_clock on the sender: compute only, communication only, compute then communicate (serial), compute with communication enqueued per bucket (overlap)",
                "completion_semantics": "every bucket's work() has completed on this rank; with a host buffer that proves the local kernel accepted the bytes, not that the peer consumed them"}

    def plan(self, point, ctx, ports, rep_dir):
        opts = point.options

        def argv_for(rank, spec, peer_arg):
            return ["--rank", str(rank), "--peers", peer_arg, "--source-rank", str(self.source(point)), "--bucket-bytes", str(point.payload_bytes),
                    "--bucket-count", str(opts.get("bucket_count", 4)), "--compute-rounds", str(opts["compute_rounds"]), "--label", "suite"]

        return _two_rank_invocations(self, point, ctx, ports, argv_for)

    def parse(self, point, group, outcome):
        obj = _json_object(outcome.stdout[self.source(point)])
        phases = {k: _number(obj, k) for k in ("compute_only_us", "comm_only_us", "serial_us", "async_overlap_us")}
        if phases["async_overlap_us"] <= 0:
            raise ParseError("the overlapped phase reports a non-positive time")
        for key in ("hidden_us", "hidden_fraction", "speedup"):
            if key not in obj or not isinstance(obj[key], (int, float)):
                raise ParseError(f"the output has no numeric '{key}'")
        tm = timing("cpu_overlap_total_time", {"mean": phases["async_overlap_us"]}, "reported", "mean", sample_count=1)
        extra = {k: {"value": v, "unit": "us"} for k, v in phases.items() if k != "async_overlap_us"}
        extra["hidden_us"] = {"value": float(obj["hidden_us"]), "unit": "us", "note": "serial - overlapped; negative or larger than the smaller phase is noise"}
        extra["hidden_fraction"] = {"value": float(obj["hidden_fraction"]), "unit": "ratio", "note": "tool definition: hidden_us / comm_only_us"}
        extra["speedup"] = {"value": float(obj["speedup"]), "unit": "ratio", "note": "serial / overlapped"}
        warnings = ["CPU-compute overlap: the compute is a CPU hash loop, not a GPU kernel", "single-shot phases (no warmup, no repetitions inside the tool); only repeated runs show the spread"] + _hidden(obj)
        ver = {"status": "not_supported", "method": None, "details": "the tool does not verify the transferred bytes"}
        return Parsed({"timing": tm, "throughput": [], "extra": extra, "warnings": warnings}, ver)


class CudaOverlap(BenchmarkAdapter):
    SCHEDULES = {"serial": "serial", "overlap": "overlap", "compute_only": "compute-only", "comm_only": "comm-only"}
    caps = Capabilities(
        name="cuda_overlap", category="overlap", description="CUDA kernel compute / TBCCL communication overlap on the CUDA sender, receiver host or Metal-shared (one schedule per run)",
        executable="tbccl_cuda_overlap_bench", operations=tuple(SCHEDULES), memory_backends=("cuda-chunked", "host", "metal-shared"), min_ranks=2, max_ranks=2, source_selectable=True,
        verification={"serial": "native", "overlap": "native", "comm_only": "native"}, source_files=("benchmarks/cuda_overlap_bench.cpp",),
        notes="The source rank must be the CUDA rank (memory_backend cuda-chunked); the other rank is host or metal-shared and does no compute. Options: compute_rounds (required; pick it once with the tool's --calibrate), bucket_count, chunk_bytes, depth. "
              "Overlap is derived afterwards from the four schedules' medians, never from submission timestamps.")

    def source(self, point):
        return point.source_rank or 0

    def data_direction(self, point):
        return {"mode": "directional", "source_rank": self.source(point)}

    def estimate_bytes(self, point):
        if point.operation == "compute_only":
            return 0
        return point.payload_bytes * int(point.options.get("bucket_count", 4)) * (point.iterations + point.warmup + 1)

    def validate_workload(self, workload, operations):
        return [] if (workload.get("options") or {}).get("compute_rounds") else ["cuda_overlap needs options.compute_rounds (calibrate it with the tool's --calibrate)"]

    def check_point(self, point, ctx):
        src = self.source(point)
        if point.ranks[src]["memory_backend"] != "cuda-chunked":
            return f"the source rank {src} must use memory_backend cuda-chunked (the CUDA sender)"
        other = point.ranks[1 - src]["memory_backend"]
        if other not in ("host", "metal-shared"):
            return f"the receiver backend must be host or metal-shared, not {other}"
        return _check_providers(point, ctx)

    def describe(self, point):
        return {"payload_semantics": "bytes per bucket; bucket_count buckets are computed by a CUDA kernel on the sender and sent with the chunked async CUDA staging to the receiver", "backend": "tbccl-async-substrate",
                "timing_method": f"schedule '{self.SCHEDULES[point.operation]}': steady_clock total per iteration on the sender, from the first bucket's compute start to the last transfer's completion; "
                                 "each bucket's compute stream is synchronized before its transfer is enqueued; the tool prints median, p95, enqueue median and max",
                "completion_semantics": "all kernels have finished and every transfer's work() has completed on the sender (no destination ACK in this tool)"}

    def plan(self, point, ctx, ports, rep_dir):
        opts = point.options

        def argv_for(rank, spec, peer_arg):
            argv = ["--rank", str(rank), "--peers", peer_arg, "--source-rank", str(self.source(point)), "--bucket-bytes", str(point.payload_bytes), "--bucket-count", str(opts.get("bucket_count", 4)),
                    "--compute-rounds", str(opts["compute_rounds"]), "--chunk-bytes", str(opts.get("chunk_bytes", 262144)), "--depth", str(opts.get("depth", 1)), "--warmup", str(point.warmup),
                    "--iterations", str(point.iterations), "--schedule", self.SCHEDULES[point.operation], "--label", "suite"]
            if rank != self.source(point):
                argv += ["--receiver-backend", spec["memory_backend"]]
            return argv

        return _two_rank_invocations(self, point, ctx, ports, argv_for)

    def parse(self, point, group, outcome):
        obj = _json_object(outcome.stdout[self.source(point)])
        if obj.get("role") != "sender" or obj.get("schedule") != self.SCHEDULES[point.operation]:
            raise ParseError("the source rank's output is not the sender's result for the requested schedule")
        stats = {"median": _number(obj, "total_median_us"), "p95": _number(obj, "total_p95_us")}
        if stats["median"] <= 0:
            raise ParseError("the sender reports a non-positive median total")
        n = int(_number(obj, "iterations"))
        tm = timing("overlap_schedule_total_time", stats, "reported", "median", percentile_method="ceil-nearest-rank", sample_count=n)
        extra = {"enqueue_median_us": {"value": _number(obj, "enqueue_median_us"), "unit": "us", "note": "submission only"}, "enqueue_max_us": {"value": _number(obj, "enqueue_max_us"), "unit": "us"},
                 "compute_rounds": {"value": _number(obj, "compute_rounds"), "unit": "count"}, "bucket_count": {"value": _number(obj, "bucket_count"), "unit": "count"}}
        warnings = ["CUDA kernel compute on the sender; the receiver does no compute", f"{n} iterations: p95 is not a meaningful tail estimate"] if n < 100 else ["CUDA kernel compute on the sender"]
        if not _flag(obj, "verify_ok"):
            raise ParseError("the sender reports verify_ok false")
        if point.operation == "compute_only":
            ver = {"status": "not_supported", "method": None, "details": "compute-only moves no data"}
        else:
            ver = {"status": "passed", "method": "native: the receiver verifies the computed bucket bytes after the timed loop (exit status 1 on mismatch)", "details": "exit status 0 on every rank"}
        return Parsed({"timing": tm, "throughput": [], "extra": extra, "warnings": warnings}, ver)


class HeteroAllReduce(BenchmarkAdapter):
    caps = Capabilities(
        name="hetero_allreduce", category="collective", description="Heterogeneous N=2 SUM float32 all-reduce: legacy synchronous reduce+broadcast versus the async tensor substrate",
        executable="tbccl_hetero_allreduce_bench", operations=("all_reduce",), algorithms=("sync", "async"), memory_backends=("host", "cuda", "metal-shared"), min_ranks=2, max_ranks=2,
        verification={"all_reduce": "native"}, source_files=("benchmarks/tbccl_hetero_allreduce_bench.cpp",),
        notes="Each rank has its own backend. 'cuda' means pinned staging for sync and CudaChunkedAsyncBackend device memory for async; sync fills its staging buffer on the host and stages host-to-device after the broadcast, "
              "so the two algorithms differ in more than the substrate. Options: chunk_bytes, root, metal_async_path. Not comparable with the legacy World all_reduce of collective_bench.")

    def estimate_bytes(self, point):
        return point.payload_bytes * 2 * (point.iterations + point.warmup + 1)

    def check_point(self, point, ctx):
        if point.payload_bytes < 4 or point.payload_bytes % 4:
            return "the payload is Float32 elements; it must be a multiple of 4 bytes"
        return _check_providers(point, ctx)

    def describe(self, point):
        return {"payload_semantics": "total Float32 tensor bytes per rank (bytes/4 elements); SUM, root 0",
                "backend": "tbccl-legacy-world" if point.algorithm == "sync" else "tbccl-async-substrate",
                "timing_method": "steady_clock around one all-reduce per iteration on each rank (sync: reduce + broadcast + host-to-device stage; async: n2_all_reduce_tensor); local input fill is outside the timer; "
                                 "the record's value is the slower rank's median; per-rank statistics are kept",
                "completion_semantics": "this rank's all-reduce call returned with the result in its backend; ranks are not synchronized to a common clock"}

    def plan(self, point, ctx, ports, rep_dir):
        opts = point.options

        def argv_for(rank, spec, peer_arg):
            argv = ["--rank", str(rank), "--peers", peer_arg, "--backend", spec["memory_backend"], "--count", str(point.payload_bytes // 4), "--chunk-bytes", str(opts.get("chunk_bytes", 262144)),
                    "--warmup", str(point.warmup), "--iterations", str(point.iterations), "--algo", point.algorithm, "--root", str(opts.get("root", 0)), "--label", "suite"]
            if spec["memory_backend"] == "metal-shared" and opts.get("metal_async_path"):
                argv += ["--metal-async-path", str(opts["metal_async_path"])]
            return argv

        return _two_rank_invocations(self, point, ctx, ports, argv_for, roles=("root", "peer"))

    def parse(self, point, group, outcome):
        algo = point.algorithm
        per_rank, verify = [], []
        for rank in (0, 1):
            obj = _json_object(outcome.stdout[rank])
            per_rank.append({"rank": rank, "statistics": {"median": _number(obj, f"{algo}_median_us"), "p95": _number(obj, f"{algo}_p95_us"), "p99": _number(obj, f"{algo}_p99_us"), "mean": _number(obj, f"{algo}_mean_us")},
                             "sample_count": point.iterations})
            verify.append(_flag(obj, f"{algo}_verify_ok"))
        if not all(verify):
            raise ParseError("a rank reports the all-reduce result failed verification")
        slow = max(per_rank, key=lambda r: r["statistics"]["median"])
        tm = timing("hetero_all_reduce_time_slowest_rank", dict(slow["statistics"]), "reported", "median", percentile_method="tool-defined (see the tool's compute_stats())", sample_count=point.iterations, per_rank=per_rank)
        warnings = [f"{point.iterations} iterations: p95/p99 are not meaningful tail estimates"] if point.iterations < 100 else []
        extra = {"slowest_rank": {"value": slow["rank"], "unit": "rank"}}
        ver = {"status": "passed", "method": "native: one untimed round whose full result vector is compared with the expected sum on each rank", "details": "both ranks report verify_ok true"}
        return Parsed({"timing": tm, "throughput": [], "extra": extra, "warnings": warnings}, ver)
