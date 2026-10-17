"""Integrations for the core TBCCL executables: small-message latency, tensor transfer and the collective benchmark.

Every adapter calls the existing executable with its existing command line and parses its existing output; no algorithm or timing lives here.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .base import BenchmarkAdapter, Capabilities, Group, Invocation, Outcome, ParseError, Parsed, Point, timing

PERCENTILE_LINEAR = "linear-interpolation-n-1"
PERCENTILE_FLOOR = "floor-nearest-rank"
DTYPE_BYTES = {"float32": 4, "int32": 4, "float64": 8, "int64": 8}
ALL_BACKENDS = ("host", "cuda-pageable", "cuda-pinned", "metal-shared", "metal-private-staged")


def provider_for(backend):
    """The name `tbccl-info` lists for the provider a tensor backend needs (both Metal backends come with the metal_shared provider's build option)."""
    return "cuda" if backend.startswith("cuda") else "metal_shared" if backend.startswith("metal") else "host"


def peers(point, ctx, ports):
    return ",".join(f"{ctx.rank_address(r)}:{port}" for r, port in zip(point.ranks, ports))


def _number(row, key):
    try:
        value = float(row[key])
    except (KeyError, ValueError, TypeError):
        raise ParseError(f"column '{key}' is missing or not a number: {row.get(key)!r}")
    if value != value or value in (float("inf"), float("-inf")) or value < 0:
        raise ParseError(f"column '{key}' has an invalid value {value!r}")
    return value


class SmallMessageLatency(BenchmarkAdapter):
    OPS = {"p2p": "round_trip_time", "oneway": "per_message_time_in_block_sender", "simul": "simultaneous_exchange_time", "allgather": "all_gather_time_rank0",
           "allreduce": "all_reduce_time_rank0", "chain": "send_plus_all_gather_time_rank0"}
    caps = Capabilities(
        name="small_message_latency", category="p2p", description="N=2 Communicator latency on host memory: p2p, one-way stream, simultaneous exchange, all_gather, all_reduce",
        executable="small_message_latency", operations=tuple(OPS), min_ranks=2, max_ranks=2,
        verification={"p2p": "native"}, source_files=("benchmarks/small_message_latency.cpp",),
        notes="Only rank 0 prints. Percentiles use floor(q*n); no p99 and no raw samples. Only the p2p mode verifies data.")

    def estimate_bytes(self, point):
        samples = point.iterations + point.warmup + 1
        return point.payload_bytes * samples * (1 if point.operation == "oneway" else 2)

    def data_direction(self, point):
        if point.operation == "oneway":
            return {"mode": "directional", "source_rank": 0}  # rank 0 sends the stream, rank 1 receives it
        return {"mode": "bidirectional", "source_rank": None}

    def check_point(self, point, ctx):
        if point.operation == "allreduce" and (point.payload_bytes < 4 or point.payload_bytes % 4):
            return "all_reduce uses Float32 elements; the payload must be a multiple of 4 bytes"
        return None

    def describe(self, point):
        sem = {"p2p": "message bytes; one round trip sends this many bytes each way",
               "oneway": "message bytes; time per message over blocks of 32 back-to-back sends, as seen by the receiver",
               "simul": "message bytes; both ranks send and receive at once",
               "allgather": "bytes contributed by each rank",
               "allreduce": "total Float32 tensor bytes per rank (bytes/4 elements)",
               "chain": "message bytes for the send, then the same bytes contributed to an all_gather"}[point.operation]
        return {"payload_semantics": sem, "timing_method": "steady_clock around each operation including wait(); barrier before warmup and before the timed loop; per-iteration percentiles by floor(q*n)",
                "completion_semantics": "every posted operation has completed (wait() returned) on rank 0", "backend": "tbccl-communicator"}

    def plan(self, point, ctx, ports, rep_dir):
        peer_arg = peers(point, ctx, ports)
        invs = []
        for rank, spec in enumerate(point.ranks):
            argv = [ctx.executable(self.caps.executable, spec["host"]), "--rank", str(rank), "--peers", peer_arg, "--modes", point.operation,
                    "--sizes", str(point.payload_bytes), "--iters", str(point.iterations), "--warmup", str(point.warmup), "--label", "suite"]
            # TBCCL_BENCH_BOTH makes the follower print its own view too (the tool's documented switch); set here explicitly because the suite removes TBCCL_* from the environment
            invs.append(Invocation(rank, "leader" if rank == 0 else "follower", spec["host"], argv, env={"TBCCL_BENCH_BOTH": "1"}, wait_for_port=ports[0] if rank == 1 else None,
                                   listens_on=ports[0] if rank == 0 else None))
        return Group(invs, ports=list(ports))

    def parse(self, point, group, outcome):
        text = Path(outcome.stdout[0]).read_text()
        found = None
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                raise ParseError(f"malformed JSON line in rank 0 output: {line[:80]!r}")
            if obj.get("mode") == point.operation and obj.get("bytes") == point.payload_bytes:
                found = obj
        if found is None:
            raise ParseError("rank 0 printed no result line for the requested mode and payload")
        other = None
        try:
            for line in Path(outcome.stdout[1]).read_text().splitlines():
                if line.strip().startswith("{"):
                    obj = json.loads(line)
                    if obj.get("mode") == point.operation and obj.get("bytes") == point.payload_bytes and str(obj.get("label", "")).endswith("/r1"):
                        other = obj
        except (KeyError, OSError, ValueError):
            other = None
        return self.measure(point, found, other)

    def measure(self, point, found, other=None):
        stats = {"median": _number(found, "median_us"), "p25": _number(found, "p25_us"), "p75": _number(found, "p75_us"), "p95": _number(found, "p95_us")}
        n = int(_number(found, "n"))
        if n < 1 or stats["median"] <= 0:
            raise ParseError("the result line reports no samples or a non-positive median")
        per_rank = [{"rank": 0, "statistics": dict(stats)}]
        if other is not None:
            try:
                per_rank.append({"rank": 1, "statistics": {"median": _number(other, "median_us"), "p25": _number(other, "p25_us"), "p75": _number(other, "p75_us"), "p95": _number(other, "p95_us")},
                                 "cpu_cores": _number(other, "cpu_cores"), "sample_count": int(_number(other, "n"))})
            except ParseError:
                per_rank = per_rank[:1]
        tm = timing(self.OPS[point.operation], stats, "reported", "median", percentile_method=PERCENTILE_FLOOR, sample_count=n, per_rank=per_rank if len(per_rank) > 1 else None)
        extra = {"cpu_cores": {"value": _number(found, "cpu_cores"), "unit": "busy cores (process CPU time / wall time, timed loop)"},
                 "sleep_ratio": {"value": _number(found, "sleep_ratio"), "unit": "ratio"}}
        warnings = []
        if point.operation == "p2p":
            ver = {"status": "passed", "method": "native: one untimed memcmp round trip after the timed loop (exit status 3 on mismatch)", "details": "exit status 0"}
        else:
            ver = {"status": "not_supported", "method": None, "details": f"small_message_latency verifies the p2p mode only, not '{point.operation}'"}
        if n < 20:
            warnings.append(f"only {n} samples; the reported p95 is not a meaningful tail estimate")
        return Parsed({"timing": tm, "throughput": [], "extra": extra, "warnings": warnings}, ver)


class TensorTransfer(BenchmarkAdapter):
    MODES = {"latency_floor": "latency-floor", "network_only": "network-only", "end_to_end": "end-to-end", "ack_calibration": "ack-calibration"}
    METRICS = {"latency_floor": "round_trip_time", "network_only": "sender_send_call_time", "end_to_end": "completion_confirmed_time",
               "ack_calibration": "host_ack_calibration_completion_time"}
    caps = Capabilities(
        name="tensor_transfer", category="p2p", description="Point-to-point tensor transfer between two ranks: latency floor, one-way streaming, device end-to-end with ACK",
        executable="tbccl_tensor_transfer_bench", operations=tuple(MODES), memory_backends=ALL_BACKENDS, min_ranks=2, max_ranks=2,
        verification={"end_to_end": "native"}, source_files=("benchmarks/tensor_transfer_bench.cpp",), source_selectable=True,
        notes="The source rank (default 0, selectable independently of the rank orientation) prints the CSV. end_to_end takes a memory backend per rank (CUDA<->Metal is a normal invocation); latency_floor and network_only are host-only.")

    def source(self, point):
        return point.source_rank or 0

    def data_direction(self, point):
        if point.operation == "latency_floor":
            return {"mode": "bidirectional", "source_rank": self.source(point)}  # the source starts each round trip; payload moves both ways
        return {"mode": "directional", "source_rank": self.source(point)}

    def estimate_bytes(self, point):
        samples = point.iterations + point.warmup + 1  # the extra one is the verified round
        return point.payload_bytes * samples * (2 if point.operation == "latency_floor" else 1)

    def check_point(self, point, ctx):
        if point.operation != "end_to_end" and any(b != "host" for b in point.backends):
            return f"'{point.operation}' always uses the host backend; {sorted(set(point.backends))} needs end_to_end"
        for spec in point.ranks:
            have = ctx.providers(spec["host"]) if ctx else None
            if have is not None and provider_for(spec["memory_backend"]) not in have:
                point.skip_state = "unsupported"
                return f"memory backend '{spec['memory_backend']}' is not available in the executable's build on host '{spec['host']}' (providers: {', '.join(sorted(have)) or 'none'})"
        return None

    def describe(self, point):
        scope = point.options.get("timing_scope", "ready")
        if point.operation == "latency_floor":
            return {"payload_semantics": "bytes per message; the round trip moves this many bytes in each direction", "backend": "tbccl-legacy-world",
                    "timing_method": "per-iteration round trip on the source rank, percentiles by linear interpolation; host buffers; no verification",
                    "completion_semantics": "the source has received the echoed payload"}
        if point.operation == "network_only":
            return {"payload_semantics": "bytes per send; the source streams this payload back-to-back to its peer", "backend": "tbccl-legacy-world",
                    "timing_method": "per-send() call latency on the source rank (time for send to return), percentiles by linear interpolation; throughput is wall-clock over the loop",
                    "completion_semantics": "send() has returned on the source; delivery is not confirmed per message"}
        if point.operation == "ack_calibration":
            return {"payload_semantics": "bytes moved host to host with the end-to-end protocol (the backend is forced to host); the tool also prints an implicit 0-byte row, which is not a round trip", "backend": "tbccl-legacy-world",
                    "timing_method": "completion-confirmed time per iteration with host buffers, timing scope 'ready'; percentiles by linear interpolation",
                    "completion_semantics": "the destination's ACK has been observed by the source (includes the ACK's return trip)"}
        return {"payload_semantics": "tensor bytes moved from the source rank's backend to the destination rank's backend", "backend": "tbccl-legacy-world",
                "timing_method": f"completion-confirmed time per iteration, timing scope '{scope}' ("
                                 + ("timer starts after source generation and producer synchronization" if scope == "ready" else "timer starts before source generation, so it includes the producer")
                                 + "); percentiles by linear interpolation",
                "completion_semantics": "the destination accelerator's completion ACK has been observed by the source (includes the ACK's return trip; not a one-way latency)"}

    def plan(self, point, ctx, ports, rep_dir):
        peer_arg = peers(point, ctx, ports)
        invs = []
        for rank, spec in enumerate(point.ranks):
            argv = [ctx.executable(self.caps.executable, spec["host"]), "--mode", self.MODES[point.operation], "--rank", str(rank), "--peers", peer_arg, "--source-rank", str(self.source(point)),
                    "--sizes", str(point.payload_bytes), "--warmup", str(point.warmup), "--iterations", str(point.iterations)]
            if point.operation == "end_to_end":
                argv += ["--local-backend", spec["memory_backend"], "--timing-scope", point.options.get("timing_scope", "ready")]
            if point.options.get("busy_poll_us"):
                argv += ["--busy-poll", str(point.options["busy_poll_us"])]
            if point.options.get("diagnostics") and point.operation == "end_to_end":
                argv += ["--diagnostics"]  # the tool prints its raw per-iteration completion samples to stderr
            invs.append(Invocation(rank, "source" if rank == 0 else "destination", spec["host"], argv, wait_for_port=ports[0] if rank == 1 else None, listens_on=ports[0] if rank == 0 else None))
        return Group(invs, ports=list(ports))

    def parse(self, point, group, outcome):
        rows = list(csv.DictReader(io.StringIO(Path(outcome.stdout[self.source(point)]).read_text())))
        mode = self.MODES[point.operation]
        row = next((r for r in rows if r.get("mode") == mode and r.get("payload_bytes", "").isdigit() and int(r["payload_bytes"]) == point.payload_bytes), None)
        if row is None:
            raise ParseError("the source rank printed no CSV row for the requested mode and payload")
        parsed = self.measure(point, row)
        if point.options.get("diagnostics") and point.operation == "end_to_end":
            samples = self.diagnostic_samples(outcome, self.source(point), point.payload_bytes)
            if samples:
                if len(samples) != parsed.measurements["timing"]["sample_count"]:
                    raise ParseError(f"the tool printed {len(samples)} raw samples for {parsed.measurements['timing']['sample_count']} iterations")
                parsed.measurements["timing"]["samples"] = samples
        if point.operation == "ack_calibration":  # the tool also measures an implicit 0-byte payload first; over a physical link that row has no round trip in it
            zero = next((r for r in rows if r.get("mode") == mode and r.get("payload_bytes") == "0"), None)
            if zero is not None:
                parsed.measurements["extra"]["zero_byte_completion_median_us"] = {"value": _number(zero, "median_us"), "unit": "us", "note": "the tool's implicit 0-byte row (host buffers). Over a physical link it completes in under 1 us, so it contains no network round trip and is NOT an ACK floor"}
        return parsed

    @staticmethod
    def diagnostic_samples(outcome, rank, payload):
        """The raw per-iteration completion_confirmed samples (us) the tool prints with --diagnostics: a TBCCL_DIAGNOSTIC JSON line on the source rank's stderr."""
        try:
            text = Path(outcome.stderr[rank]).read_text()
        except (KeyError, OSError):
            return None
        for line in text.splitlines():
            if line.startswith("TBCCL_DIAGNOSTIC "):
                try:
                    obj = json.loads(line[len("TBCCL_DIAGNOSTIC "):])
                except ValueError:
                    continue
                if obj.get("kind") == "samples" and obj.get("metric") == "completion_confirmed" and obj.get("payload_bytes") == payload and obj.get("rank") == rank:
                    return [float(v) for v in obj["values_us"]]
        return None

    def measure(self, point, row):
        stats = {"median": _number(row, "median_us"), "p95": _number(row, "p95_us"), "p99": _number(row, "p99_us"), "min": _number(row, "min_us"), "max": _number(row, "max_us")}
        n = int(_number(row, "iterations"))
        if n < 1 or stats["median"] <= 0:
            raise ParseError("the CSV row reports no samples or a non-positive median")
        tm = timing(self.METRICS[point.operation], stats, "reported", "median", percentile_method=PERCENTILE_LINEAR, sample_count=n)
        device = any(b != "host" for b in point.backends)
        if point.operation == "latency_floor":
            kind, definition = "application_payload", "2 * payload bytes / median round-trip time (both directions); decimal GB/s"
        elif point.operation == "network_only":
            kind, definition = "application_payload", "payload bytes * iterations / wall-clock seconds of the streaming loop; decimal GB/s; one direction"
        else:
            kind = "device_to_device_effective" if device else "application_payload"
            definition = "payload bytes / median completion-confirmed time (includes the ACK return trip); decimal GB/s"
        throughput = [{"kind": kind, "value": _number(row, "effective_GBps"), "unit": "GB/s", "definition": definition}] if point.payload_bytes > 0 else []
        extra = {}
        for column in ("half_rtt_us", "source_staging_us", "source_sync_us", "sender_network_us", "receiver_network_us", "destination_staging_us", "destination_sync_us",
                       "completion_confirmed_us", "producer_enqueue_us"):
            if row.get(column, "NA") not in ("NA", ""):
                extra[column] = {"value": _number(row, column), "unit": "us"}
        if "half_rtt_us" in extra:
            extra["half_rtt_us"]["note"] = "approximation: median round trip / 2; not a measured one-way latency"
        warnings = []
        if n < 100:
            warnings.append(f"{n} iterations: p95/p99 are not meaningful tail estimates")
        if point.operation == "end_to_end":
            ver = {"status": "passed", "method": "native: one untimed full-byte-verified round after the timed loop", "details": "exit status 0 on every rank"}
        else:
            ver = {"status": "not_supported", "method": None, "details": f"the '{point.operation}' mode moves host buffers and does not verify them"}
        return Parsed({"timing": tm, "throughput": throughput, "extra": extra, "warnings": warnings}, ver)


class CollectiveBench(BenchmarkAdapter):
    OPS = {"all_reduce": "all-reduce", "all_gather": "all-gather", "reduce_scatter": "reduce-scatter"}
    SEMANTICS = {"all_reduce": "total tensor bytes per rank (input and output are both this size)",
                 "all_gather": "bytes contributed by each rank (every rank receives rank_count times this)",
                 "reduce_scatter": "output segment bytes per rank (every rank contributes rank_count times this)"}
    caps = Capabilities(
        name="collective_bench", category="collective", description="all_reduce / all_gather / reduce_scatter across N ranks with selectable algorithm",
        executable="tbccl_collective_bench", operations=tuple(OPS), algorithms=("reference", "ring", "auto", "ring-pipelined"), min_ranks=2, max_ranks=None,
        datatypes=("float32", "float64", "int32", "int64"), verification={op: "native" for op in OPS}, source_files=("benchmarks/collective_bench.cpp",),
        notes="ring-pipelined is experimental and needs options.chunk_bytes. all_reduce needs an element count divisible by the rank count for every algorithm.")

    def estimate_bytes(self, point):
        runs = point.iterations + point.warmup + 2  # two correctness gates
        return point.payload_bytes * runs * (point.rank_count if point.operation == "all_gather" else 2)

    def validate_workload(self, workload, operations):
        errors = []
        algos = workload.get("algorithms") or ([workload["algorithm"]] if "algorithm" in workload else [])
        if "ring-pipelined" in algos:
            chunk = (workload.get("options") or {}).get("chunk_bytes")
            if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 1:
                errors.append("algorithm 'ring-pipelined' needs options.chunk_bytes (a positive integer)")
        if not algos:
            errors.append("'algorithms' is required (reference, ring, auto or ring-pipelined)")
        return errors

    def check_point(self, point, ctx):
        dsize = DTYPE_BYTES.get(point.datatype or "float32", 4)
        n = point.rank_count
        if point.operation in ("all_reduce", "reduce_scatter"):
            if point.payload_bytes % dsize:
                return f"payload {point.payload_bytes} B is not a multiple of the {point.datatype} size ({dsize} B)"
        if point.operation == "all_reduce" and (point.payload_bytes // dsize) % n:
            return f"{point.payload_bytes // dsize} elements are not divisible by {n} ranks (required by the tool for every algorithm)"
        if point.algorithm == "ring-pipelined" and point.operation != "all_gather":
            chunk = point.options.get("chunk_bytes", 0)
            if chunk % dsize:
                return f"chunk_bytes {chunk} is not a multiple of the {point.datatype} size"
        return None

    def describe(self, point):
        return {"payload_semantics": self.SEMANTICS[point.operation], "backend": "tbccl-legacy-world",
                "timing_method": "per iteration: barrier, then only the collective call; duration is the maximum across ranks of each rank's own monotonic clock; percentiles by linear interpolation; untimed correctness gate before and after the loop",
                "completion_semantics": "the collective call has returned on every rank (slowest rank per iteration)"}

    def plan(self, point, ctx, ports, rep_dir):
        peer_arg = peers(point, ctx, ports)
        invs = []
        for rank, spec in enumerate(point.ranks):
            argv = [ctx.executable(self.caps.executable, spec["host"]), "--rank", str(rank), "--peers", peer_arg, "--collective", self.OPS[point.operation],
                    "--algorithm", point.algorithm, "--sizes", str(point.payload_bytes), "--iterations", str(point.iterations), "--warmup", str(point.warmup)]
            if point.operation in ("all_reduce", "reduce_scatter"):
                argv += ["--datatype", point.datatype or "float32", "--op", point.reduce_op or "sum"]
            if point.algorithm == "ring-pipelined":
                argv += ["--chunk-bytes", str(point.options["chunk_bytes"])]
            if point.options.get("busy_poll_us"):
                argv += ["--busy-poll", str(point.options["busy_poll_us"])]
            invs.append(Invocation(rank, "root" if rank == 0 else "peer", spec["host"], argv, wait_for_port=ports[0] if rank > 0 else None, listens_on=ports[0] if rank == 0 else None))
        return Group(invs, ports=list(ports))

    def parse(self, point, group, outcome):
        rows = list(csv.DictReader(io.StringIO(Path(outcome.stdout[0]).read_text())))
        mode = self.OPS[point.operation]
        key = {"all_reduce": "input_bytes_per_rank", "all_gather": "input_bytes_per_rank", "reduce_scatter": "segment_bytes"}[point.operation]
        row = next((r for r in rows if r.get("collective") == mode and r.get(key, "").isdigit() and int(r[key]) == point.payload_bytes
                    and r.get("requested_algorithm") == point.algorithm), None)
        if row is None:
            raise ParseError("rank 0 printed no CSV row for the requested collective, algorithm and payload")
        return self.measure(point, row)

    def measure(self, point, row):
        if int(row["world_size"]) != point.rank_count:
            raise ParseError(f"the row reports world_size {row['world_size']}, expected {point.rank_count}")
        stats = {"median": _number(row, "median_us"), "p95": _number(row, "p95_us"), "min": _number(row, "min_us"), "max": _number(row, "max_us")}
        n = int(_number(row, "iterations"))
        if n < 1 or stats["median"] <= 0:
            raise ParseError("the CSV row reports no samples or a non-positive median")
        tm = timing("collective_duration_max_across_ranks", stats, "reported", "median", percentile_method=PERCENTILE_LINEAR, sample_count=n)
        throughput = [
            {"kind": "collective_algorithmic", "value": _number(row, "algbw_GBps"), "unit": "GB/s",
             "definition": "rank_count * segment_bytes / median duration; decimal GB/s (native definition, see collective_bench.cpp)"},
            {"kind": "collective_bus_estimate", "value": _number(row, "busbw_GBps"), "unit": "GB/s",
             "definition": "algbw * (N-1)/N, times 2 for all_reduce; an estimate for an ideal ring, not a measured bus rate"}]
        extra = {"chunk_bytes": {"value": int(_number(row, "chunk_bytes")), "unit": "B"}, "segment_bytes": {"value": int(_number(row, "segment_bytes")), "unit": "B"},
                 "output_bytes_per_rank": {"value": int(_number(row, "output_bytes_per_rank")), "unit": "B"}}
        warnings = []
        if n < 100:
            warnings.append(f"{n} iterations: p95 is not a meaningful tail estimate")
        return Parsed({"timing": tm, "throughput": throughput, "extra": extra, "warnings": warnings},
                      {"status": "passed", "method": "native: untimed correctness gate before and after the timed loop (mismatch throws, nonzero exit)", "details": "exit status 0 on every rank"},
                      algorithm_resolved=row["algorithm"], selection_reason=row.get("selection_reason") or None)
