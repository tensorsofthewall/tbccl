"""Transport reference benchmark: the raw TCP ping-pong/stream tool (no TBCCL protocol in the path), plus the baseline interface.

tb_pingpong facts the adapter encodes (see docs/development/benchmark-suite-inventory.md): the server never exits, there is no warmup and no iteration
option (the tool picks the count from the payload size), the reported round-trip time is a mean, 'gbps' is Gbit/s, and nothing is verified.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

from .base import BenchmarkAdapter, Capabilities, Group, Invocation, Outcome, ParseError, Parsed, Point, timing

STDOUT_ROW = re.compile(r"^(pingpong|stream)\s+(\d+(?:\.\d+)?)\s+(B|KiB|MiB|GiB)\s+n=\s*(\d+)\s+RTT=\s*(-|[\d.]+)(?:\s+us)?\s+([\d.]+)\s+Gbit/s\s+([\d.]+)\s+msg/s")
SIZE_UNITS = {"B": 1, "KiB": 1024, "MiB": 1024 ** 2, "GiB": 1024 ** 3}


class TcpPingPong(BenchmarkAdapter):
    caps = Capabilities(
        name="tcp_pingpong", category="network_transport", description="Raw TCP ping-pong and one-way stream between two hosts (transport reference)",
        executable="tb_pingpong", operations=("pingpong", "stream"), min_ranks=2, max_ranks=2, supports_warmup=False, supports_iterations=False,
        verification={"pingpong": "not_supported", "stream": "not_supported"}, source_files=("benchmarks/tcp_pingpong.cpp",),
        notes="Rank 0 is the server, rank 1 the client. The server never exits; the runner terminates it after the client finishes.")

    def check_point(self, point, ctx):
        if point.payload_bytes < 1:
            return "payload must be at least 1 byte"
        return None

    def estimate_bytes(self, point):
        """The tool's own iteration formulas (no option to change them): this is why it is not a lightweight smoke test."""
        size = point.payload_bytes
        if point.operation == "pingpong":
            return size * max(10, min(10000, (64 * 2**20) // size)) * 2
        return size * max(16, min(100000, (4 * 2**30) // size))

    def data_direction(self, point):
        return {"mode": "directional", "source_rank": 1} if point.operation == "stream" else {"mode": "bidirectional", "source_rank": None}

    def describe(self, point):
        if point.operation == "pingpong":
            return {"payload_semantics": "bytes per message; each round trip sends this many bytes in each direction",
                    "timing_method": "steady_clock around the whole loop; mean round trip per iteration; no warmup; iteration count chosen by the tool from the payload size",
                    "completion_semantics": "the client has received the full echo", "backend": "raw-tcp"}
        return {"payload_semantics": "bytes per message sent one way; the receiver returns one ready byte after the last message",
                "timing_method": "steady_clock around the whole stream; no warmup; iteration count chosen by the tool from the payload size",
                "completion_semantics": "the client has received the receiver's ready byte", "backend": "raw-tcp"}

    def plan(self, point, ctx, ports, rep_dir):
        server, client = point.ranks[0], point.ranks[1]
        address = ctx.rank_address(server)
        port = ports[0]
        exe_s, exe_c = ctx.executable(self.caps.executable, server["host"]), ctx.executable(self.caps.executable, client["host"])
        csv_path = str(Path(rep_dir) / "pingpong.csv") if ctx.is_local(client["host"]) else None
        server_argv = [exe_s, "server", "--bind", address, "--port", str(port)]
        client_argv = [exe_c, "client", "--host", address, "--port", str(port), "--mode", point.operation, "--sizes", str(point.payload_bytes)]
        busy = point.options.get("busy_poll_us")
        if busy:
            server_argv += ["--busy-poll", str(busy)]
            client_argv += ["--busy-poll", str(busy)]
        if csv_path:
            client_argv += ["--output", csv_path]
        return Group([Invocation(0, "server", server["host"], server_argv, persistent=True, listens_on=port),
                      Invocation(1, "client", client["host"], client_argv, output_file=csv_path, wait_for_port=port, wait_host=address)], ports=[port])

    def parse(self, point, group, outcome):
        client = group.invocations[1]
        row = None
        if client.output_file and Path(client.output_file).exists():
            with open(client.output_file, newline="") as handle:
                for candidate in csv.DictReader(handle):
                    if candidate.get("mode") == point.operation and int(candidate["size_bytes"]) == point.payload_bytes:
                        row = {"iterations": int(candidate["iterations"]), "seconds": float(candidate["seconds"]), "rtt_us": float(candidate["rtt_us"]),
                               "gbps": float(candidate["gbps"]), "mps": float(candidate["messages_per_second"])}
        else:
            text = Path(outcome.stdout[1]).read_text()
            for line in text.splitlines():
                m = STDOUT_ROW.match(line.strip())
                if m and m.group(1) == point.operation:
                    size = int(round(float(m.group(2)) * SIZE_UNITS[m.group(3)]))
                    row = {"iterations": int(m.group(4)), "seconds": None, "rtt_us": 0.0 if m.group(5) == "-" else float(m.group(5)), "gbps": float(m.group(6)),
                           "mps": float(m.group(7)), "rounded": True}
        if row is None:
            raise ParseError("no result row for the requested mode and payload in the client output")
        return self.measure(point, row)

    def measure(self, point, row):
        """Normalise one result row {iterations, seconds|None, rtt_us, gbps, mps}."""
        for key in ("rtt_us", "gbps", "mps"):
            if row[key] < 0:
                raise ParseError(f"negative {key} in the client output")
        extra = {"iterations_chosen_by_tool": {"value": row["iterations"], "unit": "count"}, "messages_per_second": {"value": row["mps"], "unit": "msg/s"}}
        if row["seconds"] is not None:
            extra["loop_seconds"] = {"value": row["seconds"], "unit": "s"}
        warnings = ["values parsed from rounded console output (no --output file on this host)"] if row.get("rounded") else []
        if point.operation == "pingpong":
            if row["rtt_us"] <= 0:
                raise ParseError("ping-pong round-trip time is not positive")
            tm = timing("round_trip_time", {"mean": row["rtt_us"]}, "reported", "mean")
            definition = "payload bytes in both directions * 8 / loop wall time / 1e9 (decimal Gbit/s); TCP application payload, not link bandwidth"
        else:
            tm = None
            definition = "payload bytes sent one way * 8 / loop wall time / 1e9 (decimal Gbit/s); TCP application payload, not link bandwidth"
        measurements = {"timing": tm, "throughput": [{"kind": "tcp_application_payload", "value": row["gbps"], "unit": "Gbit/s", "definition": definition}],
                        "extra": extra, "warnings": warnings}
        return Parsed(measurements, {"status": "not_supported", "method": None, "details": "tb_pingpong does not check payload bytes"},
                      configuration_extra={"iterations": row["iterations"], "warmup": 0, "warmup_note": "the tool has no warmup and chooses its own iteration count"})


class Iperf3(BenchmarkAdapter):
    """Bounded raw TCP reference: one iperf3 stream for a few seconds, JSON from the client, the server exits after one test (-1). It is a contextual transport
    reference, never subtracted from TBCCL numbers: different workload, different timing, different software."""
    HARD_MAX_DURATION_S = 30
    caps = Capabilities(
        name="iperf3", category="network_transport", description="Single-stream raw TCP throughput with iperf3, bounded duration, one direction at a time",
        executable="iperf3", operations=("tcp_stream",), min_ranks=2, max_ranks=2, executable_in_build=False, source_selectable=True, supports_warmup=False, supports_iterations=False,
        verification={"tcp_stream": "not_supported"}, notes="Rank 0 runs the one-shot server, rank 1 the client. The data source is chosen independently (reverse mode when rank 0 sends). "
        "iperf3 is a baseline tool found on PATH or at host.executables.iperf3; it is not a dependency of TBCCL or of the suite.")

    def validate_workload(self, workload, operations):
        errors = []
        d = (workload.get("options") or {}).get("duration_s", 3)
        if not isinstance(d, (int, float)) or isinstance(d, bool) or not 1 <= d <= self.HARD_MAX_DURATION_S:
            errors.append(f"options.duration_s must be between 1 and {self.HARD_MAX_DURATION_S} seconds")
        return errors

    def _duration(self, point):
        return float(point.options.get("duration_s", 3)), float(point.options.get("omit_s", 1))

    def estimate_bytes(self, point):
        duration, omit = self._duration(point)
        return int((duration + omit) * 5e9)  # a ceiling at 40 Gbit/s (the Thunderbolt 4 line rate); the actual volume is measured and counted instead

    def data_direction(self, point):
        return {"mode": "directional", "source_rank": point.source_rank if point.source_rank is not None else 1}

    def describe(self, point):
        duration, omit = self._duration(point)
        return {"payload_semantics": "iperf3 write block size (default); one TCP stream; throughput is TCP payload over the test duration, not link bandwidth", "backend": "iperf3",
                "timing_method": f"iperf3 interval accounting over {duration:g} s after {omit:g} s omitted; sender and receiver sides both reported",
                "completion_semantics": "test duration elapsed; receiver-side figure is the delivered goodput"}

    def plan(self, point, ctx, ports, rep_dir):
        server, client = point.ranks[0], point.ranks[1]
        port = ports[0]
        duration, omit = self._duration(point)
        src = point.source_rank if point.source_rank is not None else 1
        server_argv = [ctx.tool(self.caps.executable, server["host"]), "-s", "-1", "-B", ctx.rank_address(server), "-p", str(port)]
        client_argv = [ctx.tool(self.caps.executable, client["host"]), "-c", ctx.rank_address(server), "-p", str(port), "-B", ctx.rank_address(client), "-t", f"{duration:g}",
                       "-O", f"{omit:g}", "-P", "1", "-J", "--connect-timeout", "5000"] + (["-R"] if src == 0 else [])
        return Group([Invocation(0, "server", server["host"], server_argv, persistent=True, listens_on=port),
                      Invocation(1, "client", client["host"], client_argv, wait_for_port=port)], ports=[port])

    def parse(self, point, group, outcome):
        text = Path(outcome.stdout[1]).read_text()
        try:
            data = json.loads(text)
        except ValueError:
            raise ParseError("the iperf3 client did not print JSON")
        return self.measure(point, data)

    def measure(self, point, data):
        end, start = data.get("end") or {}, data.get("start") or {}
        if data.get("error") or not end.get("sum_sent") or not end.get("sum_received"):
            raise ParseError(f"iperf3 reported an error or no summary: {data.get('error', 'no end.sum_sent/sum_received')}")
        test = start.get("test_start") or {}
        if test.get("protocol") not in (None, "TCP"):
            raise ParseError("only TCP runs are accepted")
        sent, received = end["sum_sent"], end["sum_received"]
        for name, side in (("sent", sent), ("received", received)):
            if not isinstance(side.get("bits_per_second"), (int, float)) or side["bits_per_second"] < 0:
                raise ParseError(f"end.sum_{name}.bits_per_second is missing or invalid")
        if received["bits_per_second"] == 0:
            raise ParseError("the receiver saw no data")
        gbit = lambda side: side["bits_per_second"] / 1e9  # noqa: E731
        d = "payload bytes * 8 / seconds / 1e9 (decimal Gbit/s), as iperf3 reports it on the {} side; TCP payload, not link bandwidth"
        throughput = [{"kind": "iperf3_tcp_goodput", "value": gbit(received), "unit": "Gbit/s", "definition": d.format("receiver (delivered goodput)")},
                      {"kind": "iperf3_tcp_goodput", "value": gbit(sent), "unit": "Gbit/s", "definition": d.format("sender")}]
        cpu = end.get("cpu_utilization_percent") or {}
        extra = {"bytes_sent": {"value": sent.get("bytes"), "unit": "B"}, "bytes_received": {"value": received.get("bytes"), "unit": "B"},
                 "duration_sent": {"value": sent.get("seconds"), "unit": "s"}, "duration_received": {"value": received.get("seconds"), "unit": "s"},
                 "retransmits": {"value": sent.get("retransmits"), "unit": "segments (sender side; null where unavailable)"},
                 "parallel_streams": {"value": test.get("num_streams"), "unit": "count"}, "reverse": {"value": bool(test.get("reverse")), "unit": "bool"},
                 "cpu_host_total_percent": {"value": cpu.get("host_total"), "unit": "%"}, "cpu_remote_total_percent": {"value": cpu.get("remote_total"), "unit": "%"},
                 "iperf_version": {"value": str(start.get("version", "unknown")), "unit": "text"}, "block_size": {"value": test.get("blksize"), "unit": "B"}}
        warnings = []
        if (test.get("num_streams") or 1) != 1:
            warnings.append("more than one stream was used")
        return Parsed({"timing": None, "throughput": throughput, "extra": extra, "warnings": warnings},
                      {"status": "not_supported", "method": None, "details": "iperf3 does not verify payload content"})


class Iperf3Baseline:
    """Detection only. A baseline provider will later run `iperf3` (or Gloo, NCCL, Exo MlxRing, framework-native communication) under the same topology and
    emit result records; none is executed here, and a baseline is never described as RDMA or assumed to exist on every platform."""
    name = "iperf3"
    category = "network_transport"

    @staticmethod
    def detect():
        path = shutil.which("iperf3")
        if not path:
            return {"available": False, "path": None, "version": None}
        try:
            out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10).stdout.splitlines()
            version = out[0].strip() if out else None
        except (OSError, subprocess.TimeoutExpired):
            version = None
        return {"available": True, "path": path, "version": version}


BASELINE_PROVIDERS = {"iperf3": Iperf3Baseline, "gloo": None, "nccl": None, "exo_mlx_ring": None, "framework_native": None}


def inventoried_only():
    """Benchmarks audited in the inventory but not integrated; they appear in `list` and are rejected clearly if a configuration names them."""
    def stub(name, category, executable, description, ops):
        class _Stub(BenchmarkAdapter):
            caps = Capabilities(name=name, category=category, description=description, executable=executable, operations=ops, integrated=False)
        return _Stub()

    return [stub("async_raw_transport", "network_transport", "tbccl_async_raw_transport_bench", "Raw TCP transport baseline under the async worker (not integrated)", ("raw_transport",))]
