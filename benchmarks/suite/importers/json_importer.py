"""JSON files: iperf3 output, tools/benchmark_allreduce.py ('tbccl-benchmark/1') and a suite result record or list."""
from __future__ import annotations

import json
from pathlib import Path

from .. import schema
from ..adapters.base import Parsed, timing
from ..statistics import convert
from . import ImportFailure, ImportResult, import_run_id, unknown_environment


def import_file(path: Path, skip_bad_rows=False) -> ImportResult:
    try:
        data = json.loads(path.read_text())
    except ValueError as error:
        raise ImportFailure(f"{path.name}: malformed JSON ({error})") from error
    if isinstance(data, dict) and data.get("schema") == "tbccl-benchmark/1":
        return _allreduce_report(path, data)
    if isinstance(data, dict) and "start" in data and "end" in data and "version" in (data.get("start") or {}):
        return _iperf3(path, data)
    if isinstance(data, (dict, list)) and "schema_version" in (data[0] if isinstance(data, list) and data else data):
        from .jsonl_importer import _own
        items = data if isinstance(data, list) else [data]
        return _own(path, list(enumerate(items, start=1)))
    raise ImportFailure(f"{path.name}: unrecognised JSON; supported: iperf3 output, 'tbccl-benchmark/1' reports, suite result records")


def _require(obj, keys, where):
    missing = [k for k in keys if k not in obj]
    if missing:
        raise ImportFailure(f"{where}: missing fields: {', '.join(missing)}")


def _allreduce_report(path, data):
    _require(data, ("environment", "settings", "results"), path.name)
    env = data["environment"]
    run_id = import_run_id(path)
    records = []
    backend = env.get("backend", "unknown")
    for i, result in enumerate(data["results"]):
        _require(result, ("bytes", "seconds_per_op_median", "seconds_per_op_min", "seconds_per_op_max", "repeats", "algbw_gbps_median", "verified"), f"{path.name}: results[{i}]")
        stats = {"median": convert(float(result["seconds_per_op_median"]), "s", "us"), "min": convert(float(result["seconds_per_op_min"]), "s", "us"),
                 "max": convert(float(result["seconds_per_op_max"]), "s", "us")}
        if any(v <= 0 for v in stats.values()):
            raise ImportFailure(f"{path.name}: results[{i}]: non-positive time")
        tm = timing("all_reduce_time_per_loop_mean", stats, "reported", "median", percentile_method=None, sample_count=int(result["repeats"]))
        parsed = Parsed({"timing": tm, "throughput": [{"kind": "application_payload", "value": float(result["algbw_gbps_median"]), "unit": "GB/s",
                                                       "definition": "payload bytes / median seconds per all_reduce (rank 0); decimal GB/s; no bus-bandwidth correction"}],
                         "extra": {"verified_flag": {"value": bool(result["verified"]), "unit": "bool"}}, "warnings": []},
                        {"status": "passed" if result["verified"] else "failed", "method": "report flag: one untimed verified all_reduce per size", "details": "from the report"})
        settings = data["settings"]
        world = int(env.get("world_size", 2))
        benchmark = {"name": "allreduce_harness", "adapter": None, "category": "collective", "operation": "all_reduce", "provenance": "imported", "import_source": path.name,
                     "backend": f"torch.distributed:{backend}", "source_repository": "tbccl"}
        configuration = {"payload_bytes": int(result["bytes"]), "payload_semantics": "float32 tensor bytes per rank (torch.distributed all_reduce)", "datatype": settings.get("dtype"),
                         "reduce_op": "sum", "rank_count": world, "rank_order": [], "warmup": settings.get("warmup"), "iterations": settings.get("iters"),
                         "repetitions": int(result["repeats"]), "repetition_index": 0, "timeout_s": None,
                         "timing_method": "mean seconds per all_reduce over a timed loop of `iters` calls; statistics are over `repeats` loops (not per-call samples)",
                         "completion_semantics": "barrier after the loop; rank 0 view"}
        environment = unknown_environment(("host",) * world)
        environment["framework_versions"] = {k: str(v) for k, v in env.items() if k in ("torch", "torch_tbccl", "libtbccl", "python")}
        environment["multi_host"] = not env.get("master_addr_is_loopback", True)
        environment["topology"] = {"kind": "loopback" if env.get("master_addr_is_loopback") else "unknown", "interconnect": "loopback" if env.get("master_addr_is_loopback") else "unknown",
                                   "physical": not env.get("master_addr_is_loopback", True)}
        status = "success" if result["verified"] else "failure"
        record = schema.new_record(run_id=run_id, record_id=f"{run_id}/{i}", benchmark=benchmark, environment=environment, configuration=configuration, status=status,
                                   reason=None if result["verified"] else "the report marks this size as not verified",
                                   measurements=parsed.measurements if result["verified"] else None, verification=parsed.verification, artifacts={"source_files": [path.name]})
        records.append(record)
    if not records:
        raise ImportFailure(f"{path.name}: the report has no results")
    return ImportResult(records, "tbccl-benchmark/1", [])


def _iperf3(path, data):
    end = data["end"]
    run_id = import_run_id(path)
    _require(end, ("sum_sent", "sum_received"), f"{path.name}: end")
    start = data["start"]
    test = start.get("test_start", {})
    block = int(test.get("blksize") or 0) or None
    streams = test.get("num_streams")
    records = []
    for direction, key in (("sent", "sum_sent"), ("received", "sum_received")):
        bps = end[key].get("bits_per_second")
        if bps is None or bps < 0:
            raise ImportFailure(f"{path.name}: end.{key}.bits_per_second is missing or negative")
        value = convert(float(bps), "bit/s", "Gbit/s")
        measurements = {"timing": None, "throughput": [{"kind": "iperf3_tcp_goodput", "value": value, "unit": "Gbit/s",
                                                        "definition": f"iperf3 end.{key}: TCP payload bits per second over the test duration ({direction} side); not link bandwidth"}],
                        "extra": {"duration": {"value": float(end[key].get("seconds", 0.0)), "unit": "s"}, "streams": {"value": streams, "unit": "count"}}, "warnings": []}
        environment = unknown_environment((), transport="tcp")
        environment["framework_versions"] = {"iperf3": str(start.get("version", "unknown"))}
        if test.get("protocol") not in (None, "TCP"):
            raise ImportFailure(f"{path.name}: only TCP iperf3 runs are imported (protocol {test.get('protocol')!r})")
        benchmark = {"name": "iperf3", "adapter": None, "category": "network_transport", "operation": f"tcp_stream_{direction}", "provenance": "imported", "import_source": path.name,
                     "backend": "iperf3"}
        configuration = {"payload_bytes": block, "payload_semantics": "iperf3 write block size (--length)", "rank_count": 2, "rank_order": [], "warmup": 0, "iterations": None,
                         "repetitions": 1, "repetition_index": 0, "timeout_s": None, "timing_method": "iperf3 interval accounting over the full test duration",
                         "completion_semantics": "test duration elapsed", "options": {"streams": streams, "reverse": bool(test.get("reverse"))}}
        records.append(schema.new_record(run_id=run_id, record_id=f"{run_id}/{direction}", benchmark=benchmark, environment=environment, configuration=configuration,
                                         status="success", measurements=measurements,
                                         verification={"status": "not_supported", "method": None, "details": "iperf3 does not verify payload content"}, artifacts={"source_files": [path.name]}))
    return ImportResult(records, "iperf3", [])
