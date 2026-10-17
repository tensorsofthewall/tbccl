"""CSV results: tb_pingpong, latency_sweep raw files, tbccl_collective_bench and tbccl_tensor_transfer_bench."""
from __future__ import annotations

import csv
from pathlib import Path

from ..adapters import REGISTRY
from ..adapters.base import ParseError
from . import ImportFailure, ImportResult, import_point, import_run_id, record_from_parsed, unknown_environment

FORMATS = {
    "tcp_pingpong": ["mode", "size_bytes", "iterations", "seconds", "rtt_us", "gbps", "messages_per_second"],
    "latency_sweep_raw": ["direction", "config", "busy_poll_us", "trial", "size_bytes", "iterations", "rtt_us", "gbps", "messages_per_second"],
    "latency_sweep_summary": ["direction", "config", "busy_poll_us", "size_bytes", "trials", "min_rtt_us", "mean_rtt_us", "median_rtt_us"],
    "collective_bench": ["collective", "requested_algorithm", "algorithm", "selection_reason", "datatype", "op", "world_size", "segment_bytes", "input_bytes_per_rank",
                         "output_bytes_per_rank", "chunk_bytes", "chunks_per_segment", "iterations", "warmup", "median_us", "p95_us", "min_us", "max_us", "algbw_GBps", "busbw_GBps"],
    "tensor_transfer": ["mode", "stage", "source_rank", "source_backend", "destination_backend", "payload_bytes", "iterations", "warmup", "median_us", "p95_us", "p99_us", "min_us",
                        "max_us", "effective_GBps"],
}
COLLECTIVE_OPS = {"all-reduce": "all_reduce", "all-gather": "all_gather", "reduce-scatter": "reduce_scatter"}
TENSOR_OPS = {"latency-floor": "latency_floor", "network-only": "network_only", "end-to-end": "end_to_end"}


def detect(header):
    for name, required in FORMATS.items():
        if header[:len(required)] == required or (name in ("collective_bench", "tensor_transfer") and set(required) <= set(header)):
            return name
    for name, required in FORMATS.items():
        if name != "latency_sweep_summary" and header[:2] == required[:2]:
            missing = [c for c in required if c not in header]
            raise ImportFailure(f"looks like a {name} file but is missing columns: {', '.join(missing)}")
    raise ImportFailure("unrecognised CSV header; supported: " + ", ".join(sorted(FORMATS)))


def import_file(path: Path, skip_bad_rows=False) -> ImportResult:
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames or []
        kind = detect(header)
        rows = list(reader)
    if kind == "latency_sweep_summary":
        raise ImportFailure(f"{path.name}: this is an aggregate summary (min/mean/median per size); import the *-raw.csv files it was computed from")
    if not rows:
        raise ImportFailure(f"{path.name}: no data rows")
    run_id = import_run_id(path)
    records, skipped = [], []
    for line_no, row in enumerate(rows, start=2):
        try:
            records.append(_convert(kind, row, run_id, len(records), path.name))
        except (ParseError, ValueError, KeyError) as error:
            if not skip_bad_rows:
                raise ImportFailure(f"{path.name}:{line_no}: {error}") from error
            skipped.append({"line": line_no, "reason": str(error)})
    if not records:
        raise ImportFailure(f"{path.name}: no valid rows ({len(skipped)} rejected)")
    return ImportResult(records, kind, skipped)


def _pingpong_row(row, mode):
    if mode not in ("pingpong", "stream"):
        raise ParseError(f"unknown mode {mode!r}")
    return {"iterations": int(row["iterations"]), "seconds": float(row["seconds"]) if "seconds" in row else None, "rtt_us": float(row["rtt_us"]),
            "gbps": float(row["gbps"]), "mps": float(row["messages_per_second"])}


def _convert(kind, row, run_id, index, source):
    if kind in ("tcp_pingpong", "latency_sweep_raw"):
        mode = row.get("mode", "pingpong")
        adapter = REGISTRY["tcp_pingpong"]
        size = int(row["size_bytes"])
        point = import_point("tcp_pingpong", mode, payload_bytes=size, iterations=int(row["iterations"]))
        parsed = adapter.measure(point, _pingpong_row(row, mode))
        extra_cfg, repetition = {}, 0
        if kind == "latency_sweep_raw":
            extra_cfg = {"options": {"direction": row["direction"], "config": row["config"], "busy_poll_us": int(row["busy_poll_us"])}}
            repetition = int(row["trial"]) - 1
        return record_from_parsed(point, parsed, run_id=run_id, index=index, source_name=source, environment=unknown_environment(("host", "host")),
                                  repetition_index=max(repetition, 0), repetitions=1, extra_configuration=extra_cfg)
    if kind == "collective_bench":
        op = COLLECTIVE_OPS.get(row["collective"])
        if op is None:
            raise ParseError(f"unknown collective {row['collective']!r}")
        adapter = REGISTRY["collective_bench"]
        key = "segment_bytes" if op == "reduce_scatter" else "input_bytes_per_rank"
        n = int(row["world_size"])
        point = import_point("collective_bench", op, payload_bytes=int(row[key]), rank_count=n, backends=("host",) * n, algorithm=row["requested_algorithm"],
                             datatype=row.get("datatype") or None, reduce_op=row.get("op") or None, iterations=int(row["iterations"]), warmup=int(row["warmup"]))
        point.options = {"chunk_bytes": int(row["chunk_bytes"])} if row.get("chunk_bytes", "0") not in ("0", "") else {}
        return record_from_parsed(point, adapter.measure(point, row), run_id=run_id, index=index, source_name=source, environment=unknown_environment(("host",) * n))
    if kind == "tensor_transfer":
        op = TENSOR_OPS.get(row["mode"])
        if op is None:
            raise ParseError(f"unsupported tensor_transfer mode {row['mode']!r} (supported: {', '.join(TENSOR_OPS)})")
        backends = (row["source_backend"], row["destination_backend"])
        point = import_point("tensor_transfer", op, payload_bytes=int(row["payload_bytes"]), backends=backends, iterations=int(row["iterations"]), warmup=int(row["warmup"]))
        if row.get("timing_scope", "NA") not in ("NA", ""):
            point.options = {"timing_scope": row["timing_scope"]}
        return record_from_parsed(point, REGISTRY["tensor_transfer"].measure(point, row), run_id=run_id, index=index, source_name=source,
                                  environment=unknown_environment(backends))
    raise ImportFailure(f"unsupported CSV format {kind}")
