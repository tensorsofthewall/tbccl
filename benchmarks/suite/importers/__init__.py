"""Importers for results produced before the suite existed. They preserve provenance ('imported'), never invent fields the source lacks, and never copy network addresses."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from .. import schema
from ..adapters import REGISTRY
from ..adapters.base import Point, memory_domain_label


class ImportFailure(Exception):
    pass


class ImportResult:
    def __init__(self, records, source_format, skipped=None):
        self.records, self.source_format, self.skipped = records, source_format, skipped or []


def import_run_id(path: Path) -> str:
    return "imported-" + hashlib.sha256(Path(path).read_bytes()).hexdigest()[:10]


def unknown_environment(backends=("host", "host"), transport="tcp"):
    env = schema.empty_environment()
    env["memory_backend"] = list(backends)
    env["memory_domain"] = {"label": memory_domain_label(backends) if backends else "unknown", "per_rank": list(backends)}
    env["transport"] = transport
    return env


def import_point(benchmark, operation, *, payload_bytes, rank_count=2, backends=("host", "host"), algorithm=None, datatype=None, reduce_op=None, iterations=0, warmup=0):
    return Point(workload="imported", benchmark=benchmark, operation=operation, algorithm=algorithm, payload_bytes=payload_bytes, datatype=datatype, reduce_op=reduce_op,
                 rank_count=rank_count, ranks=[{"rank": i, "host": "unknown", "memory_backend": b} for i, b in enumerate(backends)], options={}, warmup=warmup,
                 iterations=iterations, repetitions=1, timeout_s=0.0)


def record_from_parsed(point, parsed, *, run_id, index, source_name, environment, repetition_index=0, repetitions=1, extra_benchmark=None, extra_configuration=None):
    adapter = REGISTRY[point.benchmark]
    description = adapter.describe(point)
    configuration = {"payload_bytes": point.payload_bytes, "payload_semantics": description["payload_semantics"], "datatype": point.datatype, "reduce_op": point.reduce_op,
                     "rank_count": point.rank_count, "rank_order": [], "warmup": point.warmup, "iterations": point.iterations, "repetitions": repetitions,
                     "repetition_index": repetition_index, "timeout_s": None, "timing_method": description["timing_method"], "completion_semantics": description["completion_semantics"]}
    configuration.update(parsed.configuration_extra)
    configuration.update(extra_configuration or {})
    benchmark = {"name": adapter.caps.name, "adapter": adapter.caps.name, "category": adapter.caps.category, "operation": point.operation, "provenance": "imported",
                 "import_source": source_name, "backend": description.get("backend"), "algorithm_requested": point.algorithm,
                 "algorithm_resolved": parsed.algorithm_resolved, "selection_reason": parsed.selection_reason, **(extra_benchmark or {})}
    verification = {"status": "not_run", "method": None, "details": "historical file: the verification outcome was not recorded with the results"}
    return schema.new_record(run_id=run_id, record_id=f"{run_id}/{index}", benchmark=benchmark, environment=environment, configuration=configuration, status="success",
                             measurements=parsed.measurements, verification=verification, artifacts={"source_files": [source_name]})


def apply_overrides(records, overrides):
    """--set section.path=value, value read as JSON when it parses, else as text. Only 'benchmark', 'environment' and 'configuration' may be changed."""
    for item in overrides or []:
        if "=" not in item:
            raise ImportFailure(f"--set expects PATH=VALUE, got {item!r}")
        path, raw = item.split("=", 1)
        keys = path.split(".")
        if keys[0] not in ("benchmark", "environment", "configuration") or len(keys) < 2:
            raise ImportFailure(f"--set path must start with benchmark., environment. or configuration.: {path!r}")
        try:
            value = json.loads(raw)
        except ValueError:
            value = raw
        for record in records:
            target = record
            for key in keys[:-1]:
                target = target.setdefault(key, {})
            target[keys[-1]] = copy.deepcopy(value)
    for record in records:
        env = record["environment"]
        backends = env.get("memory_backend") or []
        env["memory_domain"] = {"label": memory_domain_label(backends) if backends else "unknown", "per_rank": list(backends)}
    return records


def import_path(path, overrides=None, skip_bad_rows=False) -> ImportResult:
    from . import csv_importer, json_importer, jsonl_importer
    path = Path(path)
    if not path.exists():
        raise ImportFailure(f"{path}: file not found")
    suffix = path.suffix.lower()
    module = {".csv": csv_importer, ".json": json_importer, ".jsonl": jsonl_importer}.get(suffix)
    if module is None:
        raise ImportFailure(f"{path.name}: unsupported file type '{suffix}' (supported: .csv, .json, .jsonl)")
    result = module.import_file(path, skip_bad_rows=skip_bad_rows)
    apply_overrides(result.records, overrides)
    for record in result.records:
        errors = schema.validate_record(record)
        if errors:
            raise ImportFailure(f"{path.name}: imported record failed schema validation: {'; '.join(errors)[:300]}")
    return result
