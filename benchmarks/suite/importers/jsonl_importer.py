"""JSON Lines: small_message_latency output and the suite's own results.jsonl."""
from __future__ import annotations

import json
from pathlib import Path

from .. import schema
from ..adapters import REGISTRY
from ..adapters.base import ParseError
from . import ImportFailure, ImportResult, import_point, import_run_id, record_from_parsed, unknown_environment

REQUIRED = ("mode", "bytes", "median_us", "p25_us", "p75_us", "p95_us", "n", "cpu_cores", "sleep_ratio")


def import_file(path: Path, skip_bad_rows=False) -> ImportResult:
    lines = [(i, line) for i, line in enumerate(path.read_text().splitlines(), start=1) if line.strip()]
    if not lines:
        raise ImportFailure(f"{path.name}: empty file")
    objects, skipped = [], []
    for i, line in lines:
        try:
            objects.append((i, json.loads(line)))
        except ValueError as error:
            if not skip_bad_rows:
                raise ImportFailure(f"{path.name}:{i}: malformed JSON ({error})") from error
            skipped.append({"line": i, "reason": f"malformed JSON ({error})"})
    if objects and "schema_version" in objects[0][1]:
        return _own(path, objects)
    run_id = import_run_id(path)
    records = []
    for i, obj in objects:
        try:
            missing = [k for k in REQUIRED if k not in obj]
            if "mode" in obj and obj["mode"] in ("replay", "idle"):
                raise ParseError(f"mode '{obj['mode']}' is not a latency measurement and is not imported")
            if missing:
                raise ParseError("missing fields: " + ", ".join(missing))
            if obj["mode"] not in REGISTRY["small_message_latency"].OPS:
                raise ParseError(f"unknown mode {obj['mode']!r}")
            adapter = REGISTRY["small_message_latency"]
            point = import_point("small_message_latency", obj["mode"], payload_bytes=int(obj["bytes"]))
            extra_cfg = {"options": {"label": obj.get("label", "")}}
            records.append(record_from_parsed(point, adapter.measure(point, obj), run_id=run_id, index=len(records), source_name=path.name,
                                              environment=unknown_environment(("host", "host")), extra_configuration=extra_cfg))
        except (ParseError, ValueError, KeyError) as error:
            if not skip_bad_rows:
                raise ImportFailure(f"{path.name}:{i}: {error}") from error
            skipped.append({"line": i, "reason": str(error)})
    if not records:
        raise ImportFailure(f"{path.name}: no valid lines ({len(skipped)} rejected)")
    return ImportResult(records, "small_message_latency", skipped)


def _own(path, objects):
    records = []
    for i, obj in objects:
        try:
            schema.check_schema_version(obj)
        except schema.SchemaError as error:
            raise ImportFailure(f"{path.name}:{i}: {error}") from error
        errors = schema.validate_record(obj)
        if errors:
            raise ImportFailure(f"{path.name}:{i}: record failed validation: {'; '.join(errors)[:200]}")
        records.append(obj)
    return ImportResult(records, "suite_results", [])
