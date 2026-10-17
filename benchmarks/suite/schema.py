"""Versioned result schema: record construction, validation and the invariants a JSON schema cannot express.

A record is one workload point in one independent repetition. Only records whose status is 'success' carry measurements; failures, timeouts, skips and
unsupported points keep their raw artifacts and reason but never a number that could be mistaken for a valid measurement.
"""
from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path

from . import SUITE_VERSION

SCHEMA_VERSION = "tbccl-bench-result/1"
MANIFEST_VERSION = "tbccl-bench-manifest/1"
SCHEMA_PATH = Path(__file__).resolve().parent / "schemas" / "result-v1.json"

STATES = ("success", "failure", "timeout", "skipped", "unsupported", "invalid")
VERIFICATION = ("passed", "failed", "not_supported", "not_run")


class SchemaError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _type_ok(value, name):
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    raise ValueError(f"unsupported schema type {name}")


def validate_against(instance, schema, path="$") -> list:
    """Validator for the JSON Schema subset the result schema uses (type, enum, const, required, properties, additionalProperties, items, minimum).

    Kept in-tree so validation needs no third-party package; the tests cross-check it against `jsonschema` when that is installed.
    """
    errors = []
    if "const" in schema and instance != schema["const"]:
        errors.append(f"{path}: expected {schema['const']!r}, got {instance!r}")
        return errors
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} is not one of {schema['enum']}")
        return errors
    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(instance, n) for n in names):
            errors.append(f"{path}: expected {'/'.join(names)}, got {type(instance).__name__}")
            return errors
    if "minimum" in schema and isinstance(instance, (int, float)) and not isinstance(instance, bool) and instance < schema["minimum"]:
        errors.append(f"{path}: {instance} is below the minimum {schema['minimum']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required field '{key}'")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in props:
                errors.extend(validate_against(value, props[key], f"{path}.{key}"))
            elif extra is False:
                errors.append(f"{path}: unexpected field '{key}'")
            elif isinstance(extra, dict):
                errors.extend(validate_against(value, extra, f"{path}.{key}"))
    if isinstance(instance, list) and "items" in schema:
        for i, value in enumerate(instance):
            errors.extend(validate_against(value, schema["items"], f"{path}[{i}]"))
    return errors


def invariant_errors(record: dict) -> list:
    errors = []
    state = record.get("status", {}).get("state")
    verification = record.get("verification", {}).get("status")
    measurements = record.get("measurements")
    if state == "success":
        if not measurements:
            errors.append("a successful record must carry measurements")
        if verification == "failed":
            errors.append("a record whose verification failed cannot be a success")
        timing = (measurements or {}).get("timing")
        if timing:
            primary = timing.get("primary_statistic")
            if primary not in timing.get("statistics", {}):
                errors.append(f"primary statistic '{primary}' is not among the reported statistics")
            for name in timing["statistics"]:
                if name not in timing.get("statistics_provenance", {}):
                    errors.append(f"statistic '{name}' has no provenance")
            if timing.get("samples") is not None and timing.get("sample_count") not in (None, len(timing["samples"])):
                errors.append("sample_count does not match the number of samples")
        if measurements is not None and not (timing or measurements.get("throughput")):
            errors.append("a successful record needs a timing or a throughput measurement")
    elif measurements:
        errors.append(f"a record with status '{state}' must not carry measurements")
    return errors


def validate_record(record: dict, schema: dict | None = None) -> list:
    errors = validate_against(record, schema or load_schema())
    if not errors:
        errors.extend(invariant_errors(record))
    return errors


def require_valid(record: dict) -> dict:
    errors = validate_record(record)
    if errors:
        raise SchemaError(errors)
    return record


def check_schema_version(record: dict) -> None:
    version = record.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaError([f"unsupported result schema version {version!r}; this suite reads {SCHEMA_VERSION!r}"])


def empty_environment() -> dict:
    return {"hosts": [], "memory_backend": [], "memory_domain": {"label": "unknown", "per_rank": []},
            "topology": {"kind": "unknown", "interconnect": "unknown", "physical": False}, "transport": "unknown",
            "network_interface": None, "mtu": None, "framework_versions": {}, "multi_host": None}


def new_record(*, run_id: str, record_id: str, benchmark: dict, environment: dict, configuration: dict, status: str = "success",
               reason: str | None = None, exit_codes: dict | None = None, measurements: dict | None = None, verification: dict | None = None,
               artifacts: dict | None = None, timestamp_utc: str | None = None) -> dict:
    benchmark = {"adapter": None, "source_repository": None, "source_commit": None, "source_dirty": None, "executable": None, "executable_sha256": None,
                 "executable_version": None, "build_type": None, "algorithm_requested": None, "algorithm_resolved": None, "selection_reason": None,
                 "backend": None, "import_source": None, "suite_version": SUITE_VERSION, **benchmark}
    configuration = {"datatype": None, "reduce_op": None, "rank_order": [], "warmup": None, "iterations": None, "warmup_note": None, "timeout_s": None,
                     "options": {}, **configuration}
    record = {"schema_version": SCHEMA_VERSION, "record_id": record_id, "run_id": run_id, "timestamp_utc": timestamp_utc or utc_now(),
              "benchmark": benchmark, "environment": copy.deepcopy(environment), "configuration": configuration,
              "measurements": measurements if status == "success" else None,
              "verification": verification or {"status": "not_run", "method": None, "details": ""},
              "status": {"state": status, "reason": reason, "exit_codes": exit_codes or {}},
              "artifacts": {"commands": [], "stdout": [], "stderr": [], "raw": [], "source_files": [], **(artifacts or {})}}
    return record


def read_records(path) -> list:
    """Records from a results.jsonl, a single record JSON file or a JSON list. Rejects unknown schema versions."""
    path = Path(path)
    text = path.read_text()
    if path.suffix == ".jsonl":
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)
        records = data if isinstance(data, list) else [data]
    for record in records:
        check_schema_version(record)
    return records


def write_jsonl(path, records) -> None:
    with open(path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
