"""Baseline catalogs: loading, schema and invariant validation, privacy audit of public definitions, source-artifact provenance.

A catalog is either `definitions` (public: what is measured, how it may be compared, when it is valid; no measurements) or a `snapshot` (local: the values of one
set of campaigns, with the independent observations and the source artifacts they were computed from)."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

from .. import statistics as sstats
from ..schema import SchemaError, validate_against
from . import BASELINE_SCHEMA_VERSION

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.json"
DEFINITIONS_DIR = Path(__file__).resolve().parent / "definitions"
MEASURED_FIELDS = ("run_ids", "sample_count", "independent_lifecycle_count", "value", "variation", "observations", "pair_ids", "source_artifacts")
PRIVATE_PATTERNS = (
    (re.compile(r"\b(?!0\.0\.0\.0\b|127\.0\.0\.1\b)\d{1,3}(?:\.\d{1,3}){3}\b"), "IPv4 address"),
    (re.compile(r"\b[0-9a-fA-F]{2}(?::[0-9a-fA-F]{2}){5}\b"), "MAC address"),
    (re.compile(r"(?:^|[\s\"'=])(?:/home/|/Users/|/mnt/|/media/|~/)"), "private filesystem path"),
    (re.compile(r"\bfe80:[0-9a-fA-F:]*"), "IPv6 link-local address"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "e-mail address"),
)
VERSION_FIELDS = ("software_versions",)


def load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def load_catalog(path) -> dict:
    """Read and validate a catalog file. Raises SchemaError listing every problem."""
    doc = json.loads(Path(path).read_text())
    errors = validate_catalog(doc)
    if errors:
        raise SchemaError(errors)
    return doc


def validate_catalog(doc: dict) -> list:
    if not isinstance(doc, dict):
        return ["$: a catalog is a JSON object"]
    version = doc.get("schema_version")
    if version != BASELINE_SCHEMA_VERSION:
        return [f"unsupported baseline schema version {version!r}; this suite reads {BASELINE_SCHEMA_VERSION!r}"]
    errors = validate_against(doc, load_schema())
    if errors:
        return errors
    seen = set()
    for index, case in enumerate(doc["cases"]):
        where = f"cases[{index}] ({case['case_id']})"
        if case["case_id"] in seen:
            errors.append(f"{where}: duplicate case_id")
        seen.add(case["case_id"])
        errors += [f"{where}: {e}" for e in case_errors(case, doc["kind"])]
    if doc["kind"] == "definitions":
        errors += [f"privacy: {p}" for p in privacy_problems(doc)]
    return errors


def case_errors(case: dict, kind: str) -> list:
    errors = []
    present = [f for f in MEASURED_FIELDS if f in case]
    if kind == "definitions":
        if present:
            errors.append(f"a definition must not carry measured fields ({', '.join(present)})")
        return errors
    missing = [f for f in ("run_ids", "sample_count", "independent_lifecycle_count", "source_artifacts") if f not in case]
    status = case["validity"]["status"]
    if status in ("valid", "exploratory"):
        missing += [f for f in ("value", "observations") if f not in case or case[f] is None]
        if missing:
            errors.append("a snapshot case needs " + ", ".join(sorted(set(missing))))
    for artifact in case.get("source_artifacts", []):
        if artifact["path"].startswith(("/", "~")) or ".." in Path(artifact["path"]).parts:
            errors.append(f"source artifact path {artifact['path']!r} must be relative to the results root")
    obs = case.get("observations")
    if obs:
        finite = all(math.isfinite(v) for v in obs)
        if not finite:
            errors.append("observations contain NaN or infinity")
        units = case.get("independent_lifecycle_count")
        if units is not None and case.get("replication_unit") != "request" and len(obs) != units:
            errors.append(f"{len(obs)} observations but independent_lifecycle_count is {units}: an observation is one independent unit")
        pairs = case.get("pair_ids")
        if pairs is not None and (len(pairs) != len(obs) or len(set(pairs)) != len(pairs)):
            errors.append("pair_ids must be unique and match the observations one to one")
        if finite and case.get("value") is not None and case["aggregation"].startswith("median") and not math.isclose(case["value"], sstats.median(obs), rel_tol=1e-6, abs_tol=1e-9):
            errors.append(f"value {case['value']} is not the median of its observations ({sstats.median(obs)})")
    return errors


def privacy_problems(doc) -> list:
    """Strings in a public catalog that look like addresses, private paths or e-mail addresses. Reports category and location, never the matching text."""
    problems = []

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for i, value in enumerate(node):
                walk(value, f"{path}[{i}]")
        elif isinstance(node, str):
            for pattern, label in PRIVATE_PATTERNS:
                if pattern.search(node):
                    problems.append(f"{path}: {label}")

    walk(doc, "$")
    return problems


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_provenance(case: dict, results_root) -> list:
    """Problems with a snapshot case's source artifacts under `results_root`: missing files and hash mismatches. A case without artifacts is itself a problem."""
    root = Path(results_root)
    artifacts = case.get("source_artifacts") or []
    if not artifacts:
        return [f"{case['case_id']}: no source artifacts; the value cannot be traced to saved measurements"]
    problems = []
    for artifact in artifacts:
        path = root / artifact["path"]
        if not path.is_file():
            problems.append(f"{case['case_id']}: source artifact {artifact['path']} is missing")
        elif artifact.get("sha256") and sha256_file(path) != artifact["sha256"]:
            problems.append(f"{case['case_id']}: source artifact {artifact['path']} no longer matches its recorded SHA-256")
    return problems


def index_cases(doc: dict) -> dict:
    return {case["case_id"]: case for case in doc["cases"]}


def definitions_catalogs(directory=DEFINITIONS_DIR) -> list:
    return [load_catalog(p) for p in sorted(Path(directory).glob("*.json"))]


def case_by_id(case_id: str, directory=DEFINITIONS_DIR) -> dict:
    for doc in definitions_catalogs(directory):
        found = index_cases(doc).get(case_id)
        if found:
            return found
    raise KeyError(case_id)
