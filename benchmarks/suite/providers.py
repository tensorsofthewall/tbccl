"""Benchmark providers: the framework-specific workloads that live in the repository owning the framework integration.

The suite never imports a framework and never loads a provider into its own process. A provider is a directory (``executables.<name>_bench_dir`` on a host) holding an entry point,
``provider.py``, that the suite runs with the provider's own interpreter (``executables.<name>_python``). The entry point answers ``describe`` (and optionally ``validate``) with one
JSON object on stdout; the per-rank workload scripts it ships are launched by the runner under its usual supervision, so a provider never starts a detached process.

Three versions are kept apart: the provider protocol version (this module), the result schema version (``schema.SCHEMA_VERSION``) and the methodology version (each workload).
The protocol is specified in docs/development/benchmark-providers.md.
"""
from __future__ import annotations

import json
import re
import os
import subprocess
from pathlib import Path

PROTOCOL_VERSION = 1
SUPPORTED_PROTOCOLS = (1,)
SUPPORTED_RESULT_SCHEMAS = ("tbccl-bench-result/1",)
DESCRIBE_TIMEOUT_S = 30.0
MAX_DESCRIPTION_BYTES = 256 * 1024
ENTRY_POINT = "provider.py"

# provider id -> (executables key for the interpreter, executables key for the provider directory, adapters that use it)
PROVIDERS = {
    "torch-tbccl": {"python": "torch_python", "dir": "torch_bench_dir", "legacy_dir": "torch_bench", "adapters": ("torch_collectives", "torch_p2p", "torch_async", "torch_ddp")},
    "vllm-tbccl": {"python": "vllm_python", "dir": "vllm_bench_dir", "legacy_dir": "vllm_bench", "adapters": ("vllm_startup", "vllm_correctness", "vllm_latency", "vllm_throughput", "vllm_concurrency", "vllm_pipeline", "vllm_backend")},
    "exo-tbccl": {"python": "exo_python", "dir": "exo_bench_dir", "legacy_dir": "exo_bench", "adapters": ("exo_correctness", "exo_latency", "exo_throughput", "exo_transport_ab", "exo_stage_profile", "exo_large_model", "exo_lifecycle")},
}

AVAILABILITY = ("available", "environment_missing", "unsupported")
REQUIRED_FIELDS = {
    "provider_id": str, "provider_protocol_version": int, "provider_version": str, "supported_workloads": list, "supported_platforms": list, "supported_memory_kinds": list,
    "supported_device_pairings": list, "required_framework_versions": dict, "required_runtime_capabilities": list, "optional_dependencies": list,
    "required_model_configuration": dict, "benchmark_configuration_schema": dict, "supported_result_schema_versions": list, "entry_points": dict, "availability": dict,
}
# the keys a description may report about the environment it found; anything the provider could not determine is None, never invented
VERSION_KEYS = ("provider_version", "framework", "framework_version", "adapter_version", "libtbccl_build")


def validate_description(desc, expected_id=None) -> list:
    """Problems with a provider description (empty when it is usable by this suite)."""
    if not isinstance(desc, dict):
        return ["the description is not a JSON object"]
    problems = []
    for key, kind in REQUIRED_FIELDS.items():
        if key not in desc:
            problems.append(f"missing field '{key}'")
        elif not isinstance(desc[key], kind) or (kind is int and isinstance(desc[key], bool)):
            problems.append(f"field '{key}' must be {kind.__name__}")
    if problems:
        return problems
    if expected_id and desc["provider_id"] != expected_id:
        problems.append(f"provider_id is '{desc['provider_id']}', expected '{expected_id}'")
    if desc["provider_protocol_version"] not in SUPPORTED_PROTOCOLS:
        problems.append(f"provider protocol {desc['provider_protocol_version']} is not supported (this suite speaks {list(SUPPORTED_PROTOCOLS)})")
    if not set(desc["supported_result_schema_versions"]) & set(SUPPORTED_RESULT_SCHEMAS):
        problems.append(f"none of the provider's result schema versions {desc['supported_result_schema_versions']} is readable here ({list(SUPPORTED_RESULT_SCHEMAS)})")
    availability = desc["availability"]
    if availability.get("status") not in AVAILABILITY:
        problems.append(f"availability.status must be one of {AVAILABILITY}")
    elif availability["status"] != "available" and not availability.get("reasons"):
        problems.append("an unavailable provider must give its reasons")
    hashes = desc.get("entry_point_sha256", {})
    if not isinstance(hashes, dict) or not all(isinstance(k, str) and isinstance(v, str) and len(v) == 64 and set(v) <= set("0123456789abcdef") for k, v in hashes.items()):
        problems.append("entry_point_sha256 must map entry point names to SHA-256 hex digests")
    elif set(hashes) - set(desc["entry_points"]):
        problems.append(f"entry_point_sha256 names unknown entry points: {sorted(set(hashes) - set(desc['entry_points']))}")
    for entry, relative in desc["entry_points"].items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            problems.append(f"entry point '{entry}' must be a path relative to the provider directory")
    return problems


def parse_description(text) -> tuple:
    """(description or None, problems) from a provider's stdout. Only one JSON object is accepted: logs belong on stderr."""
    if len(text.encode("utf-8", "replace")) > MAX_DESCRIPTION_BYTES:
        return None, [f"the description is larger than {MAX_DESCRIPTION_BYTES} bytes"]
    try:
        desc = json.loads(text)
    except ValueError:
        return None, ["stdout is not a single JSON object (logs must go to stderr)"]
    return (desc, validate_description(desc)) if isinstance(desc, dict) else (None, ["stdout is not a JSON object"])


def run_describe(python, bench_dir, timeout=DESCRIBE_TIMEOUT_S, operation="describe") -> dict:
    """Run ``<python> <bench_dir>/provider.py <operation>`` locally. Returns {'returncode', 'stdout', 'stderr_tail'} or {'error'}. No shell is used and the working directory is the provider's own."""
    script = Path(bench_dir) / ENTRY_POINT
    if not script.is_file():
        return {"error": f"{script} is not a file"}
    env = {k: v for k, v in os.environ.items() if not k.startswith("TBCCL_")}  # a variable in the user's shell must not change what a provider reports
    try:
        done = subprocess.run([str(python), str(script), operation], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, cwd=str(Path(bench_dir)), env=env)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return {"returncode": done.returncode, "stdout": done.stdout, "stderr_tail": done.stderr.strip()[-300:]}


def remote_describe(host, python, bench_dir, timeout=DESCRIBE_TIMEOUT_S) -> dict:
    """The same describe on a non-local host, through the host's launcher and the suite's helper (no file is installed there)."""
    from .remote import session
    data, error = session.run_agent_json(host, ["describe-provider", "--python", python, "--dir", bench_dir, "--timeout", str(timeout)], timeout=timeout + 30, python=host.python)
    return data if data is not None else {"error": error}


def interpret(provider_id, raw) -> dict:
    """One provider's discovery entry from the raw outcome of a describe call."""
    if "error" in raw:
        return {"provider_id": provider_id, "status": "describe_failed", "reasons": [raw["error"]], "description": None}
    if raw["returncode"] != 0:
        return {"provider_id": provider_id, "status": "describe_failed", "reasons": [f"describe exited {raw['returncode']}: {raw['stderr_tail']}"], "description": None}
    desc, problems = parse_description(raw["stdout"])
    if problems:
        return {"provider_id": provider_id, "status": "incompatible", "reasons": problems, "description": desc}
    problems = validate_description(desc, expected_id=provider_id)
    if problems:
        return {"provider_id": provider_id, "status": "incompatible", "reasons": problems, "description": desc}
    return {"provider_id": provider_id, "status": desc["availability"]["status"], "reasons": list(desc["availability"].get("reasons") or []), "description": desc}


def bundled_dir(repo_root, provider_id) -> Path:
    """The compatibility copy of a provider's workload scripts that still ships with the suite (see docs/development/benchmark-providers.md)."""
    return Path(repo_root) / "benchmarks" / "suite" / PROVIDERS[provider_id]["legacy_dir"]


def bundled_hashes(repo_root, provider_id) -> dict:
    """SHA-256 of the scripts of the bundled compatibility copy, so a result produced with it names the exact scripts too."""
    import hashlib
    root = bundled_dir(repo_root, provider_id)
    return {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(root.rglob("*.py")) if "__pycache__" not in f.parts} if root.is_dir() else {}


def configured(host, provider_id) -> tuple:
    """(python, directory) as configured on a host's executables, either of which may be None. Nothing is searched for on PATH or in the current directory."""
    spec = PROVIDERS[provider_id]
    return host.executables.get(spec["python"]), host.executables.get(spec["dir"])


def discover(config, ctx, probe=True, timeout=DESCRIBE_TIMEOUT_S, remote_describe=None) -> list:
    """Status of every provider on every host of a configuration: 'native' is always built in; each external provider is judged on what its host configures.

    `remote_describe(host, python, bench_dir, timeout)` runs a describe on a non-local host (the runner passes its launcher-based helper); without it remote hosts are reported unprobed.
    """
    rows = [{"provider_id": "native", "host": None, "status": "available", "reasons": ["built in"], "source": "built-in", "description": None}]
    for provider_id in PROVIDERS:
        for host_id, host in config.hosts.items():
            python, directory = configured(host, provider_id)
            row = {"provider_id": provider_id, "host": host_id, "python": python, "directory": directory, "source": "configured" if directory else None}
            if not python or not directory:
                missing = [k for k, v in ((PROVIDERS[provider_id]["python"], python), (PROVIDERS[provider_id]["dir"], directory)) if not v]
                row.update(status="not_configured", reasons=[f"hosts.{host_id}.executables.{', '.join(missing)} not set"], description=None)
            elif not probe:
                row.update(status="configured", reasons=["not probed"], description=None)
            elif host.is_local:
                row.update(interpret(provider_id, run_describe(python, directory, timeout)))
            elif remote_describe is None:
                row.update(status="configured", reasons=["remote host not probed"], description=None)
            else:
                row.update(interpret(provider_id, remote_describe(host, python, directory, timeout)))
            rows.append(row)
    return rows


def record_entries(rows, host_ids) -> list:
    """The provenance stored in result records: what each provider on the participating hosts reported about itself and its environment. Missing provenance stays explicit."""
    out = []
    for row in rows:
        if row.get("host") not in host_ids or row.get("description") is None:
            continue
        d = row["description"]
        out.append({"provider_id": row["provider_id"], "host": row["host"], "protocol_version": d["provider_protocol_version"], "provider_version": d["provider_version"],
                    "status": row["status"], "interpreter": row.get("python"), "directory": row.get("directory"),
                    "framework_versions": {k: v for k, v in (d["availability"].get("detected") or {}).items() if k != "native_modules"},
                    "native_modules": (d["availability"].get("detected") or {}).get("native_modules"), "result_schema_versions": d["supported_result_schema_versions"],
                    "entry_point_sha256": d.get("entry_point_sha256") or {}})
    return out


_ABSOLUTE_PATH = re.compile(r"(?<![\w.~/>}-])(/(?:[^\s/:;,'\")]+/)*[^\s/:;,'\")]+)")


def _shorten(match):
    """Four or more components keep the last two. A shorter path (`/home/<user>/python`, `/mnt/<user>/venv`) keeps only the last one, because its second component can be a user name."""
    parts = match.group(1).split("/")[1:]
    if len(parts) < 2:
        return match.group(1)
    if len(parts) == 2 and parts[0] in ("home", "Users"):
        return "~"  # a home directory itself: its name is the user name
    return "<...>/" + "/".join(parts[-2:] if len(parts) >= 4 else parts[-1:])


def shareable_path(path):
    """An absolute path as it may be recorded in a result that is shared: the last two components (one for a short path), so `<...>/benchmarks/provider` and `<...>/bin/python` stay recognisable without revealing user names, mount points or directory layout. Relative values and None are unchanged."""
    if not isinstance(path, str):
        return path
    return _ABSOLUTE_PATH.sub(_shorten, path)


def shareable_entry(entry: dict) -> dict:
    """A provider provenance entry with the interpreter, directory and note stripped of absolute paths. The identity comparison uses versions and hashes, never paths."""
    out = dict(entry)
    for key in ("interpreter", "directory", "note"):
        if key in out:
            out[key] = shareable_path(out[key])
    return out


def format_rows(rows) -> str:
    lines = []
    for r in rows:
        where = f" on {r['host']}" if r.get("host") else ""
        reasons = "; ".join(r["reasons"]) if r.get("reasons") else ""
        lines.append(f"{r['provider_id']:<12} {r['status'].replace('_', ' '):<20}{where}" + (f"  ({reasons})" if reasons else ""))
    return "\n".join(lines)


def provider_of(adapter_name):
    for provider_id, spec in PROVIDERS.items():
        if adapter_name in spec["adapters"]:
            return provider_id
    return None


# ---- validate (protocol 1, optional operation): the runner stages each workload configuration on the host that will run it and asks the provider's own interpreter to check it
_DEVICE = {"host": "cpu", "cuda": "cuda", "mps": "mps", "metal": "metal"}
_INT_OPTIONS = {"torch-tbccl": ("layers", "width", "batch", "steps"), "vllm-tbccl": (), "exo-tbccl": ()}


def workload_config(point):
    """The provider-side description of one planned point (the keys of the provider's configuration schema), or None for a point no provider owns. Only values of the right type are passed on."""
    pid = provider_of(point.benchmark)
    if pid is None:
        return None
    o = point.options or {}
    name = point.benchmark.split("_", 1)[1]
    if pid == "torch-tbccl":
        config = {"workload": {"collectives": "collectives", "p2p": "p2p", "async": "async", "ddp": "ddp"}[name], "backend": o.get("backend", "tbccl"), "bytes": point.payload_bytes,
                  "warmup": point.warmup, "iterations": point.iterations, "devices": ",".join(_DEVICE.get(b, b) for b in point.backends)}
        for key in _INT_OPTIONS[pid]:
            if isinstance(o.get(key), int) and not isinstance(o.get(key), bool):
                config[key] = o[key]
        if isinstance(o.get("bucket_cap_mb"), (int, float)) and not isinstance(o.get("bucket_cap_mb"), bool):
            config["bucket_cap_mb"] = o["bucket_cap_mb"]
        if point.benchmark == "torch_ddp":
            config.pop("bytes", None)  # a DDP point's size is the gradient volume, not a float32 tensor payload
        return config
    config = {"workload": name}
    if pid == "exo-tbccl":
        if o.get("backend"):
            config["backend"] = o["backend"]
        config["tier"] = o.get("tier", "baseline")
        env = {**(o.get("env") or {}), **(o.get("server_env") or {})}
        mode = str(env.get("EXO_TBCCL_ACTIVITY_MODE", "")).lower()
        if mode and mode != "off":
            config["activity_policy"] = mode
    return config


def run_validate(python, bench_dir, configs, timeout=DESCRIBE_TIMEOUT_S) -> list:
    """Local counterpart of the helper's `validate-provider`: one raw outcome per configuration."""
    import tempfile
    script = Path(bench_dir) / ENTRY_POINT
    if not script.is_file():
        return [{"error": f"{script} is not a file"} for _ in configs]
    env = {k: v for k, v in os.environ.items() if not k.startswith("TBCCL_")}
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, config in enumerate(configs):
            path = Path(tmp) / f"config-{i}.json"
            path.write_text(json.dumps(config))
            try:
                done = subprocess.run([str(python), str(script), "validate", str(path)], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, cwd=str(Path(bench_dir)), env=env)
                out.append({"returncode": done.returncode, "stdout": done.stdout, "stderr_tail": done.stderr.strip()[-300:]})
            except (OSError, subprocess.TimeoutExpired) as error:
                out.append({"error": f"{type(error).__name__}: {error}"})
    return out


def remote_validate(host, python, bench_dir, configs, timeout=DESCRIBE_TIMEOUT_S) -> list:
    import base64
    from .remote import session
    blob = base64.b64encode(json.dumps(configs).encode()).decode()
    data, error = session.run_agent_json(host, ["validate-provider", "--python", python, "--dir", bench_dir, "--configs-b64", blob, "--timeout", str(timeout)], timeout=timeout * max(1, len(configs)) + 30, python=host.python)
    return data if isinstance(data, list) and len(data) == len(configs) else [{"error": error or "the helper returned no result"} for _ in configs]


def interpret_validation(raw) -> dict:
    """{'state': 'ok' | 'invalid' | 'unsupported' | 'failed', 'errors': [...]} for one configuration. A provider that does not implement `validate` (usage exit 2) is 'unsupported', not invalid."""
    if "error" in raw:
        return {"state": "failed", "errors": [raw["error"]]}
    try:
        body = json.loads(raw["stdout"])
    except ValueError:
        body = None
    if isinstance(body, dict) and "ok" in body:
        return {"state": "ok" if body["ok"] else "invalid", "errors": [str(e) for e in body.get("errors") or []]}
    if raw["returncode"] == 2:
        return {"state": "unsupported", "errors": ["the provider does not implement validate"]}
    return {"state": "failed", "errors": [f"validate exited {raw['returncode']} without a result: {raw['stderr_tail']}"]}
