"""Pre-run checks and read-only link-health monitoring.

Nothing here changes the system. Link health reuses scripts/tb4_health_snapshot.py (no sudo): a baseline snapshot before a physical run, a snapshot after
each workload point, and the script's own stop policy decides whether the link changed. A changed link invalidates the points since the last good snapshot
and stops the run; the suite never retries after a controller failure or an interface disappearance.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import health, inventory, providers
from .adapters import REGISTRY
from .config import unresolved_placeholders

HEALTH_SCRIPT = health.HEALTH_SCRIPT


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


class HealthMonitor:
    """Before/after link snapshots for every host that has a snapshot source. `snapshot_fn(host)` and `compare_fn(before, after)` are injectable for tests.

    With the default functions each snapshot is the wrapper described in health.py, and the comparison adds interface, MTU, address, boot, counter and GPU
    checks to the stop policy of scripts/tb4_health_snapshot.py."""

    def __init__(self, hosts, snapshot_fn=None, compare_fn=None, gpu_hosts=(), strict=False, busy=()):
        self.hosts = list(hosts)
        self.strict, self.busy = strict, tuple(busy)
        self._snapshot_fn = snapshot_fn or (lambda host: health.snapshot_host(host, self.busy))
        self._compare_fn = compare_fn
        self.gpu_hosts = set(gpu_hosts)
        self.history = []

    def snapshot(self):
        result = {}
        for host in self.hosts:
            try:
                result[host.id] = self._snapshot_fn(host)
            except Exception as error:  # a snapshot that cannot be taken is itself a stop condition
                result[host.id] = {"error": str(error)}
        return result

    def signals(self, snapshot):
        """{host_id: signals} for wrapper-style snapshots; hosts with an error or a custom snapshot are skipped."""
        return {h.id: health.evaluate(h, snapshot[h.id], h.id in self.gpu_hosts) for h in self.hosts if snapshot.get(h.id) and "error" not in snapshot[h.id] and "interface" in snapshot[h.id]}

    def baseline_problems(self, snapshot):
        failures, warnings = [], []
        for host in self.hosts:
            snap = snapshot.get(host.id)
            if not snap or "error" in snap:
                failures.append(f"{host.id}: baseline snapshot unavailable ({(snap or {}).get('error', 'missing')})")
        for host_id, sigs in self.signals(snapshot).items():
            f, w = health.preflight_problems(next(h for h in self.hosts if h.id == host_id), sigs)
            failures += f
            warnings += w
        return failures, warnings

    def compare(self, before, after):
        """Returns (stop_reasons, details). Missing or errored snapshots are stop reasons."""
        reasons, details = [], {}
        for host in self.hosts:
            b, a = before.get(host.id), after.get(host.id)
            if not b or not a or "error" in b or "error" in a:
                reasons.append(f"{host.id}: snapshot unavailable ({(a or b or {}).get('error', 'missing')})")
                continue
            if self._compare_fn:
                result = self._compare_fn(b, a)
            else:
                result = health.compare_wrapper(b, a, strict=self.strict)
                result["stop_reasons"] = result["stop_reasons"] + health.regressions(host, b, a, host.id in self.gpu_hosts)
            details[host.id] = result
            reasons.extend(f"{host.id}: {r}" for r in result.get("stop_reasons", []))
        return reasons, details


def check_environment(build_dirs=("build-release", "build-bench", "build")):
    checks = []
    v = sys.version_info
    checks.append(Check("python", "fail" if v < (3, 12) else ("ok" if v[:2] == (3, 13) else "warn"), sys.version.split()[0] + ("" if v[:2] == (3, 13) else " (the adapters and this suite are validated on CPython 3.13; use a uv-managed 3.13 on every host)")))
    mods = inventory.python_modules()
    checks.append(Check("PyYAML", "ok" if mods["yaml"] else "warn", "needed for .yaml configuration files (JSON configurations work without it)" if not mods["yaml"] else "available"))
    checks.append(Check("matplotlib", "ok" if mods["matplotlib"] else "warn", "available" if mods["matplotlib"] else "not installed; reports are produced without plots"))
    checks.append(Check("jsonschema", "ok" if mods["jsonschema"] else "warn", "available" if mods["jsonschema"] else "not installed; the built-in schema validator is used"))
    ip = inventory.Iperf3Baseline.detect()
    checks.append(Check("iperf3", "ok" if ip["available"] else "warn", ip["version"] or "not installed (future TCP baseline; detection only)"))
    return checks


def _validate_checks(pid, host, host_id, python, directory, points):
    """Ask the provider, in its own interpreter on the host that will run the points, to validate each distinct workload configuration; invalid ones fail with the provider's reasons."""
    wanted = {}
    for point in points:
        if point.skip_reason or providers.provider_of(point.benchmark) != pid or host_id not in {r["host"] for r in point.ranks}:
            continue
        config = providers.workload_config(point)
        wanted.setdefault(json.dumps(config, sort_keys=True), (config, []))[1].append(point.point_id)
    if not wanted:
        return []
    configs = [c for c, _ in wanted.values()]
    raw = providers.run_validate(python, directory, configs) if host.is_local else providers.remote_validate(host, python, directory, configs)
    checks, name = [], f"validate {pid}@{host_id}"
    problems, unsupported = [], False
    for (config, ids), result in zip(wanted.values(), raw):
        verdict = providers.interpret_validation(result)
        if verdict["state"] == "unsupported":
            unsupported = True
        elif verdict["state"] != "ok":
            problems.append(f"{ids[0]}{' (+%d more)' % (len(ids) - 1) if len(ids) > 1 else ''}: " + "; ".join(verdict["errors"]))
    if problems:
        checks.append(Check(name, "fail", f"the provider rejects {len(problems)} of {len(wanted)} workload configuration(s): " + " | ".join(problems)[:600]))
    elif unsupported:
        checks.append(Check(name, "warn", "the provider does not implement validate; configurations were not checked by it"))
    else:
        checks.append(Check(name, "ok", f"{len(wanted)} workload configuration(s) accepted by the provider"))
    return checks


def check_providers(config, points, probe):
    """One check per framework provider and participating host: configured, and (when `probe`) described and usable. Native-only configurations have none."""
    needed = {}
    for point in points:
        pid = None if point.skip_reason else providers.provider_of(point.benchmark)
        for rank in (point.ranks if pid else ()):
            needed.setdefault(pid, set()).add(rank["host"])
    checks = []
    for pid in sorted(needed):
        spec = providers.PROVIDERS[pid]
        for host_id in sorted(needed[pid]):
            host = config.hosts[host_id]
            name = f"provider {pid}@{host_id}"
            python, directory = providers.configured(host, pid)
            if not python or not directory:
                missing = [k for k, v in ((spec["python"], python), (spec["dir"], directory)) if not v]
                checks.append(Check(name, "fail", f"not configured: set hosts.{host_id}.executables.{' and '.join(missing)} (the provider directory is benchmarks/provider in the {pid} repository)"))
            elif not probe:
                checks.append(Check(name, "skip", "configured; not probed in a dry run (`providers CONFIG` shows its state)"))
            else:
                raw = providers.run_describe(python, directory) if host.is_local else providers.remote_describe(host, python, directory)
                row = providers.interpret(pid, raw)
                if row["status"] == "available":
                    d = row["description"]
                    detected = {k: v for k, v in d["availability"].get("detected", {}).items() if v and k not in ("python", "platform", "native_modules")}
                    checks.append(Check(name, "ok", f"{pid} {d['provider_version']} (protocol {d['provider_protocol_version']}); " + (", ".join(f"{k} {v}" for k, v in sorted(detected.items())) or "no framework versions reported")))
                    checks += _validate_checks(pid, host, host_id, python, directory, points)
                else:
                    checks.append(Check(name, "fail", f"{row['status'].replace('_', ' ')}: " + "; ".join(row["reasons"])[:300]))
    return checks


def check_config(config, points, ctx, allow_physical=False, dry_run=False):
    checks = []
    placeholders = unresolved_placeholders(config)
    if placeholders:
        checks.append(Check("placeholders", "warn" if dry_run else "fail", "unresolved placeholders (fill them in a local overlay): " + "; ".join(placeholders)))
    if config.physical:
        if allow_physical:
            checks.append(Check("physical opt-in", "ok", "physical run explicitly allowed (--allow-physical)"))
        else:
            checks.append(Check("physical opt-in", "warn" if dry_run else "fail", "this configuration runs between physical hosts; pass --allow-physical to run it"))
    used = {p.benchmark for p in points if not p.skip_reason}
    for name in sorted(used):
        if not REGISTRY[name].caps.executable_in_build:
            continue
        exe = REGISTRY[name].caps.executable
        for host_id in sorted({r["host"] for p in points if p.benchmark == name for r in p.ranks}):
            host = config.hosts[host_id]
            if not host.is_local:
                checks.append(Check(f"{exe}@{host_id}", "skip", "remote executable is not checked locally"))
                continue
            path = ctx.executable_or_none(exe, host_id)
            if path is None:
                checks.append(Check(f"{exe}@{host_id}", "fail", "executable not found (build it, or set build_dirs / TBCCL_BUILD_DIR)"))
            else:
                build = ctx.build(path)
                note = f"{path.name} ({build.get('version_text') or 'version unknown'}, build type {build.get('build_type') or 'unknown'})"
                if build.get("build_type") not in ("Release", "RelWithDebInfo", "MinSizeRel"):
                    checks.append(Check(f"{exe}@{host_id}", "warn", note + ": not an optimized build; timings will not represent the library"))
                else:
                    checks.append(Check(f"{exe}@{host_id}", "ok", note))
    checks += check_providers(config, points, probe=not dry_run)
    for host_id in sorted({r["host"] for p in points if not p.skip_reason for r in p.ranks}):
        host = config.hosts[host_id]
        if host.is_local:
            continue
        data_addresses = {a for a in [host.address, *host.interfaces.values()] if a and not a.startswith("<")}
        shared = sorted(a for a in data_addresses if any(a in str(part) for part in host.launcher.get("argv", [])))
        if shared:
            checks.append(Check(f"control path@{host_id}", "warn", "the launcher appears to reach this host over a rank data address; heartbeats, relayed output and cleanup checks would then share the "
                                                                    "interface being measured. Prefer a separate management path for the launcher."))
    skipped = [p for p in points if p.skip_reason]
    if skipped:
        checks.append(Check("matrix", "warn", f"{len(skipped)} of {len(points)} points cannot run (see the matrix)"))
    if config.physical:
        used = sorted({r["host"] for p in points if not p.skip_reason for r in p.ranks})
        for host_id in used:
            host = config.hosts[host_id]
            if host.health_command:
                checks.append(Check(f"health@{host_id}", "ok", "custom health command configured"))
            elif host.is_local and not host.launcher.get("supervised"):
                checks.append(Check(f"health@{host_id}", "ok" if HEALTH_SCRIPT.exists() else "fail", "link-health snapshots via scripts/tb4_health_snapshot.py" if HEALTH_SCRIPT.exists() else f"{HEALTH_SCRIPT} not found"))
            elif host.health_script:
                checks.append(Check(f"health@{host_id}", "ok", "link-health snapshots via the configured health_script on the host"))
            else:
                checks.append(Check(f"health@{host_id}", "fail", "a physical run needs link-health monitoring on every host: this host needs a health_script (path of scripts/tb4_health_snapshot.py on it) or a health_command"))
        for host_id in used:
            host = config.hosts[host_id]
            if host.is_local and sys.platform.startswith("linux"):
                aer = health.aer_observability(cmdline=health.local_cmdline())
                checks.append(Check(f"aer@{host_id}", "ok" if aer["state"] == "observable" else "warn", f"AER {aer['state']}: {aer['detail']}. {aer['statement']}"))
    return checks


def failed(checks):
    return [c for c in checks if c.status == "fail"]


def format_checks(checks):
    marks = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL", "skip": "skip"}
    return "\n".join(f"  [{marks[c.status]}] {c.name}: {c.detail}" for c in checks)
