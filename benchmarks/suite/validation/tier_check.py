"""Audit of one finished run (a campaign tier): every measurement valid and verified, health and cleanup acceptable, the data interface carried the traffic, no errors,
nothing left running. Read-only. Exit status 0 only when the tier is clean enough to continue.

    python3 -m benchmarks.suite.validation.tier_check RUN_DIR [--config CONFIG] [--overlay FILE] [--busy NAME]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import config as cfg
from .. import schema
from ..remote import session


def leftovers(config, names):
    out = {}
    for host in config.hosts.values():
        if host.is_local and not host.launcher.get("supervised"):
            from ..remote import agent
            out[host.id] = agent.competing_processes(names)
        else:
            data, error = session.run_agent_json(host, ["health", "--script", host.health_script or "/nonexistent"] + [a for n in names for a in ("--busy", n)], timeout=90, python=host.python)
            out[host.id] = (data or {}).get("competing") if data else f"unknown ({error})"
    return out


def audit(run_dir, config=None, busy=()):
    run_dir = Path(run_dir)
    records = schema.read_records(run_dir / "results.jsonl")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    states = {}
    for r in records:
        states[r["status"]["state"]] = states.get(r["status"]["state"], 0) + 1
    problems = []
    invalid = [r["record_id"] for r in records if schema.validate_record(r)]
    if invalid:
        problems.append(f"{len(invalid)} record(s) fail schema validation")
    bad = {k: v for k, v in states.items() if k in ("failure", "timeout", "invalid")}
    if bad:
        problems.append(f"measurements without a valid result: {bad}")
    if manifest["status"] not in ("success",):
        problems.append(f"run status is {manifest['status']} ({manifest.get('reason')})")
    if manifest["health"]["stop_reasons"]:
        problems.append("health stop reasons: " + "; ".join(manifest["health"]["stop_reasons"]))
    if manifest.get("cleanup_unconfirmed"):
        problems.append(f"cleanup not confirmed for {len(manifest['cleanup_unconfirmed'])} rank(s)")
    failed_verification = [r["record_id"] for r in records if r["verification"]["status"] == "failed"]
    if failed_verification:
        problems.append(f"{len(failed_verification)} record(s) failed verification")
    # interface counters over the whole run, from the first and last snapshots
    snaps = [run_dir / s for s in manifest["health"]["snapshots"]]
    summary = {}
    if len(snaps) >= 2:
        first, last = json.loads(snaps[0].read_text()), json.loads(snaps[-1].read_text())
        from .. import health
        for host_id in first:
            if "error" in first[host_id] or "error" in last.get(host_id, {"error": 1}):
                problems.append(f"{host_id}: first or last health snapshot missing")
                continue
            d = health.link_delta(first[host_id], last[host_id])
            summary[host_id] = d
            if d.get("rx_errors") or d.get("tx_errors"):
                problems.append(f"{host_id}: interface errors during the run: {d.get('rx_errors')}/{d.get('tx_errors')}")
    cpu = {}
    for r in records:
        for x in r["artifacts"].get("ranks", []):
            if x.get("cpu"):
                cpu.setdefault(x["host_id"], []).append((x["cpu"].get("child_user_s", 0) or 0) + (x["cpu"].get("child_system_s", 0) or 0))
    left = leftovers(config, busy) if config is not None and busy else {}
    for host, pids in left.items():
        if pids:
            problems.append(f"{host}: benchmark processes still present: {pids}")
    return {"run": run_dir.name, "states": states, "records": len(records), "problems": problems, "link_delta_whole_run": summary,
            "cpu_seconds_per_rank_mean": {h: round(sum(v) / len(v), 3) for h, v in cpu.items()}, "leftovers": left, "duration_s": manifest.get("duration_s")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    parser.add_argument("--config", help="a configuration (with overlays) to check for leftover processes on its hosts")
    parser.add_argument("--overlay", action="append", default=[])
    parser.add_argument("--busy", action="append", default=[], help="executable name that must not be running (repeatable)")
    args = parser.parse_args(argv)
    config = cfg.load_config(args.config, overlays=args.overlay, auto_overlay=False) if args.config else None
    report = audit(args.run_dir, config, args.busy)
    print(json.dumps(report, indent=2))
    print("TIER CLEAN" if not report["problems"] else "TIER NOT CLEAN: " + "; ".join(report["problems"]))
    return 0 if not report["problems"] else 1


if __name__ == "__main__":
    sys.exit(main())
