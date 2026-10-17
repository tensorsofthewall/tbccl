"""Build identity of a run, and what two identities say about each other.

A record carries the identity of what produced it in three places: the benchmark's executable hash and build commit, each host's TBCCL build (package, C ABI and wire versions), and for
framework workloads `environment.providers` (provider and framework versions, the SHA-256 of the compiled extension an adapter wheel bundles, the hashes of the workload scripts).
Two identities are compared field by field into exactly three states, never two: **matching** (both sides recorded the same value), **different** (both recorded, and they differ) and
**unknown** (at least one side did not record it). Unknown is not a match: a comparison between builds that were never identified must not read as a comparison between the same build.
"""
from __future__ import annotations

import os


def _host_role(host_id):
    return host_id or "-"


def identity_from_records(records) -> dict:
    """{fact: value} for the successful records of one run. A fact whose value differs between records of the same run is stored as the sorted '|'-joined set, so a run that mixed builds is not mistaken for one build."""
    seen = {}

    def put(key, value):
        if value in (None, "", {}, []):
            return
        seen.setdefault(key, set()).add(str(value))

    for r in records:
        if (r.get("status") or {}).get("state") != "success":
            continue
        b, e = r["benchmark"], r["environment"]
        name = b.get("name")
        put(f"executable.{name}.sha256", b.get("executable_sha256"))
        for h in e.get("hosts") or []:
            hb = h.get("build") or {}
            role = _host_role(h.get("host_id"))
            put(f"tbccl.{role}.package_version", hb.get("tbccl_package_version"))
            put(f"tbccl.{role}.c_abi", hb.get("c_abi"))
            put(f"tbccl.{role}.wire_protocol", hb.get("wire_protocol"))
            put(f"tbccl.{role}.executable_sha256", hb.get("executable_sha256"))
        for p in e.get("providers") or []:
            pid, role = p.get("provider_id"), _host_role(p.get("host"))
            if p.get("status") != "available":
                put(f"provider.{pid}.{role}.state", p.get("status"))  # not_configured / describe_failed / bundled_compat: the identity below is then genuinely unknown
            put(f"provider.{pid}.{role}.version", p.get("provider_version"))
            for k, v in (p.get("framework_versions") or {}).items():
                if k not in ("python", "platform"):
                    put(f"provider.{pid}.{role}.framework.{k}", v)
            for module, m in (p.get("native_modules") or {}).items():
                put(f"provider.{pid}.{role}.native.{os.path.basename(module).split('.')[0]}", (m or {}).get("sha256"))
            for script, h in (p.get("entry_point_sha256") or {}).items():
                put(f"provider.{pid}.{role}.script.{script}", h)
    return {k: "|".join(sorted(v)) for k, v in sorted(seen.items())}


def compare_identity(baseline: dict, candidate: dict) -> dict:
    """{'matching': [fact...], 'different': {fact: [baseline, candidate]}, 'unknown': {fact: the side that did NOT record it: 'baseline'|'candidate'}}, over the union of recorded facts."""
    out = {"matching": [], "different": {}, "unknown": {}}
    for key in sorted(set(baseline) | set(candidate)):
        a, b = baseline.get(key), candidate.get(key)
        if a is None or b is None:
            out["unknown"][key] = "candidate" if a is not None else "baseline"
        elif a == b:
            out["matching"].append(key)
        else:
            out["different"][key] = [a, b]
    out["known_build_on_both_sides"] = bool(baseline) and bool(candidate)
    return out


def describe(result) -> list:
    """Plain-language lines for a report."""
    lines = []
    for k, (a, b) in sorted(result["different"].items()):
        lines.append(f"{k}: {a[:16]} -> {b[:16]}" if len(a) > 32 or len(b) > 32 else f"{k}: {a} -> {b}")
    for k, side in sorted(result["unknown"].items()):
        lines.append(f"{k}: not recorded for the {side} (unknown, not assumed equal)")
    if not result["known_build_on_both_sides"]:
        lines.append("the build of at least one side was not recorded at all: a change cannot be attributed to a build")
    return lines
