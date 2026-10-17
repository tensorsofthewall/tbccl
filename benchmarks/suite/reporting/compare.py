"""Comparison of two sets of records. Only equivalent experiments are compared; everything else is labelled and explained.

Equivalence covers hardware, topology, operation, datatype, payload and its semantics, rank count and order, transport, memory backend, requested and resolved
algorithm, timing method, completion semantics, percentile definition and primary statistic. Version differences are allowed but always displayed. The result
states the size of the difference and whether it exceeds the repeat-to-repeat spread; it never says 'speedup'.
"""
from __future__ import annotations

import json

from .. import provenance_compare, statistics as sstats
from .summary import host_signature, primary, primary_throughput

IDENTITY = ("benchmark", "operation", "algorithm_requested", "payload_bytes")


def identity(r):
    b, c = r["benchmark"], r["configuration"]
    return (b["name"], b["operation"], b.get("algorithm_requested"), c["payload_bytes"], c.get("datatype"), c["rank_count"])


def fields(r):
    b, e, c = r["benchmark"], r["environment"], r["configuration"]
    timing = (r.get("measurements") or {}).get("timing") or {}
    return {"hardware": host_signature(r), "memory_backend": e["memory_backend"], "memory_domain": e["memory_domain"]["label"], "topology": [e["topology"]["kind"], e["topology"]["interconnect"], e["topology"]["physical"]],
            "transport": e["transport"], "mtu": e.get("mtu"), "algorithm_resolved": b.get("algorithm_resolved"), "payload_semantics": c["payload_semantics"], "reduce_op": c.get("reduce_op"),
            "timing_method": c["timing_method"], "completion_semantics": c["completion_semantics"],
            "options": c.get("options", {}), "data_direction": {k: (c.get("data_direction") or {}).get(k) for k in ("mode", "source_rank")}, "metric": timing.get("metric"), "primary_statistic": timing.get("primary_statistic"), "percentile_method": timing.get("percentile_method")}


def versions(r):
    b, e = r["benchmark"], r["environment"]
    return {"source_commit": b.get("source_commit"), "executable_sha256": (b.get("executable_sha256") or "")[:12] or None, "executable_version": b.get("executable_version"),
            "build_type": b.get("build_type"), "frameworks": e.get("framework_versions") or {}}


def differences(a, b):
    fa, fb = fields(a), fields(b)
    out = [f"{k}: {json.dumps(fa[k])} vs {json.dumps(fb[k])}" for k in fa if fa[k] != fb[k]]
    for name, f in (("baseline", fa), ("candidate", fb)):
        if not f["hardware"]:
            out.append(f"hardware of the {name} is unknown, so equivalence cannot be established (imported files can be completed with --set environment.hosts=...)")
    return out


def _by_identity(records):
    out = {}
    for r in records:
        out.setdefault(identity(r), []).append(r)
    return out


def _stat(records):
    ok = [r for r in records if r["status"]["state"] == "success"]
    timed = [primary(r)[0] for r in ok if primary(r)]
    tps = [primary_throughput(r)["value"] for r in ok if primary_throughput(r)]
    return ok, timed, tps


def _judge(base_vals, cand_vals):
    if not base_vals or not cand_vals:
        return "no data"
    if len(base_vals) < 3 or len(cand_vals) < 3:
        return "too few repetitions to judge against the spread"
    spread = max(sstats.repeat_variation(base_vals)["range_relative"] or 0.0, sstats.repeat_variation(cand_vals)["range_relative"] or 0.0)
    diff = abs(sstats.relative_difference(sstats.median(base_vals), sstats.median(cand_vals)))
    return "within repeat-to-repeat spread; no difference is claimed" if diff <= spread else "exceeds the repeat-to-repeat spread"


def compare(records_a, records_b, contextual=False):
    """Returns {'comparisons': [...], 'incompatible': [...], 'only_a': [...], 'only_b': [...]}. Records of state != success never enter a statistic."""
    ia, ib = _by_identity(records_a), _by_identity(records_b)
    result = {"comparisons": [], "incompatible": [], "only_a": [], "only_b": []}
    for key in sorted(set(ia) | set(ib), key=str):
        ra, rb = ia.get(key), ib.get(key)
        if ra is None:
            result["only_b"].append(key)
            continue
        if rb is None:
            result["only_a"].append(key)
            continue
        ok_a, vals_a, tp_a = _stat(ra)
        ok_b, vals_b, tp_b = _stat(rb)
        entry = {"identity": key, "baseline_ok": len(ok_a), "baseline_total": len(ra), "candidate_ok": len(ok_b), "candidate_total": len(rb), "versions_baseline": versions(ra[0]),
                 "versions_candidate": versions(rb[0])}
        if not ok_a or not ok_b:
            entry["note"] = "no successful measurement on " + ("the baseline" if not ok_a else "the candidate") + "; nothing to compare"
            result["incompatible"].append(entry)
            continue
        entry["provenance"] = provenance_compare.compare_identity(provenance_compare.identity_from_records(ok_a), provenance_compare.identity_from_records(ok_b))
        diffs = differences(ok_a[0], ok_b[0])
        entry["differences"] = diffs
        entry["equivalent"] = not diffs
        if diffs and not contextual:
            entry["note"] = "not equivalent experiments; no comparison made (use --contextual to see a labelled comparison)"
            result["incompatible"].append(entry)
            continue
        if not vals_a or not vals_b:
            entry["note"] = "no timing statistic on one side"
            result["incompatible"].append(entry)
            continue
        base, cand = sstats.median(vals_a), sstats.median(vals_b)
        entry.update(unit="us", statistic=primary(ok_a[0])[1], baseline=base, candidate=cand, relative_latency_difference=sstats.relative_difference(base, cand),
                     repeats=[len(vals_a), len(vals_b)], baseline_spread=sstats.repeat_variation(vals_a)["range_relative"], candidate_spread=sstats.repeat_variation(vals_b)["range_relative"],
                     judgement=_judge(vals_a, vals_b))
        if tp_a and tp_b:
            entry.update(throughput_unit=primary_throughput(ok_a[0])["unit"], throughput_baseline=sstats.median(tp_a), throughput_candidate=sstats.median(tp_b),
                         relative_throughput_difference=sstats.relative_difference(sstats.median(tp_a), sstats.median(tp_b)))
        entry["label"] = "equivalent" if not diffs else "CONTEXTUAL: not equivalent, differences listed; not evidence of a change"
        result["comparisons"].append(entry)
    return result


def _ver(v):
    return f"commit {v['source_commit'] and v['source_commit'][:10]}, exe {v['executable_sha256']}, {v['executable_version']}, build {v['build_type']}"


def to_markdown(result, name_a="baseline", name_b="candidate"):
    lines = [f"# Comparison: {name_a} vs {name_b}", "", "Relative difference = (candidate - baseline) / baseline, from the median over repetitions of each run's primary statistic. For latency a negative number means lower latency.", ""]
    if result["comparisons"]:
        lines += ["| experiment | stat | baseline us | candidate us | latency diff | throughput diff | reps | label | judgement |", "|---|---|---:|---:|---:|---:|---|---|---|"]
        for c in result["comparisons"]:
            name, op, algo, size, dtype, ranks = c["identity"]
            td = f"{c['relative_throughput_difference'] * 100:+.1f}%" if "relative_throughput_difference" in c else "-"
            lines.append(f"| {name} {op} {algo or ''} {size} B {dtype or ''} x{ranks} | {c['statistic']} | {c['baseline']:.2f} | {c['candidate']:.2f} | {c['relative_latency_difference'] * 100:+.1f}% | {td} | "
                         f"{c['repeats'][0]}/{c['repeats'][1]} | {c['label']} | {c['judgement']} |")
        lines.append("")
        changed = [c for c in result["comparisons"] if c["versions_baseline"] != c["versions_candidate"]]
        if changed:
            lines += ["## Version differences between the runs", ""]
            c = changed[0]
            lines += [f"- baseline: {_ver(c['versions_baseline'])}", f"- candidate: {_ver(c['versions_candidate'])}", ""]
        prov = [c for c in result["comparisons"] if provenance_compare.describe(c["provenance"])]
        if prov:
            lines += ["## Build and provider provenance", "", "Facts both runs recorded and that differ are listed with both values; facts that only one run recorded are unknown, never assumed equal.", ""]
            seen = set()
            for c in prov:
                for line in provenance_compare.describe(c["provenance"]):
                    if line not in seen:
                        seen.add(line)
                        lines.append(f"- {line}")
            lines.append("")
        known = [c for c in result["comparisons"] if c["provenance"]["matching"] and not c["provenance"]["different"] and not c["provenance"]["unknown"]]
        if known:
            lines += [f"{len(known)} comparison(s) are between builds recorded as identical on every recorded fact.", ""]
        ctx = [c for c in result["comparisons"] if c["differences"]]
        if ctx:
            lines += ["## Contextual comparisons (not equivalent)", ""] + [f"- {c['identity'][1]} {c['identity'][3]} B: " + "; ".join(c["differences"]) for c in ctx] + [""]
    if result["incompatible"]:
        lines += ["## Not compared", ""] + [f"- {c['identity'][0]} {c['identity'][1]} {c['identity'][2] or ''} {c['identity'][3]} B: {c.get('note')}" + ("".join(f"\n    - {d}" for d in c.get("differences", [])))
                                           for c in result["incompatible"]] + [""]
    for label, keys in (("Only in baseline", result["only_a"]), ("Only in candidate", result["only_b"])):
        if keys:
            lines += [f"## {label}", ""] + [f"- {k[0]} {k[1]} {k[2] or ''} {k[3]} B" for k in keys] + [""]
    if not result["comparisons"]:
        lines += ["No equivalent experiments were found, so there is nothing to compare.", ""]
    return "\n".join(lines)
