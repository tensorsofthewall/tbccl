"""Campaign report data for the B3B Thunderbolt characterization: tables (CSV and Markdown) and static figures from the normalised results of several runs (tiers).

    python3 -m benchmarks.suite.reporting.b3b RESULTS_DIR CAMPAIGN_ID [--out DIR]

Runs are selected by the campaign id in their label (``...-b3b1-t1-latency-...``). Failed, invalid and skipped measurements never enter a statistic; they are counted.
Nothing is combined across incomparable metrics: every table and figure is for one operation, one metric, one data direction or rank orientation, one memory backend.
Percentiles are only what the native tool printed (p95 for most, p99 only where the tool reports it); a global percentile is never derived.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as pystats
import sys
from collections import defaultdict
from pathlib import Path

from . import _io
from .. import statistics as sstats

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def load(results_dir, campaign):
    """[(tier, run_dir, manifest, records)] for the runs of one campaign; preliminary runs are returned with tier name ending in -prelim so callers can exclude them."""
    runs = []
    for d in sorted(Path(results_dir).glob(f"*-{campaign}-t*")):
        if not (d / "results.jsonl").exists():
            continue
        tier = d.name.split(f"-{campaign}-", 1)[1].rsplit("-", 1)[0]
        runs.append(_io.load_run(d, tier))
    return runs


def os_label(record, host_id):
    for h in record["environment"]["hosts"]:
        if h["host_id"] == host_id:
            return {"EndeavourOS": "Linux", "macOS": "macOS"}.get((h.get("os") or "?").split()[0], (h.get("os") or "?").split()[0])
    return "?"


def direction_label(record):
    d = record["configuration"].get("data_direction") or {}
    if d.get("mode") == "directional" and d.get("source_host"):
        return f"{os_label(record, d['source_host'])} -> {os_label(record, d['destination_host'])}"
    rank0 = os_label(record, record["configuration"]["rank_order"][0]) if record["configuration"].get("rank_order") else "?"
    return f"bidirectional (rank 0 on {rank0})"


def orientation_label(record):
    ro = record["configuration"].get("rank_order") or []
    return f"rank0={os_label(record, ro[0])}, rank1={os_label(record, ro[1])}" if len(ro) >= 2 else "?"


def ok(records):
    return [r for r in records if r["status"]["state"] == "success"]


def tier_of(record):
    return record.get("_tier", "")


def group(records, keyfn):
    out = defaultdict(list)
    for r in ok(records):
        out[keyfn(r)].append(r)
    return out


def primary_value(r):
    t = (r.get("measurements") or {}).get("timing")
    return t["statistics"][t["primary_statistic"]] if t else None


def stat(r, name):
    t = (r.get("measurements") or {}).get("timing")
    return t["statistics"].get(name) if t else None


def across(values):
    """Median of the per-repetition values with their spread. Repetitions are independent native invocations."""
    values = [v for v in values if v is not None]
    if not values:
        return None
    v = sstats.repeat_variation(values)
    return {"median": v["median"], "min": v["min"], "max": v["max"], "n": v["repetitions"], "range_rel": v.get("range_relative"), "cv": v.get("cv")}


def provenance_line(records):
    hosts = {}
    for r in records:
        for h in r["environment"]["hosts"]:
            b = h.get("build") or {}
            hosts[h["host_id"]] = f"{h.get('os', '?').split()[0]} {b.get('build_type', '?')} {str(b.get('build_commit') or 'unstamped')[:8]} ({b.get('status', '?')}) sha {str(b.get('executable_sha256') or '')[:8]}"
    return "; ".join(f"{k}: {v}" for k, v in sorted(hosts.items()))


# ---------------------------------------------------------------------------- tables
def latency_rows(records):
    rows = []
    for key, recs in sorted(group(records, lambda r: (tier_of(r), r["benchmark"]["operation"], r["configuration"]["payload_bytes"], orientation_label(r))).items()):
        tier, op, size, orient = key
        med, p95 = across([stat(r, "median") for r in recs]), across([stat(r, "p95") for r in recs])
        t0 = recs[0]["measurements"]["timing"]
        peer = [((r["measurements"]["timing"].get("per_rank") or [None, None])[1] or {}).get("statistics", {}).get("median") for r in recs]
        cpu = [((r["measurements"].get("extra") or {}).get("cpu_cores") or {}).get("value") for r in recs]
        rows.append({"tier": tier, "operation": op, "metric": t0["metric"], "payload_bytes": size, "orientation": orient, "repetitions": med["n"], "samples_per_repetition": t0["sample_count"],
                     "median_us": med["median"], "median_min_us": med["min"], "median_max_us": med["max"], "median_range_rel": med["range_rel"], "p95_us": p95["median"],
                     "rank1_median_us": (across(peer) or {}).get("median"), "cpu_cores_rank0": (across(cpu) or {}).get("median"), "verification": recs[0]["verification"]["status"]})
    return rows


def transfer_rows(records):
    rows = []
    cols = ("sender_network_us", "receiver_network_us", "source_staging_us", "destination_staging_us", "completion_confirmed_us")
    for key, recs in sorted(group(records, lambda r: (tier_of(r), r["benchmark"]["operation"], r["configuration"]["payload_bytes"], direction_label(r), orientation_label(r),
                                                       (r["configuration"].get("options") or {}).get("timing_scope", "ready"))).items()):
        tier, op, size, direction, orient, scope = key
        med = across([stat(r, "median") for r in recs])
        tp = across([((r["measurements"]["throughput"] or [{}])[0]).get("value") for r in recs])
        t0 = recs[0]["measurements"]["timing"]
        extra = {c: (across([((r["measurements"].get("extra") or {}).get(c) or {}).get("value") for r in recs]) or {}).get("median") for c in cols}
        rows.append({"tier": tier, "operation": op, "metric": t0["metric"], "timing_scope": scope if op == "end_to_end" else "n/a", "payload_bytes": size, "direction": direction, "orientation": orient,
                     "repetitions": med["n"], "iterations": recs[0]["configuration"]["iterations"], "median_us": med["median"], "median_min_us": med["min"], "median_max_us": med["max"],
                     "median_range_rel": med["range_rel"], "p95_us": (across([stat(r, "p95") for r in recs]) or {}).get("median"), "p99_us": (across([stat(r, "p99") for r in recs]) or {}).get("median"),
                     "min_us": (across([stat(r, "min") for r in recs]) or {}).get("median"), "max_us": (across([stat(r, "max") for r in recs]) or {}).get("median"),
                     "throughput_GBps": tp["median"], "throughput_min": tp["min"], "throughput_max": tp["max"], "throughput_definition": recs[0]["measurements"]["throughput"][0]["definition"], **extra})
    return rows


def iperf_rows(records):
    rows = []
    for key, recs in sorted(group(records, lambda r: (direction_label(r), orientation_label(r))).items()):
        recv = across([r["measurements"]["throughput"][0]["value"] for r in recs])
        send = across([r["measurements"]["throughput"][1]["value"] for r in recs])
        x = lambda name: [(r["measurements"]["extra"].get(name) or {}).get("value") for r in recs]  # noqa: E731
        rows.append({"direction": key[0], "orientation": key[1], "repetitions": recv["n"], "receiver_Gbps_median": recv["median"], "receiver_min": recv["min"], "receiver_max": recv["max"],
                     "sender_Gbps_median": send["median"], "bytes_sent_total": sum(v or 0 for v in x("bytes_sent")), "retransmits_total": sum(v or 0 for v in x("retransmits")),
                     "streams": recs[0]["measurements"]["extra"]["parallel_streams"]["value"], "cpu_host_percent_median": (across(x("cpu_host_total_percent")) or {}).get("median"),
                     "cpu_remote_percent_median": (across(x("cpu_remote_total_percent")) or {}).get("median"), "duration_s": (across(x("duration_received")) or {}).get("median")})
    return rows


def collective_rows(records):
    rows = []
    for key, recs in sorted(group(records, lambda r: (tier_of(r), r["benchmark"]["operation"], r["benchmark"].get("algorithm_requested"), r["benchmark"].get("algorithm_resolved"),
                                                       (r["configuration"].get("options") or {}).get("chunk_bytes"), r["configuration"]["payload_bytes"], orientation_label(r))).items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tier, op, req, res, chunk, size, orient = key
        med = across([stat(r, "median") for r in recs])
        tp = {t["kind"]: across([next(x["value"] for x in r["measurements"]["throughput"] if x["kind"] == t["kind"]) for r in recs]) for t in recs[0]["measurements"]["throughput"]}
        rows.append({"tier": tier, "operation": op, "requested_algorithm": req, "resolved_algorithm": res, "selection_reason": recs[0]["benchmark"].get("selection_reason") or "", "chunk_bytes": chunk or "",
                     "payload_bytes": size, "orientation": orient, "repetitions": med["n"], "iterations": recs[0]["configuration"]["iterations"], "median_us": med["median"],
                     "median_min_us": med["min"], "median_max_us": med["max"], "median_range_rel": med["range_rel"], "p95_us": (across([stat(r, "p95") for r in recs]) or {}).get("median"),
                     "algbw_GBps": tp["collective_algorithmic"]["median"], "busbw_GBps_estimate": tp["collective_bus_estimate"]["median"], "verification": recs[0]["verification"]["status"]})
    return rows


def raw_rtt_rows(records):
    """tb_pingpong ping-pong: the tool reports a MEAN round trip over its own fixed iteration count (no median, no percentiles), so it is shown as a mean and never merged with medians."""
    rows = []
    for key, recs in sorted(group(records, lambda r: (r["configuration"]["payload_bytes"], orientation_label(r))).items()):
        mean = across([stat(r, "mean") for r in recs])
        rows.append({"payload_bytes": key[0], "orientation": key[1], "repetitions": mean["n"], "mean_rtt_us": mean["median"], "mean_min_us": mean["min"], "mean_max_us": mean["max"],
                     "iterations_chosen_by_tool": recs[0]["measurements"]["extra"]["iterations_chosen_by_tool"]["value"], "throughput_Gbps": recs[0]["measurements"]["throughput"][0]["value"]})
    return rows


def run_to_run(rows, tier_a, tier_b, keys):
    """Same configuration measured in two different runs (hours or minutes apart): ratio of the medians. A large ratio means run-to-run drift, not a property of the configuration."""
    index = {tuple(r[k] for k in keys): r for r in rows if r["tier"] == tier_a}
    out = []
    for r in rows:
        if r["tier"] != tier_b:
            continue
        a = index.get(tuple(r[k] for k in keys))
        if a:
            out.append({**{k: r[k] for k in keys}, f"{tier_a}_median_us": a["median_us"], f"{tier_b}_median_us": r["median_us"], "ratio": r["median_us"] / a["median_us"],
                        f"{tier_a}_range_rel": a["median_range_rel"], f"{tier_b}_range_rel": r["median_range_rel"]})
    return out


def cpu_rows(runs):
    """Whole-process CPU seconds per rank role (not the timed loop), grouped by benchmark and host."""
    acc = defaultdict(list)
    for _, _, _, records in runs:
        for r in ok(records):
            for x in r["artifacts"].get("ranks", []):
                c = x.get("cpu")
                if c:
                    acc[(r["benchmark"]["name"], r["benchmark"]["operation"], x["host_id"], x.get("supervised"))].append(((c.get("child_user_s") or 0) + (c.get("child_system_s") or 0), c.get("supervisor_cpu_s")))
    rows = []
    for (name, op, host, sup), vals in sorted(acc.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        rows.append({"benchmark": name, "operation": op, "host": host, "supervised": sup, "ranks_sampled": len(vals), "benchmark_cpu_s_median": pystats.median(v[0] for v in vals),
                     "supervisor_cpu_s_median": pystats.median(v[1] for v in vals if v[1] is not None) if any(v[1] is not None for v in vals) else None})
    return rows


def failures(runs):
    out = []
    for tier, d, m, records in runs:
        for r in records:
            if r["status"]["state"] != "success":
                out.append({"tier": tier, "record": r["record_id"].split("/", 1)[-1], "state": r["status"]["state"], "reason": (r["status"].get("reason") or "")[:200]})
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def md_table(rows, columns, fmt=None):
    fmt = fmt or {}
    lines = ["| " + " | ".join(c for c, _ in columns) + " |", "|" + "|".join("---:" if c not in ("operation", "direction", "orientation", "resolved", "algorithm") else "---" for c, _ in columns) + "|"]
    for r in rows:
        cells = []
        for _, key in columns:
            v = r.get(key)
            f = fmt.get(key)
            cells.append("-" if v is None or v == "" else (f.format(v) if f and isinstance(v, (int, float)) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- figures
def _plt():
    """pyplot with the non-interactive backend, or None when matplotlib is not installed (tables are written without figures)."""
    if not _have_mpl():
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _setup(plt, title, xlabel, ylabel, logy=True):
    fig, ax = plt.subplots(figsize=(8.4, 5.4), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, loc="left", color=INK, fontsize=10.5)
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    return fig, ax


def _save(fig, plt, path, footer):
    import textwrap
    wrapped = "\n".join(line for part in footer.split("\n") for line in (textwrap.wrap(part, 190) or [""]))
    fig.text(0.01, 0.005, wrapped, fontsize=6.3, color=MUTED, va="bottom", ha="left")
    fig.subplots_adjust(bottom=0.1 + 0.022 * (wrapped.count("\n") + 1), left=0.11)
    fig.savefig(path, dpi=140, facecolor="#fcfcfb")
    plt.close(fig)
    return str(path)


def _footer(operation, direction, backend, metric, timing, units, reps, records):
    return (f"operation: {operation} | data direction / orientation: {direction} | memory backend: {backend} | transport: TCP over Thunderbolt 4 IP (not RDMA)\n"
            f"metric: {metric} | timing: {timing[:150]} | units: {units} | repetitions: {reps}\nbuild provenance: {provenance_line(records)[:230]}")


def _lines(ax, series, linestyle="-"):
    for i, (label, pts) in enumerate(sorted(series.items())):
        pts = sorted(pts)
        c, m = CATEGORICAL[i % len(CATEGORICAL)], MARKERS[i % len(MARKERS)]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker=m, markersize=5, linewidth=1.7, label=label, markeredgecolor="#fcfcfb", linestyle=linestyle)
        if len(pts[0]) >= 4:
            ax.fill_between([p[0] for p in pts], [p[2] for p in pts], [p[3] for p in pts], color=c, alpha=0.14, linewidth=0)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)


def figures(out, latency, transfer, iperf, collectives, runs_records):
    plt = _plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    # 1. latency vs size, one figure per operation (RTT / stream / simultaneous exchange stay separate)
    for op in sorted({r["operation"] for r in latency}):
        sub = [r for r in latency if r["operation"] == op]
        series = defaultdict(list)
        for r in sub:
            series[r["orientation"]].append((r["payload_bytes"], r["median_us"], r["median_min_us"], r["median_max_us"]))
        if not series:
            continue
        fig, ax = _setup(plt, f"small_message_latency {op}: {sub[0]['metric']}", "payload (bytes; native semantics)", "median of repetition medians (us); band = min..max")
        _lines(ax, series)
        recs = [x for x in runs_records if x["benchmark"]["name"] == "small_message_latency" and x["benchmark"]["operation"] == op]
        made.append(_save(fig, plt, out / f"latency-{op}.png", _footer(op, "see legend (rank orientation)", "host", sub[0]["metric"], recs[0]["configuration"]["timing_method"] if recs else "", "us", sub[0]["repetitions"], recs)))
    # 2. throughput vs size, per operation and direction; network-only and end-to-end are separate figures
    for op in sorted({r["operation"] for r in transfer}):
        sub = [r for r in transfer if r["operation"] == op and r["timing_scope"] in ("ready", "n/a") and "forward" not in r["orientation"] or r["operation"] == op and r["timing_scope"] in ("ready", "n/a")]
        series = defaultdict(list)
        for r in sub:
            series[f"{r['direction']}  [{r['orientation']}]"].append((r["payload_bytes"], r["throughput_GBps"], r["throughput_min"], r["throughput_max"]))
        if not series:
            continue
        fig, ax = _setup(plt, f"tensor_transfer {op}: effective payload throughput (decimal GB/s; not physical link rate)", "payload (bytes)", "payload throughput (GB/s; median, band min..max)")
        _lines(ax, series)
        recs = [x for x in runs_records if x["benchmark"]["name"] == "tensor_transfer" and x["benchmark"]["operation"] == op]
        made.append(_save(fig, plt, out / f"throughput-{op}.png", _footer(op, "see legend", "host", sub[0]["metric"], recs[0]["configuration"]["timing_method"] if recs else "", "GB/s (decimal)", sub[0]["repetitions"], recs)))
        # latency (completion) vs size for the same data
        series = defaultdict(list)
        for r in sub:
            series[f"{r['direction']}  [{r['orientation']}]"].append((r["payload_bytes"], r["median_us"], r["median_min_us"], r["median_max_us"]))
        fig, ax = _setup(plt, f"tensor_transfer {op}: {sub[0]['metric']}", "payload (bytes)", "median of repetitions (us); band = min..max")
        _lines(ax, series)
        made.append(_save(fig, plt, out / f"completion-{op}.png", _footer(op, "see legend", "host", sub[0]["metric"], recs[0]["configuration"]["timing_method"] if recs else "", "us", sub[0]["repetitions"], recs)))
    # 3. direction comparison: ratio of the two physical directions, forward orientation, per operation
    for op in sorted({r["operation"] for r in transfer}):
        by = defaultdict(dict)
        for r in transfer:
            if r["operation"] == op and r["timing_scope"] in ("ready", "n/a") and "rank0=Linux" in r["orientation"]:
                by[r["payload_bytes"]][r["direction"]] = r["median_us"]
        pts = [(s, d["Linux -> macOS"] / d["macOS -> Linux"]) for s, d in sorted(by.items()) if len(d) == 2]
        if not pts:
            continue
        fig, ax = _setup(plt, f"tensor_transfer {op}: direction asymmetry (Linux->Mac time / Mac->Linux time; above 1 = Linux->Mac slower)", "payload (bytes)", "ratio of median times", logy=False)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=CATEGORICAL[0], marker="o", linewidth=1.7, markeredgecolor="#fcfcfb")
        ax.axhline(1.0, color=MUTED, linewidth=0.8)
        recs = [x for x in runs_records if x["benchmark"]["name"] == "tensor_transfer" and x["benchmark"]["operation"] == op]
        made.append(_save(fig, plt, out / f"direction-{op}.png", _footer(op, "both physical directions, orientation rank0=Linux", "host", "ratio of the two medians", "see each series", "ratio", "see tables", recs)))
    # 4. collectives: algorithms per operation, forward orientation, completion time and algbw
    for op in sorted({r["operation"] for r in collectives}):
        series, series_bw = defaultdict(list), defaultdict(list)
        for r in collectives:
            if r["operation"] == op and "rank0=Linux" in r["orientation"]:
                label = f"{r['requested_algorithm']}" + (f" (resolved {r['resolved_algorithm']})" if r["requested_algorithm"] == "auto" else "") + (f" chunk {r['chunk_bytes']}" if r["chunk_bytes"] else "")
                series[label].append((r["payload_bytes"], r["median_us"], r["median_min_us"], r["median_max_us"]))
                series_bw[label].append((r["payload_bytes"], r["algbw_GBps"]))
        if not series:
            continue
        recs = [x for x in runs_records if x["benchmark"]["name"] == "collective_bench" and x["benchmark"]["operation"] == op]
        fig, ax = _setup(plt, f"collective_bench {op}: completion time (legacy synchronous World API, float32 SUM, 2 ranks)", "payload (bytes; per-operation native semantics)", "median of repetitions (us); band = min..max")
        _lines(ax, series)
        made.append(_save(fig, plt, out / f"collective-{op}-time.png", _footer(op, "orientation rank0=Linux", "host", "collective_duration_max_across_ranks", recs[0]["configuration"]["timing_method"] if recs else "", "us", "see tables", recs)))
        ref = {p[0]: p[1] for p in series.get("reference", [])}
        rel = {k: [(p[0], p[1] / ref[p[0]]) for p in pts if p[0] in ref and k != "reference"] for k, pts in series.items()}
        rel = {k: v for k, v in rel.items() if v}
        if rel:
            fig, ax = _setup(plt, f"collective_bench {op}: time relative to 'reference' (below 1 = faster)", "payload (bytes)", "time / reference time", logy=False)
            _lines(ax, rel)
            ax.axhline(1.0, color=MUTED, linewidth=0.8)
            made.append(_save(fig, plt, out / f"collective-{op}-relative.png", _footer(op, "orientation rank0=Linux", "host", "ratio of medians of repetition medians", "per iteration, max across ranks; legacy World API", "ratio", "3-5 per point", recs)))
        fig, ax = _setup(plt, f"collective_bench {op}: algorithmic bandwidth as the tool defines it (decimal GB/s)", "payload (bytes)", "algbw (GB/s; median)")
        _lines(ax, series_bw)
        made.append(_save(fig, plt, out / f"collective-{op}-algbw.png", _footer(op, "orientation rank0=Linux", "host", "algbw = rank_count * segment_bytes / median time", "per iteration, max across ranks", "GB/s (decimal)", "see tables", recs)))
    # 5. repeat-to-repeat variation: range of the repetition medians relative to their median, for every family
    fig, ax = _setup(plt, "repeat-to-repeat variation: (max - min) / median of the per-repetition medians", "payload (bytes)", "relative range", logy=True)
    families = {"latency (small_message_latency)": [(r["payload_bytes"], r["median_range_rel"]) for r in latency if r["median_range_rel"]],
                "transfer (tensor_transfer)": [(r["payload_bytes"], r["median_range_rel"]) for r in transfer if r["median_range_rel"]],
                "collectives (collective_bench)": [(r["payload_bytes"], r["median_range_rel"]) for r in collectives if r["median_range_rel"]]}
    for i, (label, pts) in enumerate(sorted(families.items())):
        if pts:
            ax.scatter([p[0] for p in pts], [max(p[1], 1e-4) for p in pts], color=CATEGORICAL[i], marker=MARKERS[i], s=26, alpha=0.8, label=label, edgecolors="#fcfcfb", linewidths=0.5)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    made.append(_save(fig, plt, out / "variability.png", _footer("all families", "all", "host", "range of repetition medians / their median", "per workload", "ratio", "3-5 per point", runs_records)))
    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("campaign")
    parser.add_argument("--out")
    parser.add_argument("--include-prelim", action="store_true")
    args = parser.parse_args(argv)
    everything = load(args.results_dir, args.campaign)
    runs = [r for r in everything if (args.include_prelim or "prelim" not in r[0] and "validation" not in r[0]) and r[2]["status"] == "success"]
    excluded = [{"tier": t, "run": d.name, "status": m["status"], "reason": m.get("reason"), "counts": m["counts"]} for t, d, m, _ in everything if (t, d) not in {(x[0], x[1]) for x in runs}]
    out = Path(args.out or Path(args.results_dir) / f"campaign-{args.campaign}")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    allrecs = []
    for tier, _, _, recs in runs:
        for r in recs:
            r["_tier"] = tier.split("-")[0] if tier.startswith(("t5",)) is False else tier
            r["_tier"] = tier
            allrecs.append(r)
    by_bench = lambda name: [r for r in allrecs if r["benchmark"]["name"] == name]  # noqa: E731
    lat, tr, ip, co = latency_rows(by_bench("small_message_latency")), transfer_rows(by_bench("tensor_transfer")), iperf_rows(by_bench("iperf3")), collective_rows(by_bench("collective_bench"))
    rtt = raw_rtt_rows(by_bench("tcp_pingpong"))
    r2r_t = run_to_run(tr, "t2-transfer", "t5b-confirm-transfer", ("operation", "payload_bytes", "direction", "orientation", "timing_scope"))
    r2r_c = run_to_run(co, "t4-collectives", "t5c-confirm-collectives", ("operation", "requested_algorithm", "payload_bytes", "orientation"))
    for name, rows in (("latency", lat), ("transfer", tr), ("iperf3", ip), ("collectives", co), ("raw_rtt", rtt), ("run_to_run_transfer", r2r_t), ("run_to_run_collectives", r2r_c),
                       ("cpu", cpu_rows(runs)), ("failures", failures(runs))):
        write_csv(out / "tables" / f"{name}.csv", rows)
    main = lambda rows: [r for r in rows if r["tier"] in ("t1-latency-rerun", "t2-transfer", "t4-collectives")]  # noqa: E731
    made = figures(out / "figures", main(lat), main(tr), ip, main(co), [r for r in allrecs if tier_of(r) in ("t1-latency-rerun", "t2-transfer", "t3-iperf", "t4-collectives")]) if _have_mpl() else []
    (out / "excluded_runs.json").write_text(json.dumps(excluded, indent=2))
    print(json.dumps({"campaign": args.campaign, "excluded_from_distributions": excluded, "runs": [{"tier": t, "run": d.name, "status": m["status"], "counts": m["counts"]} for t, d, m, _ in runs], "tables": str(out / "tables"),
                      "figures": made}, indent=2))
    return 0


def _have_mpl():
    import importlib.util
    return importlib.util.find_spec("matplotlib") is not None


if __name__ == "__main__":
    sys.exit(main())
