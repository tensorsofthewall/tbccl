"""Campaign report data for B6 (Exo pipeline-parallel inference: MlxTbccl through exo-tbccl against MlxRing): tables (CSV) and static figures from the runs (tiers) of a campaign.

    python3 -m benchmarks.suite.reporting.b6 RESULTS_DIR CAMPAIGN [--out DIR] [--exclude TEXT] [--include-tier PREFIX]

Every figure is computed from the raw per-request timelines the client kept (``raw/<point>/rep<k>/exo/<phase>.requests.jsonl``), never from the printed summaries. The unit of
replication is the cluster lifecycle (one point = fresh nodes, a fresh placement, a fresh model load): requests within one lifecycle are not independent lifecycles, so a comparison
between backends uses the per-lifecycle medians and states how many lifecycles each side has. Exo chooses the stage order and the layer split per start, so two lifecycles are compared only
when their placement (which host ran which stage, how many layers each) is identical, and a cell with fewer than two lifecycles on either side is reported as not compared with the reason.
Closed-loop phases, correctness gates and traced runs are never merged. Client-observed latencies are not native transfer latencies. CPU seconds and link bytes are taken from the passive
per-host sampler between the first request submitted and the last request ended in the measured phases of that lifecycle.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as pystats
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from . import _io
from ..exo_bench import metrics

LAT = re.compile(r"^lat-r(?P<pass>\d+)-(?P<prompt>p\d+)-o(?P<o>\d+)$")
MIN_LIFECYCLES = 2
SPREAD_RATIO = 1.10  # a backend difference smaller than this factor, or inside the lifecycle-to-lifecycle spread, is not reported as a difference


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def load(results_dir, campaign):
    runs = []
    for d in sorted(Path(results_dir).glob(f"*-{campaign}-*")):
        if not (d / "results.jsonl").exists() or not (d / "manifest.json").exists():
            continue
        tier = d.name.split(f"-{campaign}-", 1)[1].rsplit("-", 1)[0]
        runs.append(_io.load_run(d, tier))
    return runs


def inference(r):
    return ((r.get("measurements") or {}).get("extra") or {}).get("inference") or {}


def placement_key(r):
    """(stage hosts, layers per stage): the placement Exo chose. Two lifecycles are comparable only when this is identical."""
    cfg = r["configuration"].get("inference") or {}
    return "{}|{}".format(">".join(str(h) for h in cfg.get("stage_hosts") or []), "/".join(str(n) for n in cfg.get("stage_layers") or []))


def backend_of(r):
    return (r["configuration"].get("inference") or {}).get("backend")


def variant_of(r):
    """The experimental variant of a lifecycle ("default" is the baseline: no optional fast path)."""
    return (r["configuration"].get("inference") or {}).get("variant", "default")


def rep_dir(run_dir, r):
    _, point, rep = r["record_id"].split("/", 2)
    return Path(run_dir) / "raw" / point / rep


def read_requests(run_dir, r, phase):
    files = inference(r).get("raw_files", {}).get(phase, {})
    if "requests" not in files:
        return []
    path = rep_dir(run_dir, r) / files["requests"]
    return _io.read_jsonl(path) if path.exists() else []


def usable(r):
    return r["status"]["state"] == "success" and bool(r.get("measurements")) and bool(inference(r))


def lifecycle_rows(runs):
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            inf = inference(r)
            cfg = r["configuration"].get("inference") or {}
            out.append({"tier": tier, "run": Path(run_dir).name, "record": r["record_id"], "state": r["status"]["state"], "backend": backend_of(r), "placement": placement_key(r), "variant": variant_of(r),
                        "stage_hosts": ">".join(str(h) for h in cfg.get("stage_hosts") or []), "stage_layers": "/".join(str(n) for n in cfg.get("stage_layers") or []),
                        "load_s": inf.get("load_s"), "verification": (r.get("verification") or {}).get("status"), "reason": r["status"].get("reason")})
    return out


def cell_name(phase):
    m = LAT.match(phase)
    return (m.group("prompt"), int(m.group("o")), int(m.group("pass"))) if m else None


def lifecycle_cell_stats(run_dir, r):
    """Per (prompt, output tokens): the raw observations of this lifecycle, pooled over its passes. Requests with an undefined TPOT (one-token outputs) contribute to TTFT and E2E only."""
    cells = defaultdict(list)
    for phase in inference(r).get("phases", {}):
        c = cell_name(phase)
        if c:
            cells[(c[0], c[1])] += [q for q in read_requests(run_dir, r, phase) if q.get("status") == "ok"]
    out = {}
    for key, recs in cells.items():
        ds = [metrics.derive(q) for q in recs]
        gen = [d["output_tokens"] for d in ds if d["output_tokens"] is not None]
        row = {"n": len(recs), "generated_tokens_median": pystats.median(gen) if gen else None}
        for name, k in (("ttft", "ttft_s"), ("tpot", "tpot_s"), ("e2e", "e2e_s")):
            vals = [d[k] for d in ds if d[k] is not None]
            row[f"{name}_n"] = len(vals)
            row[f"{name}_p50_ms"] = pystats.median(vals) * 1e3 if vals else None
            s = metrics.summarize(vals)
            row[f"{name}_p95_ms"] = s["p95"] * 1e3 if s["p95"] is not None else None
        itl = [x for d in ds for x in d["itl_s"]]
        s = metrics.summarize(itl)
        row.update({"itl_n": s["n"], "itl_p50_ms": s["p50"] * 1e3 if s["p50"] is not None else None, "itl_p95_ms": s["p95"] * 1e3 if s["p95"] is not None else None,
                    "itl_p99_ms": s["p99"] * 1e3 if s["p99"] is not None else None})
        out[key] = row
    return out


def cell_rows(runs):
    """One row per lifecycle and cell (the per-lifecycle medians are the replicated unit)."""
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if not usable(r):
                continue
            for (prompt, o), row in sorted(lifecycle_cell_stats(run_dir, r).items()):
                out.append({"tier": tier, "record": r["record_id"], "backend": backend_of(r), "placement": placement_key(r), "variant": variant_of(r), "prompt": prompt, "output_tokens_cfg": o, **row})
    return out


PHASE_CONC = re.compile(r"^conc-r(?P<pass>\d+)-c(?P<c>\d+)-(?P<mode>rounds|cont)$")


def phase_rows(runs, kinds=("rounds", "continuous")):
    """One row per lifecycle and measured phase (any rounds / continuous phase, so the concurrency sweep and the large-model phases are covered), from the raw timelines: request counts, the
    distributions' medians, output tokens per second over the phase wall time and the median generated tokens per request."""
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if not usable(r):
                continue
            for name, summ in inference(r).get("phases", {}).items():
                if summ.get("kind") not in kinds:
                    continue
                recs = read_requests(run_dir, r, name)
                if not recs:
                    continue
                d = metrics.request_distributions(recs)
                t = metrics.phase_throughput(recs)
                m = PHASE_CONC.match(name)
                row = {"tier": tier, "record": r["record_id"], "backend": backend_of(r), "placement": placement_key(r), "phase": name, "kind": summ.get("kind"), "conc": summ.get("conc") or (int(m.group("c")) if m else 1),
                       "pass": int(m.group("pass")) if m else None, "requests": t["requests"], "ok": t["successful"], "output_tokens_total": t["output_tokens"],
                       "output_tokens_per_s": (t["rates"] or {}).get("output_tokens_per_s"), "requests_per_s": (t["rates"] or {}).get("requests_per_s"), "wall_s": t["wall_s"]}
                for key in ("ttft_s", "tpot_s", "e2e_s"):
                    row[key[:-2] + "_p50_ms"] = d[key]["p50"] * 1e3 if d[key]["p50"] is not None else None
                    row[key[:-2] + "_n"] = d[key]["n"]
                out.append(row)
    return out


def concurrency_rows(rows):
    """Per placement, backend and concurrency: the median over passes and lifecycles of output tokens/s and TPOT, for the rounds and continuous loops separately (never merged)."""
    groups = defaultdict(list)
    for x in rows:
        if x["pass"] is not None:
            groups[(x["placement"], x["backend"], x["conc"], "rounds" if x["kind"] == "rounds" else "continuous")].append(x)
    out = []
    for (placement, backend, conc, mode), xs in sorted(groups.items()):
        tp = [x["output_tokens_per_s"] for x in xs if x["output_tokens_per_s"] is not None]
        tpot = [x["tpot_p50_ms"] for x in xs if x["tpot_p50_ms"] is not None]
        out.append({"placement": placement, "backend": backend, "conc": conc, "loop": mode, "lifecycles": len({x["record"] for x in xs}), "phases": len(xs),
                    "output_tokens_per_s_median": pystats.median(tp) if tp else None, "output_tokens_per_s_min": min(tp) if tp else None, "output_tokens_per_s_max": max(tp) if tp else None,
                    "tpot_p50_ms_median": pystats.median(tpot) if tpot else None})
    return out


def spread(values):
    return max(values) / min(values) if values and min(values) > 0 else None


def ab_rows(cells):
    """MlxTbccl against MlxRing per cell and identical placement, from the per-lifecycle medians. 'Not compared' says why instead of a number."""
    groups = defaultdict(lambda: defaultdict(list))
    for c in cells:
        if c.get("variant", "default") != "default":  # a fast-path variant never enters the baseline comparison
            continue
        groups[(c["placement"], c["prompt"], c["output_tokens_cfg"])][c["backend"]].append(c)
    out = []
    for (placement, prompt, o), by in sorted(groups.items()):
        for metric in ("ttft", "tpot", "e2e"):
            row = {"placement": placement, "prompt": prompt, "output_tokens_cfg": o, "metric": metric}
            lives = {b: [c[f"{metric}_p50_ms"] for c in by.get(b, []) if c.get(f"{metric}_p50_ms") is not None] for b in ("MlxRing", "MlxTbccl")}
            row.update({"ring_lifecycles": len(lives["MlxRing"]), "tbccl_lifecycles": len(lives["MlxTbccl"])})
            if min(len(v) for v in lives.values()) < MIN_LIFECYCLES:
                row["not_compared"] = f"fewer than {MIN_LIFECYCLES} lifecycles on a side with this placement"
                out.append(row)
                continue
            a, b = pystats.median(lives["MlxRing"]), pystats.median(lives["MlxTbccl"])
            sp = max(spread(lives["MlxRing"]) or 1.0, spread(lives["MlxTbccl"]) or 1.0)
            ratio = b / a
            row.update({"ring_p50_ms": a, "tbccl_p50_ms": b, "ratio_tbccl_over_ring": ratio, "lifecycle_spread_max": sp,
                        "exceeds_variation": bool(abs(ratio - 1.0) > max(sp - 1.0, SPREAD_RATIO - 1.0))})
            out.append(row)
    return out


def variant_rows(cells):
    """Each experimental variant of MlxTbccl against the default MlxTbccl lifecycles (and MlxRing as the reference) per identical placement and cell, from per-lifecycle medians; fewer than two
    lifecycles on either side of a comparison is 'not compared'."""
    groups = defaultdict(lambda: defaultdict(list))
    for c in cells:
        key = "MlxRing" if c["backend"] == "MlxRing" else c.get("variant", "default")
        if c["backend"] == "MlxRing" and c.get("variant", "default") != "default":
            continue
        groups[(c["placement"], c["prompt"], c["output_tokens_cfg"])][key].append(c)
    out = []
    for (placement, prompt, o), by in sorted(groups.items()):
        for variant in sorted(k for k in by if k not in ("default", "MlxRing")):
            for metric in ("tpot", "e2e", "ttft"):
                vals = {k: [c[f"{metric}_p50_ms"] for c in by.get(k, []) if c.get(f"{metric}_p50_ms") is not None] for k in (variant, "default", "MlxRing")}
                row = {"placement": placement, "prompt": prompt, "output_tokens_cfg": o, "metric": metric, "variant": variant, "variant_lifecycles": len(vals[variant]), "default_lifecycles": len(vals["default"]),
                       "ring_lifecycles": len(vals["MlxRing"])}
                if min(len(vals[variant]), len(vals["default"])) < MIN_LIFECYCLES:
                    row["not_compared"] = f"fewer than {MIN_LIFECYCLES} lifecycles on a side with this placement"
                    out.append(row)
                    continue
                v, d = pystats.median(vals[variant]), pystats.median(vals["default"])
                sp = max(spread(vals[variant]) or 1.0, spread(vals["default"]) or 1.0)
                row.update({"variant_p50_ms": v, "default_p50_ms": d, "ratio_variant_over_default": v / d, "lifecycle_spread_max": sp,
                            "exceeds_variation": bool(abs(v / d - 1.0) > max(sp - 1.0, SPREAD_RATIO - 1.0))})
                if len(vals["MlxRing"]) >= 1:
                    row["ring_p50_ms"] = pystats.median(vals["MlxRing"])
                out.append(row)
    return out


def utc_s(text):
    """Seconds since the epoch of an ISO 8601 UTC timestamp ('Z' or a numeric offset, optional fraction). Raises ValueError for anything else."""
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError(f"timestamp {text!r} has no time zone")
    return int(parsed.timestamp())


def sampler_series(run_dir, r):
    """Host -> sample rows of this lifecycle: from the files the adapter wrote, else from the raw per-rank outputs of an older run."""
    inf = inference(r)
    rd = rep_dir(run_dir, r)
    series = {}
    files = (inf.get("raw_files") or {}).get("samples") or {}
    for host, rel in files.items():
        p = rd / rel
        if p.exists():
            series[host] = _io.read_jsonl(p)
    if series:
        return series
    for p in sorted(rd.glob("rank*.stdout")):
        rows = []
        for line in p.read_text(errors="replace").splitlines():
            if line.startswith("TBCCL_EXO_SAMPLE "):
                try:
                    rows.append(json.loads(line[len("TBCCL_EXO_SAMPLE "):]))
                except ValueError:
                    continue
        if rows:
            series[rows[0].get("role", p.name)] = rows
    return series


def measured_window(run_dir, r):
    """(first submission wall time, last end wall time, generated tokens, successful requests) over the measured phases (lat-* cells) of a lifecycle; None when there are none."""
    starts, ends, tokens, n = [], [], 0, 0
    for phase in inference(r).get("phases", {}):
        if not cell_name(phase):
            continue
        for q in read_requests(run_dir, r, phase):
            if q.get("status") != "ok":
                continue
            d = metrics.derive(q)
            starts.append(q["wall_submit"])
            ends.append(q["wall_submit"] + (q["t_end_ns"] - q["t_submit_ns"]) / 1e9)
            tokens += d["output_tokens"] or 0
            n += 1
    return (min(starts), max(ends), tokens, n) if starts else None


def resource_rows(runs):
    """CPU seconds, GPU utilisation and link bytes per generated token over the measured window of each lifecycle, per host."""
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if not usable(r):
                continue
            win = measured_window(run_dir, r)
            if not win:
                continue
            t0, t1, tokens, n = win
            for host, rows in sampler_series(run_dir, r).items():
                rows = [x for x in rows if "wall_utc" in x]
                inside = [x for x in rows if t0 - 1 <= utc_s(x["wall_utc"]) <= t1 + 1]
                if len(inside) < 3 or not tokens:
                    continue
                interval = pystats.median(b["t_mono_s"] - a["t_mono_s"] for a, b in zip(inside, inside[1:]))
                proc = [sum(p["cpu_percent"] for p in x.get("processes", [])) for x in inside]
                cpu_s = sum(proc) / 100.0 * interval
                links = [x["link"] for x in inside if x.get("link")]
                gpu = [float(x["gpu"]["utilization_gpu_percent"]) for x in inside if x.get("gpu") and x["gpu"].get("utilization_gpu_percent") not in (None, "")]
                out.append({"tier": tier, "record": r["record_id"], "backend": backend_of(r), "placement": placement_key(r), "variant": variant_of(r), "host": host, "window_s": t1 - t0, "samples": len(inside),
                            "output_tokens": tokens, "requests": n, "process_cpu_seconds": cpu_s, "cpu_seconds_per_output_token": cpu_s / tokens,
                            "process_cpu_percent_median": pystats.median(proc), "gpu_utilization_median": pystats.median(gpu) if gpu else None,
                            "process_threads_max": max((sum(p["threads"] for p in x.get("processes", [])) for x in inside), default=None),
                            "link_tx_bytes": links[-1]["tx_bytes"] - links[0]["tx_bytes"] if len(links) > 1 else None, "link_rx_bytes": links[-1]["rx_bytes"] - links[0]["rx_bytes"] if len(links) > 1 else None,
                            "link_bytes_per_output_token": ((links[-1]["tx_bytes"] - links[0]["tx_bytes"]) + (links[-1]["rx_bytes"] - links[0]["rx_bytes"])) / tokens if len(links) > 1 else None})
    return out


def resource_ab(rows):
    """Per placement and host: median over lifecycles of CPU seconds per output token, by backend."""
    groups = defaultdict(lambda: defaultdict(list))
    for x in rows:
        if x.get("variant", "default") != "default":
            continue
        groups[(x["placement"], x["host"])][x["backend"]].append(x)
    out = []
    for (placement, host), by in sorted(groups.items()):
        row = {"placement": placement, "host": host, "ring_lifecycles": len(by.get("MlxRing", [])), "tbccl_lifecycles": len(by.get("MlxTbccl", []))}
        if min(row["ring_lifecycles"], row["tbccl_lifecycles"]) < MIN_LIFECYCLES:
            row["not_compared"] = f"fewer than {MIN_LIFECYCLES} lifecycles on a side with this placement"
            out.append(row)
            continue
        for key in ("cpu_seconds_per_output_token", "link_bytes_per_output_token", "process_cpu_percent_median"):
            a = pystats.median(x[key] for x in by["MlxRing"] if x.get(key) is not None)
            b = pystats.median(x[key] for x in by["MlxTbccl"] if x.get(key) is not None)
            row.update({f"ring_{key}": a, f"tbccl_{key}": b, f"{key}_ratio": b / a if a else None})
        out.append(row)
    return out


def gate_rows(runs):
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            for name, ph in (inference(r).get("phases") or {}).items():
                chk = ph.get("check") or ph.get("cancel")
                if chk:
                    out.append({"tier": tier, "record": r["record_id"], "backend": backend_of(r), "placement": placement_key(r), "phase": name, "passed": chk.get("passed"),
                                "outcomes": json.dumps(chk.get("outcome") or {"cancelled": chk.get("cancelled"), "completed": chk.get("completed")}), "tie_margin": (chk.get("tie_margin"))})
    return out


def excluded_rows(everything, used):
    return [{"tier": t, "run": d.name, "status": m["status"], "reason": m.get("reason"), "counts": m["counts"]} for t, d, m, _ in everything if (t, d) not in {(x[0], x[1]) for x in used}]


def figures(out, ab):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made = []
    rows = [r for r in ab if r.get("ratio_tbccl_over_ring") and r["metric"] == "tpot"]
    if rows:
        out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        for placement in sorted({r["placement"] for r in rows}):
            pts = sorted((r["output_tokens_cfg"], r["prompt"], r["ratio_tbccl_over_ring"]) for r in rows if r["placement"] == placement)
            ax.plot(range(len(pts)), [p[2] for p in pts], marker="o", label=placement)
            ax.set_xticks(range(len(pts)))
            ax.set_xticklabels([f"{p[1]}/o{p[0]}" for p in pts], rotation=60, fontsize=7)
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_ylabel("TPOT, MlxTbccl / MlxRing (median of lifecycle medians)")
        ax.set_title("Exo decode time per output token, by placement")
        ax.legend(fontsize=7)
        fig.tight_layout()
        p = out / "tpot_ratio.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        made.append(str(p))
    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("campaign")
    parser.add_argument("--out")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--include-tier", action="append", default=[], help="use only runs whose tier starts with this (repeatable)")
    args = parser.parse_args(argv)
    everything = load(args.results_dir, args.campaign)
    runs = [r for r in everything if not any(x in r[1].name for x in args.exclude) and r[2]["status"] in ("success", "partial")
            and (not args.include_tier or any(r[0].startswith(t) for t in args.include_tier))]
    out = Path(args.out or Path(args.results_dir) / f"campaign-{args.campaign}")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    cells = cell_rows(runs)
    ab = ab_rows(cells)
    res = resource_rows(runs)
    prows = phase_rows(runs)
    for name, data in (("lifecycles", lifecycle_rows(runs)), ("cells", cells), ("ab_latency", ab), ("resources", res), ("resources_ab", resource_ab(res)), ("gates", gate_rows(runs)),
                       ("phases", prows), ("concurrency", concurrency_rows(prows)), ("variants", variant_rows(cells))):
        write_csv(out / "tables" / f"{name}.csv", data)
    excluded = excluded_rows(everything, runs)
    (out / "excluded_runs.json").write_text(json.dumps(excluded, indent=2))
    made = figures(out / "figures", ab)
    print(json.dumps({"campaign": args.campaign, "runs": [{"tier": t, "run": d.name, "status": m["status"], "counts": m["counts"]} for t, d, m, _ in runs], "excluded": excluded,
                      "tables": str(out / "tables"), "figures": made}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
