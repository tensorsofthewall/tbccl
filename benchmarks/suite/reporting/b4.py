"""Campaign report data for B4 (PyTorch communication and DDP over torch-tbccl): tables (CSV) and static figures from the normalised results of several runs (tiers).

    python3 -m benchmarks.suite.reporting.b4 RESULTS_DIR CAMPAIGN [--out DIR] [--exclude TEXT]

Native TBCCL results (``tensor_transfer``, ``small_message_latency``), PyTorch adapter results (``torch_*``) and DDP training results are kept in separate tables and never merged
into one statistic. A framework-versus-native figure is only produced from points that were run as interleaved A,B,B,A pairs with the same hosts, direction, size and memory kind, and it is
reported as an observed difference of two measurements, not as an isolated adapter cost. Raw per-iteration samples are kept; repetitions are never averaged across modes.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as pystats
import sys
from collections import defaultdict
from pathlib import Path

from . import _io
from . import b3b
from .b3b import CATEGORICAL, MUTED, INK, across, group, ok, stat

TORCH = ("torch_collectives", "torch_p2p", "torch_async", "torch_ddp")
SPLIT_RATIO = 1.5  # repetition medians that differ by more than this factor are reported as separate modes


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


def os_of(r, host_id):
    return b3b.os_label(r, host_id)


def backend_of(r, host_id):
    for x in r["artifacts"].get("ranks", []):
        if x["host_id"] == host_id:
            return x["memory_backend"]
    return "?"


def pairing(r):
    """cpu/cuda/mps naming of the two machines, Linux side first (independent of rank numbering)."""
    by = {os_of(r, h["host_id"]): h["host_id"] for h in r["environment"]["hosts"]}
    name = {"host": "cpu", "cuda": "cuda", "mps": "mps", "cuda-pinned": "cuda", "metal-shared": "mps"}
    return f"{name.get(backend_of(r, by.get('Linux', '')), '?')}/{name.get(backend_of(r, by.get('macOS', '')), '?')}"


def direction(r):
    d = r["configuration"].get("data_direction") or {}
    if d.get("mode") == "directional" and d.get("source_host"):
        return f"{os_of(r, d['source_host'])} -> {os_of(r, d['destination_host'])}"
    return "symmetric"


def opt(r, k, default=None):
    return (r["configuration"].get("options") or {}).get(k, default)


def extra(r, k):
    return ((r["measurements"].get("extra") or {}).get(k) or {}).get("value")


def timing(r):
    return r["measurements"]["timing"]


def backend_name(r):
    return opt(r, "backend", "tbccl")


def spread_rows(values):
    a = across(values)
    return a


def modes(meds):
    """Per-repetition medians split into modes when they differ by more than SPLIT_RATIO (so that a bimodal configuration is not summarized by one number)."""
    v = sorted(m for m in meds if m is not None)
    if len(v) < 2 or v[-1] / v[0] <= SPLIT_RATIO:
        return None
    k = max(range(1, len(v)), key=lambda i: v[i] / v[i - 1])
    lo, hi = v[:k], v[k:]
    return {"low_mode_us": pystats.median(lo), "low_count": len(lo), "high_mode_us": pystats.median(hi), "high_count": len(hi)}


def comm_rows(records, names):
    rows = []
    keyfn = lambda r: (r["_tier"], r["benchmark"]["name"], r["benchmark"]["operation"], backend_name(r), pairing(r), direction(r), r["configuration"]["payload_bytes"])  # noqa: E731
    for key, recs in sorted(group([r for r in records if r["benchmark"]["name"] in names], keyfn).items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tier, bench, op, backend, pair, dirn, size = key
        meds = [stat(r, "median") for r in recs]
        med = across(meds)
        t0 = timing(recs[0])
        pooled = sorted(x for r in recs for x in (timing(r).get("samples") or []))
        row = {"tier": tier, "benchmark": bench, "operation": op, "backend": backend, "pairing": pair, "direction": dirn, "payload_bytes": size, "repetitions": med["n"],
               "iterations": recs[0]["configuration"]["iterations"], "metric": t0["metric"], "median_us": med["median"], "median_min_us": med["min"], "median_max_us": med["max"],
               "median_range_rel": med["range_rel"], "p95_us": (across([stat(r, "p95") for r in recs]) or {}).get("median"), "p99_us": (across([stat(r, "p99") for r in recs]) or {}).get("median"),
               "pooled_samples": len(pooled), "pooled_p99_us": pooled[int(0.99 * (len(pooled) - 1))] if len(pooled) >= 100 else None,
               "repetition_medians_us": ";".join(f"{m:.1f}" for m in meds), "throughput_GBps": (across([(r["measurements"]["throughput"] or [{}])[0].get("value") for r in recs]) or {}).get("median"),
               "verification": recs[0]["verification"]["status"]}
        m = modes(meds)
        if m:
            row.update({"bimodal": True, **m})
        else:
            row["bimodal"] = False
        for k in ("device_ready_median_us", "dev_sync_median_us_rank0", "dev_sync_median_us_rank1", "submit_median_us_rank0", "submit_median_us_rank1", "wait_median_us_rank0", "wait_median_us_rank1",
                  "callback_median_us_rank0", "callback_median_us_rank1", "source_submit_median_us", "source_wait_median_us", "source_send_median_us", "destination_recv_median_us", "source_dev_sync_median_us"):
            v = across([extra(r, k) for r in recs])
            if v and v["median"] is not None:
                row[k] = v["median"]
        for k in ("callbacks_seen_rank0", "complete_at_submit_rank0", "complete_at_submit_rank1"):
            vals = [extra(r, k) for r in recs if extra(r, k) is not None]
            if vals:
                row[k + "_total"] = sum(vals)
        cpu = [(pr.get("cpu_cores")) for r in recs for pr in (timing(r).get("per_rank") or [])]
        cpu = [c for c in cpu if c is not None]
        row["cpu_cores_median"] = pystats.median(cpu) if cpu else None
        rows.append(row)
    return rows


AB_TAG = re.compile(r"^b(\d+)-(.+)-(A|B)(\d)$")


def framework_rows(records):
    """Matched native (A) and torch-tbccl (B) points from the interleaved A,B,B,A blocks: the same hosts, source, size, memory kinds and completion semantics (ACK-confirmed one-way transfer,
    all_reduce / all_gather completion, ping-pong round trip). Both arms' repetition medians are kept; the difference is an observed difference of two measurements."""
    arms = defaultdict(lambda: {"A": [], "B": []})
    for r in ok(records):
        tag = opt(r, "tag")
        m = AB_TAG.match(tag or "")
        if not m:
            continue
        block, ident, arm, _ = m.groups()
        op = r["benchmark"]["operation"]
        family = {"end_to_end": "p2p_ack", "p2p_ack": "p2p_ack", "allreduce": "all_reduce", "all_reduce": "all_reduce", "allgather": "all_gather", "all_gather": "all_gather",
                  "p2p": "p2p_rtt", "p2p_rtt": "p2p_rtt"}[op]
        arms[(family, ident, r["configuration"]["payload_bytes"])][arm].append((int(block), r))
    rows = []
    for (family, ident, size), d in sorted(arms.items()):
        if not d["A"] or not d["B"]:
            continue
        am, bm = [stat(r, "median") for _, r in d["A"]], [stat(r, "median") for _, r in d["B"]]
        a, b = across(am), across(bm)
        r0 = d["B"][0][1]
        rows.append({"family": family, "identity": ident, "pairing": pairing(r0) if r0["benchmark"]["name"].startswith("torch") else pairing(d["A"][0][1]), "payload_bytes": size,
                     "native_benchmark": d["A"][0][1]["benchmark"]["name"], "torch_benchmark": r0["benchmark"]["name"], "native_n": a["n"], "torch_n": b["n"],
                     "native_median_us": a["median"], "native_min_us": a["min"], "native_max_us": a["max"], "native_repetition_medians_us": ";".join(f"{v:.1f}" for v in am),
                     "torch_median_us": b["median"], "torch_min_us": b["min"], "torch_max_us": b["max"], "torch_repetition_medians_us": ";".join(f"{v:.1f}" for v in bm),
                     "difference_us": b["median"] - a["median"], "ratio": b["median"] / a["median"] if a["median"] else None,
                     "native_bimodal": modes(am) is not None, "torch_bimodal": modes(bm) is not None})
    return rows


def ddp_rows(records):
    rows = []
    for key, recs in sorted(group([r for r in records if r["benchmark"]["name"] == "torch_ddp"], lambda r: (r["_tier"], backend_name(r), pairing(r), opt(r, "tag"))).items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tier, backend, pair, tag = key
        st = across([stat(r, "median") for r in recs])
        sps = across([next(t["value"] for t in r["measurements"]["throughput"] if t["kind"] == "training_samples_per_second") for r in recs])
        row = {"tier": tier, "backend": backend, "pairing": pair, "tag": tag, "model": extra(recs[0], "model"), "layers": opt(recs[0], "layers"), "width": opt(recs[0], "width"),
               "batch_per_rank": opt(recs[0], "batch"), "bucket_cap_mb": opt(recs[0], "bucket_cap_mb", 25), "grad_bytes": extra(recs[0], "grad_bytes_per_step"), "repetitions": st["n"],
               "step_median_us": st["median"], "step_min_us": st["min"], "step_max_us": st["max"], "step_range_rel": st["range_rel"], "samples_per_s": sps["median"], "samples_per_s_min": sps["min"],
               "samples_per_s_max": sps["max"], "steps_per_s": 1e6 / st["median"], "compute_imbalance_ratio": (across([extra(r, "compute_imbalance_ratio") for r in recs]) or {}).get("median"),
               "slowest_compute_rank": pystats.mode([extra(r, "slowest_compute_rank") for r in recs]), "ddp_bucket_sizes": extra(recs[0], "ddp_bucket_sizes"),
               "grad_rel_err_max": max(extra(r, "correctness_max_grad_rel_err") for r in recs), "param_rel_err_max": max(extra(r, "correctness_max_param_rel_err") for r in recs),
               "cross_rank_abs_diff_max": max(extra(r, "correctness_cross_rank_max_abs_diff") for r in recs)}
        for rk in (0, 1):
            row[f"rank{rk}_device"] = extra(recs[0], f"rank{rk}_device")
            for k in ("forward", "backward", "optimizer", "phased_step", "nosync_forward", "nosync_backward", "nosync_optimizer", "nosync_step"):
                row[f"rank{rk}_{k}_us"] = (across([extra(r, f"rank{rk}_{k}_median_us") for r in recs]) or {}).get("median")
            row[f"rank{rk}_exposed_comm_inferred_us"] = (across([extra(r, f"rank{rk}_exposed_communication_inferred_us") for r in recs]) or {}).get("median")
            row[f"rank{rk}_peak_memory_MiB"] = (across([extra(r, f"rank{rk}_peak_memory_bytes") for r in recs]) or {}).get("median", 0) / 2**20
            row[f"rank{rk}_cpu_cores"] = (across([extra(r, f"rank{rk}_cpu_cores") for r in recs]) or {}).get("median")
            gpu = [extra(r, f"rank{rk}_gpu_utilization_median_percent") for r in recs if extra(r, f"rank{rk}_gpu_utilization_median_percent") is not None]
            if gpu:
                row[f"rank{rk}_gpu_utilization_percent"] = pystats.median(gpu)
        rows.append(row)
    return rows


def profile_rows(records):
    rows = []
    for r in ok(records):
        if r["benchmark"]["name"] != "torch_ddp" or not opt(r, "profile"):
            continue
        for rk in (0, 1):
            prof = extra(r, f"rank{rk}_profile")
            if not prof:
                continue
            p = json.loads(prof)
            ts = p.get("trace_summary") or {}
            rows.append({"tier": r["_tier"], "pairing": pairing(r), "tag": opt(r, "tag"), "repetition": r["configuration"]["repetition_index"], "rank": rk, "device": extra(r, f"rank{rk}_device"),
                         "trace": p.get("trace"), **{k: v for k, v in ts.items() if k not in ("layout_bytes", "note")}, "bucket_layout_bytes": ";".join(str(x) for x in ts.get("layout_bytes", [])),
                         "gpu_throttle_reasons": ";".join(p.get("gpu_throttle_reasons", []))})
    return rows


def run_to_run(rows, tier_a, tier_b, keys, value, spread):
    index = {tuple(r.get(k) for k in keys): r for r in rows if r["tier"] == tier_a}
    out = []
    for r in rows:
        a = index.get(tuple(r.get(k) for k in keys)) if r["tier"] == tier_b else None
        if a and a[value]:
            out.append({**{k: r.get(k) for k in keys}, "main_median": a[value], "confirm_median": r[value], "ratio_confirm_over_main": r[value] / a[value]})
    return out


# ---------------------------------------------------------------------------- figures
def _plt():
    return b3b._plt()


def _foot(op, direction_text, backend, metric, timing_text, units, reps, records):
    return (f"operation: {op} | pairing / data direction: {direction_text} | framework: PyTorch {backend} | transport: TCP over Thunderbolt 4 IP (not RDMA)\n"
            f"metric: {metric} | timing: {timing_text[:150]} | units: {units} | repetitions: {reps}\nbuild provenance: {b3b.provenance_line(records)[:230]}")


def figures(out, comm, fw, asyn, ddp, profile, recs):
    plt = _plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    colls = [r for r in comm if r["benchmark"] == "torch_collectives" and r["tier"].endswith("t1-collectives")]
    tc = [x for x in recs if x["benchmark"]["name"] == "torch_collectives"]
    for op in ("all_reduce", "all_gather", "broadcast"):
        sub = [r for r in colls if r["operation"] == op and r["direction"] in ("symmetric", "Linux -> macOS")]
        if not sub:
            continue
        series = defaultdict(list)
        for r in sub:
            series[r["pairing"]].append((r["payload_bytes"], r["median_us"], r["median_min_us"], r["median_max_us"]))
        fig, ax = b3b._setup(plt, f"PyTorch {op} over torch-tbccl: blocking completion time vs tensor size (float32, 2 ranks)", "tensor bytes", "median of repetition medians (us); band = min..max")
        b3b._lines(ax, series)
        ax.legend(frameon=False, fontsize=7, labelcolor=MUTED, loc="upper left")
        made.append(b3b._save(fig, plt, out / f"collective-{op}.png", _foot(op, "Linux -> macOS" if op == "broadcast" else "symmetric", "torch-tbccl", "per-iteration maximum over ranks of the call time",
                                                                             tc[0]["configuration"]["timing_method"] if tc else "", "us", "3", tc)))
    p2p = [r for r in comm if r["benchmark"] == "torch_p2p" and r["operation"] == "p2p_ack" and r["tier"].endswith("t1-collectives")]
    if p2p:
        for dirn in sorted({r["direction"] for r in p2p}):
            series = defaultdict(list)
            for r in p2p:
                if r["direction"] == dirn:
                    series[r["pairing"]].append((r["payload_bytes"], r["median_us"], r["median_min_us"], r["median_max_us"]))
            fig, ax = b3b._setup(plt, f"PyTorch send/recv with 1-byte ACK, {dirn}: completion-confirmed time", "tensor bytes", "median of repetition medians (us); band = min..max")
            b3b._lines(ax, series)
            ax.legend(frameon=False, fontsize=7, labelcolor=MUTED, loc="upper left")
            tp = [x for x in recs if x["benchmark"]["name"] == "torch_p2p"]
            made.append(b3b._save(fig, plt, out / f"p2p-ack-{dirn.replace(' ', '').replace('>', 'to').replace('-', '')}.png", _foot("p2p_ack", dirn, "torch-tbccl", "completion-confirmed time on the source",
                                                                                                                                     tp[0]["configuration"]["timing_method"] if tp else "", "us", "3", tp)))
    # framework versus native
    if fw:
        fams = sorted({r["family"] for r in fw})
        fig, axes = plt.subplots(1, len(fams), figsize=(5.2 * len(fams), 5.2), facecolor="#fcfcfb", squeeze=False)
        for ax, fam in zip(axes[0], fams):
            ax.set_facecolor("#fcfcfb")
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            groups = defaultdict(list)
            for r in fw:
                if r["family"] == fam:
                    groups[(r["pairing"], r["identity"].split("-s")[-1] if "-s" in r["identity"] else "")].append((r["payload_bytes"], r["ratio"]))
            for i, (k, pts) in enumerate(sorted(groups.items())):
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="os^Dvp"[i % 6], color=CATEGORICAL[i % 8], label=f"{k[0]}" + (f" src{k[1]}" if k[1] else ""), markersize=5, linewidth=1.5)
            ax.axhline(1.0, color=MUTED, linewidth=0.8)
            ax.set_title(f"{fam}: torch-tbccl / native (median ratio)", loc="left", color=INK, fontsize=9.5)
            ax.set_xlabel("tensor bytes", color=MUTED)
            ax.set_ylabel("ratio of medians (interleaved A,B,B,A)", color=MUTED)
            ax.legend(frameon=False, fontsize=6.5, labelcolor=MUTED)
            ax.grid(True, color=b3b.GRID, linewidth=0.8, which="both")
        made.append(b3b._save(fig, plt, out / "framework-vs-native-ratio.png", _foot("native vs PyTorch (matched)", "see legend", "torch-tbccl and native tbccl", "ratio torch / native of medians of repetition medians",
                                                                                      "native: tensor_transfer end_to_end, small_message_latency; torch: p2p_ack, all_reduce, all_gather, p2p_rtt", "ratio", "6 per arm", recs)))
        fig, ax = b3b._setup(plt, "framework vs native, absolute medians (host memory CPU/CPU shown; native = solid, torch-tbccl = dashed)", "tensor bytes", "median of repetition medians (us)")
        for i, (fam, ident) in enumerate(sorted({(r["family"], r["identity"]) for r in fw if r["pairing"] == "cpu/cpu"})[:6]):
            pts = sorted((r["payload_bytes"], r["native_median_us"], r["torch_median_us"]) for r in fw if r["family"] == fam and r["identity"] == ident)
            c = CATEGORICAL[i % 8]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker="o", markersize=4, label=f"{fam} {ident} native")
            ax.plot([p[0] for p in pts], [p[2] for p in pts], color=c, marker="s", markersize=4, linestyle="--", label=f"{fam} {ident} torch-tbccl")
        ax.legend(frameon=False, fontsize=6, labelcolor=MUTED, ncol=2)
        made.append(b3b._save(fig, plt, out / "framework-vs-native-absolute.png", _foot("native vs PyTorch (matched)", "see legend", "torch-tbccl and native tbccl", "median completion time", "see tables", "us", "6 per arm", recs)))
    # async
    a_rows = [r for r in asyn if r["tier"].endswith("t3-async") and r["operation"] == "all_reduce_async"]
    if a_rows:
        ta = [x for x in recs if x["benchmark"]["name"] == "torch_async"]
        for pair in sorted({r["pairing"] for r in a_rows}):
            sub = sorted((r for r in a_rows if r["pairing"] == pair), key=lambda r: r["payload_bytes"])
            fig, ax = b3b._setup(plt, f"PyTorch all_reduce(async_op=True), {pair}: submission vs wait vs lifetime", "tensor bytes", "median (us)")
            for i, (lab, key) in enumerate((("submission (call returned)", "submit_median_us_rank0"), ("time inside Work.wait()", "wait_median_us_rank0"), ("lifetime (submit start to wait return, max over ranks)", "median_us"),
                                            ("completion callback (rank 0)", "callback_median_us_rank0"))):
                pts = [(r["payload_bytes"], r.get(key)) for r in sub if r.get(key)]
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=CATEGORICAL[i], marker="os^D"[i], label=lab, markersize=5, linewidth=1.6)
            ax.legend(frameon=False, fontsize=7, labelcolor=MUTED, loc="upper left")
            made.append(b3b._save(fig, plt, out / f"async-allreduce-{pair.replace('/', '-')}.png", _foot("all_reduce_async", "symmetric", "torch-tbccl", "submit, wait, lifetime, callback", ta[0]["configuration"]["timing_method"] if ta else "", "us", "3", ta)))
    # DDP
    main = [r for r in ddp if r["tier"].startswith("t4") and r["backend"] == "tbccl"]
    if main:
        td = [x for x in recs if x["benchmark"]["name"] == "torch_ddp"]
        cfgs = sorted(main, key=lambda r: (r["tag"], r["pairing"]))
        fig, ax = plt.subplots(figsize=(max(9, 0.9 * len(cfgs) * 2), 5.8), facecolor="#fcfcfb")
        w = 0.38
        for ci, r in enumerate(cfgs):
            for rk in (0, 1):
                x = ci + (rk - 0.5) * w
                comp_f, comp_b, comp_o = r[f"rank{rk}_nosync_forward_us"], r[f"rank{rk}_nosync_backward_us"], r[f"rank{rk}_nosync_optimizer_us"]
                exp = max(0.0, r[f"rank{rk}_backward_us"] - r[f"rank{rk}_nosync_backward_us"])
                bottom = 0
                for val, col, lab in ((comp_f, CATEGORICAL[0], "forward (compute)"), (comp_b, CATEGORICAL[1], "backward (compute, no_sync)"), (comp_o, CATEGORICAL[2], "optimizer"), (exp, CATEGORICAL[3], "exposed communication (inferred)")):
                    ax.bar(x, val / 1e3, w * 0.92, bottom=bottom, color=col, label=lab if ci == 0 and rk == 0 else None, edgecolor="#fcfcfb")
                    bottom += val / 1e3
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels([f"{r['pairing']}\n{r['tag']}\n(rank0 | rank1)" for r in cfgs], fontsize=6.5, color=MUTED)
        ax.set_ylabel("median per-step phase time (ms); left bar rank 0, right bar rank 1", color=MUTED)
        ax.set_title("DDP step-time breakdown per rank: compute from the no_sync loop, exposed communication inferred (backward with DDP minus backward without)", loc="left", color=INK, fontsize=9.5)
        ax.legend(frameon=False, fontsize=7, labelcolor=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        made.append(b3b._save(fig, plt, out / "ddp-step-breakdown.png", _foot("DDP training step", "both ranks", "torch-tbccl", "phase medians per rank", td[0]["configuration"]["timing_method"] if td else "", "ms", "3", td)))
        fig, ax = plt.subplots(figsize=(9, 5.4), facecolor="#fcfcfb")
        labels = [f"{r['pairing']}\n{r['tag']}" for r in cfgs]
        for rk in (0, 1):
            ax.bar([i + (rk - 0.5) * 0.38 for i in range(len(cfgs))], [r[f"rank{rk}_nosync_step_us"] / 1e3 for r in cfgs], 0.36, label=f"rank {rk} compute-only step (no_sync)", color=CATEGORICAL[rk], edgecolor="#fcfcfb")
        ax.plot(range(len(cfgs)), [r["step_median_us"] / 1e3 for r in cfgs], color=INK, marker="o", linestyle="none", label="DDP step (slowest rank, free loop)")
        ax.set_xticks(range(len(cfgs)))
        ax.set_xticklabels(labels, fontsize=6.5, color=MUTED)
        ax.set_ylabel("median step time (ms)", color=MUTED)
        ax.set_title("Per-rank compute-only step time against the DDP step time (rank imbalance)", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=7, labelcolor=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        made.append(b3b._save(fig, plt, out / "ddp-per-rank-step.png", _foot("DDP training step", "both ranks", "torch-tbccl", "compute-only step per rank vs DDP step", "see tables", "ms", "3", td)))
        fig, ax = plt.subplots(figsize=(9, 5.4), facecolor="#fcfcfb")
        for i, tag_group in enumerate(sorted({r["pairing"] for r in main})):
            sub = sorted((r for r in main if r["pairing"] == tag_group), key=lambda r: (r["layers"], r["batch_per_rank"], r["bucket_cap_mb"]))
            ax.plot(range(len(sub)), [r["samples_per_s"] for r in sub], marker="os^Dvp"[i % 6], color=CATEGORICAL[i % 8], label=tag_group)
            for j, r in enumerate(sub):
                ax.annotate(r["tag"], (j, r["samples_per_s"]), fontsize=5.5, color=MUTED, rotation=30)
        ax.set_yscale("log")
        ax.set_xlabel("configuration index (model, batch per rank, bucket cap; labelled)", color=MUTED)
        ax.set_ylabel("samples per second (global batch / slowest-rank step)", color=MUTED)
        ax.set_title("DDP training throughput by configuration", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=7, labelcolor=MUTED)
        ax.grid(True, color=b3b.GRID, linewidth=0.8, which="both")
        made.append(b3b._save(fig, plt, out / "ddp-throughput.png", _foot("DDP training step", "both ranks", "torch-tbccl", "samples/s", "see tables", "samples/s", "3", td)))
    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("campaign")
    parser.add_argument("--out")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--include-prelim", action="store_true")
    args = parser.parse_args(argv)
    everything = load(args.results_dir, args.campaign)
    runs = [r for r in everything if (args.include_prelim or "prelim" not in r[0] and "validation" not in r[0]) and r[2]["status"] == "success" and not any(x in r[1].name for x in args.exclude)]
    excluded = [{"tier": t, "run": d.name, "status": m["status"], "reason": m.get("reason"), "counts": m["counts"]} for t, d, m, _ in everything if (t, d) not in {(x[0], x[1]) for x in runs}]
    out = Path(args.out or Path(args.results_dir) / f"campaign-{args.campaign}")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    allrecs = []
    for tier, _, _, recs in runs:
        for r in recs:
            r["_tier"] = tier
            allrecs.append(r)
    comm = comm_rows(allrecs, ("torch_collectives", "torch_p2p", "torch_async"))
    asyn = [r for r in comm if r["benchmark"] == "torch_async"]
    fw = framework_rows(allrecs)
    ddp = ddp_rows(allrecs)
    prof = profile_rows(allrecs)
    r2r = {"run_to_run_comm": run_to_run(comm, "t1-collectives", "t6-confirm", ("benchmark", "operation", "pairing", "direction", "payload_bytes"), "median_us", "median_range_rel"),
           "run_to_run_ddp": run_to_run(ddp, "t4-ddp", "t6-confirm", ("pairing", "tag"), "step_median_us", "step_range_rel")}
    for name, rows in (*r2r.items(), ("communication", comm), ("framework_vs_native", fw), ("ddp", ddp), ("ddp_profile", prof), ("cpu", b3b.cpu_rows(runs)), ("failures", b3b.failures(runs))):
        write_csv(out / "tables" / f"{name}.csv", rows)
    made = figures(out / "figures", comm, fw, asyn, ddp, prof, allrecs) if b3b._have_mpl() else []
    (out / "excluded_runs.json").write_text(json.dumps(excluded, indent=2))
    print(json.dumps({"campaign": args.campaign, "excluded_from_distributions": excluded, "runs": [{"tier": t, "run": d.name, "status": m["status"], "counts": m["counts"]} for t, d, m, _ in runs],
                      "tables": str(out / "tables"), "figures": made}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
