"""Campaign report data for the B3C heterogeneous CUDA / Metal characterization: tables (CSV) and static figures from the normalised results of several runs (tiers).

    python3 -m benchmarks.suite.reporting.b3c RESULTS_DIR CAMPAIGN [CAMPAIGN ...] [--out DIR] [--b3b-tables DIR]

Runs are selected by the campaign ids in their label (``...-b3c1-t1-transfer-...``). Preliminary, validation, aborted and failed runs never enter a distribution; they are listed.
Nothing is combined across incomparable metrics: a table or figure is for one operation, one metric, one data direction, one memory-backend pair. Host-memory controls are taken from the
SAME run as the device points they are compared with; the older B3B host-memory numbers appear only as a labelled cross-campaign column. A time difference between a device and a host
configuration is an observed incremental path cost, not an isolated staging cost, and overlapping stage columns are never summed.
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
from . import b3b
from .b3b import CATEGORICAL, MUTED, INK, across, group, ok, stat

STAGE_COLUMNS = ("source_staging_us", "source_sync_us", "sender_network_us", "receiver_network_us", "destination_staging_us", "destination_sync_us", "producer_enqueue_us")
CUDA_OS, METAL_OS = "Linux", "macOS"
NO_COPY_BELOW_US = 0.1  # a staging call this short is a no-op (shared or host memory) at the clock's resolution (about 20-40 ns per reading); no throughput is derived from it


def load(results_dir, campaigns):
    """[(tier, run_dir, manifest, records)]: tier is the label after the campaign id (``t1-transfer``, ``staging-linux``, ``t0-validation-rerun``)."""
    runs = []
    for campaign in campaigns:
        for d in sorted(Path(results_dir).glob(f"*-{campaign}-*")):
            if not (d / "results.jsonl").exists() or not (d / "manifest.json").exists():
                continue
            tier = d.name.split(f"-{campaign}-", 1)[1].rsplit("-", 1)[0]
            runs.append(_io.load_run(d, f"{campaign}:{tier}"))
    return runs


def os_of(record, host_id):
    return b3b.os_label(record, host_id)


def backend_of(record, host_id):
    for x in record["artifacts"].get("ranks", []):
        if x["host_id"] == host_id:
            return x["memory_backend"]
    return "?"


def hosts_by_os(record):
    return {os_of(record, h["host_id"]): h["host_id"] for h in record["environment"]["hosts"]}


def pair_label(record):
    """Source backend -> destination backend for a directional benchmark; for a bidirectional one the Linux-side and macOS-side backends (independent of rank numbering)."""
    d = record["configuration"].get("data_direction") or {}
    if d.get("mode") == "directional" and d.get("source_host"):
        return f"{backend_of(record, d['source_host'])} -> {backend_of(record, d['destination_host'])}"
    by = hosts_by_os(record)
    return f"{backend_of(record, by.get(CUDA_OS, ''))} | {backend_of(record, by.get(METAL_OS, ''))}"


def device_pair(record):
    """The Linux-side and macOS-side backends regardless of data direction: the identity of the memory configuration."""
    by = hosts_by_os(record)
    return f"{backend_of(record, by.get(CUDA_OS, ''))} / {backend_of(record, by.get(METAL_OS, ''))}"


def opt(record, name, default=None):
    return (record["configuration"].get("options") or {}).get(name, default)


def extra(r, name):
    return ((r["measurements"].get("extra") or {}).get(name) or {}).get("value")


def tp_value(r, kind=None):
    for t in r["measurements"].get("throughput") or []:
        if kind is None or t["kind"] == kind:
            return t["value"]
    return None


def timing(r):
    return r["measurements"]["timing"]


def write_csv(path, rows):
    """Like the B3B writer but with the union of the columns, since optional columns (a host control, a reference) exist only on some rows."""
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------- tables
def staging_rows(records):
    rows = []
    for key, recs in sorted(group(records, lambda r: (os_of(r, r["artifacts"]["ranks"][0]["host_id"]), r["artifacts"]["ranks"][0]["memory_backend"],
                                                       r["benchmark"]["operation"], r["configuration"]["payload_bytes"])).items()):
        host_os, backend, op, size = key
        med = across([stat(r, "median") for r in recs])
        tp = across([tp_value(r) for r in recs])
        t0 = timing(recs[0])
        rows.append({"host_os": host_os, "memory_backend": backend, "operation": op, "metric": t0["metric"], "payload_bytes": size, "repetitions": med["n"], "iterations": t0["sample_count"],
                     "median_us": med["median"], "median_min_us": med["min"], "median_max_us": med["max"], "median_range_rel": med["range_rel"],
                     "p95_us": (across([stat(r, "p95") for r in recs]) or {}).get("median"), "p99_us": (across([stat(r, "p99") for r in recs]) or {}).get("median"),
                     "throughput_GBps": tp["median"] if tp and med["median"] >= NO_COPY_BELOW_US else None, "below_clock_resolution": med["median"] is not None and med["median"] < NO_COPY_BELOW_US,
                     "verification": recs[0]["verification"]["status"]})
    return rows


def transfer_rows(records, b3b_reference=None, staging=None):
    rows = []
    local = {(r["host_os"], r["memory_backend"], r["operation"], r["payload_bytes"]): r["median_us"] for r in (staging or [])}
    keyfn = lambda r: (r["_tier"], pair_label(r), b3b.direction_label(r), b3b.orientation_label(r), opt(r, "timing_scope", "ready"), r["benchmark"]["operation"], r["configuration"]["payload_bytes"])  # noqa: E731
    grouped = group(records, keyfn)
    host_index = {}
    for key, recs in grouped.items():
        if key[1] == "host -> host" and key[5] == "end_to_end":
            host_index[(key[0],) + key[2:5] + key[6:]] = across([stat(r, "median") for r in recs])
    for key, recs in sorted(grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tier, pair, direction, orient, scope, op, size = key
        med = across([stat(r, "median") for r in recs])
        tp = across([tp_value(r) for r in recs])
        t0 = timing(recs[0])
        row = {"tier": tier, "operation": op, "pair": pair, "direction": direction, "orientation": orient, "timing_scope": scope if op == "end_to_end" else "n/a", "payload_bytes": size,
               "repetitions": med["n"], "iterations": recs[0]["configuration"]["iterations"], "median_us": med["median"], "median_min_us": med["min"], "median_max_us": med["max"],
               "median_range_rel": med["range_rel"], "p95_us": (across([stat(r, "p95") for r in recs]) or {}).get("median"), "p99_us": (across([stat(r, "p99") for r in recs]) or {}).get("median"),
               "throughput_GBps": tp["median"] if tp else None, "throughput_min": tp["min"] if tp else None, "throughput_max": tp["max"] if tp else None,
               "verification": recs[0]["verification"]["status"], "metric": t0["metric"]}
        for c in STAGE_COLUMNS:
            row[c] = (across([extra(r, c) for r in recs]) or {}).get("median")
        zero = across([extra(r, "zero_byte_completion_median_us") for r in recs])
        row["zero_byte_row_median_us"] = zero["median"] if zero else None  # the tool's implicit 0-byte row: no network round trip, not an ACK floor
        src_os, dst_os = (direction.split(" -> ") + ["?"])[:2] if " -> " in direction else (None, None)
        src_b, dst_b = (pair.split(" -> ") + ["?"])[:2] if " -> " in pair else (None, None)
        host = host_index.get((tier, direction, orient, scope, size))
        if op == "end_to_end" and pair != "host -> host" and host:
            row["host_control_median_us"] = host["median"]
            row["incremental_vs_host_us"] = med["median"] - host["median"]
            row["ratio_vs_host"] = med["median"] / host["median"] if host["median"] else None
            if scope == "ready" and src_os and local:
                d2h = 0.0 if src_b == "host" else local.get((src_os, src_b, "staging_d2h", size))
                h2d = 0.0 if dst_b == "host" else local.get((dst_os, dst_b, "staging_h2d", size))
                if d2h is not None and h2d is not None:
                    row["local_staging_sum_us"] = d2h + h2d  # source d2h + destination h2d from the separate local-only runs; shared memory counts as 0
                    row["incremental_minus_local_staging_us"] = row["incremental_vs_host_us"] - (d2h + h2d)
        if b3b_reference and op == "end_to_end" and scope == "ready" and "rank0=Linux" in orient:
            ref = b3b_reference.get((direction, size))
            if ref:
                row["b3b_host_reference_median_us"] = ref
        rows.append(row)
    return rows


def b3b_end_to_end_reference(tables_dir):
    path = Path(tables_dir) / "transfer.csv" if tables_dir else None
    if not path or not path.exists():
        return {}
    ref = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["operation"] == "end_to_end" and row["tier"] == "t2-transfer" and row["timing_scope"] == "ready" and row["orientation"].startswith("rank0=Linux"):
                ref[(row["direction"], int(row["payload_bytes"]))] = float(row["median_us"])
    return ref


def async_rows(records):
    rows = []
    keyfn = lambda r: (r["_tier"], pair_label(r), b3b.direction_label(r), b3b.orientation_label(r), opt(r, "chunk_bytes", 0), opt(r, "depth", 2), opt(r, "metal_async_path", "adapter"), r["configuration"]["payload_bytes"])  # noqa: E731
    for key, recs in sorted(group(records, keyfn).items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tier, pair, direction, orient, chunk, depth, path, size = key
        med = across([stat(r, "median") for r in recs])
        enq = across([extra(r, "enqueue_median_us") for r in recs])
        tp = across([tp_value(r) for r in recs])
        samples = [s for r in recs for s in (timing(r).get("samples") or [])]
        rows.append({"tier": tier, "pair": pair, "direction": direction, "orientation": orient, "chunk_bytes": chunk, "pipeline_depth": depth, "metal_async_path": path, "payload_bytes": size,
                     "repetitions": med["n"], "iterations": recs[0]["configuration"]["iterations"], "completion_median_us": med["median"], "completion_min_us": med["min"], "completion_max_us": med["max"],
                     "completion_range_rel": med["range_rel"], "completion_p95_us": (across([stat(r, "p95") for r in recs]) or {}).get("median"), "pooled_samples": len(samples),
                     "pooled_sample_p99_us": sorted(samples)[int(0.99 * (len(samples) - 1))] if len(samples) >= 100 else None,
                     "enqueue_median_us": enq["median"], "enqueue_min_us": enq["min"], "enqueue_max_us": enq["max"], "throughput_GBps": tp["median"] if tp else None,
                     "verification": recs[0]["verification"]["status"]})
    return rows


def sync_vs_async(transfer, asyn):
    """Synchronous tool (end_to_end) against the asynchronous tool at the same payload, direction, orientation and memory pair; chunk 0 only (the sync tool stages the whole tensor).
    cuda-chunked has no synchronous counterpart (device memory, per-chunk staging); it is compared with cuda-pinned and labelled."""
    index = {(r["tier"].split(":")[0], r["pair"], r["direction"], r["orientation"], r["payload_bytes"]): r for r in transfer
             if r["operation"] == "end_to_end" and r["timing_scope"] == "ready" and r["tier"].endswith("t1-transfer")}
    out = []
    for a in (x for x in asyn if x["tier"].endswith("t2-async")):
        if a["chunk_bytes"] not in (0, "0", None) and "cuda-chunked" not in a["pair"]:
            continue
        for sync_pair, note in ((a["pair"], "same memory pair"), (a["pair"].replace("cuda-chunked", "cuda-pinned"), "cuda-chunked compared with cuda-pinned: not the same memory")):
            if note.startswith("cuda-chunked") and "cuda-chunked" not in a["pair"]:
                continue
            s = index.get((a["tier"].split(":")[0], sync_pair, a["direction"], a["orientation"], a["payload_bytes"]))
            if s:
                out.append({"pair_async": a["pair"], "pair_sync": sync_pair, "match": note, "direction": a["direction"], "orientation": a["orientation"], "chunk_bytes": a["chunk_bytes"], "metal_async_path": a["metal_async_path"],
                            "payload_bytes": a["payload_bytes"], "sync_completion_median_us": s["median_us"], "async_completion_median_us": a["completion_median_us"],
                            "async_over_sync": a["completion_median_us"] / s["median_us"] if s["median_us"] else None, "async_enqueue_median_us": a["enqueue_median_us"],
                            "sync_range_rel": s["median_range_rel"], "async_range_rel": a["completion_range_rel"]})
                break
    return out


def cpu_overlap_rows(records):
    rows = []
    for key, recs in sorted(group(records, lambda r: (r["_tier"], b3b.direction_label(r), r["configuration"]["payload_bytes"], opt(r, "bucket_count", 4), opt(r, "compute_rounds"))).items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tier, direction, size, buckets, rounds = key
        phases = {p: across([extra(r, p) for r in recs]) for p in ("compute_only_us", "comm_only_us", "serial_us")}
        phases["async_overlap_us"] = across([stat(r, "mean") for r in recs])
        hidden = across([extra(r, "hidden_fraction") for r in recs])
        per_run = [extra(r, "hidden_fraction") for r in recs]
        invalid = sum(1 for v in per_run if v is None or v < 0 or v > 1)
        rows.append({"tier": tier, "compute_on": direction.split(" -> ")[0], "bucket_bytes": size, "bucket_count": buckets, "compute_rounds": rounds, "repetitions": len(recs),
                     **{f"{p}_median": phases[p]["median"] for p in phases}, **{f"{p}_range_rel": phases[p]["range_rel"] for p in phases},
                     "hidden_fraction_median": hidden["median"], "hidden_fraction_min": hidden["min"], "hidden_fraction_max": hidden["max"], "runs_outside_0_1": invalid,
                     "note": "CPU hash-loop compute; single-shot phases per run; a fraction outside [0, 1] is noise, not overlap"})
    return rows


def cuda_overlap_rows(records):
    """pipeline_bound_us is DERIVED from the measured compute-only and comm-only medians, assuming equal buckets: the first bucket's compute (or the last bucket's transfer) cannot overlap with anything,
    so the best a bucket pipeline can do is max(C, M) + min(C, M) / buckets. overlap_over_pipeline_bound near 1 means the overlap is as good as the schedule allows; it is not a native output.
    Combine the four schedules (same bucket size, count, compute setting and receiver) into serial/overlap figures computed from measured total times only."""
    rows = []
    byconf = defaultdict(dict)
    for r in ok(records):
        key = (r["_tier"], device_pair(r), b3b.orientation_label(r), r["configuration"]["payload_bytes"], opt(r, "bucket_count", 4), opt(r, "compute_rounds"), opt(r, "chunk_bytes"), opt(r, "depth"))
        byconf[key].setdefault(r["benchmark"]["operation"], []).append(r)
    for key, ops in sorted(byconf.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        if not {"serial", "overlap", "compute_only", "comm_only"} <= set(ops):
            continue
        v = {op: across([stat(r, "median") for r in recs]) for op, recs in ops.items()}
        c, m, s, o = v["compute_only"]["median"], v["comm_only"]["median"], v["serial"]["median"], v["overlap"]["median"]
        saved = s - o
        spread = max(x["range_rel"] or 0 for x in v.values())
        frac = saved / min(c, m) if min(c, m) > 0 else None
        notes = []
        if frac is None or frac < 0 or frac > 1:
            notes.append(f"overlap fraction {frac:.2f} is outside [0, 1]" if frac is not None else "undefined")
        if o < max(c, m) * (1 - spread):
            notes.append("overlapped total is below the larger of compute-only and comm-only by more than the repetition spread")
        if abs(s - (c + m)) / (c + m) > max(0.1, spread):
            notes.append(f"serial total differs from compute-only + comm-only by {100 * (s - (c + m)) / (c + m):+.0f}%")
        rows.append({"tier": key[0], "memory_pair": key[1], "orientation": key[2], "bucket_bytes": key[3], "bucket_count": key[4], "compute_rounds": key[5], "chunk_bytes": key[6], "depth": key[7],
                     "repetitions": min(x["n"] for x in v.values()), "compute_only_us": c, "comm_only_us": m, "serial_us": s, "overlap_us": o,
                     "compute_only_range_rel": v["compute_only"]["range_rel"], "comm_only_range_rel": v["comm_only"]["range_rel"], "serial_range_rel": v["serial"]["range_rel"], "overlap_range_rel": v["overlap"]["range_rel"],
                     "time_saved_us": saved, "serial_over_overlap": s / o if o else None,
                     "pipeline_bound_us": max(c, m) + min(c, m) / key[4], "overlap_over_pipeline_bound": o / (max(c, m) + min(c, m) / key[4]), "overlap_fraction_of_smaller_phase": frac,
                     "enqueue_median_us_overlap": (across([extra(r, "enqueue_median_us") for r in ops["overlap"]]) or {}).get("median"),
                     "valid": not notes, "invalid_reason": "; ".join(notes)})
    return rows


def hetero_rows(records):
    rows = []
    keyfn = lambda r: (r["_tier"], device_pair(r), b3b.orientation_label(r), r["configuration"]["payload_bytes"], r["benchmark"]["algorithm_requested"])  # noqa: E731
    for key, recs in sorted(group(records, keyfn).items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tier, pair, orient, size, algo = key
        med = across([stat(r, "median") for r in recs])
        per = defaultdict(list)
        for r in recs:
            for p in timing(r).get("per_rank") or []:
                per[p["rank"]].append(p["statistics"]["median"])
        rows.append({"tier": tier, "memory_pair": pair, "orientation": orient, "algorithm": algo, "payload_bytes": size, "repetitions": med["n"], "iterations": recs[0]["configuration"]["iterations"],
                     "slowest_rank_median_us": med["median"], "median_min_us": med["min"], "median_max_us": med["max"], "median_range_rel": med["range_rel"],
                     "rank0_median_us": pystats.median(per[0]) if per[0] else None, "rank1_median_us": pystats.median(per[1]) if per[1] else None,
                     "p95_us": (across([stat(r, "p95") for r in recs]) or {}).get("median"), "verification": recs[0]["verification"]["status"]})
    by = {(r["tier"], r["memory_pair"], r["orientation"], r["payload_bytes"], r["algorithm"]): r for r in rows}
    for r in rows:
        if r["algorithm"] == "async":
            s = by.get((r["tier"], r["memory_pair"], r["orientation"], r["payload_bytes"], "sync"))
            r["async_over_sync"] = r["slowest_rank_median_us"] / s["slowest_rank_median_us"] if s else None
    return rows


def sample_mode_rows(records, split_us=180.0):
    """Per configuration, from the tool's raw per-iteration samples (--diagnostics): the repetition medians, the pooled percentiles and how many repetitions sit above split_us.
    A configuration whose repetition medians fall in two clusters is bimodal ACROSS processes: the median of such a configuration describes neither mode."""
    rows = []
    for key, recs in sorted(group([r for r in records if timing(r).get("samples")], lambda r: (r["_tier"], pair_label(r), b3b.direction_label(r), b3b.orientation_label(r), r["configuration"]["payload_bytes"])).items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        tier, pair, direction, orient, size = key
        meds = [stat(r, "median") for r in recs]
        pooled = sorted(x for r in recs for x in timing(r)["samples"])
        q = lambda f: pooled[int(f * (len(pooled) - 1))]  # noqa: E731
        rows.append({"tier": tier, "pair": pair, "direction": direction, "orientation": orient, "payload_bytes": size, "repetitions": len(recs), "samples_pooled": len(pooled),
                     "repetition_medians_us": ";".join(f"{m:.1f}" for m in meds), "repetitions_above_split": sum(1 for m in meds if m > split_us), "split_us": split_us,
                     "p05_us": q(0.05), "p25_us": q(0.25), "p50_us": q(0.5), "p75_us": q(0.75), "p95_us": q(0.95), "p99_us": q(0.99), "fraction_samples_above_split": sum(1 for x in pooled if x > split_us) / len(pooled)})
    return rows


def run_to_run(rows, tier_a, tier_b, keys, value, spread):
    """The same configuration measured in the main run (3 repetitions) and the later confirmation run (5): ratio of the medians. A large ratio is drift between runs, not a property of the configuration."""
    index = {tuple(r.get(k) for k in keys): r for r in rows if r["tier"] == tier_a}
    out = []
    for r in rows:
        a = index.get(tuple(r.get(k) for k in keys)) if r["tier"] == tier_b else None
        if a and a[value]:
            out.append({**{k: r.get(k) for k in keys}, "main_median": a[value], "confirm_median": r[value], "ratio_confirm_over_main": r[value] / a[value], "main_range_rel": a[spread], "confirm_range_rel": r[spread],
                        "main_repetitions": a["repetitions"], "confirm_repetitions": r["repetitions"]})
    return out


def link_rows(runs):
    rows = []
    for tier, d, m, records in runs:
        agg = defaultdict(lambda: defaultdict(float))
        for r in records:
            for host, delta in (r["artifacts"].get("link_delta") or {}).items():
                for k in ("rx_bytes", "tx_bytes", "rx_errors", "tx_errors", "rx_dropped", "tx_dropped", "drops", "tcp_retransmitted_segments"):
                    if isinstance(delta.get(k), (int, float)):
                        agg[host][k] += delta[k]
        for host, vals in sorted(agg.items()):
            rows.append({"run": d.name, "tier": tier, "host": host, "records": len(records), **{k: int(v) for k, v in sorted(vals.items())}})
    return rows


# ---------------------------------------------------------------------------- figures
def _have_mpl():
    import importlib.util
    return importlib.util.find_spec("matplotlib") is not None


def _prov(records):
    return b3b.provenance_line(records)


def _foot(op, direction, backend, metric, timing_text, units, reps, records, transport="TCP over Thunderbolt 4 IP (not RDMA)"):
    return (f"operation: {op} | data direction / orientation: {direction} | memory backend: {backend} | transport: {transport}\n"
            f"metric: {metric} | timing: {timing_text[:150]} | units: {units} | repetitions: {reps}\nbuild provenance: {_prov(records)[:230]}")


def _small_legend(ax):
    ax.legend(frameon=False, fontsize=6.5, labelcolor=MUTED, ncol=2, loc="upper left")


def figures(out, staging, transfer, asyn, sva, cuda_ov, cpu_ov, hetero, recs):
    plt = b3b._plt()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    made = []
    e2e = [r for r in transfer if r["operation"] == "end_to_end" and r["timing_scope"] == "ready" and r["tier"].endswith("t1-transfer")]
    fwd = [r for r in e2e if r["orientation"].startswith("rank0=Linux")]
    tr_recs = [x for x in recs if x["benchmark"]["name"] == "tensor_transfer" and x["benchmark"]["operation"] == "end_to_end"]
    tm = tr_recs[0]["configuration"]["timing_method"] if tr_recs else ""
    # 1/2. host vs device end-to-end completion time and throughput, per physical direction
    for direction in sorted({r["direction"] for r in fwd}):
        sub = [r for r in fwd if r["direction"] == direction and r["pair"] in ("host -> host", "cuda-pinned -> metal-shared", "cuda-pageable -> metal-shared", "cuda-pinned -> metal-private-staged", "cuda-pinned -> host", "host -> metal-shared",
                                                                               "metal-shared -> cuda-pinned", "metal-shared -> cuda-pageable", "metal-private-staged -> cuda-pinned", "host -> cuda-pinned", "metal-shared -> host")]
        if not sub:
            continue
        series, series_tp = defaultdict(list), defaultdict(list)
        for r in sub:
            series[r["pair"]].append((r["payload_bytes"], r["median_us"], r["median_min_us"], r["median_max_us"]))
            if r["throughput_GBps"]:
                series_tp[r["pair"]].append((r["payload_bytes"], r["throughput_GBps"], r["throughput_min"], r["throughput_max"]))
        fig, ax = b3b._setup(plt, f"tensor_transfer end_to_end, {direction}: device vs host completion-confirmed time", "payload (bytes)", "median of repetition medians (us); band = min..max")
        b3b._lines(ax, series)
        _small_legend(ax)
        made.append(b3b._save(fig, plt, out / f"host-vs-device-e2e-{direction.replace(' ', '').replace('>', 'to').replace('-', '')}.png", _foot("end_to_end", direction + " (rank 0 on Linux)", "legend: source -> destination", "completion_confirmed_time", tm, "us", "3", tr_recs)))
        fig, ax = b3b._setup(plt, f"tensor_transfer end_to_end, {direction}: effective payload throughput (decimal GB/s, includes the ACK return trip)", "payload (bytes)", "GB/s (median; band min..max)")
        b3b._lines(ax, series_tp)
        _small_legend(ax)
        made.append(b3b._save(fig, plt, out / f"device-throughput-{direction.replace(' ', '').replace('>', 'to').replace('-', '')}.png", _foot("end_to_end", direction + " (rank 0 on Linux)", "legend: source -> destination", "payload / median completion-confirmed time", tm, "GB/s (decimal)", "3", tr_recs)))
    # 3. staging overhead vs payload: local staging medians (separate hosts) and the observed incremental device-vs-host cost
    series = defaultdict(list)
    for r in staging:
        if r["median_us"] and r["median_us"] >= NO_COPY_BELOW_US:
            series[f"{r['host_os']} {r['memory_backend']} {r['operation'].replace('staging_', '')}"].append((r["payload_bytes"], r["median_us"], r["median_min_us"], r["median_max_us"]))
    if series:
        st_recs = [x for x in recs if x["benchmark"]["name"] == "tensor_staging"]
        fig, ax = b3b._setup(plt, "local staging only (single process, no network): time per staging call", "payload (bytes)", "median of repetition medians (us); band = min..max")
        b3b._lines(ax, series)
        _small_legend(ax)
        made.append(b3b._save(fig, plt, out / "staging-local.png", _foot("staging d2h / h2d", "none (local)", "legend: host OS, backend, direction", "device_to_host / host_to_device staging time",
                                                                      st_recs[0]["configuration"]["timing_method"] if st_recs else "", "us (no-op staging below 0.1 us omitted)", "3", st_recs, transport="none (local staging in one process, no network)")))
    inc = defaultdict(list)
    for r in fwd:
        if "incremental_vs_host_us" in r and r["incremental_vs_host_us"] is not None:
            inc[f"{r['pair']}"].append((r["payload_bytes"], max(r["incremental_vs_host_us"], 0.01)))
    if inc:
        fig, ax = b3b._setup(plt, "observed incremental end_to_end cost over the host-memory control (same run, same direction): not an isolated staging cost", "payload (bytes)", "device median - host median (us; negative values omitted)")
        b3b._lines(ax, inc)
        _small_legend(ax)
        made.append(b3b._save(fig, plt, out / "incremental-device-cost.png", _foot("end_to_end", "all four direction/pair combinations at rank0=Linux", "legend", "device median - host median", tm, "us", "3", tr_recs)))
    # 4. CUDA -> Metal versus Metal -> CUDA
    by = defaultdict(dict)
    for r in fwd:
        a, b = (r["pair"].split(" -> ") + ["?"])[:2]
        if a == "?" or b == "?":
            continue
        by[(frozenset((a, b)), r["payload_bytes"])][r["direction"]] = (r["median_us"], a, b)
    pts = defaultdict(list)
    for (names, size), d in sorted(by.items(), key=lambda kv: (sorted(kv[0][0]), kv[0][1])):
        if len(d) == 2 and "Linux -> macOS" in d and "macOS -> Linux" in d:
            pts[" / ".join(sorted(names))].append((size, d["Linux -> macOS"][0] / d["macOS -> Linux"][0]))
    if pts:
        fig, ax = b3b._setup(plt, "end_to_end: CUDA host -> Metal host time / Metal host -> CUDA host time (above 1 = CUDA -> Metal slower)", "payload (bytes)", "ratio of median times", logy=False)
        b3b._lines(ax, pts)
        _small_legend(ax)
        ax.axhline(1.0, color=MUTED, linewidth=0.8)
        made.append(b3b._save(fig, plt, out / "direction-ratio.png", _foot("end_to_end", "Linux -> macOS vs macOS -> Linux, rank0=Linux", "legend: the two memory kinds involved", "ratio of medians", tm, "ratio", "3", tr_recs)))
    # 5. sync vs async: completion time and enqueue time at identical payload / direction / memory
    for direction in sorted({r["direction"] for r in sva}):
        series = defaultdict(list)
        for r in sva:
            if r["direction"] == direction and "rank0=Linux" in r["orientation"]:
                series[f"sync end_to_end {r['pair_sync']}"].append((r["payload_bytes"], r["sync_completion_median_us"]))
                series[f"async completion {r['pair_async']} (chunk {r['chunk_bytes']}, {r['metal_async_path']})"].append((r["payload_bytes"], r["async_completion_median_us"]))
                series[f"async enqueue only {r['pair_async']} (chunk {r['chunk_bytes']}, {r['metal_async_path']})"].append((r["payload_bytes"], max(r["async_enqueue_median_us"], 0.01)))
        if not series:
            continue
        as_recs = [x for x in recs if x["benchmark"]["name"] == "async_transfer"]
        fig, ax = b3b._setup(plt, f"sync vs async, {direction}: completion (ACK-confirmed) and submission time", "payload (bytes)", "median (us)")
        b3b._lines(ax, series)
        _small_legend(ax)
        made.append(b3b._save(fig, plt, out / f"sync-vs-async-{direction.replace(' ', '').replace('>', 'to').replace('-', '')}.png",
                              _foot("end_to_end vs async transfer", direction, "legend", "completion: ACK-confirmed; enqueue: enqueue() return only (not a transfer time)", as_recs[0]["configuration"]["timing_method"] if as_recs else "", "us", "3", as_recs or recs)))
    # 6. serial vs overlapped
    if cuda_ov:
        cu_recs = [x for x in recs if x["benchmark"]["name"] == "cuda_overlap"]
        cuda_ov = sorted(cuda_ov, key=lambda r: (r["memory_pair"], int(r["compute_rounds"]), r["tier"]))
        labels = [f"{r['memory_pair'].split(' / ')[1]}\nrounds {r['compute_rounds']}\n{r['tier'].split(':')[1].split('-')[0]} n={r['repetitions']}" for r in cuda_ov]
        fig, ax = plt.subplots(figsize=(9.2, 5.6), facecolor="#fcfcfb")
        width = 0.2
        for i, (name, col) in enumerate((("compute only", "compute_only_us"), ("comm only", "comm_only_us"), ("serial", "serial_us"), ("overlapped", "overlap_us"))):
            ax.bar([x + (i - 1.5) * width for x in range(len(cuda_ov))], [r[col] for r in cuda_ov], width, label=name, color=CATEGORICAL[i], edgecolor="#fcfcfb")
        ax.set_xticks(range(len(cuda_ov)))
        ax.set_xticklabels(labels, fontsize=7.5, color=MUTED)
        ax.set_ylabel("median total time per iteration (us)", color=MUTED)
        ax.set_title("CUDA kernel compute + TBCCL communication: serialized vs overlapped (Linux CUDA sender)", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        made.append(b3b._save(fig, plt, out / "overlap-cuda-serial-vs-overlapped.png", _foot("cuda_overlap (4 schedules)", "Linux CUDA sender -> receiver", "cuda-chunked -> legend receiver", "overlap_schedule_total_time",
                                                                                           cu_recs[0]["configuration"]["timing_method"] if cu_recs else "", "us", "3", cu_recs)))
    if cpu_ov:
        cp_recs = [x for x in recs if x["benchmark"]["name"] == "async_overlap"]
        fig, ax = plt.subplots(figsize=(8.4, 5.2), facecolor="#fcfcfb")
        for i, (name, col) in enumerate((("compute only", "compute_only_us_median"), ("comm only", "comm_only_us_median"), ("serial", "serial_us_median"), ("overlapped", "async_overlap_us_median"))):
            ax.bar([x + (i - 1.5) * 0.2 for x in range(len(cpu_ov))], [r[col] for r in cpu_ov], 0.2, label=name, color=CATEGORICAL[i], edgecolor="#fcfcfb")
        ax.set_xticks(range(len(cpu_ov)))
        ax.set_xticklabels([f"{r['compute_on']}\n{r['bucket_bytes'] // 1024} KiB x{r['bucket_count']}" for r in cpu_ov], fontsize=7.5, color=MUTED)
        ax.set_ylabel("median of single-shot phases (us)", color=MUTED)
        ax.set_title("CPU compute (hash loop) + communication, host buffers: this is CPU-compute overlap, not GPU compute", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
        made.append(b3b._save(fig, plt, out / "overlap-cpu.png", _foot("async_overlap", "rank0=Linux sender", "host -> host", "four single-shot phase times", cp_recs[0]["configuration"]["timing_method"] if cp_recs else "", "us", "3", cp_recs)))
    # 7. hetero all-reduce: sync vs async
    for orient in sorted({r["orientation"] for r in hetero}):
        series = defaultdict(list)
        for r in hetero:
            if r["orientation"] == orient:
                series[f"{r['algorithm']} {r['memory_pair']}"].append((r["payload_bytes"], r["slowest_rank_median_us"], r["median_min_us"], r["median_max_us"]))
        if not series:
            continue
        h_recs = [x for x in recs if x["benchmark"]["name"] == "hetero_allreduce"]
        fig, ax = b3b._setup(plt, f"hetero_allreduce ({orient}): sync reduce+broadcast vs async substrate, float32 SUM, 2 ranks", "payload (bytes)", "slower rank's median (us); band = min..max")
        b3b._lines(ax, series)
        _small_legend(ax)
        made.append(b3b._save(fig, plt, out / f"hetero-allreduce-{'linux0' if 'rank0=Linux' in orient else 'mac0'}.png",
                              _foot("hetero all_reduce", orient, "legend: Linux-side / macOS-side backend", "hetero_all_reduce_time_slowest_rank", h_recs[0]["configuration"]["timing_method"] if h_recs else "", "us", "3", h_recs)))
    # 8. async: enqueue vs completion share
    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("campaigns", nargs="+")
    parser.add_argument("--out")
    parser.add_argument("--b3b-tables", help="the B3B campaign tables directory, for the labelled cross-campaign host reference")
    parser.add_argument("--include-prelim", action="store_true")
    parser.add_argument("--exclude", action="append", default=[], help="leave out runs whose directory name contains this text (repeatable); they are listed as excluded")
    args = parser.parse_args(argv)
    everything = load(args.results_dir, args.campaigns)
    runs = [r for r in everything if (args.include_prelim or "prelim" not in r[0] and "validation" not in r[0]) and r[2]["status"] == "success" and not any(x in r[1].name for x in args.exclude)]
    excluded = [{"tier": t, "run": d.name, "status": m["status"], "reason": m.get("reason"), "counts": m["counts"]} for t, d, m, _ in everything if (t, d) not in {(x[0], x[1]) for x in runs}]
    out = Path(args.out or Path(args.results_dir) / f"campaign-{'-'.join(args.campaigns)}")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    allrecs = []
    for tier, _, _, recs in runs:
        for r in recs:
            r["_tier"] = tier
            allrecs.append(r)
    by_bench = lambda name: [r for r in allrecs if r["benchmark"]["name"] == name]  # noqa: E731
    stg = staging_rows(by_bench("tensor_staging"))
    tr = transfer_rows(by_bench("tensor_transfer"), b3b_end_to_end_reference(args.b3b_tables), stg)
    asy = async_rows(by_bench("async_transfer"))
    sva = sync_vs_async(tr, asy)
    cov, cpu_ov, het = cuda_overlap_rows(by_bench("cuda_overlap")), cpu_overlap_rows(by_bench("async_overlap")), hetero_rows(by_bench("hetero_allreduce"))
    r2r = {"run_to_run_transfer": run_to_run(tr, "b3c1:t1-transfer", "b3c1:t5-confirm", ("operation", "pair", "direction", "orientation", "timing_scope", "payload_bytes"), "median_us", "median_range_rel"),
           "run_to_run_async": run_to_run(asy, "b3c1:t2-async", "b3c1:t5-confirm", ("pair", "direction", "orientation", "chunk_bytes", "metal_async_path", "payload_bytes"), "completion_median_us", "completion_range_rel"),
           "run_to_run_hetero": run_to_run(het, "b3c1:t4-hetero", "b3c1:t5-confirm", ("memory_pair", "orientation", "algorithm", "payload_bytes"), "slowest_rank_median_us", "median_range_rel")}
    for name, rows in (*r2r.items(), ("staging", stg), ("transfer", tr), ("async", asy), ("sync_vs_async", sva), ("cuda_overlap", cov), ("cpu_overlap", cpu_ov), ("hetero_allreduce", het), ("cpu", b3b.cpu_rows(runs)),
                       ("link_deltas", link_rows(runs)), ("sample_modes", sample_mode_rows(by_bench("tensor_transfer"))), ("failures", b3b.failures(runs))):
        write_csv(out / "tables" / f"{name}.csv", rows)
    made = figures(out / "figures", stg, tr, asy, sva, cov, cpu_ov, het, allrecs) if _have_mpl() else []
    (out / "excluded_runs.json").write_text(json.dumps(excluded, indent=2))
    print(json.dumps({"campaigns": args.campaigns, "excluded_from_distributions": excluded, "runs": [{"tier": t, "run": d.name, "status": m["status"], "counts": m["counts"]} for t, d, m, _ in runs],
                      "tables": str(out / "tables"), "figures": made}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
