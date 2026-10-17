"""Campaign report data for B5 (vLLM pipeline-parallel inference over vllm-tbccl): tables (CSV) and static figures from the runs (tiers) of a campaign.

    python3 -m benchmarks.suite.reporting.b5 RESULTS_DIR CAMPAIGN [--out DIR] [--exclude TEXT]

Every figure is computed from the raw per-request timelines the client kept (``raw/<point>/rep<k>/vllm/<phase>.requests.jsonl``), pooled over the passes of a cell; the per-pass
medians are kept next to the pooled distribution so the repeat-to-repeat spread is visible. Closed-loop (rounds / continuous) and open-loop phases, single-node references and
the pipelines are never merged into one distribution. A percentile with too few observations is withheld (``metrics.MIN_SAMPLES``) and never plotted. Client-observed latencies
are not native transfer latencies, and historical native (B3B/B3C) or PyTorch (B4) results are not merged into these tables.
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
from ..stream import metrics
from . import b3b
from .b3b import CATEGORICAL, GRID, INK, MUTED

LAT = re.compile(r"^(?P<kind>lat|pre|dec)-r(?P<pass>\d+)-p(?P<p>\d+)-o(?P<o>\d+)$")
CONC = re.compile(r"^conc-(?:r(?P<pass>\d+)-c)?(?P<c>\d+)-(?P<mode>rounds|cont)$")  # the baseline preset names its continuous phases conc-<c>-cont (one pass)
SPREAD_RATIO = 1.10  # an orientation / pairing difference smaller than this factor, or inside the pass-to-pass spread, is not reported as a difference


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


def config_label(r):
    inf = r["configuration"].get("inference") or {}
    stages = ((r["measurements"] or {}).get("extra") or {}).get("inference", {}).get("placement", {}).get("stages") or []
    devices = inf.get("stage_devices") or [s["device"] for s in stages]
    if len(devices) == 1:
        return f"single-{devices[0]}"
    return ">".join({"host": "cpu"}.get(d, d) for d in devices)


def api_host_os(r):
    stages = r["measurements"]["extra"]["inference"]["placement"]["stages"]
    return b3b.os_label(r, stages[0]["host"])


def rep_dir(run_dir, r):
    _, point, rep = r["record_id"].split("/", 2)
    return Path(run_dir) / "raw" / point / rep


def read_requests(run_dir, r, phase):
    files = r["measurements"]["extra"]["inference"]["raw_files"].get(phase, {})
    if "requests" not in files:
        return []
    path = rep_dir(run_dir, r) / files["requests"]
    return _io.read_jsonl(path) if path.exists() else []


def dist_row(records):
    d = metrics.request_distributions(records)
    t = metrics.phase_throughput(records)
    row = {"requests": t["requests"], "ok": t["successful"], "error_rate": t["error_rate"], "timeouts": t["timeouts"], "cancellations": t["cancellations"]}
    for name, key in (("ttft", "ttft_s"), ("tpot", "tpot_s"), ("e2e", "e2e_s"), ("itl", "itl_s")):
        s = d[key]
        row[f"{name}_n"] = s["n"]
        for q in ("p50", "p95", "p99"):
            row[f"{name}_{q}_ms"] = s[q] * 1e3 if s[q] is not None else None
        row[f"{name}_mean_ms"] = s["mean"] * 1e3 if s["mean"] is not None else None
    row["tpot_undefined"] = d["tpot_undefined_requests"]
    row["batched_intervals"] = d["batched_intervals"]
    row["generated_tokens"] = t["output_tokens"]
    if t["rates"]:
        row.update({k: v for k, v in t["rates"].items()})
    row["wall_s"] = t["wall_s"]
    return row


def backend_rows(runs):
    """The stage-boundary ping-pong points: one row per repetition and size (round trips of the tensor dict through vllm-tbccl, measured on rank 0)."""
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if r["status"]["state"] != "success" or not r.get("measurements"):
                continue
            b = r["measurements"]["extra"]["inference"].get("boundary")
            if not b:
                continue
            devices = r["measurements"]["extra"]["inference"]["devices"]
            for tokens, v in sorted(b.items(), key=lambda kv: int(kv[0])):
                out.append({"tier": tier, "record": r["record_id"], "rank0_device": devices[0], "rank1_device": devices[1], "tokens": int(tokens), "bytes_per_direction": v["bytes_per_direction"],
                            "rtt_p50_us": v["rtt_us"]["p50"], "rtt_p95_us": v["rtt_us"]["p95"], "rtt_min_us": v["rtt_us"]["min"], "rtt_n": v["rtt_us"]["n"]})
    return out


def phase_rows(runs):
    """One row per (point, repetition, phase) of every measured phase, derived from the raw timelines (never from the printed summaries)."""
    rows = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if r["status"]["state"] != "success" or not r.get("measurements"):
                continue
            inf = r["measurements"]["extra"]["inference"]
            for name, summ in inf.get("phases", {}).items():
                m = LAT.match(name) or CONC.match(name)
                kind = summ.get("kind")
                if kind not in ("rounds", "continuous", "open_loop", "check") and not m:
                    continue
                recs = read_requests(run_dir, r, name)
                if not recs:
                    continue
                base = {"tier": tier, "run": Path(run_dir).name, "record": r["record_id"], "config": config_label(r), "api_host": api_host_os(r), "phase": name, "phase_kind": kind,
                        "loop": summ.get("loop"), "concurrency": summ.get("concurrency"), "prompt": summ.get("prompt"), "max_tokens": summ.get("max_tokens"), "repetition": r["configuration"].get("repetition_index")}
                if m:
                    gd = m.groupdict()
                    base.update({"cell_kind": gd.get("kind") or "conc", "pass": int(gd.get("pass") or 1), "prompt_tokens": int(gd["p"]) if gd.get("p") else None,
                                 "output_tokens_cfg": int(gd["o"]) if gd.get("o") else None, "conc": int(gd["c"]) if gd.get("c") else None, "mode": gd.get("mode")})
                rows.append({**base, **dist_row(recs)})
    return rows


def cell_rows(runs, rows):
    """Latency cells (pooled over passes and repetitions) with the per-pass medians that show repeat-to-repeat spread."""
    by_cell = defaultdict(list)
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if r["status"]["state"] != "success" or not r.get("measurements"):
                continue
            inf = r["measurements"]["extra"]["inference"]
            for name in inf.get("phases", {}):
                m = LAT.match(name)
                if m:
                    gd = m.groupdict()
                    by_cell[(config_label(r), api_host_os(r), int(gd["p"]), int(gd["o"]))].append((f"{tier}/{name}", read_requests(run_dir, r, name), gd["kind"]))
    out = []
    for (config, api, p, o), items in sorted(by_cell.items()):
        allrecs = [x for _, recs, _ in items for x in recs]
        kinds = "+".join(sorted({k for _, _, k in items}))  # the same prompt/output cell appears in the latency, prefill and decode grids; they are one cell (all concurrency 1)
        row = {"config": config, "api_host": api, "cell_kind": kinds, "prompt_tokens": p, "output_tokens": o, "passes": len(items), **dist_row(allrecs)}
        for metric, key in (("ttft", "ttft_s"), ("tpot", "tpot_s"), ("e2e", "e2e_s")):
            meds = []
            for _, recs, _ in items:
                vals = [metrics.derive(x)[key] for x in recs if x.get("status") == "ok" and metrics.derive(x)[key] is not None]
                if vals:
                    meds.append(pystats.median(vals) * 1e3)
            row[f"{metric}_pass_medians_ms"] = ";".join(f"{x:.2f}" for x in meds)
            row[f"{metric}_pass_spread"] = (max(meds) / min(meds)) if len(meds) > 1 and min(meds) > 0 else None
        out.append(row)
    return out


def concurrency_rows(rows):
    """Throughput and latency against concurrency, pooled over the passes of each (config, mode, concurrency); round-based and continuous loops are separate series."""
    groups = defaultdict(list)
    for r in rows:
        if r.get("cell_kind") == "conc":
            groups[(r["config"], r["mode"], r["conc"])].append(r)
    out = []
    for (config, mode, conc), items in sorted(groups.items()):
        tok = [x["output_tokens_per_s"] for x in items if x.get("output_tokens_per_s") is not None]
        req = [x["requests_per_s"] for x in items if x.get("requests_per_s") is not None]
        row = {"config": config, "mode": mode, "concurrency": conc, "passes": len(items), "output_tokens_per_s_median": pystats.median(tok) if tok else None,
               "output_tokens_per_s_min": min(tok) if tok else None, "output_tokens_per_s_max": max(tok) if tok else None,
               "requests_per_s_median": pystats.median(req) if req else None, "failed": sum(x["requests"] - x["ok"] for x in items)}
        for k in ("ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms", "tpot_p95_ms", "e2e_p50_ms", "e2e_p95_ms", "itl_p50_ms", "itl_p95_ms"):
            vals = [x[k] for x in items if x.get(k) is not None]
            row[k] = pystats.median(vals) if vals else None
        out.append(row)
    return out


def ratio_rows(cells):
    """Orientation / pairing comparisons per cell: ratio of pooled medians, and whether it exceeds both the pass-to-pass spread of either arm and the stated tolerance."""
    idx = {(c["config"], c["prompt_tokens"], c["output_tokens"]): c for c in cells}
    out = []
    pairs = (("cuda>metal", "metal>cuda"), ("cuda>metal", "cuda>cpu"))
    for a, b in pairs:
        for (config, p, o), ca in idx.items():
            if config != a or (b, p, o) not in idx:
                continue
            cb = idx[(b, p, o)]
            kind = ca["cell_kind"]
            for metric in ("ttft", "tpot", "e2e"):
                ma, mb = ca.get(f"{metric}_p50_ms"), cb.get(f"{metric}_p50_ms")
                if not ma or not mb:
                    continue
                spread = max(ca.get(f"{metric}_pass_spread") or 1.0, cb.get(f"{metric}_pass_spread") or 1.0)
                ratio = mb / ma
                out.append({"a": a, "b": b, "cell_kind": kind, "prompt_tokens": p, "output_tokens": o, "metric": metric, "a_p50_ms": ma, "b_p50_ms": mb, "ratio_b_over_a": ratio,
                            "pass_spread_max": spread, "exceeds_variation": bool(abs(ratio - 1.0) > max(spread - 1.0, SPREAD_RATIO - 1.0))})
    return out


def itl_samples(runs, prompt=128, tokens=64):
    """Pooled raw ITL intervals (seconds) of the prompt/output cell per configuration, from the raw timelines."""
    out = defaultdict(list)
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if r["status"]["state"] != "success" or not r.get("measurements"):
                continue
            for name in r["measurements"]["extra"]["inference"].get("phases", {}):
                m = LAT.match(name)
                if m and m["kind"] in ("lat", "dec") and int(m["p"]) == prompt and int(m["o"]) == tokens:
                    for rec in read_requests(run_dir, r, name):
                        if rec.get("status") == "ok":
                            out[config_label(r)].extend(metrics.derive(rec)["itl_s"])
    return out


def step_class(e):
    """decode1: one token for one request; decode: one token per request for several requests; prefill: any step that schedules more tokens than requests."""
    if e.get("tokens") is None or e.get("requests") is None:
        return "all"
    if e["tokens"] == e["requests"]:
        return "decode1" if e["requests"] == 1 else "decode"
    return "prefill"


def trace_rows(runs):
    """Per-process timing events of the diagnostic runs, summarised by event and step class. Durations are measured inside one process on its own monotonic clock; hosts are never
    compared, and 'step_forward' of a downstream Metal stage and 'step' of a downstream stage include the blocking wait for the upstream stage's activation."""
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if r["status"]["state"] != "success" or not r.get("measurements"):
                continue
            tr = r["measurements"]["extra"]["inference"].get("trace")
            if not tr:
                continue
            devices = {s["host"]: s["device"] for s in r["measurements"]["extra"]["inference"]["placement"]["stages"]}
            stage_of = {s["host"]: s["stage"] for s in r["measurements"]["extra"]["inference"]["placement"]["stages"]}
            for host, files in tr["files"].items():
                if not isinstance(files, list):
                    continue
                groups = defaultdict(list)
                for rel in files:
                    path = rep_dir(run_dir, r) / rel
                    if not path.exists():
                        continue
                    if rel.endswith(".jsonl"):
                        for e in _io.read_jsonl(path):
                            groups[("hook", e["kind"], step_class(e) if e["kind"] == "step" else "all")].append(((e["done_ns"] - e["entry_ns"]) / 1e6, e.get("bytes")))
                    else:
                        for e in json.loads(path.read_text()):
                            groups[("tbccl", e["op"], "all")].append(((e["done_ns"] - e["entry_ns"]) / 1e6, e.get("bytes")))
                for (source, event, cls), vals in sorted(groups.items()):
                    d = metrics.summarize([v[0] for v in vals])
                    b = [v[1] for v in vals if v[1]]
                    out.append({"tier": tier, "record": r["record_id"], "config": config_label(r), "host": host, "stage": stage_of.get(host), "device": devices.get(host), "source": source, "event": event,
                                "step_class": cls, "n": len(vals), "bytes_p50": pystats.median(b) if b else None, "dur_p50_ms": d["p50"], "dur_p90_ms": d["p90"], "dur_p95_ms": d["p95"],
                                "dur_mean_ms": d["mean"], "dur_total_s": sum(v[0] for v in vals) / 1e3})
    return out


def decomposition_rows(runs, traces):
    """One row per traced point: the client's TPOT of the single-request decode cell next to the median of each stage's single-token decode step, side by side and never summed (stage
    timelines overlap, and a downstream stage's step includes its wait for the upstream activation)."""
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if r["status"]["state"] != "success" or not r.get("measurements") or not r["measurements"]["extra"]["inference"].get("trace"):
                continue
            inf = r["measurements"]["extra"]["inference"]
            cell = [read_requests(run_dir, r, name) for name in inf["phases"] if LAT.match(name) and LAT.match(name)["p"] == "128" and LAT.match(name)["o"] == "32"]
            recs = [x for c in cell for x in c if x.get("status") == "ok"]
            tpot = [metrics.derive(x)["tpot_s"] for x in recs if metrics.derive(x)["tpot_s"] is not None]
            row = {"tier": tier, "record": r["record_id"], "config": config_label(r), "api_host": api_host_os(r), "client_tpot_p50_ms": pystats.median(tpot) * 1e3 if tpot else None, "client_tpot_n": len(tpot)}
            for t in traces:
                if t["record"] != r["record_id"] or t["event"] == "step_execute":  # vllm-tbccl's step_execute counts empty scheduler polls too; the hook's classified steps are used instead
                    continue
                key = f"stage{t['stage']}_{t['device']}_{t['event']}" + (f"_{t['step_class']}" if t["step_class"] != "all" else "")
                row[key + "_p50_ms"] = t["dur_p50_ms"]
            out.append(row)
    return out


SERVER_FAMILIES = {"queue": "vllm:request_queue_time_seconds", "prefill": "vllm:request_prefill_time_seconds", "decode": "vllm:request_decode_time_seconds", "inference": "vllm:request_inference_time_seconds",
                   "e2e": "vllm:e2e_request_latency_seconds", "ttft": "vllm:time_to_first_token_seconds", "itl": "vllm:inter_token_latency_seconds"}


def _family(values, base, suffix):
    return sum(v for k, v in values.items() if k.split("{")[0] == base + suffix)


def server_rows(runs):
    """What the server itself reports during each measured phase (the API host's /metrics, scraped about twice a second): queue depth, and the mean of each server-side latency
    family over the phase (delta of the histogram sum / delta of its count between the first and last scrape). Client-observed and server-observed figures are different measurements."""
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if r["status"]["state"] != "success" or not r.get("measurements"):
                continue
            inf = r["measurements"]["extra"]["inference"]
            for name, files in inf.get("raw_files", {}).items():
                if "server_metrics" not in files:
                    continue
                path = rep_dir(run_dir, r) / files["server_metrics"]
                samples = _io.read_jsonl(path) if path.exists() else []
                if len(samples) < 2:
                    continue
                first, last = samples[0]["values"], samples[-1]["values"]
                running = [sum(v for k, v in x["values"].items() if k.startswith("vllm:num_requests_running")) for x in samples]
                waiting = [sum(v for k, v in x["values"].items() if k.startswith("vllm:num_requests_waiting{")) for x in samples]
                row = {"tier": tier, "record": r["record_id"], "config": config_label(r), "phase": name, "scrapes": len(samples), "running_max": max(running), "running_mean": sum(running) / len(running),
                       "waiting_max": max(waiting), "waiting_mean": sum(waiting) / len(waiting), "kv_cache_usage_max": max((sum(v for k, v in x["values"].items() if k.startswith("vllm:kv_cache_usage_perc")) for x in samples), default=None)}
                for label, base in SERVER_FAMILIES.items():
                    dn = _family(last, base, "_count") - _family(first, base, "_count")
                    ds = _family(last, base, "_sum") - _family(first, base, "_sum")
                    row[f"server_{label}_mean_ms"] = (ds / dn * 1e3) if dn > 0 else None
                    row[f"server_{label}_n"] = dn
                out.append(row)
    return out


def confirm_rows(runs):
    """One row per engine lifecycle of the confirmation tier: the medians of the key cells of that lifecycle (TTFT of p16/o1 and p128/o32, TPOT of p128/o32, output tokens/s at concurrency 1, 4 and 12)."""
    out = []
    for tier, run_dir, manifest, records in runs:
        if not tier.startswith("t9"):
            continue
        for r in records:
            if r["status"]["state"] != "success" or not r.get("measurements"):
                continue
            inf = r["measurements"]["extra"]["inference"]
            row = {"record": r["record_id"].split("/")[1][-24:], "config": config_label(r), "api_host": api_host_os(r), "order_index": len(out)}
            vals = defaultdict(list)
            for name in inf.get("phases", {}):
                m = LAT.match(name)
                if m:
                    for rec in read_requests(run_dir, r, name):
                        if rec.get("status") == "ok":
                            d = metrics.derive(rec)
                            cell = f"p{m['p']}o{m['o']}"
                            vals[("ttft", cell)].append(d["ttft_s"] * 1e3)
                            if d["tpot_s"] is not None:
                                vals[("tpot", cell)].append(d["tpot_s"] * 1e3)
                c = CONC.match(name)
                if c and c["mode"] == "cont":
                    th = metrics.phase_throughput(read_requests(run_dir, r, name))
                    if th["rates"]:
                        vals[("tok_s", f"c{c['c']}")].append(th["rates"]["output_tokens_per_s"])
            for (kind, cell), v in sorted(vals.items()):
                row[f"{kind}_{cell}"] = pystats.median(v)
            out.append(row)
    return out


def confirm_summary(rows):
    """Per configuration and key figure: the median of the per-lifecycle medians, their range and relative spread (max/min)."""
    out = []
    keys = sorted({k for r in rows for k in r if k.startswith(("ttft_", "tpot_", "tok_s_"))})
    for config in sorted({r["config"] for r in rows}):
        for key in keys:
            v = [r[key] for r in rows if r["config"] == config and r.get(key) is not None]
            if v:
                out.append({"config": config, "figure": key, "lifecycles": len(v), "median_of_medians": pystats.median(v), "min": min(v), "max": max(v), "spread_max_over_min": max(v) / min(v) if min(v) > 0 else None})
    return out


def sampler_series(runs, tier_prefix):
    """GPU memory and utilisation over time from the passive sampler of the CUDA host, for the runs of one tier: {config: [(seconds since the sampler started, MiB, percent), ...]}."""
    out = {}
    for tier, run_dir, manifest, records in runs:
        if not tier.startswith(tier_prefix):
            continue
        for r in records:
            if r["status"]["state"] != "success" or not r.get("measurements"):
                continue
            roles = {x["role"]: x for x in r["artifacts"].get("ranks", [])}
            for role, x in roles.items():
                if not role.startswith("sampler@") or not x.get("stdout"):
                    continue
                path = rep_dir(run_dir, r) / Path(x["stdout"]).name
                rows = []
                for line in path.read_text(errors="replace").splitlines() if path.exists() else []:
                    if line.startswith("TBCCL_VLLM_SAMPLE ") and '"gpu"' in line:
                        e = json.loads(line.split(" ", 1)[1])
                        g = e.get("gpu") or {}
                        try:
                            rows.append((e["t_mono_s"], float(g["memory_used_mib"]), float(g["utilization_gpu_percent"])))
                        except (KeyError, TypeError, ValueError):
                            continue
                if rows:
                    out[config_label(r)] = rows
    return out


def resource_rows(runs):
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if r["status"]["state"] != "success" or not r.get("measurements"):
                continue
            ev = r["measurements"]["extra"]["inference"].get("transport_evidence")
            if not ev:  # the boundary ping-pong has no sampler
                continue
            for host, s in ev["samples"].items():
                g = s.get("gpu") or {}
                t = s.get("cpu_thermal") or {}
                out.append({"tier": tier, "record": r["record_id"], "config": config_label(r), "host": host, "device": ev["hosts"][host]["device"], "samples": s["samples"],
                            "cpu_percent_median": (s["cpu_percent"] or {}).get("median"), "cpu_percent_max": (s["cpu_percent"] or {}).get("max"), "mem_used_percent_max": s["mem_used_percent_max"],
                            "process_rss_mib_max": s["process_rss_mib_max"], "gpu_temperature_c_max": g.get("temperature_c_max"), "gpu_utilization_median": (g.get("utilization_percent") or {}).get("median"),
                            "gpu_memory_used_mib_max": g.get("memory_used_mib_max"), "gpu_throttle_reasons": ",".join(g.get("throttle_reasons_seen") or []),
                            "cpu_pkg_temp_max_c": t.get("x86_pkg_temp_max"), "cpu_package_throttle_events": (t.get("package_throttle_count_last") or 0) - (t.get("package_throttle_count_first") or 0) if t else None,
                            "link_tx_bytes": (s.get("link_delta") or {}).get("tx_bytes"), "link_rx_bytes": (s.get("link_delta") or {}).get("rx_bytes"),
                            "link_errors": sum(((s.get("link_delta") or {}).get(k) or 0) for k in ("rx_errors", "tx_errors", "rx_dropped", "tx_dropped")), "engine_lost": s.get("engine_lost")})
    return out


def gate_rows(runs):
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            if r["status"]["state"] != "success" or not r.get("measurements"):
                continue
            for name, b in r["measurements"]["extra"]["inference"].get("phases", {}).items():
                ck = b.get("check") or b.get("cancel")
                if not ck:
                    continue
                oc = ck["outcome"] if b.get("check") else ck["followup"]["outcome"]
                out.append({"tier": tier, "config": config_label(r), "phase": name, "kind": b["kind"], "concurrency": ck.get("concurrency"), "passed": ck["passed"], **oc,
                            "details": json.dumps(ck.get("details") or ck.get("followup", {}).get("details") or [])})
    return out


def lifecycle_rows(runs):
    out = []
    for tier, run_dir, manifest, records in runs:
        for r in records:
            ranks = r["artifacts"].get("ranks", [])
            unconfirmed = [x["role"] for x in ranks if x.get("supervised") and x.get("cleanup_confirmed") is not True]
            stops = {x["role"]: x.get("stop_s") for x in ranks if x.get("stop_s") is not None}
            out.append({"tier": tier, "record": r["record_id"], "state": r["status"]["state"], "reason": (r["status"].get("reason") or "")[:200], "group_wall_s": r["artifacts"].get("wall_s"),
                        "exit_codes": json.dumps(r["status"].get("exit_codes")), "stop_s": json.dumps(stops), "cleanup_unconfirmed": ",".join(unconfirmed),
                        "terminated_by_runner": ",".join(x["role"] for x in r["artifacts"].get("ranks", []) if x.get("terminated_by_runner")),
                        "link_delta": json.dumps(r["artifacts"].get("link_delta") or {})})
    return out


def figures(out, cells, conc, itl, ratios, resources, series=None, decomposition=None):
    plt = b3b._plt()
    if plt is None:
        return []
    out.mkdir(parents=True, exist_ok=True)
    made = []
    colors = {}

    def color(config):
        return colors.setdefault(config, CATEGORICAL[len(colors) % len(CATEGORICAL)])

    def finish(fig, ax, name, title, xlabel, ylabel, legend=True, ticks=None):
        import textwrap
        ax.set_title("\n".join(textwrap.wrap(title, 72)), fontsize=9, color=INK, loc="left")
        if ticks:
            from matplotlib.ticker import FixedLocator, FixedFormatter
            ax.xaxis.set_major_locator(FixedLocator(ticks))
            ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
            ax.minorticks_off()
        ax.set_xlabel(xlabel, color=MUTED)
        ax.set_ylabel(ylabel, color=MUTED)
        ax.grid(True, color=GRID, linewidth=0.6)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8, frameon=False)
        fig.tight_layout()
        path = out / f"{name}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        made.append(str(path))

    # TTFT versus prompt tokens (output of 1 or 8 tokens): the prefill-focused cells, median with the 5th-95th spread of the pooled observations when there are enough
    pre = [c for c in cells if c["output_tokens"] in (1, 8)]
    if pre:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for config in sorted({c["config"] for c in pre}):
            for o, style in ((1, "-"), (8, "--")):
                pts = sorted((c["prompt_tokens"], c["ttft_p50_ms"]) for c in pre if c["config"] == config and c["output_tokens"] == o and c["ttft_p50_ms"] is not None)
                if len(pts) > 1:
                    ax.plot(*zip(*pts), style, marker="o", color=color(config), label=f"{config}, {o} output token(s)")
        ax.set_xscale("log")
        finish(fig, ax, "ttft_vs_input_tokens", "TTFT versus prompt tokens (median, closed loop, concurrency 1)", "prompt tokens", "TTFT (ms)", ticks=sorted({c["prompt_tokens"] for c in pre}))
    dec = [c for c in cells if c["prompt_tokens"] == 128 and c["tpot_p50_ms"] is not None and c["output_tokens"] > 1]
    if dec:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        for config in sorted({c["config"] for c in dec}):
            pts = sorted((c["output_tokens"], c["tpot_p50_ms"]) for c in dec if c["config"] == config)
            if len(pts) > 1:
                ax.plot(*zip(*pts), "-", marker="o", color=color(config), label=config)
        ax.set_xscale("log")
        finish(fig, ax, "tpot_vs_output_tokens", "TPOT versus output tokens (prompt 128, median)", "output tokens", "TPOT (ms)", ticks=sorted({c["output_tokens"] for c in dec}))
    if itl:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        labels, data = [], []
        for config in sorted(itl):
            labels.append(f"{config} (n={len(itl[config])})")
            data.append([v * 1e3 for v in itl[config]])
        ax.boxplot(data, showfliers=False, whis=(5, 95))
        for i, vals in enumerate(data, start=1):
            summary = metrics.summarize(vals)
            note = f"p50 {summary['p50']:.1f}\np95 {summary['p95']:.1f}" + (f"\np99 {summary['p99']:.1f}" if summary["p99"] is not None else "")
            ax.annotate(note, (i, summary["p50"]), xytext=(34, 0), textcoords="offset points", fontsize=7, color=MUTED, va="center")
        ax.set_xticks(range(1, len(labels) + 1), labels, fontsize=7)
        finish(fig, ax, "itl_distributions", "ITL: pooled raw intervals, prompt 128 / 64 output tokens (box: quartiles; whiskers: 5th-95th percentile)", "configuration", "ITL (ms)", legend=False)
    if conc:
        for metric, ylabel, name in (("output_tokens_per_s_median", "output tokens/s", "throughput_vs_concurrency"), ("e2e_p50_ms", "E2E latency, median (ms)", "e2e_vs_concurrency")):
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            for config in sorted({c["config"] for c in conc}):
                for mode, style in (("rounds", "-"), ("cont", "--")):
                    pts = sorted((c["concurrency"], c[metric]) for c in conc if c["config"] == config and c["mode"] == mode and c.get(metric) is not None)
                    if pts:
                        ax.plot(*zip(*pts), style, marker="o", color=color(config), label=f"{config} ({'rounds' if mode == 'rounds' else 'continuous'})")
            finish(fig, ax, name, f"{ylabel.split(',')[0]} versus closed-loop concurrency (p128/o32)", "concurrency", ylabel)
        for lat, name in (("ttft_p50_ms", "ttft_vs_throughput"), ("tpot_p50_ms", "tpot_vs_throughput")):
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            for config in sorted({c["config"] for c in conc}):
                pts = sorted((c["output_tokens_per_s_median"], c[lat], c["concurrency"]) for c in conc if c["config"] == config and c["mode"] == "cont" and c.get(lat) and c["output_tokens_per_s_median"])
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", marker="o", color=color(config), label=config)
                    for x, y, cc in pts:
                        ax.annotate(str(cc), (x, y), fontsize=7, color=MUTED, xytext=(3, 3), textcoords="offset points")
            finish(fig, ax, name, f"{lat.split('_')[0].upper()} versus output throughput (continuous closed loop; labels = concurrency)", "output tokens/s", f"{lat.split('_')[0].upper()} median (ms)")
    if ratios:
        for metric in ("ttft", "tpot"):
            rs = [r for r in ratios if r["metric"] == metric and r["a"] == "cuda>metal" and r["b"] == "metal>cuda" and r["output_tokens"] in (32, 64)]
            if rs:
                fig, ax = plt.subplots(figsize=(6.4, 4.0))
                labels = [f"p{r['prompt_tokens']}/o{r['output_tokens']}" for r in rs]
                ax.bar(range(len(rs)), [r["ratio_b_over_a"] for r in rs], color=[CATEGORICAL[0] if r["exceeds_variation"] else MUTED for r in rs])
                ax.axhline(1.0, color=INK, linewidth=0.8)
                ax.set_xticks(range(len(rs)), labels, rotation=45, fontsize=7)
                finish(fig, ax, f"order_{metric}_ratio", f"Metal-first / CUDA-first {metric.upper()} ratio per cell (blue: beyond the pass-to-pass spread)", "cell", "ratio of medians", legend=False)
        rs = [r for r in ratios if r["a"] == "cuda>metal" and r["b"] == "cuda>cpu" and r["metric"] == "tpot"]
        if rs:
            fig, ax = plt.subplots(figsize=(6.4, 4.0))
            ax.bar(range(len(rs)), [r["ratio_b_over_a"] for r in rs], color=CATEGORICAL[1])
            ax.set_xticks(range(len(rs)), [f"p{r['prompt_tokens']}/o{r['output_tokens']}" for r in rs], rotation=45, fontsize=7)
            ax.axhline(1.0, color=INK, linewidth=0.8)
            finish(fig, ax, "cpu_vs_metal_tpot_ratio", "CUDA+CPU / CUDA+Metal TPOT ratio per cell", "cell", "ratio of medians", legend=False)
    for config, rows_ in sorted((series or {}).items()):
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot([r[0] for r in rows_], [r[1] for r in rows_], color=CATEGORICAL[2], label="GPU memory used (MiB)")
        ax2 = ax.twinx()
        ax2.plot([r[0] for r in rows_], [r[2] for r in rows_], color=CATEGORICAL[1], linewidth=0.7, label="GPU utilisation (%)")
        ax2.set_ylabel("GPU utilisation (%)", color=MUTED)
        finish(fig, ax, f"gpu_memory_series_{config.replace('>', '-')}", f"GPU memory and utilisation over the concurrency sweep ({config}, nvidia-smi once per second)", "seconds since the sampler started", "GPU memory used (MiB)", legend=False)
    if decomposition:
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        labels, bars = [], []
        for d in decomposition:
            comps = [("client TPOT", d.get("client_tpot_p50_ms"))] + [(k.replace("_p50_ms", "").replace("stage", "s"), float(v)) for k, v in d.items()
                                                                     if k.endswith("_p50_ms") and k != "client_tpot_p50_ms" and v not in (None, "") and ("decode1" in k or "forward" in k or "sample" in k or "recv" in k or "send" in k)]
            labels.append(d["config"])
            bars.append(comps)
        row_h, y = 0.8, 0.0
        yticks, ylabels = [], []
        for config, comps in zip(labels, bars):
            for name, value in comps:
                if value is None:
                    continue
                ax.barh(y, value, color=CATEGORICAL[0] if name == "client TPOT" else CATEGORICAL[2], height=0.7)
                ax.text(value + 0.2, y, f"{value:.1f}", va="center", fontsize=6, color=MUTED)
                yticks.append(y)
                ylabels.append(f"{config}: {name}")
                y += 1
            y += 0.8
        ax.set_yticks(yticks, ylabels, fontsize=6)
        ax.invert_yaxis()
        finish(fig, ax, "stage_components", "Median stage timings of single-token decode steps next to the client's TPOT (traced runs; separate clocks, overlapping, NOT additive; a downstream stage's step includes its wait)", "milliseconds", "", legend=False)
    mem = [r for r in resources if r.get("gpu_memory_used_mib_max") is not None]
    if mem:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        xs = [r["record"].split("/")[1][:28] for r in mem]
        ax.bar(range(len(mem)), [r["gpu_memory_used_mib_max"] for r in mem], color=CATEGORICAL[2])
        ax.set_xticks(range(len(mem)), xs, rotation=60, fontsize=6)
        finish(fig, ax, "gpu_memory", "GPU memory used (max over the run, nvidia-smi) per measured point", "point", "MiB", legend=False)
    return made


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir")
    parser.add_argument("campaign")
    parser.add_argument("--out")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--include-tier", action="append", default=[], help="use only runs whose tier starts with this (repeatable); instrumented, remote-client and confirmation runs are reported separately")
    args = parser.parse_args(argv)
    everything = load(args.results_dir, args.campaign)
    runs = [r for r in everything if not any(x in r[1].name for x in args.exclude) and r[2]["status"] in ("success", "partial")
            and (not args.include_tier or any(r[0].startswith(t) for t in args.include_tier))]
    excluded = [{"tier": t, "run": d.name, "status": m["status"], "reason": m.get("reason"), "counts": m["counts"]} for t, d, m, _ in everything if (t, d) not in {(x[0], x[1]) for x in runs}]
    out = Path(args.out or Path(args.results_dir) / f"campaign-{args.campaign}")
    (out / "tables").mkdir(parents=True, exist_ok=True)
    rows = phase_rows(runs)
    cells = cell_rows(runs, rows)
    conc = concurrency_rows(rows)
    ratios = ratio_rows(cells)
    resources = resource_rows(runs)
    traces = trace_rows(runs)
    confirm = confirm_rows(runs)
    for name, data in (("backend_boundary", backend_rows(runs)), ("confirm_lifecycles", confirm), ("confirm_summary", confirm_summary(confirm)), ("server_metrics", server_rows(runs)), ("trace_events", traces), ("stage_decomposition", decomposition_rows(runs, traces)), ("phases", rows), ("latency_cells", cells), ("concurrency", conc), ("order_and_pairing_ratios", ratios), ("resources", resources), ("gates", gate_rows(runs)), ("lifecycle", lifecycle_rows(everything))):
        write_csv(out / "tables" / f"{name}.csv", data)
    made = figures(out / "figures", cells, conc, itl_samples(runs), ratios, resources, sampler_series(runs, "t3-"), decomposition_rows(runs, traces))
    (out / "excluded_runs.json").write_text(json.dumps(excluded, indent=2))
    print(json.dumps({"campaign": args.campaign, "runs": [{"tier": t, "run": d.name, "status": m["status"], "counts": m["counts"]} for t, d, m, _ in runs], "excluded": excluded,
                      "tables": str(out / "tables"), "figures": made}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
