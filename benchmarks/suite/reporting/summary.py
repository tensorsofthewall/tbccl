"""Grouping of records into points and Markdown / CSV summaries. Failed and incomplete points are listed, never averaged into a result."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from .. import statistics as sstats


def host_signature(record):
    hosts = record["environment"]["hosts"]
    return [(h.get("os", "") or "unknown").split(" ")[0] + "/" + (h.get("arch") or "unknown") + "/" + (h.get("cpu_model") or "unknown") + "/" + (h.get("gpu_model") or "none")
            for h in sorted(hosts, key=lambda h: min(h["ranks"]) if h["ranks"] else 0)]


def point_key(record):
    """Everything that makes two records repetitions of the same experiment."""
    b, e, c = record["benchmark"], record["environment"], record["configuration"]
    timing = (record.get("measurements") or {}).get("timing") or {}
    return json.dumps([b["name"], b["operation"], b.get("algorithm_requested"), b.get("algorithm_resolved"), c["payload_bytes"], c.get("datatype"), c.get("reduce_op"),
                       c["rank_count"], e["memory_backend"], e["topology"]["kind"], e["topology"]["interconnect"], e["transport"], c["timing_method"], c["completion_semantics"],
                       json.dumps(c.get("options", {}), sort_keys=True), {k: (c.get("data_direction") or {}).get(k) for k in ("mode", "source_rank")}, host_signature(record), timing.get("metric"), timing.get("primary_statistic")], sort_keys=True)


def primary(record):
    """(value, statistic name, unit) of the record's primary timing, or None."""
    timing = (record.get("measurements") or {}).get("timing")
    if not timing:
        return None
    return timing["statistics"][timing["primary_statistic"]], timing["primary_statistic"], timing["unit"]


def primary_throughput(record):
    t = (record.get("measurements") or {}).get("throughput") or []
    return t[0] if t else None


def group_records(records):
    groups = {}
    for r in records:
        groups.setdefault(point_key(r), []).append(r)
    return groups


def summarize(records):
    rows = []
    for key, recs in group_records(records).items():
        ok = [r for r in recs if r["status"]["state"] == "success"]
        first = recs[0]
        b, e, c = first["benchmark"], first["environment"], first["configuration"]
        states = {}
        for r in recs:
            states[r["status"]["state"]] = states.get(r["status"]["state"], 0) + 1
        row = {"benchmark": b["name"], "operation": b["operation"], "backend": b.get("backend") or "", "orientation": c.get("orientation") or "forward", "algorithm_requested": b.get("algorithm_requested") or "", "algorithm_resolved": b.get("algorithm_resolved") or "",
               "memory_domain": e["memory_domain"]["label"], "memory_backend": "/".join(e["memory_backend"]), "topology": e["topology"]["kind"], "interconnect": e["topology"]["interconnect"],
               "transport": e["transport"], "rank_count": c["rank_count"], "datatype": c.get("datatype") or "", "payload_bytes": c["payload_bytes"],
               "provenance": b["provenance"], "records": len(recs), "successful": len(ok), "states": ",".join(f"{k}={v}" for k, v in sorted(states.items())),
               "metric": "", "statistic": "", "unit": "us", "value_median": None, "value_min": None, "value_max": None, "repeat_cv": None,
               "throughput_kind": "", "throughput_value": None, "throughput_unit": "", "verification": ",".join(sorted({r["verification"]["status"] for r in recs})), "warnings": ""}
        timed = [primary(r) for r in ok if primary(r)]
        if timed:
            vals = [t[0] for t in timed]
            var = sstats.repeat_variation(vals)
            row.update(metric=ok[0]["measurements"]["timing"]["metric"], statistic=timed[0][1], value_median=var["median"], value_min=var["min"], value_max=var["max"], repeat_cv=var.get("cv"))
        tps = [primary_throughput(r) for r in ok if primary_throughput(r)]
        if tps:
            row.update(throughput_kind=tps[0]["kind"], throughput_value=sstats.median([t["value"] for t in tps]), throughput_unit=tps[0]["unit"])
        warns = sorted({w for r in ok for w in r["measurements"].get("warnings", [])})
        if len(ok) < 3 and ok:
            warns.append(f"only {len(ok)} successful repetition(s): spread is not meaningful")
        row["warnings"] = " | ".join(warns)
        rows.append(row)
    rows.sort(key=lambda r: (r["benchmark"], r["operation"], r["algorithm_requested"], r["rank_count"], r["payload_bytes"]))
    return rows


def to_csv(rows):
    if not rows:
        return ""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def _fmt(v, digits=2):
    return "-" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))


def host_rows(records):
    """One entry per host of the runs that used several hosts or supervised ranks: hardware, build identity and TBCCL / ABI / wire versions."""
    multi = [r for r in records if r["environment"].get("multi_host") or any(x.get("supervised") for x in r["artifacts"].get("ranks", []))]
    hosts = {}
    for r in multi:
        for h in r["environment"]["hosts"]:
            hosts.setdefault(h["host_id"], h)
    out = []
    for hid, h in sorted(hosts.items()):
        b = h.get("build") or {}
        out.append({"host": hid, "ranks": h["ranks"], "os": h.get("os"), "cpu": h.get("cpu_model"), "gpu": h.get("gpu_model"), "build_type": b.get("build_type"), "compiler": b.get("compiler"),
                    "provenance": b.get("status") or "unknown", "build_commit": b.get("build_commit"), "executable_sha256": b.get("executable_sha256"),
                    "tbccl_version": b.get("tbccl_package_version"), "c_abi": b.get("c_abi"), "wire_protocol": b.get("wire_protocol")})
    return out


def cleanup_state(records):
    """('none', []) without supervised ranks, ('confirmed', []) when every supervised rank was confirmed clean, else ('not_confirmed', [the ranks that were not])."""
    supervised = [(r["record_id"].split("/", 1)[-1], x) for r in records for x in r["artifacts"].get("ranks", []) if x.get("supervised")]
    if not supervised:
        return "none", []
    bad = [{"record": rid, "rank": x["rank"], "host": x["host_id"], "note": x.get("cleanup_note")} for rid, x in supervised if x.get("cleanup_confirmed") is not True]
    return ("not_confirmed", bad) if bad else ("confirmed", [])


def _hosts_section(records, manifest):
    """Per-host hardware, build provenance and cleanup status for runs that used more than one host (or any supervised rank)."""
    multi = [r for r in records if r["environment"].get("multi_host") or any(x.get("supervised") for x in r["artifacts"].get("ranks", []))]
    if not multi:
        return []
    hosts = {}
    for r in multi:
        for h in r["environment"]["hosts"]:
            hosts.setdefault(h["host_id"], h)
    lines = ["## Hosts", "", "| host | ranks | OS | CPU | GPU | build | provenance | executable sha256 | TBCCL / ABI / wire |", "|---|---|---|---|---|---|---|---|---|"]
    for hid, h in sorted(hosts.items()):
        b = h.get("build") or {}
        lines.append(f"| {hid} | {','.join(map(str, h['ranks']))} | {h.get('os') or '-'} | {h.get('cpu_model') or '-'} | {h.get('gpu_model') or '-'} | {b.get('build_type') or '-'}, {b.get('compiler') or '-'} | "
                     f"{b.get('status') or 'unknown'}{' (commit ' + b['build_commit'][:10] + ')' if b.get('build_commit') else ''} | {(b.get('executable_sha256') or '-')[:12]} | "
                     f"{b.get('tbccl_package_version') or '-'} / {b.get('c_abi') or '-'} / {b.get('wire_protocol') or '-'} |")
    lines.append("")
    unconfirmed = [(r["record_id"].split("/", 1)[-1], x) for r in records for x in r["artifacts"].get("ranks", []) if x.get("supervised") and x.get("cleanup_confirmed") is not True]
    if unconfirmed:
        lines += ["**Cleanup NOT confirmed** for supervised ranks (the measurement itself is unaffected, but leftover processes are not ruled out):", ""]
        lines += [f"- `{rid}` rank {x['rank']} on {x['host_id']}: {x.get('cleanup_note')}" for rid, x in unconfirmed[:20]]
        lines.append("")
    elif any(x.get("supervised") for r in records for x in r["artifacts"].get("ranks", [])):
        lines += ["Cleanup was confirmed by the host supervisor and by a separate check for every supervised rank.", ""]
    return lines


def providers_used(records):
    """The distinct provider entries of a run (what each framework provider reported about itself), ordered by provider and host. Empty for a native-only run."""
    seen = {}
    for r in records:
        for p in r["environment"].get("providers") or []:
            seen.setdefault((p["provider_id"], p.get("host"), p.get("status"), p.get("provider_version"), json.dumps(p.get("framework_versions") or {}, sort_keys=True)), p)
    return [seen[k] for k in sorted(seen, key=lambda k: tuple(str(x) for x in k))]


def _native_text(entry):
    """The compiled extension(s) an adapter wheel ships, by hash; unavailable when the provider could not report them (not installed, or an older provider)."""
    modules = entry.get("native_modules")
    return ", ".join(f"{Path(name).name.split('.')[0]} {m['sha256'][:10]}" for name, m in sorted(modules.items())) if modules else "unavailable"


def _providers_section(records):
    entries = providers_used(records)
    if not entries:
        return []
    lines = ["## Framework providers", "", "| provider | host | state | provider version | framework versions | compiled extension (sha256) | scripts (sha256) |", "|---|---|---|---|---|---|---|"]
    for p in entries:
        fv = {k: v for k, v in (p.get("framework_versions") or {}).items() if v and k not in ("python", "platform")}
        lines.append(f"| {p['provider_id']} | {p.get('host') or '-'} | {p.get('status') or '-'} | {p.get('provider_version') or 'unavailable'} | "
                     f"{', '.join(f'{k} {v}' for k, v in sorted(fv.items())) or 'unavailable'} | {_native_text(p)} | {', '.join(f'{k} {h[:10]}' for k, h in sorted((p.get('entry_point_sha256') or {}).items())) or 'unavailable'} |")
    lines.append("")
    if any(p.get("status") == "bundled_compat" for p in entries):
        lines += ["`bundled_compat`: these results were produced by the compatibility copy of the workload scripts shipped with the suite; framework versions were not collected.", ""]
    return lines


def _commit_text(manifest, quote):
    """The repository commit of the run, or a statement that there is none (a source export without Git metadata); a commit is never invented."""
    repo = (manifest or {}).get("repository") or {}
    if not repo.get("commit"):
        return "unavailable (the source tree has no Git metadata)"
    return f"{quote}{repo['commit']}{quote}" + (" (uncommitted changes)" if repo.get("dirty") else "")


def to_markdown(rows, records, title="Benchmark summary", manifest=None):
    lines = [f"# {title}", ""]
    if manifest:
        lines += [f"- run: `{manifest.get('run_id')}` ({manifest.get('status')}); suite {manifest.get('suite_version')}", f"- repository commit: {_commit_text(manifest, '`')}", ""]
    lines += _hosts_section(records, manifest)
    lines += _providers_section(records)
    states = {}
    for r in records:
        states[r["status"]["state"]] = states.get(r["status"]["state"], 0) + 1
    lines += ["Records: " + ", ".join(f"{k} {v}" for k, v in sorted(states.items())), ""]
    lines += ["Latencies are in microseconds. The statistic column names what the value is (median of the per-repetition medians, or the mean for tools that report only a mean).",
              "Loopback results compare builds on one machine; they are not link measurements. Throughput is as defined by the native tool (see the record's `definition`).", ""]
    for provenance in ("native", "imported"):
        sub = [r for r in rows if r["provenance"] == provenance]
        if not sub:
            continue
        lines += [f"## {'Measured by this suite' if provenance == 'native' else 'Imported historical results'}", "",
                  "| benchmark | operation | backend | orientation | algorithm (resolved) | domain | topology | ranks | payload B | stat | value us | min..max | cv | throughput | ok/records | verification |",
                  "|---|---|---|---|---|---|---|---:|---:|---|---:|---|---:|---|---|---|"]
        for r in sub:
            algo = r["algorithm_requested"] + (f" ({r['algorithm_resolved']})" if r["algorithm_resolved"] and r["algorithm_resolved"] != r["algorithm_requested"] else "")
            tp = f"{r['throughput_value']:.3g} {r['throughput_unit']}" if r["throughput_value"] is not None else "-"
            spread = f"{_fmt(r['value_min'])}..{_fmt(r['value_max'])}" if r["value_median"] is not None else "-"
            lines.append(f"| {r['benchmark']} | {r['operation']} | {r['backend'] or '-'} | {r['orientation']} | {algo or '-'} | {r['memory_domain']} | {r['topology']} | {r['rank_count']} | {r['payload_bytes']} | {r['statistic'] or '-'} | "
                         f"{_fmt(r['value_median'])} | {spread} | {_fmt(r['repeat_cv'], 3)} | {tp} | {r['successful']}/{r['records']} | {r['verification']} |")
        lines.append("")
    bad = [r for r in records if r["status"]["state"] != "success"]
    if bad:
        lines += ["## Points without a valid measurement", "", "| record | state | reason |", "|---|---|---|"]
        for r in bad:
            ranks = "; ".join(f"rank {x['rank']}@{x['host_id']}: exit {x.get('exit_code')}" for x in r["artifacts"].get("ranks", []))
            lines.append(f"| `{r['record_id'].split('/', 1)[-1]}` | {r['status']['state']} | {(r['status'].get('reason') or '').replace('|', '/')[:160]}{' [' + ranks + ']' if ranks else ''} |")
        lines.append("")
    counts = {}
    for r in records:
        if r["status"]["state"] == "success":
            for w in r["measurements"].get("warnings", []):
                key = f"{r['benchmark']['name']} {r['benchmark']['operation']}: {w}"
                counts[key] = counts.get(key, 0) + 1
    if counts:
        lines += ["## Warnings", ""] + [f"- {k} (x{v})" for k, v in sorted(counts.items())[:40]] + [""]
    return "\n".join(lines)


REPORT_VERSION = "tbccl-bench-report/1"


def invalid_points(records):
    return [{"record": r["record_id"].split("/", 1)[-1], "state": r["status"]["state"], "reason": (r["status"].get("reason") or "")[:300]} for r in records if r["status"]["state"] != "success"]


def to_json(rows, records, title="Benchmark summary", manifest=None):
    """The same content as the Markdown report as one JSON document. Failed and incomplete points are listed under `points_without_valid_measurement`, never inside `results`."""
    states = {}
    for r in records:
        states[r["status"]["state"]] = states.get(r["status"]["state"], 0) + 1
    run = {k: (manifest or {}).get(k) for k in ("run_id", "status", "suite_version", "started_utc", "finished_utc")}
    run["repository"] = (manifest or {}).get("repository")
    return json.dumps({"report_version": REPORT_VERSION, "title": title, "run": run, "record_states": states, "hosts": host_rows(records), "cleanup": dict(zip(("state", "unconfirmed"), cleanup_state(records))), "providers": providers_used(records), "results": rows,
                       "points_without_valid_measurement": invalid_points(records)}, indent=2, sort_keys=True, default=str) + "\n"


_CSS = ("body{font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px;background:#fff;color:#0b0b0b}"
        "h1{font-size:1.5rem}h2{font-size:1.2rem;margin-top:2rem}table{border-collapse:collapse;width:100%;font-size:.85rem}"
        "th,td{border-bottom:1px solid #e6e5e1;padding:4px 8px;text-align:left;vertical-align:top}th{background:#f5f4f1}td.n{text-align:right;font-variant-numeric:tabular-nums}"
        ".wrap{overflow-x:auto}.note{color:#52514e}.bad{color:#b3261e}img{max-width:100%;height:auto}code{font-size:.85em}"
        "@media (prefers-color-scheme:dark){body{background:#151513;color:#f2f1ee}th{background:#242421}th,td{border-color:#35342f}.note{color:#a5a39c}.bad{color:#f2a19b}}")


def to_html(rows, records, title="Benchmark summary", manifest=None, images=()):
    """A single self-contained HTML page: inline style, no scripts, no external requests (a content-security policy forbids them). `images` is a list of (caption, PNG bytes) embedded as data URIs."""
    import base64
    from html import escape as e
    out = ["<!doctype html>", "<html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">",
           "<meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'; img-src data:\">", f"<title>{e(title)}</title><style>{_CSS}</style></head><body>", f"<h1>{e(title)}</h1>"]
    if manifest:
        out.append(f"<p class=\"note\">run <code>{e(str(manifest.get('run_id')))}</code> ({e(str(manifest.get('status')))}); suite {e(str(manifest.get('suite_version')))}; "
                   f"repository commit {e(_commit_text(manifest, ''))}</p>")
    states = {}
    for r in records:
        states[r["status"]["state"]] = states.get(r["status"]["state"], 0) + 1
    out.append("<p>Records: " + ", ".join(f"{e(k)} {v}" for k, v in sorted(states.items())) + "</p>")
    out.append("<p class=\"note\">Latencies are in microseconds. A value is the median over independent repetitions of each repetition's primary statistic. Loopback results compare builds on one machine; they are not link measurements. "
               "Failed, timed-out and unverified points are listed separately and never averaged into a result.</p>")
    hosts = host_rows(records)
    if hosts:
        out.append("<h2>Hosts</h2><div class=\"wrap\"><table><tr><th>host</th><th>ranks</th><th>OS</th><th>CPU</th><th>GPU</th><th>build</th><th>provenance</th><th>executable sha256</th><th>TBCCL / ABI / wire</th></tr>")
        for h in hosts:
            prov = f"{h['provenance']}" + (f" (commit {h['build_commit'][:10]})" if h["build_commit"] else "")
            out.append("<tr>" + "".join(f"<td>{e(str(c))}</td>" for c in (h["host"], ",".join(map(str, h["ranks"])), h["os"] or "-", h["cpu"] or "-", h["gpu"] or "-", f"{h['build_type'] or '-'}, {h['compiler'] or '-'}", prov,
                                                                        (h["executable_sha256"] or "-")[:12], f"{h['tbccl_version'] or '-'} / {h['c_abi'] or '-'} / {h['wire_protocol'] or '-'}")) + "</tr>")
        out.append("</table></div>")
    state, unconfirmed = cleanup_state(records)
    if state == "confirmed":
        out.append("<p class=\"note\">Cleanup was confirmed by the host supervisor and by a separate check for every supervised rank.</p>")
    elif state == "not_confirmed":
        out.append("<p class=\"bad\"><strong>Cleanup NOT confirmed</strong> for supervised ranks (the measurement itself is unaffected, but leftover processes are not ruled out):</p><ul>"
                   + "".join(f"<li><code>{e(u['record'])}</code> rank {u['rank']} on {e(str(u['host']))}: {e(str(u['note']))}</li>" for u in unconfirmed[:20]) + "</ul>")
    entries = providers_used(records)
    if entries:
        out.append("<h2>Framework providers</h2><div class=\"wrap\"><table><tr><th>provider</th><th>host</th><th>state</th><th>provider version</th><th>framework versions</th><th>compiled extension (sha256)</th></tr>")
        for p in entries:
            fv = {k: v for k, v in (p.get("framework_versions") or {}).items() if v and k not in ("python", "platform")}
            out.append(f"<tr><td>{e(p['provider_id'])}</td><td>{e(str(p.get('host') or '-'))}</td><td>{e(str(p.get('status') or '-'))}</td><td>{e(str(p.get('provider_version') or 'unavailable'))}</td>"
                       f"<td>{e(', '.join(f'{k} {v}' for k, v in sorted(fv.items())) or 'unavailable')}</td><td>{e(_native_text(p))}</td></tr>")
        out.append("</table></div>")
    out.append("<h2>Results</h2><div class=\"wrap\"><table><tr><th>benchmark</th><th>operation</th><th>backend</th><th>orientation</th><th>algorithm</th><th>domain</th><th>topology</th><th>ranks</th><th>payload B</th><th>statistic</th><th>value</th><th>min..max</th><th>cv</th><th>throughput</th><th>ok/records</th><th>verification</th><th>provenance</th></tr>")
    for r in rows:
        algo = r["algorithm_requested"] + (f" ({r['algorithm_resolved']})" if r["algorithm_resolved"] and r["algorithm_resolved"] != r["algorithm_requested"] else "")
        tp = f"{r['throughput_value']:.3g} {r['throughput_unit']}" if r["throughput_value"] is not None else "-"
        spread = f"{_fmt(r['value_min'])}..{_fmt(r['value_max'])}" if r["value_median"] is not None else "-"
        cells = [r["benchmark"], r["operation"], r["backend"] or "-", r["orientation"], algo or "-", r["memory_domain"], r["topology"], r["rank_count"], r["payload_bytes"], r["statistic"] or "-", _fmt(r["value_median"]), spread,
                 _fmt(r["repeat_cv"], 3), tp, f"{r['successful']}/{r['records']}", r["verification"], r["provenance"]]
        out.append("<tr>" + "".join(f"<td{' class=\"n\"' if i in (7, 8, 10, 12) else ''}>{e(str(c))}</td>" for i, c in enumerate(cells)) + "</tr>")
    out.append("</table></div>")
    bad = invalid_points(records)
    if bad:
        out.append("<h2>Points without a valid measurement</h2><div class=\"wrap\"><table><tr><th>record</th><th>state</th><th>reason</th></tr>")
        out += [f"<tr><td><code>{e(b['record'])}</code></td><td class=\"bad\">{e(b['state'])}</td><td>{e(b['reason'])}</td></tr>" for b in bad]
        out.append("</table></div>")
    for caption, png in images:
        out.append(f"<h2>{e(caption)}</h2><img alt=\"{e(caption)}\" src=\"data:image/png;base64,{base64.b64encode(png).decode()}\">")
    out.append("</body></html>")
    return "\n".join(out) + "\n"
