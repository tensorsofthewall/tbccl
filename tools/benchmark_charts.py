#!/usr/bin/env python3
"""Render the benchmark charts of the documentation from curated reference data.

    python3 tools/benchmark_charts.py           # write docs/benchmarks/charts/*.html
    python3 tools/benchmark_charts.py --check   # fail if a committed chart is stale or unknown

Input is `docs/benchmarks/chart-data.json`: values copied from the published tables, each with the exact published text and where it was published. The
Markdown tables stay authoritative; this file only feeds the pictures. Output is one HTML fragment per chart (a figure with an inline SVG) that the page
includes with the `raw` directive, so the chart takes its colours from the page's tokens and follows the light and dark themes without scripts, fonts
or remote files. Standard library only; output is a pure function of the data (no timestamps, no paths, no random ids).

The same file is used by all four TBCCL repositories; only the data file differs.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from html import escape
from pathlib import Path

SCHEMA = "tbccl-doc-chart-data/1"
ACCENTS = ("tbccl", "torch", "vllm", "exo")
BAR_STYLES = ("solid", "hatch", "solid-neutral", "hatch-accent")
MARKERS = ("circle", "square", "triangle", "diamond")
DASHES = {"solid": "", "dashed": "7 4", "dotted": "2 3"}
CLASSIFICATIONS = ("improvement", "regression", "indeterminate", "incompatible", "insufficient_evidence", "invalid_measurement")
WIDTH = 600
MAX_LABEL = 30


class ChartDataError(ValueError):
    pass


def _need(cond, where, message):
    if not cond:
        raise ChartDataError(f"{where}: {message}")


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _n(x):
    """Coordinates with at most two decimals and no trailing zeros, so the output is stable."""
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _nice_ceiling(v):
    exp = 10 ** math.floor(math.log10(v))
    for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * exp >= v:
            return m * exp
    return 10 * exp


def _ticks(top):
    exp = 10 ** math.floor(math.log10(top))
    for step in (exp / 2, exp, exp * 2, exp * 2.5, exp * 5):
        if top / step <= 6:
            break
    count = int(round(top / step))
    return [step * i for i in range(count + 1)]


def _fmt_tick(v):
    if v == int(v):
        return f"{int(v):,}"
    return f"{v:g}"


def validate(data):
    _need(isinstance(data, dict) and data.get("schema") == SCHEMA, "chart data", f"schema must be {SCHEMA!r}")
    _need(data.get("accent") in ACCENTS, "chart data", f"accent must be one of {ACCENTS}")
    _need(isinstance(data.get("provenance"), str) and data["provenance"], "chart data", "provenance statement is required")
    ids = set()
    charts = data.get("charts")
    _need(isinstance(charts, list) and charts, "chart data", "charts must be a non-empty list")
    for c in charts:
        cid = c.get("id")
        _need(isinstance(cid, str) and re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", cid or ""), "chart", f"bad id {cid!r}")
        _need(cid not in ids, cid, "duplicate id")
        ids.add(cid)
        sec = c.get("section")
        _need(isinstance(sec, str) and sec.strip() or (isinstance(sec, list) and sec and all(isinstance(x, str) and x.strip() for x in sec)), cid,
              "section is required: the heading of the page section holding the table, or a list of headings")
        for key in ("title", "summary", "conditions", "page", "statistic"):
            _need(isinstance(c.get(key), str) and c[key].strip(), cid, f"{key} is required")
        _need(c.get("kind") in ("bars", "lines"), cid, "kind must be bars or lines")
        _need("classification" not in c or c["classification"] in CLASSIFICATIONS, cid, f"unknown classification {c.get('classification')!r}")
        (_validate_bars if c["kind"] == "bars" else _validate_lines)(c)
    return data


def _check_point(where, p):
    _need(_num(p.get("value")) and p["value"] > 0, where, "value must be a positive number")
    _need(isinstance(p.get("published"), str) and p["published"], where, "the exact published text is required")
    _need(isinstance(p.get("exact"), bool), where, "exact must say whether value equals the published figure")
    if "min" in p or "max" in p:
        _need(_num(p.get("min")) and _num(p.get("max")), where, "min and max go together and must be numbers")
        _need(p["min"] <= p["value"] <= p["max"], where, f"value {p['value']} is outside its range {p['min']}..{p['max']}")
        _need(p["min"] > 0, where, "range must be positive")
        _need(isinstance(p.get("range_published"), str) and p["range_published"], where, "the range as published is required beside min and max")
    t = p.get("table")
    _need(isinstance(t, dict) and all(isinstance(t.get(k), str) and t[k].strip() for k in ("section", "row", "column")), where,
          "table must name the section, the row (its first cell) and the column header the value was read from")
    n = p.get("n")
    _need((isinstance(n, int) and not isinstance(n, bool) and n >= 1) or (isinstance(n, str) and n.strip()), where,
          "n (independent repetitions or lifecycles) must be an integer of 1 or more, or the published text such as '3-6'; never leave it out")
    if "classification" in p:
        _need(p["classification"] in CLASSIFICATIONS, where, f"unknown classification {p['classification']!r}")


def _validate_bars(c):
    cid = c["id"]
    series = c.get("series")
    _need(isinstance(series, list) and series, cid, "series is required")
    names = set()
    for s in series:
        _need(s.get("style") in BAR_STYLES, cid, f"series style must be one of {BAR_STYLES}")
        _need(isinstance(s.get("id"), str) and s["id"] not in names, cid, "series ids must be unique strings")
        names.add(s["id"])
    panels = c.get("panels")
    _need(isinstance(panels, list) and panels, cid, "panels is required")
    for pi, panel in enumerate(panels):
        where = f"{cid} panel {pi}"
        _need(isinstance(panel.get("unit"), str) and panel["unit"], where, "unit is required")
        _need(panel.get("direction") in ("lower is better", "higher is better", "neither"), where, "direction must be 'lower is better', 'higher is better' or 'neither'")
        for line in panel.get("note", []):
            _need(isinstance(line, str) and len(line) <= 80, where, "note lines must be strings of at most 80 characters")
        groups = panel.get("groups")
        _need(isinstance(groups, list) and groups, where, "groups is required")
        for g in groups:
            _need(isinstance(g.get("bars"), list) and g["bars"], where, "each group needs bars")
            for b in g["bars"]:
                bw = f"{where} / {g.get('label', '')} / {b.get('label')}"
                _need(isinstance(b.get("label"), str) and 0 < len(b["label"]) <= MAX_LABEL, bw, f"label is required and at most {MAX_LABEL} characters")
                _need(b.get("series") in names, bw, "unknown series")
                _check_point(bw, b)


def _validate_lines(c):
    cid = c["id"]
    for key in ("x_label", "y_label", "y_unit"):
        _need(isinstance(c.get(key), str) and c[key], cid, f"{key} is required")
    _need(c.get("x_scale") in ("linear", "log"), cid, "x_scale must be linear or log; it is never guessed")
    _need(c.get("y_scale") in ("log", "linear"), cid, "y_scale must be log or linear")
    ticks = c.get("x_ticks")
    _need(isinstance(ticks, list) and len(ticks) >= 2, cid, "x_ticks needs at least two entries")
    for t in ticks:
        _need(_num(t.get("x")) and t["x"] >= 0 and (c["x_scale"] == "linear" or t["x"] > 0) and isinstance(t.get("label"), str), cid,
              "each x tick needs a label and a positive x (a non-negative x on a linear axis)")
    xs = [t["x"] for t in ticks]
    _need(xs == sorted(xs) and len(set(xs)) == len(xs), cid, "x_ticks must be strictly increasing")
    series = c.get("series")
    _need(isinstance(series, list) and series, cid, "series is required")
    for s in series:
        sw = f"{cid} / {s.get('label')}"
        _need(isinstance(s.get("label"), str) and 0 < len(s["label"]) <= 60, sw, "label is required")
        _need(s.get("marker") in MARKERS and s.get("dash") in DASHES, sw, f"marker must be one of {MARKERS} and dash one of {tuple(DASHES)}")
        _need(s.get("color") in ("accent", "neutral"), sw, "color must be accent or neutral")
        _need(isinstance(s.get("points"), list) and s["points"], sw, "points is required")
        last = -1
        for p in s["points"]:
            _need(_num(p.get("x")) and p["x"] >= 0 and (c["x_scale"] == "linear" or p["x"] > 0), sw, "x must be a positive number (non-negative on a linear axis)")
            _need(p["x"] in xs, sw, f"x={p['x']} has no tick, so the reader cannot read its payload")
            _need(p["x"] > last, sw, "points must be in increasing x")
            last = p["x"]
            _check_point(f"{sw} @ {p['x']}", p)


# ---------------------------------------------------------------- bars

def _bar_fill(style, cid, accent):
    return {
        "solid": f"var(--tb-accent-{accent})",
        "solid-neutral": "var(--color-foreground-muted)",
        "hatch": f"url(#{cid}-hatch)",
        "hatch-accent": f"url(#{cid}-hatch-accent)",
    }[style]


def _bars_svg(c, accent):
    cid = c["id"]
    series = {s["id"]: s for s in c["series"]}
    label_w, left, right, row, bar_h = 196, 210, 96, 28, 17
    plot_w = WIDTH - left - right
    y = 8
    out = []
    for panel in c["panels"]:
        top = max(b.get("max", b["value"]) for g in panel["groups"] for b in g["bars"])
        ceiling = _nice_ceiling(top)
        scale = plot_w / ceiling
        head, rows = [], []
        if panel.get("title"):
            head.append(f'<text class="tb-chart-heading" x="0" y="{_n(y + 14)}">{escape(panel["title"])}</text>')
            y += 24
        for line in panel.get("note", []):
            head.append(f'<text class="tb-chart-note" x="0" y="{_n(y + 12)}">{escape(line)}</text>')
            y += 18
        if panel.get("note"):
            y += 4
        plot_top = y
        for g in panel["groups"]:
            if g.get("label"):
                rows.append(f'<text class="tb-chart-group" x="0" y="{_n(y + 14)}">{escape(g["label"])}</text>')
                y += 22
            for b in g["bars"]:
                s = series[b["series"]]
                cy = y + row / 2
                rows.append(f'<text class="tb-chart-label" x="{label_w}" y="{_n(cy + 4.5)}" text-anchor="end">{escape(b["label"])}</text>')
                w = b["value"] * scale
                rows.append(f'<rect class="tb-chart-bar" x="{left}" y="{_n(cy - bar_h / 2)}" width="{_n(w)}" height="{bar_h}" style="fill:{_bar_fill(s["style"], cid, accent)}"/>')
                end = left + w
                if "min" in b:
                    x1, x2 = left + b["min"] * scale, left + b["max"] * scale
                    rows.append(f'<path class="tb-chart-range" d="M{_n(x1)} {_n(cy)}H{_n(x2)}M{_n(x1)} {_n(cy - 5)}V{_n(cy + 5)}M{_n(x2)} {_n(cy - 5)}V{_n(cy + 5)}"/>')
                    end = x2
                rows.append(f'<text class="tb-chart-value" x="{_n(end + 7)}" y="{_n(cy + 4.5)}">{escape(b["published"])}</text>')
                y += row
            y += 6
        grid, ticks = [], []
        for t in _ticks(ceiling):
            x = left + t * scale
            grid.append(f'<path class="tb-chart-grid" d="M{_n(x)} {_n(plot_top)}V{_n(y)}"/>')
            ticks.append(f'<text class="tb-chart-tick" x="{_n(x)}" y="{_n(y + 16)}" text-anchor="middle">{escape(_fmt_tick(t))}</text>')
        foot = [f'<path class="tb-chart-axis" d="M{left} {_n(plot_top)}V{_n(y)}"/>',
                f'<text class="tb-chart-unit" x="{_n(left + plot_w / 2)}" y="{_n(y + 34)}" text-anchor="middle">{escape(panel["unit"])} ({escape(panel["direction"])})</text>']
        out += head + grid + rows + ticks + foot
        y += 48
    defs = (f'<defs><pattern id="{cid}-hatch" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="tb-chart-hatch-bg"/><path class="tb-chart-hatch-line" d="M0 0V6"/></pattern>'
            f'<pattern id="{cid}-hatch-accent" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)"><rect width="6" height="6" class="tb-chart-hatch-bg"/><path class="tb-chart-hatch-line" style="stroke:var(--tb-accent-{accent})" d="M0 0V6"/></pattern></defs>')
    return y, defs, "".join(out)


# ---------------------------------------------------------------- lines

def _marker(kind, x, y, style):
    r = 5
    if kind == "circle":
        return f'<circle class="tb-chart-marker" cx="{_n(x)}" cy="{_n(y)}" r="{r}" style="{style}"/>'
    if kind == "square":
        return f'<rect class="tb-chart-marker" x="{_n(x - r)}" y="{_n(y - r)}" width="{2 * r}" height="{2 * r}" style="{style}"/>'
    if kind == "triangle":
        return f'<path class="tb-chart-marker" d="M{_n(x)} {_n(y - r - 1)}L{_n(x + r + 1)} {_n(y + r)}H{_n(x - r - 1)}z" style="{style}"/>'
    return f'<path class="tb-chart-marker" d="M{_n(x)} {_n(y - r - 1)}L{_n(x + r + 1)} {_n(y)}L{_n(x)} {_n(y + r + 1)}L{_n(x - r - 1)} {_n(y)}z" style="{style}"/>'


def _lines_svg(c, accent):
    left, right, top, plot_h = 78, 24, 14, 250
    plot_w = WIDTH - left - right
    xs = [t["x"] for t in c["x_ticks"]]
    pad = 14
    if c["x_scale"] == "log":
        frac = lambda v: (math.log2(v) - math.log2(xs[0])) / (math.log2(xs[-1]) - math.log2(xs[0]))
    else:
        frac = lambda v: (v - xs[0]) / (xs[-1] - xs[0])
    lx = lambda v: left + pad + frac(v) * (plot_w - 2 * pad)
    vals = [p.get(k, p["value"]) for s in c["series"] for p in s["points"] for k in ("value", "min", "max") if k in p or k == "value"]
    lo, hi = min(vals), max(vals)
    body = []
    if c["y_scale"] == "log":
        d0, d1 = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
        if d0 == d1:
            d1 += 1
        ly = lambda v: top + plot_h - (math.log10(v) - d0) / (d1 - d0) * plot_h
        ticks = [10.0 ** k for k in range(d0, d1 + 1)]
        minor = [m * 10.0 ** k for k in range(d0, d1) for m in (2, 5)]
    else:
        ceiling = _nice_ceiling(hi)
        ly = lambda v: top + plot_h - v / ceiling * plot_h
        ticks, minor = _ticks(ceiling), []
    for m in minor:
        body.append(f'<path class="tb-chart-grid tb-chart-minor" d="M{left} {_n(ly(m))}H{WIDTH - right}"/>')
    for t in ticks:
        body.append(f'<path class="tb-chart-grid" d="M{left} {_n(ly(t))}H{WIDTH - right}"/>')
        body.append(f'<text class="tb-chart-tick" x="{left - 8}" y="{_n(ly(t) + 4)}" text-anchor="end">{escape(_fmt_tick(t))}</text>')
    for t in c["x_ticks"]:
        body.append(f'<path class="tb-chart-grid" d="M{_n(lx(t["x"]))} {top}V{top + plot_h}"/>')
        body.append(f'<text class="tb-chart-tick" x="{_n(lx(t["x"]))}" y="{top + plot_h + 17}" text-anchor="middle">{escape(t["label"])}</text>')
    body.append(f'<path class="tb-chart-axis" d="M{left} {top}V{top + plot_h}H{WIDTH - right}"/>')
    body.append(f'<text class="tb-chart-unit" x="{_n(left + plot_w / 2)}" y="{top + plot_h + 40}" text-anchor="middle">{escape(c["x_label"])}{" (logarithmic)" if c["x_scale"] == "log" else ""}</text>')
    body.append(f'<text class="tb-chart-unit" transform="translate(14 {_n(top + plot_h / 2)}) rotate(-90)" text-anchor="middle">{escape(c["y_label"])}, {escape(c["y_unit"])}{" (logarithmic)" if c["y_scale"] == "log" else ""}</text>')
    for s in c["series"]:
        color = f"var(--tb-accent-{accent})" if s["color"] == "accent" else "var(--color-foreground-primary)"
        pts = [(lx(p["x"]), ly(p["value"])) for p in s["points"]]
        if len(pts) > 1:
            path = "M" + "L".join(f"{_n(x)} {_n(y)}" for x, y in pts)
            dash = f' stroke-dasharray="{DASHES[s["dash"]]}"' if DASHES[s["dash"]] else ""
            body.append(f'<path class="tb-chart-line" d="{path}" style="stroke:{color}"{dash}/>')
        for p, (x, y) in zip(s["points"], pts):
            if "min" in p:
                body.append(f'<path class="tb-chart-range" d="M{_n(x)} {_n(ly(p["min"]))}V{_n(ly(p["max"]))}M{_n(x - 4)} {_n(ly(p["min"]))}H{_n(x + 4)}M{_n(x - 4)} {_n(ly(p["max"]))}H{_n(x + 4)}"/>')
            body.append(_marker(s["marker"], x, y, f"fill:{color}" if s["marker"] in ("circle", "triangle") else f"fill:var(--color-background-primary);stroke:{color}"))
    y = top + plot_h + 62
    for s in c["series"]:
        color = f"var(--tb-accent-{accent})" if s["color"] == "accent" else "var(--color-foreground-primary)"
        dash = f' stroke-dasharray="{DASHES[s["dash"]]}"' if DASHES[s["dash"]] else ""
        body.append(f'<path class="tb-chart-line" d="M{left} {y}H{left + 36}" style="stroke:{color}"{dash}/>')
        body.append(_marker(s["marker"], left + 18, y, f"fill:{color}" if s["marker"] in ("circle", "triangle") else f"fill:var(--color-background-primary);stroke:{color}"))
        body.append(f'<text class="tb-chart-label" x="{left + 46}" y="{y + 4.5}">{escape(s["label"])}</text>')
        y += 24
    return y - 8, "", "".join(body)


# ---------------------------------------------------------------- page fragment

def render(c, accent):
    cid = c["id"]
    height, defs, body = (_bars_svg if c["kind"] == "bars" else _lines_svg)(c, accent)
    label = f'Chart: {c["title"]}'
    svg = (f'<svg viewBox="0 0 {WIDTH} {_n(height)}" role="img" aria-labelledby="{cid}-t {cid}-d" xmlns="http://www.w3.org/2000/svg">'
           f'<title id="{cid}-t">{escape(c["title"])}</title><desc id="{cid}-d">{escape(c["summary"])}</desc>{defs}{body}</svg>')
    return (f'<figure class="tb-chart" id="chart-{cid}">\n'
            f'<p class="tb-chart-title">{escape(c["title"])}</p>\n'
            f'<div class="tb-chart-scroll" role="region" tabindex="0" aria-label="{escape(label)} (scrolls sideways on narrow screens)">{svg}</div>\n'
            f'<p class="tb-chart-hint">Scroll the chart sideways to see all of it, or read the table below.</p>\n'
            f'<figcaption>{escape(c["conditions"])} Values are {escape(c["statistic"])}. Exact figures are in the table below the chart.</figcaption>\n'
            f'</figure>\n')


def build(data):
    validate(data)
    return {f'{c["id"]}.html': render(c, data["accent"]) for c in data["charts"]}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", type=Path, default=root / "docs" / "benchmarks" / "chart-data.json")
    ap.add_argument("--out", type=Path, default=root / "docs" / "benchmarks" / "charts")
    ap.add_argument("--check", action="store_true", help="do not write; fail when a committed chart differs from what the data produces")
    args = ap.parse_args(argv)
    try:
        files = build(json.loads(args.data.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ChartDataError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if args.check:
        bad = [n for n, text in files.items() if not (args.out / n).is_file() or (args.out / n).read_text(encoding="utf-8") != text]
        extra = sorted(p.name for p in args.out.glob("*.html") if p.name not in files) if args.out.is_dir() else []
        for n in bad:
            print(f"stale or missing: {args.out / n}", file=sys.stderr)
        for n in extra:
            print(f"not produced by the data: {args.out / n}", file=sys.stderr)
        if bad or extra:
            print("run: python3 tools/benchmark_charts.py", file=sys.stderr)
            return 1
        print(f"{len(files)} chart(s) up to date")
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    for n, text in files.items():
        (args.out / n).write_text(text, encoding="utf-8")
    print(f"wrote {len(files)} chart(s) to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
