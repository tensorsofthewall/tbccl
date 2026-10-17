"""Static plots (matplotlib, optional). Four kinds: message size vs latency, message size vs bandwidth, baseline vs candidate, repetition variability.

Series use the fixed categorical order of the default palette, each with its own marker so identity never rests on colour alone. Axes are logarithmic where the
data spans decades. Every figure carries a footer naming hardware, operation, memory backend, transport, units, benchmark version and the source result files.
"""
from __future__ import annotations

import importlib.util

from .. import statistics as sstats
from .summary import group_records, host_signature, primary, primary_throughput

CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "h"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def available():
    return importlib.util.find_spec("matplotlib") is not None


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _series_label(r):
    b = r["benchmark"]
    algo = b.get("algorithm_requested")
    resolved = b.get("algorithm_resolved")
    algo = f"{algo}->{resolved}" if algo == "auto" and resolved else algo
    return f"{b['operation']}" + (f" [{algo}]" if algo else "") + (f" {r['configuration']['datatype']}" if r["configuration"].get("datatype") else "") + f" x{r['configuration']['rank_count']}"


def _footer(records, sources):
    r = records[0]
    e, b = r["environment"], r["benchmark"]
    hw = " | ".join(sorted(set(host_signature(r))))
    versions = sorted({f"{x['benchmark'].get('source_commit') and x['benchmark']['source_commit'][:8]} / {x['benchmark'].get('executable_version')}" for x in records})
    ops = sorted({x["benchmark"]["operation"] for x in records})
    return (f"hardware: {hw}\noperation: {', '.join(ops)} | memory backend: {'/'.join(e['memory_backend'])} ({e['memory_domain']['label']}) | transport: {e['transport']} | topology: {e['topology']['kind']}"
            f" ({e['topology']['interconnect']})\nbenchmark version: {'; '.join(versions)} | sources: {', '.join(sorted(sources))}")


def _style(ax, xlabel, ylabel, title):
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.set_title(title, loc="left", color=INK, fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8, which="both")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def _series(records, value_fn):
    """{label: [(payload, [per-repetition values])]} from successful records only."""
    out = {}
    for key, recs in group_records([r for r in records if r["status"]["state"] == "success"]).items():
        vals = [v for v in (value_fn(r) for r in recs) if v is not None]
        if vals:
            out.setdefault(_series_label(recs[0]), []).append((recs[0]["configuration"]["payload_bytes"], vals))
    for pts in out.values():
        pts.sort()
    return out


def _lines(ax, series):
    for i, (label, pts) in enumerate(sorted(series.items())[:8]):
        c, m = CATEGORICAL[i % len(CATEGORICAL)], MARKERS[i % len(MARKERS)]
        xs = [p[0] for p in pts]
        ys = [sstats.median(p[1]) for p in pts]
        ax.plot(xs, ys, color=c, marker=m, markersize=5, linewidth=1.8, label=label, markeredgecolor="#fcfcfb")
        if any(len(p[1]) > 1 for p in pts):
            ax.fill_between(xs, [min(p[1]) for p in pts], [max(p[1]) for p in pts], color=c, alpha=0.15, linewidth=0)
    if len(series) > 8:
        ax.text(0.99, 0.01, f"{len(series) - 8} more series not drawn (limit of 8 per figure; see summary.csv)", transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)


def _save(fig, plt, records, sources, path):
    fig.text(0.01, 0.005, _footer(records, sources), fontsize=6.5, color=MUTED, va="bottom", ha="left")
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(path, dpi=140, facecolor="#fcfcfb")
    plt.close(fig)
    return path


def plot_latency(records, path, sources):
    plt = _plt()
    series = _series(records, lambda r: primary(r)[0] if primary(r) else None)
    if not series:
        return None
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _lines(ax, series)
    _style(ax, "payload (bytes, see each benchmark's payload semantics)", "latency, median of repetitions (us)", f"{records[0]['benchmark']['name']}: message size vs latency")
    return _save(fig, plt, records, sources, path)


def plot_bandwidth(records, path, sources):
    plt = _plt()
    units = {primary_throughput(r)["unit"] for r in records if r["status"]["state"] == "success" and primary_throughput(r)}
    if len(units) != 1:
        return None
    series = _series(records, lambda r: primary_throughput(r)["value"] if primary_throughput(r) else None)
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    _lines(ax, series)
    _style(ax, "payload (bytes)", f"reported throughput ({units.pop()}; definition per benchmark)", f"{records[0]['benchmark']['name']}: message size vs bandwidth")
    return _save(fig, plt, records, sources, path)


def plot_baseline_candidate(result, records_a, records_b, path, sources, names=("baseline", "candidate")):
    plt = _plt()
    comps = [c for c in result["comparisons"] if c["equivalent"]]
    if not comps:
        return None
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, (idx, label) in enumerate(((0, names[0]), (1, names[1]))):
        pts = sorted((c["identity"][3], c["baseline"] if idx == 0 else c["candidate"]) for c in comps)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=CATEGORICAL[i], marker=MARKERS[i], linewidth=1.8, markersize=5, label=label, markeredgecolor="#fcfcfb")
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    _style(ax, "payload (bytes)", "latency, median of repetitions (us)", "Baseline vs candidate (equivalent experiments only)")
    return _save(fig, plt, records_a + records_b, sources, path)


def plot_variability(records, path, sources):
    plt = _plt()
    groups = []
    for key, recs in group_records([r for r in records if r["status"]["state"] == "success"]).items():
        vals = [primary(r)[0] for r in recs if primary(r)]
        if len(vals) > 1:
            groups.append((recs[0]["configuration"]["payload_bytes"], _series_label(recs[0]), vals))
    if not groups:
        return None
    labels = sorted({g[1] for g in groups})
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for g in groups:
        i = labels.index(g[1])
        med = sstats.median(g[2])
        ax.scatter([g[0]] * len(g[2]), [v / med for v in g[2]], color=CATEGORICAL[i % 8], marker=MARKERS[i % 8], s=22, alpha=0.85, edgecolors="#fcfcfb", linewidths=0.6)
    for i, label in enumerate(labels[:8]):
        ax.scatter([], [], color=CATEGORICAL[i % 8], marker=MARKERS[i % 8], label=label)
    ax.axhline(1.0, color=MUTED, linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("payload (bytes)", color=MUTED)
    ax.set_ylabel("repetition value / median of repetitions", color=MUTED)
    ax.set_title("Repetition variability (one point per independent repetition)", loc="left", color=INK, fontsize=11)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8, labelcolor=MUTED)
    return _save(fig, plt, records, sources, path)


def make_all(records, out_dir, sources, baseline=None, comparison=None, names=("baseline", "candidate")):
    """One set of figures per benchmark, so each figure has one unit, one set of semantics and at most eight series."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    short = [f"{Path(p).parent.name}/{Path(p).name}" for p in sources]
    made = []
    for name in sorted({r["benchmark"]["name"] for r in records}):
        subset = [r for r in records if r["benchmark"]["name"] == name]
        made += [plot_latency(subset, out / f"latency-vs-size-{name}.png", short), plot_bandwidth(subset, out / f"bandwidth-vs-size-{name}.png", short),
                 plot_variability(subset, out / f"repetition-variability-{name}.png", short)]
        if baseline is not None and comparison is not None:
            sub = {**comparison, "comparisons": [c for c in comparison["comparisons"] if c["identity"][0] == name]}
            made.append(plot_baseline_candidate(sub, [r for r in baseline if r["benchmark"]["name"] == name], subset, out / f"baseline-vs-candidate-{name}.png", short, names))
    return [str(p) for p in made if p]
