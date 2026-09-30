#!/usr/bin/env python3
"""Phase 31 Part K-N/Z: training ON vs. OFF latency-distribution comparison.

Standard-library only (no scipy) -- Fisher's exact test (two-sided, via
the hypergeometric distribution and math.comb) and the Wilson score
interval are both simple enough to implement directly rather than adding
a dependency for this one phase (plan Part AA item 101/Part E item 28).

Every function here takes plain lists of per-iteration latency samples in
microseconds (as produced by parsing a TBCCL_DIAGNOSTIC transfer_trace,
the same shape analyze_tb4_trace.py's other modes already consume) and
plain event counts -- it never re-derives them from raw trace files
itself, keeping this module focused on the statistics.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_tb4_trace as analyze  # noqa: E402

DEFAULT_BINS = [
    (0, 200), (200, 225), (225, 250), (250, 275), (275, 300),
    (300, 325), (325, 350), (350, 375), (375, 400), (400, 450),
    (450, 500), (500, 800), (800, 1000), (1000, math.inf),
]


def bin_label(lo, hi):
    if hi == math.inf:
        return f">={lo}"
    return f"{lo}-{hi}"


def histogram(samples_us, bins=DEFAULT_BINS):
    """Counts (not fractions) per bin, left-inclusive/right-exclusive,
    identical bin edges must be used for both conditions being compared
    (plan Part K item 46's explicit requirement)."""
    counts = {bin_label(lo, hi): 0 for lo, hi in bins}
    for value in samples_us:
        for lo, hi in bins:
            if lo <= value < hi:
                counts[bin_label(lo, hi)] += 1
                break
    return counts


def mode_fractions(samples_us, low_max=300, intermediate_max=500, high_max=800):
    """Part M item 58: fraction <300us, 300-500us, 500-800us, >=800us.
    Boundaries are the plan's own stated defaults, not fit to the data."""
    n = len(samples_us)
    if n == 0:
        return {"low": None, "intermediate": None, "high": None, "tail": None, "n": 0}
    low = sum(1 for v in samples_us if v < low_max)
    intermediate = sum(1 for v in samples_us if low_max <= v < intermediate_max)
    high = sum(1 for v in samples_us if intermediate_max <= v < high_max)
    tail = sum(1 for v in samples_us if v >= high_max)
    return {"low": low / n, "intermediate": intermediate / n, "high": high / n,
            "tail": tail / n, "n": n}


def threshold_fractions(samples_us, thresholds=(500, 800, 1000)):
    n = len(samples_us)
    if n == 0:
        return {t: None for t in thresholds}
    return {t: sum(1 for v in samples_us if v >= t) / n for t in thresholds}


def summary_stats(samples_us):
    if not samples_us:
        return {"n": 0, "median_us": None, "p75_us": None, "p90_us": None,
                "p95_us": None, "p99_us": None, "max_us": None}
    return {
        "n": len(samples_us),
        "median_us": statistics.median(samples_us),
        "p75_us": analyze.percentile(samples_us, .75),
        "p90_us": analyze.percentile(samples_us, .90),
        "p95_us": analyze.percentile(samples_us, .95),
        "p99_us": analyze.percentile(samples_us, .99),
        "max_us": max(samples_us),
    }


def empirical_cdf(samples_us):
    """Returns sorted (value, cumulative_fraction) pairs -- no plotting,
    just the data (plan Part M item 59: "no plot required")."""
    ordered = sorted(samples_us)
    n = len(ordered)
    return [(v, (i + 1) / n) for i, v in enumerate(ordered)]


def wilson_interval(successes, n, z=1.959963984540054):
    """95% Wilson score interval for a binomial proportion. Standard
    closed-form (no scipy needed); z=1.96 for 95% confidence."""
    if n == 0:
        return {"point": None, "low": None, "high": None, "n": 0}
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half_width = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return {"point": p, "low": max(0.0, center - half_width),
            "high": min(1.0, center + half_width), "n": n, "successes": successes}


def _hypergeom_pmf(k, a_total, b_total, n_draw):
    """P(a_successes = k) under the hypergeometric null for a 2x2 table
    margin-fixed at (a_total successes available, b_total failures
    available, n_draw drawn) -- the standard Fisher's-exact building
    block, via math.comb (exact integer arithmetic, no gamma-function
    floating-point error)."""
    total = a_total + b_total
    if k < 0 or k > n_draw or k > a_total or (n_draw - k) > b_total:
        return 0.0
    return (math.comb(a_total, k) * math.comb(b_total, n_draw - k)) / math.comb(total, n_draw)


def fisher_exact_two_sided(table):
    """table = [[a, b], [c, d]] -- e.g. [[tail_on, non_tail_on],
    [tail_off, non_tail_off]]. Returns the two-sided p-value via the
    standard "sum probabilities <= observed" definition, computed exactly
    over the hypergeometric distribution (no scipy dependency)."""
    (a, b), (c, d) = table
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    n_draw = col1  # draws from row1+row2 population, col1 "successes" available
    observed = _hypergeom_pmf(a, row1, row2, col1)
    p_value = 0.0
    k_min = max(0, col1 - row2)
    k_max = min(row1, col1)
    for k in range(k_min, k_max + 1):
        p_k = _hypergeom_pmf(k, row1, row2, col1)
        if p_k <= observed * (1 + 1e-9):
            p_value += p_k
    return min(1.0, p_value)


def event_rate_comparison(on_events, on_n, off_events, off_n):
    """Part L: absolute risk difference, event-rate ratio (with a
    continuity-corrected variant clearly labeled when either count is
    zero, per item 55's explicit instruction), Fisher's exact test, and
    Wilson intervals for both proportions."""
    on_rate = on_events / on_n if on_n else None
    off_rate = off_events / off_n if off_n else None
    result = {
        "on": {"events": on_events, "n": on_n, "rate": on_rate,
               "wilson": wilson_interval(on_events, on_n)},
        "off": {"events": off_events, "n": off_n, "rate": off_rate,
                "wilson": wilson_interval(off_events, off_n)},
        "absolute_risk_difference": (on_rate - off_rate) if (on_rate is not None and off_rate is not None) else None,
        "fisher_exact_p_value": fisher_exact_two_sided(
            [[on_events, on_n - on_events], [off_events, off_n - off_events]]),
    }
    if on_rate and off_rate:
        result["risk_ratio_on_over_off"] = on_rate / off_rate
        result["risk_ratio_continuity_corrected"] = False
    elif off_events == 0 and on_rate:
        # Zero-event denominator: report a continuity-corrected ratio,
        # clearly labeled, rather than a literal division by zero or an
        # unqualified "infinite" ratio (plan Part L item 55).
        corrected_off_rate = 0.5 / off_n
        result["risk_ratio_on_over_off"] = on_rate / corrected_off_rate
        result["risk_ratio_continuity_corrected"] = True
    else:
        result["risk_ratio_on_over_off"] = None
        result["risk_ratio_continuity_corrected"] = None
    return result


def irq_cadence_table(baselines_by_condition):
    """baselines_by_condition: {condition_name: per_vector_baseline dict
    as produced by analyze_tb4_trace.nhi_cadence_report}. Returns a flat
    comparison table, TX=177/RX=178 (Phase 29/30's confirmed roles on
    this boot -- not re-derived here)."""
    rows = []
    for condition, baseline in baselines_by_condition.items():
        tx = (baseline or {}).get("177", {})
        rx = (baseline or {}).get("178", {})
        rows.append({
            "condition": condition,
            "tx_median_us": tx.get("median_us"), "tx_p95_us": tx.get("p95_us"),
            "rx_median_us": rx.get("median_us"), "rx_p95_us": rx.get("p95_us"),
        })
    return rows


def condition_table(conditions):
    """conditions: {name: samples_us list}. Produces the Part Z item 100
    summary table as a list of row dicts."""
    rows = []
    for name, samples in conditions.items():
        stats = summary_stats(samples)
        modes = mode_fractions(samples)
        thresholds = threshold_fractions(samples)
        rows.append({
            "condition": name, "n": stats["n"], "median_us": stats["median_us"],
            "p95_us": stats["p95_us"], "fraction_300_500us": modes["intermediate"],
            "fraction_ge_800us": thresholds[800], "count_ge_1000us":
                sum(1 for v in samples if v >= 1000) if samples else 0,
            "tail_rate": thresholds[1000],
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--on-samples", type=Path, required=True,
                         help="JSON file: flat list of ON-condition latency samples (us)")
    parser.add_argument("--off-samples", type=Path, nargs="+", required=True,
                         help="one or more JSON files: flat lists of OFF-condition samples, concatenated")
    parser.add_argument("--slow-threshold-us", type=float, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    on_samples = json.loads(args.on_samples.read_text())
    off_samples = []
    for path in args.off_samples:
        off_samples.extend(json.loads(path.read_text()))

    on_events = sum(1 for v in on_samples if v >= args.slow_threshold_us)
    off_events = sum(1 for v in off_samples if v >= args.slow_threshold_us)

    result = {
        "table": condition_table({"training_on": on_samples, "training_off": off_samples}),
        "histogram_on": histogram(on_samples),
        "histogram_off": histogram(off_samples),
        "mode_fractions_on": mode_fractions(on_samples),
        "mode_fractions_off": mode_fractions(off_samples),
        "event_rate_comparison": event_rate_comparison(
            on_events, len(on_samples), off_events, len(off_samples)),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
