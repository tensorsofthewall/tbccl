"""Regression comparison of a candidate measurement against a baseline measurement of the same case.

Method (also in README.md): the unit of replication is the independent lifecycle (process pair, engine or cluster lifecycle, or repetition), never the request or
iteration inside one. A case carries one observation per independent unit; the candidate is compared with the baseline in this order:

  1. validity            invalid or superseded measurements are never compared                                            -> invalid_measurement
  2. comparability       a dimension that defines the experiment differs or is unrecorded                                 -> incompatible
  3. sufficiency         too few independent units for the claimed statistic, or the smallest p-value attainable with these
                         unit counts cannot reach the significance level                                                  -> insufficient_evidence
  4. effect              the difference of the medians must exceed a tolerance derived from the repeat-to-repeat spread
                         of the baseline (or a documented benchmark-specific tolerance) and must be statistically separated
                         by an exact permutation (rank-sum) test, or an exact signed-rank test where the observations are
                         paired (interleaved blocks keep their pairing)                                                     -> improvement | regression
  5. otherwise           the data neither show a change nor show that nothing changed                                       -> indeterminate

No fixed percentage threshold is used. A difference of medians alone is never called significant, and `indeterminate` is not a pass.
"""
from __future__ import annotations

import itertools
import math

from .. import provenance_compare as _pc, statistics as sstats
from . import comparability

CLASSIFICATIONS = ("improvement", "regression", "indeterminate", "incompatible", "insufficient_evidence", "invalid_measurement")
DEFAULT_ALPHA = 0.05
DEFAULT_MIN_UNITS = 3
PERMUTATION_LIMIT = 300_000
TAIL_MINIMUM = {90: 10, 95: 20, 99: 100}  # observations needed inside one unit for a percentile to be reported (as in the campaign reports)


# ---------------------------------------------------------------- exact tests

def _midranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def rank_sum_p(x, y):
    """Two-sided p-value of the Wilcoxon rank-sum test for independent samples: exact (all assignments, mid-ranks for ties) when the number of assignments is at most
    PERMUTATION_LIMIT, otherwise the normal approximation with tie correction. Returns (p, method)."""
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        raise ValueError("both samples need at least one observation")
    pooled = list(x) + list(y)
    ranks = _midranks(pooled)
    observed = sum(ranks[:n])
    expected = n * (n + m + 1) / 2.0
    if math.comb(n + m, n) <= PERMUTATION_LIMIT:
        extreme = total = 0
        threshold = abs(observed - expected) - 1e-9
        for combo in itertools.combinations(range(n + m), n):
            total += 1
            if abs(sum(ranks[i] for i in combo) - expected) >= threshold:
                extreme += 1
        return extreme / total, "exact permutation"
    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    tie = sum(c ** 3 - c for c in counts.values())
    var = n * m / 12.0 * ((n + m + 1) - tie / ((n + m) * (n + m - 1)))
    if var <= 0:
        return 1.0, "normal approximation"
    z = (abs(observed - expected) - 0.5) / math.sqrt(var)
    return math.erfc(max(z, 0.0) / math.sqrt(2.0)), "normal approximation"


def signed_rank_p(differences):
    """Two-sided exact p-value of the Wilcoxon signed-rank test on paired differences (zeros dropped, mid-ranks). Returns (p, method, n_effective)."""
    d = [v for v in differences if v != 0]
    n = len(d)
    if n == 0:
        return 1.0, "exact signed-rank", 0
    ranks2 = [int(round(2 * r)) for r in _midranks([abs(v) for v in d])]  # doubled ranks are integers
    total = sum(ranks2)
    dist = {0: 1}
    for r in ranks2:
        nxt = {}
        for s, c in dist.items():
            nxt[s] = nxt.get(s, 0) + c
            nxt[s + r] = nxt.get(s + r, 0) + c
        dist = nxt
    observed = sum(r for r, v in zip(ranks2, d) if v > 0)
    centre = total / 2.0
    extreme = sum(c for s, c in dist.items() if abs(s - centre) >= abs(observed - centre))
    return min(1.0, extreme / 2 ** n), "exact signed-rank", n


def min_attainable_p(n, m):
    """Smallest two-sided p-value an exact rank-sum test can reach with n and m independent units (no ties)."""
    return min(1.0, 2.0 / math.comb(n + m, n))


def min_attainable_p_paired(pairs):
    return min(1.0, 2.0 / 2 ** pairs) if pairs else 1.0


# ---------------------------------------------------------------- units and validity

def _convert(value, from_unit, to_unit):
    if from_unit == to_unit:
        return value
    return sstats.convert(value, from_unit, to_unit)


def validity_problems(case) -> list:
    problems = []
    status = case["validity"]["status"]
    if status in ("invalid", "superseded"):
        problems.append(f"validity status is {status!r}")
    if case.get("warmup_excluded") is False:
        problems.append("warm-up requests or iterations are included in the measurement")
    name = case["metric_name"].lower()
    if ("tpot" in name or "inter_token" in name) and case.get("output_tokens") is not None and case["output_tokens"] < 2:
        problems.append("time per output token is undefined for one-token outputs")
    if "derived_one_way" in (case.get("timing_scope") or ""):
        problems.append("a one-way latency derived by halving a round trip is not a measurement; only the round trip is")
    obs = case.get("observations")
    if obs is not None and any((not isinstance(v, (int, float))) or not math.isfinite(v) or v < 0 for v in obs):
        problems.append("observations contain NaN, infinity or negative values")
    if case.get("instrumentation") == "traced" and case["validity"]["status"] == "valid" and "pooled" in " ".join(case["validity"].get("caveats", [])):
        problems.append("traced runs are pooled with untraced runs")
    return problems


def _percentile_rank(aggregation):
    if aggregation.startswith("percentile:"):
        try:
            return int(float(aggregation.split(":", 1)[1]))
        except ValueError:
            return None
    return None


def _sufficiency(base, cand, min_units):
    reasons = []
    for label, case in (("baseline", base), ("candidate", cand)):
        units = case.get("independent_lifecycle_count")
        if units is None or units < min_units:
            reasons.append(f"the {label} has {units} independent unit(s); at least {min_units} are required")
        q = _percentile_rank(case["aggregation"])
        if q is not None:
            need = TAIL_MINIMUM.get(q, 0)
            n = case.get("sample_count") or 0
            if n < need:
                reasons.append(f"the {label} reports p{q} from {n} observations; at least {need} are needed")
        if case.get("replication_unit") == "request":
            reasons.append(f"the {label} replicates requests inside one lifecycle, which are not independent units")
    return reasons


# ---------------------------------------------------------------- comparison

def derive_tolerance(observation_sets) -> dict:
    """Noise floor from repeated measurements: the largest relative range (max-min over the median) of the given sets of independent observations."""
    ranges = []
    for obs in observation_sets:
        if obs and len(obs) >= 2:
            v = sstats.repeat_variation(obs)["range_relative"]
            if v is not None:
                ranges.append(v)
    if not ranges:
        return {"relative": None, "basis": "no set with at least two independent observations"}
    return {"relative": max(ranges), "basis": "largest relative range (max-min over median) across independent units of the compared sets"}


def aggregate_by_lifecycle(samples_by_lifecycle: dict, statistic: str = "median") -> dict:
    """One observation per lifecycle from raw per-request samples. Returns {'observations': [...], 'unit_ids': [...], 'sample_count': n}; the median of these is NOT the
    median of all requests pooled, and the number of independent units is the number of lifecycles."""
    if statistic != "median":
        raise ValueError("only the median is supported for per-lifecycle aggregation")
    ids = sorted(samples_by_lifecycle)
    return {"observations": [sstats.median(samples_by_lifecycle[i]) for i in ids if samples_by_lifecycle[i]], "unit_ids": [i for i in ids if samples_by_lifecycle[i]],
            "sample_count": sum(len(v) for v in samples_by_lifecycle.values())}


def _versions(a, b):
    """Software facts both cases recorded that differ: {fact: [baseline, candidate]}. A fact only one side recorded is unknown and is reported by _provenance, not here."""
    va, vb = a.get("software_versions") or {}, b.get("software_versions") or {}
    return {k: [va[k], vb[k]] for k in sorted(set(va) & set(vb)) if va[k] != vb[k]}


def _provenance(a, b):
    """matching / different / unknown facts of the two cases' recorded software identity (see provenance_compare)."""
    return _pc.compare_identity({k: str(v) for k, v in (a.get("software_versions") or {}).items() if v not in (None, "")},
                                {k: str(v) for k, v in (b.get("software_versions") or {}).items() if v not in (None, "")})


def compare(baseline: dict, candidate: dict, purpose: str = "regression", tolerance: float | None = None, alpha: float = DEFAULT_ALPHA,
            min_units: int | None = None) -> dict:
    """Classify `candidate` against `baseline`. `tolerance` is an optional benchmark-specific minimum relative effect (documented per case, derived from repeated
    measurements); without it the baseline's own repeat-to-repeat spread is the noise floor."""
    result = {"case_id": baseline["case_id"], "candidate_case_id": candidate["case_id"], "purpose": purpose, "classification": None, "reasons": [], "labels": [],
              "metric": baseline["metric_name"], "unit": baseline["unit"], "software_version_differences": _versions(baseline, candidate),
              "software_provenance": _provenance(baseline, candidate),
              "corrections": [c for case in (baseline, candidate) for c in case.get("corrections", [])]}

    def done(classification, *reasons, **extra):
        result["classification"] = classification
        result["reasons"] = list(reasons) + result["reasons"]
        result.update(extra)
        return result

    problems = [f"baseline: {p}" for p in validity_problems(baseline)] + [f"candidate: {p}" for p in validity_problems(candidate)]
    if problems:
        return done("invalid_measurement", *problems)
    fit = comparability.check(baseline, candidate, purpose)
    result["labels"] = list(fit["labels"])
    if not result["software_provenance"]["known_build_on_both_sides"]:
        result["labels"].append("build identity not recorded for at least one side: a change cannot be attributed to a build")
    elif result["software_provenance"]["unknown"]:
        result["labels"].append("some build facts were recorded for only one side (unknown, not assumed equal)")
    if not fit["compatible"]:
        return done("incompatible", *[f"{m['dimension']}: {m['baseline']!r} against {m['candidate']!r}" + (" (not recorded)" if m["kind"] == "missing" else "") for m in fit["mismatches"]],
                    *fit["explanations"], mismatches=fit["mismatches"], speedup=None)
    if baseline["direction"] != candidate["direction"]:
        return done("incompatible", "the metric directions differ", speedup=None)
    try:
        scale = lambda v: _convert(v, candidate["unit"], baseline["unit"])  # noqa: E731
        scale(1.0)
    except ValueError:
        return done("incompatible", f"units {baseline['unit']!r} and {candidate['unit']!r} cannot be converted", speedup=None)
    base_obs, cand_obs = list(baseline.get("observations") or []), [scale(v) for v in (candidate.get("observations") or [])]
    needed = min_units or baseline.get("min_independent_units") or DEFAULT_MIN_UNITS
    short = _sufficiency(baseline, candidate, needed)
    if short or not base_obs or not cand_obs:
        return done("insufficient_evidence", *(short or ["a side has no observations"]), baseline_units=len(base_obs), candidate_units=len(cand_obs))
    bmed, cmed = sstats.median(base_obs), sstats.median(cand_obs)
    rel = sstats.relative_difference(bmed, cmed) if bmed else None
    summary = {"baseline_median": bmed, "candidate_median": cmed, "relative_difference": rel, "baseline_range": [min(base_obs), max(base_obs)],
               "candidate_range": [min(cand_obs), max(cand_obs)], "baseline_units": len(base_obs), "candidate_units": len(cand_obs)}
    paired = bool(baseline.get("pair_ids") and candidate.get("pair_ids"))
    pair_noise = None
    if paired:
        bmap, cmap = dict(zip(baseline["pair_ids"], base_obs)), dict(zip(candidate["pair_ids"], cand_obs))
        shared = sorted(set(bmap) & set(cmap))
        if len(shared) < 2:
            paired = False
            result["reasons"].append("fewer than two shared pair ids: the observations are compared unpaired")
        else:
            diffs = [cmap[k] - bmap[k] for k in shared]
            p, method, n_eff = signed_rank_p(diffs)
            floor = min_attainable_p_paired(len(shared))
            rel_pairs = [(cmap[k] - bmap[k]) / bmap[k] for k in shared if bmap[k]]
            summary.update(pairs=len(shared), paired_differences=diffs)
            if rel_pairs:
                rel = sstats.median(rel_pairs)  # the effect is the typical within-pair change: block-to-block differences cancel
                summary["relative_difference"] = rel
                summary["relative_difference_basis"] = "median of within-pair relative differences"
                pair_noise = max(rel_pairs) - min(rel_pairs)
    if not paired:
        p, method = rank_sum_p(base_obs, cand_obs)
        floor = min_attainable_p(len(base_obs), len(cand_obs))
    summary.update(p_value=p, test=("paired " if paired else "") + method, min_attainable_p=floor, alpha=alpha)
    result.update(summary)
    if tolerance is not None:
        noise, source = tolerance, "benchmark-specific"
    elif paired and "relative_difference_basis" in summary:
        noise, source = pair_noise, "spread (max-min) of the within-pair relative differences"
    else:
        noise, source = derive_tolerance([base_obs])["relative"], "baseline repeat-to-repeat spread"
    result["tolerance"] = {"relative": noise, "source": source}
    if floor > alpha:
        return done("insufficient_evidence", f"with {len(base_obs)} baseline and {len(cand_obs)} candidate independent units the smallest attainable two-sided p-value "
                    f"is {floor:.3f}, above the significance level {alpha}")
    if rel is None:
        return done("indeterminate", "the baseline median is zero; a relative difference is undefined")
    separated = p <= alpha
    exceeds = noise is None or abs(rel) > noise
    if separated and exceeds and rel != 0:
        worse = (rel > 0) == (baseline["direction"] == "lower_is_better")
        return done("regression" if worse else "improvement",
                    f"median {'increased' if rel > 0 else 'decreased'} by {abs(rel) * 100:.1f}% (p={p:.4f}, {summary['test']}); exceeds the tolerance "
                    f"{(noise or 0) * 100:.1f}%", speedup=None)
    why = []
    if not separated:
        why.append(f"the observations are not statistically separated (p={p:.3f}, {summary['test']})")
    if not exceeds:
        why.append(f"the difference of {abs(rel) * 100:.1f}% does not exceed the tolerance of {noise * 100:.1f}%")
    return done("indeterminate", *why, "indeterminate is neither a pass nor evidence of no change")


def contextual_view(a: dict, b: dict) -> dict:
    """Both measurements side by side, labelled, for display only: no ratio, difference or speedup is computed."""
    fit = comparability.check(a, b, "regression")
    return {"baseline": {"case_id": a["case_id"], "value": a.get("value"), "unit": a["unit"], "metric": a["metric_name"]},
            "candidate": {"case_id": b["case_id"], "value": b.get("value"), "unit": b["unit"], "metric": b["metric_name"]},
            "comparable": fit["compatible"], "differences": [m["dimension"] for m in fit["mismatches"]], "explanations": fit["explanations"],
            "label": "CONTEXTUAL: shown side by side; not a comparison of equivalent experiments", "ratio": None}


def to_markdown(results, title="Regression comparison") -> str:
    lines = [f"# {title}", "",
             "Classification is based on independent lifecycles, a tolerance derived from the baseline's repeat-to-repeat spread and an exact test; `indeterminate` and "
             "`insufficient_evidence` are not passes.", "", "| case | classification | median change | p | units (base/cand) | note |", "|---|---|---:|---:|---|---|"]
    for r in results:
        rel = f"{r['relative_difference'] * 100:+.1f}%" if r.get("relative_difference") is not None else "-"
        p = f"{r['p_value']:.3f}" if r.get("p_value") is not None else "-"
        units = f"{r.get('baseline_units', '-')}/{r.get('candidate_units', '-')}"
        note = "; ".join(r["reasons"][:2]).replace("|", "/")
        lines.append(f"| {r['case_id']} | {r['classification']} | {rel} | {p} | {units} | {note} |")
    labels = sorted({label for r in results for label in r.get("labels", [])})
    if labels:
        lines += ["", "Labels: " + "; ".join(labels)]
    changed = [r for r in results if r.get("software_version_differences")]
    if changed:
        lines += ["", "## Software version differences", ""] + [f"- {r['case_id']}: " + ", ".join(f"{k}: {v[0]} -> {v[1]}" for k, v in r["software_version_differences"].items()) for r in changed]
    unknown = [r for r in results if r.get("software_provenance", {}).get("unknown")]
    if unknown:
        lines += ["", "## Software facts recorded for only one side (unknown, not assumed equal)", ""] + [f"- {r['case_id']}: " + ", ".join(sorted(r["software_provenance"]["unknown"])) for r in unknown]
    corrected = [r for r in results if r.get("corrections")]
    if corrected:
        lines += ["", "## Later corrections that apply", ""] + [f"- {r['case_id']}: {c['id']}: {c['statement']}" for r in corrected for c in r["corrections"]]
    if not results:
        lines += ["", "No comparisons were made."]
    return "\n".join(lines) + "\n"
