"""Shared statistics with explicit definitions.

Percentiles use linear interpolation between order statistics at fractional rank q*(n-1) (the NumPy default, "type 7"). The native tools disagree on this
(small_message_latency uses floor(q*n)); results carry the method that produced them and this module never converts one to the other.

Aggregation rules enforced here:
  * a percentile is computed only from raw samples, never from medians or other percentiles;
  * per-rank aggregates are never merged into a global percentile;
  * ranks' raw samples are combined only iteration-by-iteration (maximum for one collective), and only when every rank has the same number of samples.
"""
from __future__ import annotations

import math
import statistics as _stats

PERCENTILE_LINEAR = "linear-interpolation-n-1"
PERCENTILE_FLOOR_RANK = "floor-nearest-rank"
MIN_TAIL_SAMPLES = 5


class AggregationError(ValueError):
    """Raised when asked to combine data in a way that would produce a meaningless number."""


def _clean(samples):
    values = [float(v) for v in samples]
    if any(math.isnan(v) or math.isinf(v) for v in values):
        raise ValueError("samples contain NaN or infinity")
    return sorted(values)


def percentile(samples, q: float) -> float:
    """q is a fraction in [0, 1]."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be a fraction between 0 and 1")
    values = _clean(samples)
    if not values:
        raise ValueError("no samples")
    index = q * (len(values) - 1)
    lo = int(math.floor(index))
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (index - lo) * (values[hi] - values[lo])


def median(samples) -> float:
    return percentile(samples, 0.5)


def stdev(samples) -> float | None:
    values = _clean(samples)
    return _stats.stdev(values) if len(values) > 1 else None


def tail_warning(sample_count: int, q: float) -> str | None:
    """A tail percentile needs enough samples beyond it; fewer than MIN_TAIL_SAMPLES in the tail makes it an extreme-value estimate."""
    tail = sample_count * (1.0 - q)
    if tail < MIN_TAIL_SAMPLES:
        return (f"p{q * 100:g} from {sample_count} samples has only {tail:.1f} samples above it; "
                f"at least {math.ceil(MIN_TAIL_SAMPLES / (1.0 - q))} are needed for a meaningful estimate")
    return None


def summarize(samples, quantiles=(0.95, 0.99)) -> dict:
    """Statistics of raw samples. Returns values plus warnings; every value here is 'computed' provenance."""
    values = _clean(samples)
    if not values:
        raise ValueError("no samples")
    out = {"n": len(values), "min": values[0], "max": values[-1], "mean": _stats.fmean(values), "median": median(values)}
    sd = stdev(values)
    if sd is not None:
        out["stdev"] = sd
    warnings = []
    for q in quantiles:
        out[f"p{q * 100:g}"] = percentile(values, q)
        warning = tail_warning(len(values), q)
        if warning:
            warnings.append(warning)
    out["warnings"] = warnings
    return out


def repeat_variation(values) -> dict:
    """Spread of one per-repetition value (for example each repetition's median) across independent repetitions."""
    vals = _clean(values)
    if not vals:
        raise ValueError("no values")
    mid = median(vals)
    sd = stdev(vals)
    out = {"repetitions": len(vals), "median": mid, "min": vals[0], "max": vals[-1],
           "range_relative": (vals[-1] - vals[0]) / mid if mid else None}
    if sd is not None:
        out["stdev"] = sd
        out["cv"] = sd / _stats.fmean(vals) if _stats.fmean(vals) else None
    return out


def relative_difference(baseline: float, candidate: float) -> float:
    """(candidate - baseline) / baseline. For latency a negative value is an improvement."""
    if baseline == 0:
        raise ZeroDivisionError("baseline is zero")
    return (candidate - baseline) / baseline


relative_latency_difference = relative_difference
relative_throughput_difference = relative_difference


def per_iteration_max(per_rank_samples: dict) -> list:
    """Duration of one collective as seen by the slowest rank, iteration by iteration. Valid only for durations measured with each rank's own monotonic clock."""
    lengths = {len(v) for v in per_rank_samples.values()}
    if len(lengths) != 1 or not lengths.copy().pop():
        raise AggregationError("per-iteration maximum needs every rank to have the same, non-zero number of samples")
    return [max(column) for column in zip(*per_rank_samples.values())]


def pool_samples(per_rank_samples: dict) -> list:
    """Concatenate ranks' raw samples into one population (for example both directions of a symmetric exchange)."""
    pooled = []
    for values in per_rank_samples.values():
        pooled.extend(values)
    if not pooled:
        raise AggregationError("no samples to pool")
    return pooled


def combine_aggregates(per_rank_aggregates: dict, name: str):
    """Refuses to build a global percentile (or median) from per-rank aggregates; that quantity is not defined by them."""
    raise AggregationError(f"cannot derive a global '{name}' from per-rank aggregates; keep them per rank or supply the raw samples")


_TIME = {"s": 1e6, "ms": 1e3, "us": 1.0, "ns": 1e-3}
_RATE_BYTES = {"B/s": 1.0, "KB/s": 1e3, "MB/s": 1e6, "GB/s": 1e9, "KiB/s": 1024.0, "MiB/s": 1024.0 ** 2, "GiB/s": 1024.0 ** 3}
_RATE_BITS = {"bit/s": 1.0, "Kbit/s": 1e3, "Mbit/s": 1e6, "Gbit/s": 1e9}


def convert(value: float, from_unit: str, to_unit: str) -> float:
    """Unit conversion for times and rates. Decimal (GB/s) and binary (GiB/s) units are distinct; bits and bytes differ by a factor of 8."""
    if from_unit == to_unit:
        return value
    if from_unit in _TIME and to_unit in _TIME:
        return value * _TIME[from_unit] / _TIME[to_unit]
    bits_per_second = None
    if from_unit in _RATE_BYTES:
        bits_per_second = value * _RATE_BYTES[from_unit] * 8.0
    elif from_unit in _RATE_BITS:
        bits_per_second = value * _RATE_BITS[from_unit]
    if bits_per_second is not None:
        if to_unit in _RATE_BYTES:
            return bits_per_second / 8.0 / _RATE_BYTES[to_unit]
        if to_unit in _RATE_BITS:
            return bits_per_second / _RATE_BITS[to_unit]
    raise ValueError(f"cannot convert {from_unit!r} to {to_unit!r}")
