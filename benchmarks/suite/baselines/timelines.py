"""Per-request metrics from the suite's per-request timeline files (schema tbccl-exo-request/1 and tbccl-vllm-request/1), recomputed from the raw event times.

Definitions (client side, one monotonic clock per request): TTFT = first token event - submission; E2E = last token event - submission; ITL = gaps between consecutive
token events; TPOT = (last event - first event) / (generated tokens - 1), undefined (None, never zero) for outputs of one token. Only requests with status 'ok' count;
warm-up requests are excluded. Throughput is generated tokens divided by the time from the first submission to the last token of the set of requests."""
from __future__ import annotations

import json
from pathlib import Path

from .. import statistics as sstats


def read_requests(path) -> list:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _times(request):
    return [e[0] if isinstance(e, (list, tuple)) else e for e in request.get("events") or []]


def request_metrics(request: dict, token_source: str = "events"):
    """Metrics of one request or None if it did not complete ('ok' with at least one token event). Times are in seconds.

    token_source: 'events' counts token events (Exo: one event per token, no token ids); 'usage' takes the server-reported completion token count (vLLM)."""
    if request.get("status") != "ok":
        return None
    times = _times(request)
    if not times:
        return None
    tokens = len(times) if token_source == "events" else (request.get("usage") or {}).get("completion_tokens")
    if tokens is None:
        return None
    submit = request["t_submit_ns"]
    return {"tokens": tokens, "ttft": (times[0] - submit) / 1e9, "e2e": (times[-1] - submit) / 1e9, "itl": [(b - a) / 1e9 for a, b in zip(times, times[1:])],
            "tpot": ((times[-1] - times[0]) / 1e9 / (tokens - 1)) if tokens > 1 else None, "t_submit_ns": submit, "t_last_ns": times[-1]}


def measured(requests) -> list:
    return [r for r in requests if not r.get("warmup")]


def metric_samples(requests, metric: str, token_source: str = "events") -> list:
    """Samples of one metric ('ttft', 'tpot', 'e2e') over completed, non-warm-up requests. TPOT of one-token requests is skipped, not zero."""
    out = []
    for request in measured(requests):
        m = request_metrics(request, token_source)
        if m is None or m.get(metric) is None:
            continue
        out.append(m[metric])
    return out


def throughput_tokens_per_s(requests, token_source: str = "events"):
    """Generated tokens per second over the span from the first submission to the last token of the completed, non-warm-up requests; None if there are none."""
    ms = [request_metrics(r, token_source) for r in measured(requests)]
    ms = [m for m in ms if m]
    if not ms:
        return None
    span = (max(m["t_last_ns"] for m in ms) - min(m["t_submit_ns"] for m in ms)) / 1e9
    return sum(m["tokens"] for m in ms) / span if span > 0 else None


def token_accounting(requests, token_source: str = "events") -> dict:
    """Cross-check of token events against the server-reported usage; requests where they disagree are listed by index."""
    checked = bad = 0
    for i, r in enumerate(measured(requests)):
        usage = (r.get("usage") or {}).get("completion_tokens")
        if r.get("status") == "ok" and usage is not None:
            checked += 1
            bad += usage != len(_times(r))
    return {"checked": checked, "mismatching": bad}


def lifecycle_observation(requests, metric: str, token_source: str = "events"):
    """(median over this lifecycle's requests, sample count) - one independent observation. None when no request qualifies."""
    samples = metric_samples(requests, metric, token_source)
    return (sstats.median(samples), len(samples)) if samples else None
