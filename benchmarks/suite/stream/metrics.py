"""Request timelines and the latency definitions used by the vLLM benchmarks (standard library only, so it runs on any host and in offline tests).

A timeline is the raw record of one streamed completion: the client-side monotonic submission time, every output-carrying server-sent event with its monotonic arrival
time, the usage the server reported, and how the request ended. Every figure in a result is derived from these records, never from previously aggregated values.

Definitions (all durations from one host's monotonic clock; timestamps of different hosts are never subtracted):

  output event   a server-sent event that carries at least one generated token. The token count of an event is, in order of preference, the increase of the server's
                 cumulative completion-token count (``continuous_usage_stats``), the length of the event's ``token_ids``, or - flagged as assumed - one when only text arrived.
  output tokens  the server-reported ``usage.completion_tokens`` of the final event, never the configured maximum; the token ids and the event counts are cross-checked.
  TTFT           first output event time - submission time.
  E2E            last output event time - submission time (the stream end is recorded separately as ``complete``).
  TPOT           (last output event time - first output event time) / (output tokens - tokens carried by the first event). With one token in the first event this is the
                 usual (t_last - t_first) / (N - 1). Undefined (None, never zero) when no token follows the first event.
  ITL            the intervals between successive output events. Per-token only while every event carries one token; intervals that end in a multi-token event are kept
                 as ``batched`` intervals and are not counted as per-token ITL.
"""
from __future__ import annotations

import json
import math

TIMELINE_SCHEMA = "tbccl-vllm-request/1"
MIN_SAMPLES = {"p50": 1, "p90": 10, "p95": 20, "p99": 100}  # a percentile is reported only with enough observations to mean something


class StreamError(Exception):
    pass


def parse_sse_line(raw):
    """One line of a server-sent-event stream -> ('data', object) | ('done', None) | ('ignore', None). Malformed data raises StreamError."""
    line = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else raw
    line = line.strip()
    if not line or line.startswith(":"):
        return "ignore", None
    if not line.startswith("data:"):
        return "ignore", None
    payload = line[5:].strip()
    if payload == "[DONE]":
        return "done", None
    try:
        return "data", json.loads(payload)
    except ValueError:
        raise StreamError(f"malformed event payload: {payload[:80]!r}")


def event_content(obj, previous_completion_tokens):
    """What one decoded event adds: (new_tokens, token_ids, completion_tokens_so_far, assumed, finish_reason, usage).

    ``new_tokens`` is None when the event carries no generated token (a usage-only or finish-only event)."""
    usage = obj.get("usage") or None
    cum = usage.get("completion_tokens") if usage else None
    choices = obj.get("choices") or []
    ids, text, finish = None, "", None
    if choices:
        ch = choices[0]
        ids = ch.get("token_ids")
        text = ch.get("text") or ""
        finish = ch.get("finish_reason")
    new, assumed = None, False
    if cum is not None and previous_completion_tokens is not None:
        new = cum - previous_completion_tokens
    elif cum is not None and ids is not None:
        new = len(ids)
    elif ids is not None:
        new = len(ids)
    elif text:
        new, assumed = 1, True
    if new is not None and new <= 0:
        new = None
    return new, ids, cum, assumed, finish, usage


class Timeline:
    """Accumulates one request's stream. ``now`` timestamps are integers in nanoseconds from the caller's monotonic clock."""

    def __init__(self, meta, t_submit_ns):
        self.meta = dict(meta)
        self.t_submit_ns = t_submit_ns
        self.t_response_ns = None
        self.events = []  # [t_ns, new_tokens, cumulative completion tokens or None]
        self.token_ids = []
        self.assumed = False
        self.usage = None
        self.finish_reason = None
        self.t_done_ns = None
        self.t_end_ns = None
        self.status = "partial"
        self.error = None
        self.http_status = None
        self._cum = 0  # completion tokens seen so far (the server's cumulative count when it reports one)
        self.events_without_tokens = 0

    def on_event(self, kind, obj, t_ns):
        if kind == "done":
            self.t_done_ns = t_ns
            return
        new, ids, cum, assumed, finish, usage = event_content(obj, self._cum)
        if usage and usage.get("completion_tokens") is not None:
            self.usage = {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
        if finish:
            self.finish_reason = finish
        if cum is not None:
            self._cum = cum
        if new is None:
            self.events_without_tokens += 1
            return
        if ids:
            self.token_ids.extend(ids)
        self.assumed |= assumed
        if cum is None:
            self._cum += new
        self.events.append([t_ns, new, cum])

    def finish(self, status, t_ns, error=None):
        self.status, self.t_end_ns, self.error = status, t_ns, error

    def to_record(self):
        rec = {"schema": TIMELINE_SCHEMA, **self.meta, "t_submit_ns": self.t_submit_ns, "t_response_ns": self.t_response_ns, "events": self.events,
               "usage": self.usage, "finish_reason": self.finish_reason, "t_done_ns": self.t_done_ns, "t_end_ns": self.t_end_ns, "status": self.status,
               "error": self.error, "http_status": self.http_status, "token_count_assumed": self.assumed, "events_without_tokens": self.events_without_tokens}
        rec["token_ids"] = self.token_ids or None
        return rec


def derive(rec):
    """Per-request figures (seconds) from a raw timeline record. Fields that are undefined for this request are None, never zero."""
    out = {"ok": rec.get("status") == "ok", "output_tokens": None, "token_source": None, "ttft_s": None, "tpot_s": None, "e2e_s": None, "complete_s": None,
           "itl_s": [], "batched_intervals": 0, "event_count": len(rec.get("events") or []), "problems": []}
    events = rec.get("events") or []
    t0 = rec["t_submit_ns"]
    usage = rec.get("usage") or {}
    summed = sum(e[1] for e in events) if events else 0
    ids = rec.get("token_ids")
    if usage.get("completion_tokens") is not None:
        out["output_tokens"], out["token_source"] = usage["completion_tokens"], "usage"
        if ids is not None and len(ids) != usage["completion_tokens"]:
            out["problems"].append(f"token_ids ({len(ids)}) disagree with usage ({usage['completion_tokens']})")
        if events and summed != usage["completion_tokens"]:
            out["problems"].append(f"event token counts ({summed}) disagree with usage ({usage['completion_tokens']})")
    elif ids is not None:
        out["output_tokens"], out["token_source"] = len(ids), "token_ids"
    elif events and not rec.get("token_count_assumed"):
        out["output_tokens"], out["token_source"] = summed, "events"
    elif events:
        out["output_tokens"], out["token_source"] = summed, "assumed_one_per_event"
        out["problems"].append("token counts assumed from text events")
    if rec.get("t_done_ns") is not None:
        out["complete_s"] = (rec["t_done_ns"] - t0) / 1e9
    if not events:
        return out
    out["ttft_s"] = (events[0][0] - t0) / 1e9
    out["e2e_s"] = (events[-1][0] - t0) / 1e9
    n_total = out["output_tokens"]
    if n_total is not None and len(events) >= 1:
        later = n_total - events[0][1]
        if later >= 1 and len(events) >= 2:
            out["tpot_s"] = (events[-1][0] - events[0][0]) / 1e9 / later
    for prev, cur in zip(events, events[1:]):
        dt = (cur[0] - prev[0]) / 1e9
        if cur[1] == 1:
            out["itl_s"].append(dt)
        else:
            out["batched_intervals"] += 1
    return out


def percentile(sorted_values, f):
    n = len(sorted_values)
    if n == 0:
        return None
    pos = f * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def summarize(values):
    """Distribution summary of raw observations. Percentiles with too few observations are None with the reason recorded, so no plot or claim can quote them."""
    v = sorted(x for x in values if x is not None)
    n = len(v)
    out = {"n": n, "min": v[0] if n else None, "max": v[-1] if n else None, "mean": (sum(v) / n) if n else None, "withheld": {}}
    for name, f in (("p50", .5), ("p90", .9), ("p95", .95), ("p99", .99)):
        if n >= MIN_SAMPLES[name]:
            out[name] = percentile(v, f)
        else:
            out[name] = None
            out["withheld"][name] = f"needs at least {MIN_SAMPLES[name]} observations, have {n}"
    return out


def phase_throughput(records, wall_s=None):
    """Aggregate rates of one phase from its raw timelines: successful requests/s, token rates, error and timeout counts. ``wall_s`` defaults to first submission -> last end."""
    ok = [r for r in records if r.get("status") == "ok"]
    starts = [r["t_submit_ns"] for r in records]
    ends = [r["t_end_ns"] for r in records if r.get("t_end_ns") is not None]
    if wall_s is None and starts and ends:
        wall_s = (max(ends) - min(starts)) / 1e9
    derived = [derive(r) for r in ok]
    out_tokens = sum(d["output_tokens"] or 0 for d in derived)
    in_tokens = sum(((r.get("usage") or {}).get("prompt_tokens") or r.get("prompt_tokens_planned") or 0) for r in ok)
    counts = {}
    for r in records:
        counts[r.get("status")] = counts.get(r.get("status"), 0) + 1
    rates = None
    if wall_s and wall_s > 0:
        rates = {"requests_per_s": len(ok) / wall_s, "input_tokens_per_s": in_tokens / wall_s, "output_tokens_per_s": out_tokens / wall_s,
                 "total_tokens_per_s": (in_tokens + out_tokens) / wall_s}
    total = len(records)
    return {"wall_s": wall_s, "requests": total, "successful": len(ok), "status_counts": counts, "error_rate": (total - len(ok)) / total if total else None,
            "timeouts": counts.get("timeout", 0), "cancellations": counts.get("cancelled", 0), "input_tokens": in_tokens, "output_tokens": out_tokens, "rates": rates}


def request_distributions(records):
    """TTFT, TPOT, E2E, ITL pooled over the successful requests' raw observations (seconds)."""
    ttft, tpot, e2e, itl, ntok = [], [], [], [], []
    undefined_tpot = 0
    batched = 0
    for r in records:
        if r.get("status") != "ok":
            continue
        d = derive(r)
        if d["ttft_s"] is not None:
            ttft.append(d["ttft_s"])
        if d["e2e_s"] is not None:
            e2e.append(d["e2e_s"])
        if d["tpot_s"] is not None:
            tpot.append(d["tpot_s"])
        else:
            undefined_tpot += 1
        itl.extend(d["itl_s"])
        batched += d["batched_intervals"]
        if d["output_tokens"] is not None:
            ntok.append(d["output_tokens"])
    return {"ttft_s": summarize(ttft), "tpot_s": summarize(tpot), "e2e_s": summarize(e2e), "itl_s": summarize(itl), "output_tokens": summarize(ntok),
            "tpot_undefined_requests": undefined_tpot, "batched_intervals": batched}
