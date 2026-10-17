"""Request timelines for Exo's chat-completions stream (standard library only, so it runs on any host and in offline tests).

Exo streams one server-sent event per generated token (a ``TokenChunk``): ``choices[0].delta.content`` holds the decoded text of that token, which can be an empty string,
``choices[0].logprobs.content`` (only when logprobs were requested) names the token, ``choices[0].finish_reason`` is set on the last token's event and ``usage`` is attached
to that event. Exo exposes no token ids. The definitions of TTFT, TPOT, ITL and E2E are those of ``vllm_bench.metrics`` (one source of truth for every framework benchmark);
this module only decides what an Exo event contributes:

  token event    an event with ``logprobs.content`` entries (each entry is one token), or - when logprobs were not requested - an event whose delta carries a ``content`` (or
                 ``reasoning_content``) field, even an empty one: Exo sends one event per generated token and a token can decode to an empty string (observed: 14 text events for 16
                 reported completion tokens). Without logprobs the count is therefore an assumption (flagged), and the cross-check against the reported usage reports any difference.
  output tokens  the server-reported ``usage.completion_tokens`` of the final event, never the configured maximum.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _shared():
    """The vLLM benchmark's metrics module, loaded by path under its own name: this module also runs as a flat script module next to the client on a remote host, where a package import is not available."""
    path = Path(__file__).resolve().parent.parent / "vllm_bench" / "metrics.py"
    spec = importlib.util.spec_from_file_location("_tbccl_stream_metrics", path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_m = _shared()
StreamError, Timeline, derive, parse_sse_line, phase_throughput, request_distributions, summarize = (
    _m.StreamError, _m.Timeline, _m.derive, _m.parse_sse_line, _m.phase_throughput, _m.request_distributions, _m.summarize)

TIMELINE_SCHEMA = "tbccl-exo-request/1"


def event_content(obj):
    """What one decoded event adds: (new_tokens, token_texts, margins, assumed, finish_reason, usage, error). ``new_tokens`` is None when the event carries no visible token."""
    usage = obj.get("usage") or None
    error = obj.get("error") or None
    choices = obj.get("choices") or []
    texts, margins, text, finish, assumed, new, has_field = None, None, "", None, False, None, False
    if choices:
        ch = choices[0]
        delta = ch.get("delta") or {}
        text = delta.get("content") or delta.get("reasoning_content") or ""
        has_field = "content" in delta or "reasoning_content" in delta
        finish = ch.get("finish_reason")
        lp = (ch.get("logprobs") or {}).get("content") or []
        if lp:
            new = len(lp)
            texts = [item.get("token") for item in lp]
            margins = [_margin(item.get("top_logprobs")) for item in lp]
        elif has_field:
            new, assumed = 1, True
            texts = [text]
    return new, texts, margins, assumed, finish, usage, error


def _margin(top):
    """Log-probability gap between the best and the second-best candidate of one position (None without two candidates)."""
    vals = sorted((t.get("logprob") for t in top or [] if t.get("logprob") is not None), reverse=True)
    return vals[0] - vals[1] if len(vals) >= 2 else None


class ExoTimeline(Timeline):
    """One request's stream. Timestamps are integer nanoseconds of the caller's monotonic clock (``time.monotonic_ns``)."""

    def __init__(self, meta, t_submit_ns):
        super().__init__(meta, t_submit_ns)
        self.token_texts = []
        self.margins = []
        self.stream_error = None

    def on_event(self, kind, obj, t_ns):
        if kind == "done":
            self.t_done_ns = t_ns
            return
        new, texts, margins, assumed, finish, usage, error = event_content(obj)
        if error:
            self.stream_error = error
        if usage and usage.get("completion_tokens") is not None:
            self.usage = {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")}
        if finish:
            self.finish_reason = finish
        if new is None:
            self.events_without_tokens += 1
            return
        self.assumed |= assumed
        self._cum += new
        self.token_texts.extend(texts or [])
        self.margins.extend(margins or [])
        self.events.append([t_ns, new, None])

    def to_record(self):
        rec = super().to_record()
        rec["schema"] = TIMELINE_SCHEMA
        rec["token_texts"] = self.token_texts or None
        rec["margins"] = self.margins or None
        rec["stream_error"] = self.stream_error
        return rec


def accounting(rec):
    """Token bookkeeping of one raw record: the figures that must agree (events, usage, configured limit) and what they say about the stream."""
    d = derive(rec)
    usage_tokens = (rec.get("usage") or {}).get("completion_tokens")
    events = rec.get("events") or []
    seen = sum(e[1] for e in events)
    limit = rec.get("max_tokens")
    problems = list(d["problems"])
    if rec.get("status") == "ok":
        if rec.get("finish_reason") is None:
            problems.append("missing finish_reason")
        if usage_tokens is None:
            problems.append("missing usage")
        elif seen != usage_tokens:
            problems.append(f"token events ({seen}) differ from usage ({usage_tokens})")
        if rec.get("t_done_ns") is None:
            problems.append("stream ended without [DONE]")
        if rec.get("stream_error"):
            problems.append("error event in the stream")
        if rec.get("finish_reason") == "length" and limit is not None and usage_tokens is not None and usage_tokens != limit:
            problems.append(f"finish_reason length with {usage_tokens} tokens, limit {limit}")
    return {"output_tokens": d["output_tokens"], "token_events": seen, "usage_tokens": usage_tokens, "max_tokens": limit, "finish_reason": rec.get("finish_reason"),
            "multi_token_events": sum(1 for e in events if e[1] != 1), "problems": problems}


def compare_tokens(tokens, reference, margins, tie_margin):
    """Classify a generated token sequence against a greedy reference (token texts: Exo exposes no ids).

    ``exact``       identical sequences.
    ``near_tie``    the first difference is at a position where the reference's own top-1/top-2 log-probability margin is below ``tie_margin``: a different but equally valid greedy
                    choice, not corruption. The first differing index and that margin are kept.
    ``mismatch``    anything else: a divergence at a position where the reference was confident, an unknown margin, or a truncated/extended sequence with identical shared prefix.
    """
    if tokens is None or reference is None:
        return {"outcome": "mismatch", "reason": "missing token texts"}
    n = min(len(tokens), len(reference))
    for i in range(n):
        if tokens[i] != reference[i]:
            margin = margins[i] if margins and i < len(margins) else None
            if margin is not None and margin < tie_margin:
                return {"outcome": "near_tie", "first_difference": i, "reference_margin": margin, "reference_token": reference[i], "token": tokens[i]}
            return {"outcome": "mismatch", "first_difference": i, "reference_margin": margin, "reference_token": reference[i], "token": tokens[i], "reason": "divergence at a confident position"}
    if len(tokens) != len(reference):
        return {"outcome": "mismatch", "reason": f"length {len(tokens)} differs from the reference length {len(reference)} with an identical shared prefix"}
    return {"outcome": "exact"}


def check_summary(outcomes):
    """Counts of outcomes of one check phase and whether it passed (no mismatch, no failed request)."""
    counts = {}
    for o in outcomes:
        counts[o["outcome"]] = counts.get(o["outcome"], 0) + 1
    return {"passed": counts.get("mismatch", 0) == 0 and counts.get("failed", 0) == 0 and bool(outcomes), "outcome": counts}
