"""Latency definitions, SSE parsing and token accounting of the vLLM benchmark client (offline: no vLLM, no network)."""
import json
import unittest

from benchmarks.suite.tests.common import approx
from benchmarks.suite.stream import metrics


def sse(obj):
    return ("data: " + json.dumps(obj) + "\n").encode()


def token_event(n_cum, ids, finish=None, prompt=16):
    return {"choices": [{"index": 0, "text": "x", "token_ids": ids, "finish_reason": finish}], "usage": {"prompt_tokens": prompt, "completion_tokens": n_cum, "total_tokens": prompt + n_cum}}


def timeline(events, submit=0, status="ok", usage=True, done=True, ids=True):
    """events: [(t_ms, cumulative tokens, [ids])]. Feeds a Timeline the way the client does."""
    tl = metrics.Timeline({"phase": "t", "request_index": 0}, submit)
    for t_ms, cum, tok in events:
        obj = token_event(cum, tok if ids else None)
        if not usage:
            obj.pop("usage")
        kind, parsed = metrics.parse_sse_line(sse(obj))
        tl.on_event(kind, parsed, int(t_ms * 1e6))
    if usage:  # the final usage-only event
        total = events[-1][1] if events else 0
        tl.on_event("data", {"choices": [], "usage": {"prompt_tokens": 16, "completion_tokens": total}}, int(events[-1][0] * 1e6) + 1 if events else 1)
    if done:
        tl.on_event("done", None, int((events[-1][0] if events else 0) * 1e6) + 2)
    tl.finish(status, int(((events[-1][0] if events else 0) + 1) * 1e6))
    return tl.to_record()


class Metrics(unittest.TestCase):
    def test_parse_sse_line_kinds(self):
        assert metrics.parse_sse_line(b"\n") == ("ignore", None)
        assert metrics.parse_sse_line(b": keep-alive\n") == ("ignore", None)
        assert metrics.parse_sse_line(b"event: x\n") == ("ignore", None)
        assert metrics.parse_sse_line(b"data: [DONE]\n") == ("done", None)
        kind, obj = metrics.parse_sse_line(sse({"a": 1}))
        assert kind == "data" and obj == {"a": 1}
        with self.assertRaises(metrics.StreamError):
            metrics.parse_sse_line(b"data: {not json\n")


    def test_event_content_prefers_the_cumulative_usage(self):
        new, ids, cum, assumed, finish, usage = metrics.event_content(token_event(3, [7, 8], finish="length"), 1)
        assert (new, ids, cum, assumed, finish) == (2, [7, 8], 3, False, "length") and usage["completion_tokens"] == 3
        # no usage: the token ids give the count; text alone is an assumption and says so
        obj = {"choices": [{"text": "a", "token_ids": [5]}]}
        assert metrics.event_content(obj, 0)[0] == 1
        new, _, _, assumed, _, _ = metrics.event_content({"choices": [{"text": "a"}]}, 0)
        assert (new, assumed) == (1, True)
        # a usage-only event or an empty finish event carries no token
        assert metrics.event_content({"choices": [], "usage": {"completion_tokens": 3}}, 3)[0] is None
        assert metrics.event_content({"choices": [{"text": "", "finish_reason": "stop"}]}, 0)[0] is None


    def test_ttft_tpot_itl_e2e_one_token_per_event(self):
        rec = timeline([(100, 1, [1]), (110, 2, [2]), (122, 3, [3]), (131, 4, [4])])
        d = metrics.derive(rec)
        assert d["output_tokens"] == 4 and d["token_source"] == "usage"
        assert d["ttft_s"] == approx(0.100)
        assert d["e2e_s"] == approx(0.131)
        assert d["tpot_s"] == approx((0.131 - 0.100) / 3)
        assert d["itl_s"] == approx([0.010, 0.012, 0.009])
        assert d["batched_intervals"] == 0 and d["problems"] == []


    def test_one_token_completion_has_ttft_but_no_tpot_or_itl(self):
        d = metrics.derive(timeline([(80, 1, [9])]))
        assert d["ttft_s"] == approx(0.080) and d["e2e_s"] == approx(0.080)
        assert d["tpot_s"] is None and d["itl_s"] == []  # undefined, never zero


    def test_multi_token_events_are_not_turned_into_per_token_timestamps(self):
        # the second event carries three tokens: its interval is a batched interval, not three ITL samples
        d = metrics.derive(timeline([(100, 1, [1]), (130, 4, [2, 3, 4]), (140, 5, [5])]))
        assert d["output_tokens"] == 5 and d["event_count"] == 3
        assert d["batched_intervals"] == 1 and d["itl_s"] == approx([0.010])
        assert d["tpot_s"] == approx((0.140 - 0.100) / 4)  # (t_last - t_first) / (tokens after the first event)


    def test_first_event_with_several_tokens_changes_the_tpot_divisor(self):
        d = metrics.derive(timeline([(100, 3, [1, 2, 3]), (112, 4, [4])]))
        assert d["tpot_s"] == approx(0.012 / 1)  # one token follows the first event
        assert metrics.derive(timeline([(100, 3, [1, 2, 3])]))["tpot_s"] is None  # nothing follows it


    def test_actual_tokens_come_from_usage_not_from_the_configured_maximum(self):
        rec = timeline([(10, 1, [1]), (20, 2, [2])])
        rec["max_tokens"] = 64
        assert metrics.derive(rec)["output_tokens"] == 2


    def test_missing_usage_falls_back_to_token_ids_then_events_with_a_flag(self):
        rec = timeline([(10, 1, [1]), (20, 2, [2])], usage=False)
        d = metrics.derive(rec)
        assert d["output_tokens"] == 2 and d["token_source"] == "token_ids"
        rec = timeline([(10, 1, None), (20, 2, None)], usage=False, ids=False)
        rec["token_count_assumed"] = True
        d = metrics.derive(rec)
        assert d["token_source"] == "assumed_one_per_event" and d["problems"]


    def test_disagreeing_counts_are_reported(self):
        rec = timeline([(10, 1, [1]), (20, 2, [2])])
        rec["usage"]["completion_tokens"] = 3
        problems = metrics.derive(rec)["problems"]
        assert any("disagree" in p for p in problems)


    def test_failed_and_partial_requests_have_no_latency_when_nothing_arrived(self):
        tl = metrics.Timeline({"phase": "t"}, 5)
        tl.finish("error", 9, "HTTP 500")
        d = metrics.derive(tl.to_record())
        assert d["ok"] is False and d["ttft_s"] is None and d["tpot_s"] is None
        partial = timeline([(10, 1, [1])], status="partial", done=False)
        assert metrics.derive(partial)["ok"] is False


    def test_summarize_withholds_percentiles_without_enough_observations(self):
        s = metrics.summarize([float(i) for i in range(1, 11)])
        assert s["n"] == 10 and s["p50"] == approx(5.5) and s["p90"] is not None
        assert s["p95"] is None and s["p99"] is None
        assert "p99" in s["withheld"] and "100" in s["withheld"]["p99"]
        big = metrics.summarize(list(range(200)))
        assert big["p99"] is not None and big["withheld"] == {}
        empty = metrics.summarize([])
        assert empty["n"] == 0 and empty["p50"] is None


    def test_distributions_are_pooled_from_raw_observations_not_from_medians(self):
        a = timeline([(100, 1, [1]), (110, 2, [2])])
        b = timeline([(300, 1, [1]), (330, 2, [2])])
        d = metrics.request_distributions([a, b])
        assert d["ttft_s"]["n"] == 2 and d["itl_s"]["n"] == 2
        assert d["ttft_s"]["mean"] == approx(0.2)
        one = timeline([(50, 1, [1])])
        assert metrics.request_distributions([one])["tpot_undefined_requests"] == 1


    def test_phase_throughput_counts_only_successful_requests_and_actual_tokens(self):
        ok1 = timeline([(100, 1, [1]), (200, 2, [2])], submit=0)
        ok2 = timeline([(150, 1, [1]), (250, 2, [2]), (300, 3, [3])], submit=0)
        bad = metrics.Timeline({"phase": "t"}, 0)
        bad.finish("timeout", int(400e6), "timed out")
        t = metrics.phase_throughput([ok1, ok2, bad.to_record()])
        assert t["successful"] == 2 and t["requests"] == 3 and t["timeouts"] == 1
        assert t["output_tokens"] == 5
        assert t["error_rate"] == approx(1 / 3)
        assert t["wall_s"] == approx(0.4) and t["rates"]["output_tokens_per_s"] == approx(12.5)
        assert t["rates"]["requests_per_s"] == approx(5.0)


if __name__ == "__main__":
    unittest.main()
