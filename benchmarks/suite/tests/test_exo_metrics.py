"""Exo stream parsing, token accounting and latency definitions (offline: no Exo, no network). The fixtures are real streams captured from a single-node Exo v2 instance."""
import json
import unittest

from benchmarks.suite.stream import exo_metrics as metrics
from benchmarks.suite.tests.common import FIXTURES, approx

EXO = FIXTURES / "exo"


def feed(lines, step_ns=10_000_000, submit=0, first_extra_ns=0):
    """Feed raw SSE lines to a timeline the way the client does; event i arrives at submit + first_extra + (i+1) * step."""
    tl = metrics.ExoTimeline({"phase": "t", "index": 0, "max_tokens": 6}, submit)
    n = 0
    for raw in lines:
        kind, obj = metrics.parse_sse_line(raw)
        if kind == "ignore":
            continue
        n += 1
        tl.on_event(kind, obj, submit + first_extra_ns + n * step_ns)
    return tl


def fixture(name):
    return (EXO / name).read_text().splitlines()


def data(delta="x", finish=None, usage=None, logprobs=True):
    ch = {"index": 0, "delta": {"role": "assistant", "content": delta}}
    if logprobs:
        ch["logprobs"] = {"content": [{"token": delta, "logprob": -0.5, "top_logprobs": [{"token": delta, "logprob": -0.5}, {"token": "y", "logprob": -2.0}]}]}
    if finish:
        ch["finish_reason"] = finish
    obj = {"choices": [ch]}
    if usage:
        obj["usage"] = usage
    return "data: " + json.dumps(obj)


class RealStreams(unittest.TestCase):
    def test_logprob_stream_is_one_token_per_event(self):
        tl = feed(fixture("chat_stream_logprobs.sse"))
        rec = tl.to_record()
        self.assertEqual(len(rec["events"]), 6)
        self.assertTrue(all(e[1] == 1 for e in rec["events"]))
        self.assertEqual(rec["finish_reason"], "length")
        self.assertEqual(rec["usage"]["completion_tokens"], 6)
        self.assertEqual(rec["token_texts"], ["One", ",", " two", ",", " three", ","])
        self.assertEqual(len(rec["margins"]), 6)
        self.assertIsNotNone(rec["t_done_ns"])
        self.assertFalse(rec["token_count_assumed"])

    def test_plain_stream_counts_text_events_and_flags_the_assumption(self):
        tl = feed(fixture("chat_stream_plain.sse"))
        rec = tl.to_record()
        self.assertEqual(len(rec["events"]), 6)
        self.assertTrue(rec["token_count_assumed"])
        self.assertEqual(metrics.accounting(dict(rec, status="ok"))["problems"], [])

    def test_comment_lines_are_ignored(self):
        for raw in fixture("chat_stream_logprobs.sse"):
            if raw.startswith(":"):
                self.assertEqual(metrics.parse_sse_line(raw)[0], "ignore")


class Definitions(unittest.TestCase):
    def ok(self, tl):
        rec = tl.to_record()
        rec["status"] = "ok"
        return rec

    def test_ttft_tpot_itl_e2e_from_known_arrival_times(self):
        lines = [data("a"), data("b"), data("c"), data("d", finish="length", usage={"prompt_tokens": 20, "completion_tokens": 4}), "data: [DONE]"]
        rec = self.ok(feed(lines, step_ns=10_000_000, first_extra_ns=40_000_000))
        d = metrics.derive(rec)
        self.assertEqual(d["output_tokens"], 4)
        self.assertEqual(d["token_source"], "usage")
        approx(self, d["ttft_s"], 0.050)
        approx(self, d["e2e_s"], 0.080)
        approx(self, d["tpot_s"], 0.010)
        self.assertEqual(len(d["itl_s"]), 3)
        self.assertEqual(d["problems"], [])

    def test_one_token_response_has_no_tpot_or_itl(self):
        rec = self.ok(feed([data("a", finish="length", usage={"prompt_tokens": 20, "completion_tokens": 1}), "data: [DONE]"]))
        d = metrics.derive(rec)
        self.assertIsNone(d["tpot_s"])
        self.assertEqual(d["itl_s"], [])
        self.assertEqual(metrics.request_distributions([rec])["tpot_undefined_requests"], 1)

    def test_output_tokens_are_the_reported_usage_not_the_limit(self):
        rec = self.ok(feed([data("a"), data("b", finish="stop", usage={"prompt_tokens": 20, "completion_tokens": 2}), "data: [DONE]"]))
        rec["max_tokens"] = 64
        acc = metrics.accounting(rec)
        self.assertEqual(acc["output_tokens"], 2)
        self.assertEqual(acc["max_tokens"], 64)
        self.assertEqual(acc["problems"], [])


class Accounting(unittest.TestCase):
    def acc(self, lines, **extra):
        rec = feed(lines).to_record()
        rec["status"] = "ok"
        rec.update(extra)
        return metrics.accounting(rec)["problems"]

    def test_missing_finish_reason(self):
        self.assertIn("missing finish_reason", self.acc([data("a", usage={"prompt_tokens": 3, "completion_tokens": 1}), "data: [DONE]"]))

    def test_missing_usage(self):
        self.assertIn("missing usage", self.acc([data("a", finish="stop"), "data: [DONE]"]))

    def test_truncated_stream_without_done(self):
        self.assertIn("stream ended without [DONE]", self.acc([data("a", finish="stop", usage={"prompt_tokens": 3, "completion_tokens": 1})]))

    def test_empty_text_token_without_logprobs_is_counted_from_its_event(self):
        lines = [data("a", logprobs=False), data("", logprobs=False), data("b", finish="stop", usage={"prompt_tokens": 3, "completion_tokens": 3}, logprobs=False), "data: [DONE]"]
        self.assertEqual(self.acc(lines), [])
        rec = feed(lines).to_record()
        self.assertEqual(len(rec["events"]), 3)  # the empty-text event is a token event with its own arrival time
        self.assertTrue(rec["token_count_assumed"])

    def test_an_event_without_any_delta_content_is_not_a_token(self):
        tl = feed(['data: {"choices": [{"index": 0, "delta": {"role": "assistant"}}]}', data("a", finish="stop", usage={"prompt_tokens": 3, "completion_tokens": 1}, logprobs=False), "data: [DONE]"])
        self.assertEqual((len(tl.events), tl.events_without_tokens), (1, 1))

    def test_a_miscounted_stream_is_still_caught_by_the_usage_cross_check(self):
        lines = [data("a", logprobs=False), data("b", finish="stop", usage={"prompt_tokens": 3, "completion_tokens": 5}, logprobs=False), "data: [DONE]"]
        self.assertTrue(any("differ from usage" in p for p in self.acc(lines)))

    def test_empty_text_token_with_logprobs_is_counted(self):
        lines = [data("a"), data(""), data("b", finish="stop", usage={"prompt_tokens": 3, "completion_tokens": 3}), "data: [DONE]"]
        self.assertEqual(self.acc(lines), [])

    def test_length_finish_with_fewer_tokens_than_the_limit(self):
        lines = [data("a", finish="length", usage={"prompt_tokens": 3, "completion_tokens": 1}), "data: [DONE]"]
        self.assertTrue(any("finish_reason length" in p for p in self.acc(lines, max_tokens=6)))

    def test_error_event_is_recorded(self):
        tl = feed(['data: {"error": {"message": "boom"}}', "data: [DONE]"])
        self.assertEqual(tl.stream_error, {"message": "boom"})

    def test_malformed_event_raises(self):
        with self.assertRaises(metrics.StreamError):
            metrics.parse_sse_line("data: {not json")


class TokenComparison(unittest.TestCase):
    REF = ["One", ",", " two", ",", " three"]
    MARGINS = [3.6, 2.9, 0.625, 13.1, 13.1]

    def test_identical_sequences_are_exact(self):
        self.assertEqual(metrics.compare_tokens(list(self.REF), self.REF, self.MARGINS, 0.251)["outcome"], "exact")

    def test_difference_at_a_confident_position_is_a_mismatch(self):
        res = metrics.compare_tokens(["One", ";", " two", ",", " three"], self.REF, self.MARGINS, 0.251)
        self.assertEqual((res["outcome"], res["first_difference"]), ("mismatch", 1))

    def test_difference_at_a_near_tie_is_classified_with_its_margin(self):
        margins = [3.6, 2.9, 0.125, 13.1, 13.1]
        res = metrics.compare_tokens(["One", ",", " Two", ",", " three"], self.REF, margins, 0.251)
        self.assertEqual((res["outcome"], res["first_difference"], res["reference_margin"]), ("near_tie", 2, 0.125))

    def test_the_tolerance_is_not_widened_by_the_caller(self):
        res = metrics.compare_tokens(["One", ",", " Two", ",", " three"], self.REF, self.MARGINS, 0.251)  # margin 0.625 is above the tie margin
        self.assertEqual(res["outcome"], "mismatch")

    def test_unknown_margin_is_never_a_near_tie(self):
        self.assertEqual(metrics.compare_tokens(["X"], ["Y"], None, 0.251)["outcome"], "mismatch")

    def test_truncated_or_extended_output_with_an_identical_prefix_is_a_mismatch(self):
        self.assertEqual(metrics.compare_tokens(self.REF[:3], self.REF, self.MARGINS, 0.251)["outcome"], "mismatch")
        self.assertEqual(metrics.compare_tokens(self.REF + ["!"], self.REF, self.MARGINS, 0.251)["outcome"], "mismatch")

    def test_missing_tokens(self):
        self.assertEqual(metrics.compare_tokens(None, self.REF, self.MARGINS, 0.251)["outcome"], "mismatch")

    def test_summary_passes_only_without_mismatch_or_failed_requests(self):
        ok = [{"outcome": "exact"}, {"outcome": "near_tie"}]
        self.assertTrue(metrics.check_summary(ok)["passed"])
        self.assertFalse(metrics.check_summary(ok + [{"outcome": "mismatch"}])["passed"])
        self.assertFalse(metrics.check_summary(ok + [{"outcome": "failed"}])["passed"])
        self.assertFalse(metrics.check_summary([])["passed"])


if __name__ == "__main__":
    unittest.main()
