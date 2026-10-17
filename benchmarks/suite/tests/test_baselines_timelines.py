"""Per-request metrics recomputed from raw event times (synthetic timelines)."""
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.suite.baselines import timelines as tl

S = 1_000_000_000


def req(submit_ms, event_ms, status="ok", usage=None, warmup=False):
    r = {"schema": "tbccl-exo-request/1", "t_submit_ns": submit_ms * 1_000_000, "events": [[t * 1_000_000, 1, None] for t in event_ms], "status": status, "warmup": warmup}
    if usage is not None:
        r["usage"] = {"completion_tokens": usage}
    return r


class Metrics(unittest.TestCase):
    def test_definitions(self):
        m = tl.request_metrics(req(0, [100, 110, 130, 150]))
        self.assertAlmostEqual(m["ttft"], 0.100)
        self.assertAlmostEqual(m["e2e"], 0.150)
        self.assertAlmostEqual(m["tpot"], 0.050 / 3)
        self.assertEqual([round(x, 3) for x in m["itl"]], [0.01, 0.02, 0.02])
        self.assertEqual(m["tokens"], 4)

    def test_one_token_has_no_tpot(self):
        m = tl.request_metrics(req(0, [100]))
        self.assertIsNone(m["tpot"])
        self.assertEqual(m["itl"], [])
        self.assertEqual(tl.metric_samples([req(0, [100]), req(0, [100, 120])], "tpot"), [0.020])

    def test_usage_token_count_can_be_the_source(self):
        m = tl.request_metrics(req(0, [100, 160], usage=3), "usage")
        self.assertAlmostEqual(m["tpot"], 0.060 / 2)  # three tokens reported, two intervals
        self.assertIsNone(tl.request_metrics(req(0, [100, 160]), "usage"))  # no usage: not guessed

    def test_failed_empty_and_warmup_requests_are_excluded(self):
        requests = [req(0, [100, 120]), req(0, [100, 120], status="timeout"), req(0, [], status="ok"), req(0, [100, 200], warmup=True)]
        self.assertEqual(tl.metric_samples(requests, "ttft"), [0.100])
        self.assertIsNone(tl.request_metrics(requests[1]))
        self.assertIsNone(tl.request_metrics(requests[2]))

    def test_throughput_counts_generated_tokens_over_the_whole_span(self):
        requests = [req(0, [100, 200, 300, 400]), req(50, [150, 250, 350, 450, 550])]
        self.assertAlmostEqual(tl.throughput_tokens_per_s(requests), 9 / 0.550)
        self.assertIsNone(tl.throughput_tokens_per_s([]))
        self.assertIsNone(tl.throughput_tokens_per_s([req(0, [1], status="error")]))

    def test_token_accounting_reports_disagreement(self):
        requests = [req(0, [100, 110, 120], usage=3), req(0, [100, 110], usage=3), req(0, [100], usage=None)]
        self.assertEqual(tl.token_accounting(requests), {"checked": 2, "mismatching": 1})

    def test_lifecycle_observation_is_one_median_and_its_sample_count(self):
        requests = [req(0, [100, 110, 120]), req(0, [100, 120, 140]), req(0, [100, 130, 160])]
        value, n = tl.lifecycle_observation(requests, "tpot")
        self.assertAlmostEqual(value, 0.020)
        self.assertEqual(n, 3)
        self.assertIsNone(tl.lifecycle_observation([], "tpot"))

    def test_read_requests_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "x.requests.jsonl"
            path.write_text(json.dumps(req(0, [1, 2])) + "\n\n" + json.dumps(req(0, [3])) + "\n")
            self.assertEqual(len(tl.read_requests(path)), 2)

    def test_event_may_be_a_bare_timestamp(self):
        r = {"t_submit_ns": 0, "events": [100_000_000, 150_000_000], "status": "ok"}
        self.assertAlmostEqual(tl.request_metrics(r)["tpot"], 0.050)


if __name__ == "__main__":
    unittest.main()
