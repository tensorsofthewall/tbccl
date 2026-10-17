"""Records written by the earlier campaigns (B3B native, B3C device, B4 PyTorch, B5 vLLM, B6 Exo) stay readable, summarisable and comparable after the provider architecture change. The fixtures
are sanitised derivatives of saved results: whitelisted fields, synthetic hosts and documentation addresses, no paths or raw samples."""
import json
import re
import unittest
from pathlib import Path

from benchmarks.suite import schema
from benchmarks.suite.reporting import compare, summary
from benchmarks.suite.tests.test_report_formats import PROVIDER, with_provider

HISTORICAL = Path(__file__).resolve().parent / "fixtures" / "historical"
CAMPAIGNS = ("b3b", "b3c", "b4", "b5", "b6")


def load(name):
    return schema.read_records(HISTORICAL / f"{name}.jsonl")


class Readable(unittest.TestCase):
    def test_every_historical_record_is_schema_valid_and_version_1(self):
        for name in CAMPAIGNS:
            records = load(name)
            self.assertTrue(records, name)
            for r in records:
                self.assertEqual(r["schema_version"], schema.SCHEMA_VERSION)
                self.assertEqual(schema.validate_record(r), [], f"{name}: {r['benchmark']['name']}")
                self.assertNotIn("providers", r["environment"])  # written before providers existed

    def test_each_campaign_keeps_its_characteristic_content(self):
        names = {c: {r["benchmark"]["name"] for r in load(c)} for c in CAMPAIGNS}
        self.assertTrue(names["b3b"] & {"tensor_transfer", "small_message_latency", "collective_bench"})
        self.assertTrue(any(n.startswith("torch_") for n in names["b4"]))
        self.assertTrue(any(n.startswith("vllm_") for n in names["b5"]))
        self.assertTrue(any(n.startswith("exo_") for n in names["b6"]))
        for r in load("b6"):
            inf = r["configuration"]["inference"]
            self.assertIn(inf["backend"], ("MlxRing", "MlxTbccl"))
            self.assertTrue(inf.get("stage_layers") and inf.get("stage_hosts"))  # the placement identity the comparator needs
        for r in load("b5"):
            self.assertTrue(r["configuration"]["inference"].get("stage_devices"))  # B5 recorded the device of each stage

    def test_all_reports_render_for_every_campaign_and_for_all_of_them_together(self):
        everything = []
        for name in CAMPAIGNS:
            records = load(name)
            everything += records
            rows = summary.summarize(records)
            self.assertTrue(rows, name)
            md, page = summary.to_markdown(rows, records, name), summary.to_html(rows, records, name)
            self.assertIn("## ", md)
            self.assertNotIn("Framework providers", md)  # no provider data was recorded then
            doc = json.loads(summary.to_json(rows, records, name))
            self.assertEqual(len(doc["results"]), len(rows))
            self.assertNotRegex(page, r"(?i)<script|src=\"https?:")
        self.assertEqual(len(summary.summarize(everything)), sum(len(summary.summarize(load(c))) for c in CAMPAIGNS))

    def test_old_and_new_records_can_share_a_run(self):
        old = load("b4")
        new = with_provider([json.loads(json.dumps(r)) for r in old])
        mixed = old + new
        md = summary.to_markdown(summary.summarize(mixed), mixed)
        self.assertEqual(md.count("Framework providers"), 1)
        for r in new:
            self.assertEqual(schema.validate_record(r), [])  # the additive environment.providers field does not break validation

    def test_the_comparator_still_accepts_equivalent_historical_runs_and_refuses_mixed_backends(self):
        b4 = load("b4")
        result = compare.compare(b4, json.loads(json.dumps(b4)))
        self.assertTrue(result["comparisons"])
        self.assertEqual(result["incompatible"], [])
        b6 = load("b6")
        by_backend = {r["configuration"]["inference"]["backend"]: r for r in b6}
        if len(by_backend) == 2:
            mixed = compare.compare([by_backend["MlxRing"]], [by_backend["MlxTbccl"]])
            for c in mixed["comparisons"]:
                self.assertNotEqual(c["label"], "equivalent", "a MlxRing record must not be called equivalent to a MlxTbccl one")

    def test_fixtures_hold_no_private_values(self):
        for path in HISTORICAL.glob("*.jsonl"):
            text = path.read_text()
            self.assertEqual(set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)) - {"192.0.2.1", "127.0.0.1"}, set(), path.name)
            for marker in ("/mnt/", "/home/", "/Users/", "@", "ssh", ".local"):
                self.assertNotIn(marker, text, f"{path.name}: {marker}")


if __name__ == "__main__":
    unittest.main()
