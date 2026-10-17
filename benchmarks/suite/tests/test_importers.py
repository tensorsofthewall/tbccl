import json
import unittest

from benchmarks.suite import schema
from benchmarks.suite.importers import ImportFailure, import_path
from benchmarks.suite.tests.common import FIXTURES, REPO, SuiteTestCase


class Csv(SuiteTestCase):
    def write(self, name, text):
        p = self.dir / name
        p.write_text(text)
        return p

    def test_tcp_pingpong_fixture(self):
        r = import_path(FIXTURES / "tcp_pingpong.csv")
        self.assertEqual(r.source_format, "tcp_pingpong")
        self.assertEqual(len(r.records), 3)
        first = r.records[0]
        self.assertEqual(schema.validate_record(first), [])
        self.assertEqual(first["benchmark"]["provenance"], "imported")
        self.assertEqual(first["environment"]["topology"]["kind"], "unknown")
        self.assertEqual(first["environment"]["hosts"], [])
        self.assertEqual(first["verification"]["status"], "not_run")
        self.assertIsNone(r.records[1]["measurements"]["timing"])  # stream rows carry no latency

    def test_collective_and_tensor_fixtures(self):
        c = import_path(FIXTURES / "collective_all_reduce_w2.csv")
        self.assertEqual({x["benchmark"]["algorithm_resolved"] for x in c.records}, {"reference"})
        self.assertEqual(c.records[0]["configuration"]["rank_count"], 2)
        t = import_path(FIXTURES / "tensor_latency_floor.csv")
        self.assertEqual(t.records[0]["environment"]["memory_domain"]["label"], "cpu-cpu")
        self.assertEqual(t.records[0]["benchmark"]["operation"], "latency_floor")

    def test_missing_columns_are_named(self):
        p = self.write("c.csv", "collective,requested_algorithm,algorithm,datatype,op,world_size\nall-gather,ring,ring,,,2\n")
        with self.assertRaises(ImportFailure) as ctx:
            import_path(p)
        self.assertIn("missing columns", str(ctx.exception))
        with self.assertRaises(ImportFailure):
            import_path(self.write("x.csv", "a,b,c\n1,2,3\n"))

    def test_malformed_measurements_reject_the_file_unless_skipping(self):
        text = "mode,size_bytes,iterations,seconds,rtt_us,gbps,messages_per_second\npingpong,64,10,1.0,abc,0.1,5\npingpong,64,10,1.0,12.0,0.1,5\npingpong,64,10,1.0,-3,0.1,5\n"
        p = self.write("t.csv", text)
        with self.assertRaises(ImportFailure) as ctx:
            import_path(p)
        self.assertIn("t.csv:2", str(ctx.exception))
        r = import_path(p, skip_bad_rows=True)
        self.assertEqual(len(r.records), 1)
        self.assertEqual([s["line"] for s in r.skipped], [2, 4])

    def test_empty_and_unsupported_files(self):
        with self.assertRaises(ImportFailure):
            import_path(self.write("e.csv", "mode,size_bytes,iterations,seconds,rtt_us,gbps,messages_per_second\n"))
        with self.assertRaises(ImportFailure):
            import_path(self.write("e.txt", "x"))
        with self.assertRaises(ImportFailure):
            import_path(self.dir / "missing.csv")

    def test_latency_sweep_raw_and_summary(self):
        raw = REPO / "results/latency-v1/linux-to-mac-normal-raw.csv"
        r = import_path(raw)
        self.assertEqual(r.source_format, "latency_sweep_raw")
        self.assertEqual(r.records[0]["configuration"]["options"]["direction"], "linux-to-mac")
        with self.assertRaises(ImportFailure) as ctx:
            import_path(REPO / "results/latency-v1/summary.csv")
        self.assertIn("aggregate", str(ctx.exception))

    def test_historical_repository_results_import_without_addresses(self):
        for name in ("linux-to-mac.csv", "mac-to-linux.csv"):
            r = import_path(REPO / "results" / name)
            self.assertTrue(r.records)
            self.assertNotIn("192.168", json.dumps(r.records))
        iperf = import_path(REPO / "results/iperf-linux-to-mac-mtu9000.json")
        self.assertEqual(iperf.source_format, "iperf3")
        text = json.dumps(iperf.records)
        self.assertNotIn("192.168", text)
        self.assertEqual(iperf.records[0]["measurements"]["throughput"][0]["unit"], "Gbit/s")
        self.assertEqual(iperf.records[0]["measurements"]["throughput"][0]["kind"], "iperf3_tcp_goodput")

    def test_overrides_add_context_and_recompute_the_domain(self):
        r = import_path(FIXTURES / "tcp_pingpong.csv", overrides=["environment.topology.kind=direct-link", "environment.topology.interconnect=thunderbolt4-ip", 'environment.memory_backend=["host","metal-shared"]'])
        env = r.records[0]["environment"]
        self.assertEqual((env["topology"]["kind"], env["memory_domain"]["label"]), ("direct-link", "cpu-metal"))
        with self.assertRaises(ImportFailure):
            import_path(FIXTURES / "tcp_pingpong.csv", overrides=["nonsense"])
        with self.assertRaises(ImportFailure):
            import_path(FIXTURES / "tcp_pingpong.csv", overrides=["status.state=success"])


class JsonAndJsonl(SuiteTestCase):
    def test_small_message_jsonl_keeps_the_source_percentile_method(self):
        r = import_path(FIXTURES / "small_message_latency.jsonl")
        self.assertEqual(len(r.records), 4)
        t = r.records[0]["measurements"]["timing"]
        self.assertEqual(t["percentile_method"], "floor-nearest-rank")
        self.assertNotIn("p99", t["statistics"])
        self.assertEqual(t["statistics_provenance"]["median"], "reported")

    def test_jsonl_malformed_line_and_missing_fields(self):
        p = self.dir / "s.jsonl"
        p.write_text('{"mode":"p2p","bytes":64,"median_us":1,"p25_us":1,"p75_us":1,"p95_us":1,"n":5,"cpu_cores":1,"sleep_ratio":1}\n{oops\n')
        with self.assertRaises(ImportFailure) as ctx:
            import_path(p)
        self.assertIn("s.jsonl:2", str(ctx.exception))
        self.assertEqual(len(import_path(p, skip_bad_rows=True).records), 1)
        p.write_text('{"mode":"idle","seconds":3,"cpu_cores":0.1}\n')
        with self.assertRaises(ImportFailure):
            import_path(p)

    def test_allreduce_harness_converts_units_and_refuses_unverified_sizes_as_valid(self):
        r = import_path(FIXTURES / "allreduce_harness.json")
        ok, bad = r.records
        t = ok["measurements"]["timing"]
        self.assertAlmostEqual(t["statistics"]["median"], 100.0)  # 0.0001 s
        self.assertEqual(t["sample_count"], 5)
        self.assertEqual(ok["environment"]["framework_versions"]["torch"], "2.13.0")
        self.assertEqual((bad["status"]["state"], bad["measurements"]), ("failure", None))

    def test_incompatible_schema_versions_are_rejected(self):
        p = self.dir / "r.json"
        p.write_text(json.dumps({"schema_version": "tbccl-bench-result/2", "x": 1}))
        with self.assertRaises(ImportFailure) as ctx:
            import_path(p)
        self.assertNotEqual(str(ctx.exception), "")
        p.write_text("{not json")
        with self.assertRaises(ImportFailure):
            import_path(p)
        p.write_text(json.dumps({"hello": 1}))
        with self.assertRaises(ImportFailure):
            import_path(p)

    def test_own_results_round_trip(self):
        rec = import_path(FIXTURES / "tcp_pingpong.csv").records[0]
        p = self.dir / "own.jsonl"
        schema.write_jsonl(p, [rec])
        self.assertEqual(import_path(p).records, [rec])
        self.assertEqual(schema.read_records(p), [rec])


if __name__ == "__main__":
    unittest.main()
