"""Parsing of the native executables' real output (saved fixtures) and the command lines the adapters build."""
import unittest
from pathlib import Path

from benchmarks.suite import runner as run
from benchmarks.suite.adapters import REGISTRY
from benchmarks.suite.adapters.base import Outcome, ParseError, Point
from benchmarks.suite.tests.common import FIXTURES, SuiteTestCase, make_config


def point(bench, op, size, backends=("host", "host"), algorithm=None, dtype=None, **kw):
    return Point(workload="t", benchmark=bench, operation=op, algorithm=algorithm, payload_bytes=size, datatype=dtype, reduce_op="sum" if dtype else None, rank_count=len(backends),
                 ranks=[{"rank": i, "host": "local", "memory_backend": b} for i, b in enumerate(backends)], options=kw.pop("options", {}), warmup=3, iterations=20, repetitions=1,
                 timeout_s=30)


def outcome(path):
    return Outcome({0: 0}, set(), False, None, {0: str(path)}, {}, 0.1)


class Parsing(SuiteTestCase):
    def parse(self, bench, p, fixture, text=None):
        path = FIXTURES / fixture if fixture else None
        if text is not None:
            path = self.dir / "out.txt"
            path.write_text(text)
        return REGISTRY[bench].parse(p, None, outcome(path))

    def test_collective_row_keeps_native_semantics_and_resolved_algorithm(self):
        r = self.parse("collective_bench", point("collective_bench", "all_reduce", 4096, algorithm="auto", dtype="float32"), "collective_all_reduce_w2.csv")
        t = r.measurements["timing"]
        self.assertEqual((t["statistics"]["median"], t["statistics"]["p95"], t["sample_count"]), (16.677, 19.1947, 20))
        self.assertEqual(t["statistics_provenance"], {"median": "reported", "p95": "reported", "min": "reported", "max": "reported"})
        self.assertEqual((r.algorithm_resolved, r.selection_reason), ("reference", "auto_unmeasured_world_size"))
        kinds = {x["kind"]: x for x in r.measurements["throughput"]}
        self.assertAlmostEqual(kinds["collective_algorithmic"]["value"], 0.245608)
        self.assertEqual(kinds["collective_bus_estimate"]["unit"], "GB/s")
        self.assertEqual(r.verification["status"], "passed")
        self.assertIn("p95 is not a meaningful", r.measurements["warnings"][0])

    def test_reduce_scatter_payload_is_the_output_segment(self):
        r = self.parse("collective_bench", point("collective_bench", "reduce_scatter", 4096, ("host",) * 3, "ring", "float32"), "collective_reduce_scatter_w3.csv")
        self.assertEqual(r.algorithm_resolved, "ring")
        self.assertIsNone(r.selection_reason)

    def test_collective_wrong_payload_or_algorithm_is_rejected(self):
        with self.assertRaises(ParseError):
            self.parse("collective_bench", point("collective_bench", "all_reduce", 8192, algorithm="auto", dtype="float32"), "collective_all_reduce_w2.csv")
        with self.assertRaises(ParseError):
            self.parse("collective_bench", point("collective_bench", "all_reduce", 4096, algorithm="ring", dtype="float32"), "collective_all_reduce_w2.csv")
        with self.assertRaises(ParseError):  # world size mismatch
            self.parse("collective_bench", point("collective_bench", "all_reduce", 4096, ("host",) * 3, "auto", "float32"), "collective_all_reduce_w2.csv")

    def test_tensor_latency_floor_marks_half_rtt_as_approximation(self):
        r = self.parse("tensor_transfer", point("tensor_transfer", "latency_floor", 4096), "tensor_latency_floor.csv")
        t = r.measurements["timing"]
        self.assertEqual((t["metric"], t["statistics"]["median"], t["statistics"]["p99"]), ("round_trip_time", 46.674, 91.932))
        self.assertIn("approximation", r.measurements["extra"]["half_rtt_us"]["note"])
        self.assertEqual(r.verification["status"], "not_supported")
        self.assertNotIn("source_staging_us", r.measurements["extra"])

    def test_tensor_network_only_is_send_call_time_not_delivery(self):
        r = self.parse("tensor_transfer", point("tensor_transfer", "network_only", 4096), "tensor_network_only.csv")
        self.assertEqual(r.measurements["timing"]["metric"], "sender_send_call_time")
        self.assertIn("wall-clock", r.measurements["throughput"][0]["definition"])

    def test_small_message_percentile_method_and_missing_fields(self):
        r = self.parse("small_message_latency", point("small_message_latency", "p2p", 4096), "small_message_latency.jsonl")
        t = r.measurements["timing"]
        self.assertEqual((t["percentile_method"], t["statistics"]["median"], t["sample_count"]), ("floor-nearest-rank", 121.96, 50))
        self.assertNotIn("p99", t["statistics"])
        self.assertEqual(r.verification["status"], "passed")
        r2 = self.parse("small_message_latency", point("small_message_latency", "allgather", 64), "small_message_latency.jsonl")
        self.assertEqual(r2.verification["status"], "not_supported")
        with self.assertRaises(ParseError):
            self.parse("small_message_latency", point("small_message_latency", "p2p", 4096), None, text='{"mode":"p2p","bytes":4096,"median_us":1.0}\n')
        with self.assertRaises(ParseError):
            self.parse("small_message_latency", point("small_message_latency", "p2p", 4096), None, text='{"mode":"p2p","bytes":4096,"median_us":-1,"p25_us":1,"p75_us":1,"p95_us":1,"n":5,"cpu_cores":1,"sleep_ratio":1}\n')

    def test_the_followers_view_is_kept_per_rank_and_labelled(self):
        from benchmarks.suite.adapters.base import Group, Invocation
        rank0 = '{"label":"suite","mode":"oneway","bytes":64,"median_us":1.6,"p25_us":1.4,"p75_us":1.9,"p95_us":2.9,"n":6,"cpu_cores":0.5,"sleep_ratio":1.0}\n'
        rank1 = '{"label":"suite/r1","mode":"oneway","bytes":64,"median_us":9.5,"p25_us":9.0,"p75_us":10.0,"p95_us":11.0,"n":6,"cpu_cores":0.4,"sleep_ratio":1.0}\n'
        (self.dir / "r0").write_text(rank0)
        (self.dir / "r1").write_text(rank1)
        out = Outcome({0: 0, 1: 0}, set(), False, None, {0: str(self.dir / "r0"), 1: str(self.dir / "r1")}, {}, 0.1)
        r = REGISTRY["small_message_latency"].parse(point("small_message_latency", "oneway", 64), None, out)
        t = r.measurements["timing"]
        self.assertEqual(t["metric"], "per_message_time_in_block_sender")
        self.assertEqual([x["rank"] for x in t["per_rank"]], [0, 1])
        self.assertEqual(t["per_rank"][1]["statistics"]["median"], 9.5)
        self.assertEqual(t["statistics"]["median"], 1.6)  # the primary figure stays rank 0's, as the tool prints it
        (self.dir / "r1").write_text("")
        self.assertIsNone(REGISTRY["small_message_latency"].parse(point("small_message_latency", "oneway", 64), None, out).measurements["timing"]["per_rank"])

    def test_the_follower_switch_is_passed_explicitly(self):
        ctx = run.RunContext(make_config())
        ctx.executable = lambda name, host: f"/x/{name}"
        g = REGISTRY["small_message_latency"].plan(point("small_message_latency", "p2p", 64), ctx, [1, 2], "/tmp")
        self.assertTrue(all(i.env == {"TBCCL_BENCH_BOTH": "1"} for i in g.invocations))

    def test_malformed_or_empty_output(self):
        with self.assertRaises(ParseError):
            self.parse("collective_bench", point("collective_bench", "all_gather", 4096, algorithm="ring"), None, text="collective,requested_algorithm\nall-gather,ring\n")
        with self.assertRaises(ParseError):
            self.parse("tensor_transfer", point("tensor_transfer", "latency_floor", 64), None, text="")
        with self.assertRaises(ParseError):
            self.parse("small_message_latency", point("small_message_latency", "p2p", 64), None, text="not json {\n{broken")

    def test_tcp_pingpong_is_mean_only_gbit_and_not_verified(self):
        adapter = REGISTRY["tcp_pingpong"]
        r = adapter.measure(point("tcp_pingpong", "pingpong", 64), {"iterations": 10000, "seconds": 0.1262, "rtt_us": 12.62, "gbps": 0.0812, "mps": 79213.0})
        t = r.measurements["timing"]
        self.assertEqual(list(t["statistics"]), ["mean"])
        self.assertEqual(r.measurements["throughput"][0]["unit"], "Gbit/s")
        self.assertIn("not link bandwidth", r.measurements["throughput"][0]["definition"])
        self.assertEqual((r.verification["status"], r.configuration_extra["warmup"], r.configuration_extra["iterations"]), ("not_supported", 0, 10000))
        s = adapter.measure(point("tcp_pingpong", "stream", 64), {"iterations": 1, "seconds": 1.0, "rtt_us": 0.0, "gbps": 0.3, "mps": 5.0})
        self.assertIsNone(s.measurements["timing"])
        with self.assertRaises(ParseError):
            adapter.measure(point("tcp_pingpong", "pingpong", 64), {"iterations": 1, "seconds": 1.0, "rtt_us": 0.0, "gbps": 0.3, "mps": 5.0})

    def test_tcp_stdout_fallback_for_remote_clients(self):
        out = self.dir / "c.stdout"
        out.write_text("Connected\npingpong     4.0 KiB  n=  10000  RTT=     15.43 us     4.248 Gbit/s       64821 msg/s\n")
        from benchmarks.suite.adapters.base import Group, Invocation
        g = Group([Invocation(0, "server", "local", []), Invocation(1, "client", "local", [])], [])
        r = REGISTRY["tcp_pingpong"].parse(point("tcp_pingpong", "pingpong", 4096), g, Outcome({}, set(), False, None, {1: str(out)}, {}, 0))
        self.assertEqual(r.measurements["timing"]["statistics"]["mean"], 15.43)
        self.assertTrue(r.measurements["warnings"])


class CommandLines(SuiteTestCase):
    def plan(self, p, ports):
        config = make_config()
        ctx = run.RunContext(config)
        ctx.executable = lambda name, host: f"/x/{name}"
        return REGISTRY[p.benchmark].plan(p, ctx, ports, self.dir)

    def test_collective_commands_use_one_port_per_rank_and_scrubbed_flags(self):
        g = self.plan(point("collective_bench", "all_reduce", 4096, ("host",) * 3, "ring-pipelined", "float32", options={"chunk_bytes": 1024}), [1, 2, 3])
        a = g.invocations[1].argv
        self.assertEqual(a[a.index("--peers") + 1], "127.0.0.1:1,127.0.0.1:2,127.0.0.1:3")
        self.assertEqual(a[a.index("--chunk-bytes") + 1], "1024")
        self.assertEqual(a[a.index("--op") + 1], "sum")
        self.assertEqual(g.invocations[1].wait_for_port, 1)
        self.assertIsNone(g.invocations[0].wait_for_port)
        gather = self.plan(point("collective_bench", "all_gather", 64, algorithm="ring"), [1, 2]).invocations[0].argv
        self.assertNotIn("--datatype", gather)

    def test_tcp_server_is_persistent_and_client_waits_for_it(self):
        g = self.plan(point("tcp_pingpong", "pingpong", 64), [7000])
        self.assertTrue(g.invocations[0].persistent)
        self.assertEqual(g.invocations[1].wait_for_port, 7000)
        self.assertIn("--output", g.invocations[1].argv)

    def test_end_to_end_gets_a_backend_per_rank(self):
        g = self.plan(point("tensor_transfer", "end_to_end", 4096, ("cuda-pinned", "metal-shared")), [1, 2])
        backends = [i.argv[i.argv.index("--local-backend") + 1] for i in g.invocations]
        self.assertEqual(backends, ["cuda-pinned", "metal-shared"])
        self.assertNotIn("--local-backend", self.plan(point("tensor_transfer", "latency_floor", 64), [1, 2]).invocations[0].argv)

    def test_remote_hosts_use_the_configured_launcher_prefix(self):
        config = make_config(hosts={"local": {"address": "127.0.0.1"}, "far": {"address": "192.0.2.9", "launcher": {"kind": "prefix", "argv": ["launch", "far"], "shell_join": True}}},
                             topology={"kind": "direct-link", "interconnect": "x", "transport": "tcp", "physical": True}, ranks=[{"rank": 0, "host": "local"}, {"rank": 1, "host": "far"}])
        argv = run.wrap_for_host(config.hosts["far"], ["/b/tool", "--a", "b c"], {"K": "v"})
        self.assertEqual(argv[:2], ["launch", "far"])
        self.assertEqual(len(argv), 3)
        self.assertIn("K=v", argv[2])
        self.assertIn("'b c'", argv[2])
        self.assertEqual(run.wrap_for_host(config.hosts["local"], ["/b/tool"], {}), ["/b/tool"])


if __name__ == "__main__":
    unittest.main()
