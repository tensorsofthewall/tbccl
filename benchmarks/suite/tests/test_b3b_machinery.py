"""B3B machinery: data direction independent of rank orientation, volume limits, iperf3 provider, strict link policy, link deltas, data-path and route checks, CPU capture."""
import copy
import json
import sys
import unittest

from benchmarks.suite import config as cfg
from benchmarks.suite import health, provenance, schema
from benchmarks.suite import runner as run
from benchmarks.suite.adapters import REGISTRY
from benchmarks.suite.adapters.base import Outcome, ParseError
from benchmarks.suite.preflight import HealthMonitor
from benchmarks.suite.tests.common import BASE, SuiteTestCase
from benchmarks.suite.tests.test_multihost import two_hosts
from benchmarks.suite.tests.test_remote import two_host_config


def conf(workloads, **kw):
    raw = copy.deepcopy(BASE)
    raw["workloads"] = workloads
    raw.update(kw)
    return cfg.build_config(raw)


class DataDirection(unittest.TestCase):
    def points(self, **wl):
        w = {"name": "t", "benchmark": "tensor_transfer", "operation": "end_to_end", "sizes": [4096], "memory_backend": "host", "orientations": ["forward", "reverse"], "source_ranks": [0, 1]}
        w.update(wl)
        return cfg.expand_matrix(two_hosts(workloads=[w], ranks=[{"rank": 0, "host": "alpha"}, {"rank": 1, "host": "beta"}]))

    def test_orientation_and_source_are_independent_dimensions(self):
        pts = self.points()
        self.assertEqual(len(pts), 4)
        combos = {(p.orientation, p.source_rank, p.ranks[p.source_rank]["host"]) for p in pts}
        self.assertEqual(combos, {("forward", 0, "alpha"), ("forward", 1, "beta"), ("reverse", 0, "beta"), ("reverse", 1, "alpha")})
        self.assertEqual(len({p.point_id for p in pts}), 4)

    def test_the_same_physical_direction_can_come_from_two_orientations(self):
        pts = {(p.orientation, p.source_rank): p.ranks[p.source_rank]["host"] for p in self.points()}
        self.assertEqual(pts[("forward", 0)], pts[("reverse", 1)])  # alpha is the source in both, with different rank numbers

    def test_records_state_source_and_destination(self):
        config = two_hosts(workloads=[{"name": "t", "benchmark": "tensor_transfer", "operation": "end_to_end", "sizes": [4096], "memory_backend": "host", "source_ranks": [1]}],
                           ranks=[{"rank": 0, "host": "alpha"}, {"rank": 1, "host": "beta"}])
        runner = run.Runner(config, output_dir="/nonexistent", log=lambda *_: None)
        d = runner._configuration(cfg.expand_matrix(config)[0], REGISTRY["tensor_transfer"].describe(cfg.expand_matrix(config)[0]))["data_direction"]
        self.assertEqual((d["mode"], d["source_rank"], d["source_host"], d["destination_rank"], d["destination_host"]), ("directional", 1, "beta", 0, "alpha"))

    def test_fixed_direction_benchmarks_reject_source_ranks_and_bad_values(self):
        with self.assertRaises(cfg.ConfigError) as e:
            conf([{"benchmark": "collective_bench", "operation": "all_gather", "algorithms": ["ring"], "sizes": [64], "source_ranks": [0]}])
        self.assertIn("fixed data direction", str(e.exception))
        with self.assertRaises(cfg.ConfigError):
            conf([{"benchmark": "tensor_transfer", "operation": "end_to_end", "sizes": [64], "source_ranks": []}])
        pts = conf([{"benchmark": "tensor_transfer", "operation": "end_to_end", "sizes": [64], "source_ranks": [5]}])
        self.assertIn("does not exist", cfg.expand_matrix(pts)[0].skip_reason)

    def test_the_source_rank_reaches_the_command_line_and_the_parser(self):
        from benchmarks.suite.tests.test_adapters import outcome, point
        p = point("tensor_transfer", "network_only", 4096)
        p.source_rank = 1
        ctx = run.RunContext(conf([{"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [4]}]))
        ctx.executable = lambda name, host: f"/x/{name}"
        g = REGISTRY["tensor_transfer"].plan(p, ctx, [1, 2], "/tmp")
        self.assertTrue(all(i.argv[i.argv.index("--source-rank") + 1] == "1" for i in g.invocations))
        from benchmarks.suite.tests.common import FIXTURES
        out = Outcome({1: 0}, set(), False, None, {1: str(FIXTURES / "tensor_network_only.csv")}, {}, 0.1)
        self.assertTrue(REGISTRY["tensor_transfer"].parse(p, g, out).measurements["timing"])  # read from the SOURCE rank's stdout, not rank 0's

    def test_small_message_oneway_direction_is_rank0_to_rank1_and_round_trips_are_bidirectional(self):
        a = REGISTRY["small_message_latency"]
        from benchmarks.suite.tests.test_adapters import point
        self.assertEqual(a.data_direction(point("small_message_latency", "oneway", 64)), {"mode": "directional", "source_rank": 0})
        self.assertEqual(a.data_direction(point("small_message_latency", "p2p", 64))["mode"], "bidirectional")

    def test_direction_is_part_of_the_comparison_key(self):
        from benchmarks.suite.reporting import compare
        from benchmarks.suite.tests.test_compare_report import rec
        a, b = rec(100, run="a"), rec(100, run="b")
        for r, src in ((a, 0), (b, 1)):
            r["configuration"]["data_direction"] = {"mode": "directional", "source_rank": src}
        self.assertTrue(any("data_direction" in d for d in compare.differences(a, b)))


class Limits(unittest.TestCase):
    def test_per_point_limit_skips_points_with_a_reason(self):
        c = conf([{"benchmark": "tensor_transfer", "operation": "network_only", "sizes": [4096, 16 * 2**20], "warmup": 20, "iterations": 30}], limits={"max_point_bytes": 100 * 2**20})
        pts = cfg.expand_matrix(c)
        self.assertEqual([bool(p.skip_reason) for p in pts], [False, True])
        self.assertIn("exceeds limits.max_point_bytes", pts[1].skip_reason)

    def test_estimates_follow_the_native_tools(self):
        from benchmarks.suite.tests.test_adapters import point
        self.assertEqual(REGISTRY["tcp_pingpong"].estimate_bytes(point("tcp_pingpong", "pingpong", 64)), 64 * 10000 * 2)  # the tool's fixed iteration formula
        self.assertEqual(REGISTRY["tcp_pingpong"].estimate_bytes(point("tcp_pingpong", "stream", 1 << 20)), (1 << 20) * 4096)
        p = point("tensor_transfer", "network_only", 1 << 20)
        self.assertEqual(REGISTRY["tensor_transfer"].estimate_bytes(p), (1 << 20) * (20 + 3 + 1))

    def test_limit_values_are_validated(self):
        for bad in (0, -1, "x"):
            with self.assertRaises(cfg.ConfigError):
                conf([{"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [4]}], limits={"max_total_bytes": bad})

    def test_the_run_stops_when_the_total_volume_or_duration_limit_is_reached(self):
        class R(SuiteTestCase):
            def runTest(self):
                pass
        t = R()
        t.setUp()
        self.addCleanup(t.tmp.cleanup)
        wl = [{"name": f"w{i}", "benchmark": "fake", "operations": ["ok"], "sizes": [8 + i], "rank_counts": [2], "repetitions": 1} for i in range(3)]
        from benchmarks.suite.adapters import REGISTRY as REG
        FakeA = REG["fake"]
        FakeA.estimate_bytes = lambda point: 1000
        config = two_host_config(workloads=wl, limits={"max_total_bytes": 1500})
        manifest, records = run.Runner(config, output_dir=t.dir / "o", log=lambda *_: None).run()
        self.assertEqual([r["status"]["state"] for r in records], ["success", "success", "skipped"])
        self.assertIn("total volume limit", records[2]["status"]["reason"])
        self.assertEqual(manifest["status"], "aborted")
        config = two_host_config(workloads=wl, limits={"max_campaign_s": 0.001})
        manifest, records = run.Runner(config, output_dir=t.dir / "o2", log=lambda *_: None).run()
        self.assertTrue(all(r["status"]["state"] == "skipped" for r in records[1:]))
        self.assertIn("duration limit", manifest["reason"])


class Iperf3(unittest.TestCase):
    GOOD = {"start": {"version": "iperf 3.21", "test_start": {"protocol": "TCP", "num_streams": 1, "blksize": 131072, "duration": 3, "omit": 1, "reverse": 0}},
            "end": {"sum_sent": {"bytes": 3000000000, "seconds": 3.0, "bits_per_second": 8e9, "retransmits": 2}, "sum_received": {"bytes": 2999000000, "seconds": 3.0, "bits_per_second": 7.99e9},
                    "cpu_utilization_percent": {"host_total": 21.5, "remote_total": 18.0}}}

    def point(self, source=1):
        from benchmarks.suite.tests.test_adapters import point
        p = point("iperf3", "tcp_stream", 131072)
        p.source_rank = source
        return p

    def test_parse_keeps_sender_and_receiver_bytes_retransmits_and_cpu(self):
        r = REGISTRY["iperf3"].measure(self.point(), copy.deepcopy(self.GOOD))
        t = r.measurements["throughput"]
        self.assertEqual([x["unit"] for x in t], ["Gbit/s", "Gbit/s"])
        self.assertAlmostEqual(t[0]["value"], 7.99)  # the receiver (delivered) figure is first
        self.assertIn("receiver", t[0]["definition"])
        self.assertAlmostEqual(t[1]["value"], 8.0)
        e = r.measurements["extra"]
        self.assertEqual((e["bytes_sent"]["value"], e["retransmits"]["value"], e["parallel_streams"]["value"], e["cpu_host_total_percent"]["value"]), (3000000000, 2, 1, 21.5))
        self.assertIsNone(r.measurements["timing"])
        self.assertEqual(r.verification["status"], "not_supported")

    def test_parse_rejects_errors_empty_and_non_tcp_results(self):
        a = REGISTRY["iperf3"]
        for mutate in (lambda d: d.update(error="unable to connect"), lambda d: d["end"].pop("sum_received"),
                       lambda d: d["end"]["sum_received"].update(bits_per_second=0), lambda d: d["start"]["test_start"].update(protocol="UDP"),
                       lambda d: d["end"]["sum_sent"].update(bits_per_second=-1)):
            d = copy.deepcopy(self.GOOD)
            mutate(d)
            with self.assertRaises(ParseError):
                a.measure(self.point(), d)

    def test_more_than_one_stream_is_flagged(self):
        d = copy.deepcopy(self.GOOD)
        d["start"]["test_start"]["num_streams"] = 4
        self.assertTrue(REGISTRY["iperf3"].measure(self.point(), d).measurements["warnings"])

    def test_commands_are_bounded_one_shot_bound_and_directional(self):
        ctx = run.RunContext(two_hosts(workloads=[{"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [4], "memory_backend": "host"}]))
        ctx.tool = lambda name, host: f"/x/{name}"
        for source, reverse in ((1, False), (0, True)):
            p = self.point(source)
            p.ranks = [{"rank": 0, "host": "alpha", "interface": "link", "memory_backend": "host"}, {"rank": 1, "host": "beta", "interface": "link", "memory_backend": "host"}]
            g = REGISTRY["iperf3"].plan(p, ctx, [36001], "/tmp")
            server, client = g.invocations[0].argv, g.invocations[1].argv
            self.assertIn("-1", server)  # one test, then the server exits by itself
            self.assertEqual(server[server.index("-B") + 1], "192.0.2.10")
            self.assertEqual(client[client.index("-B") + 1], "192.0.2.20")  # the client binds its own data address
            self.assertEqual(client[client.index("-t") + 1], "3")
            self.assertEqual(client[client.index("-P") + 1], "1")
            self.assertIn("-J", client)
            self.assertEqual("-R" in client, reverse)
            self.assertTrue(g.invocations[0].persistent and g.invocations[0].listens_on == 36001 and g.invocations[1].wait_for_port == 36001)

    def test_duration_is_capped_and_the_estimate_is_a_ceiling(self):
        with self.assertRaises(cfg.ConfigError):
            conf([{"benchmark": "iperf3", "operation": "tcp_stream", "sizes": [131072], "options": {"duration_s": 300}}])
        p = self.point()
        p.options = {"duration_s": 3, "omit_s": 1}
        self.assertEqual(REGISTRY["iperf3"].estimate_bytes(p), int(4 * 5e9))

    def test_the_tool_is_found_through_the_host_configuration_not_the_build_tree(self):
        c = two_hosts(workloads=[{"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [4], "memory_backend": "host"}])
        ctx = run.RunContext(c)
        self.assertEqual(ctx.tool("iperf3", "beta"), "iperf3")  # remote: on the host's PATH
        c.hosts["beta"].executables = {"iperf3": "/opt/x/iperf3"}
        self.assertEqual(ctx.tool("iperf3", "beta"), "/opt/x/iperf3")

    def test_end_to_end_with_a_stand_in_iperf3(self):
        """The whole path (supervised server, client JSON, parse, record) with a script that behaves like the two iperf3 roles."""
        import os
        import stat
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            fake = Path(tmp) / "iperf3"
            fake.write_text("#!%s\nimport sys, json, socket, time\nargs = sys.argv[1:]\nif '-s' in args:\n    s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
                            "    s.bind((args[args.index('-B') + 1], int(args[args.index('-p') + 1]))); s.listen(1); s.settimeout(20)\n    c, _ = s.accept(); time.sleep(0.2)\nelse:\n"
                            "    socket.create_connection((args[args.index('-c') + 1], int(args[args.index('-p') + 1])), timeout=5).close()\n    print(json.dumps(%r))\n" % (sys.executable, self.GOOD))
            fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
            wl = [{"name": "i", "benchmark": "iperf3", "operation": "tcp_stream", "sizes": [131072], "source_ranks": [0, 1], "repetitions": 1, "memory_backend": "host"}]
            config = two_host_config(workloads=wl)
            for h in config.hosts.values():
                h.executables = {"iperf3": str(fake)}
            manifest, records = run.Runner(config, output_dir=Path(tmp) / "o", log=lambda *_: None).run()
            self.assertEqual([r["status"]["state"] for r in records], ["success", "success"], [r["status"]["reason"] for r in records])
            for r in records:
                self.assertEqual(schema.validate_record(r), [])
                self.assertEqual(r["benchmark"]["name"], "iperf3")
            self.assertEqual({r["configuration"]["data_direction"]["source_rank"] for r in records}, {0, 1})


class StrictLinkPolicy(unittest.TestCase):
    def wrapper(self, **kw):
        return {"interface": {"present": True, "operstate": "up", "mtu": 9000, "addresses": ["192.0.2.1/24"]}, "gpu": None, "snapshot": {"boot_id": "1"}, "snapshot_error": None, **kw}

    def compare(self, script_result, strict):
        return health.compare_wrapper(self.wrapper(), self.wrapper(), script_compare=lambda b, a: script_result, strict=strict)["stop_reasons"]

    def test_any_new_correctable_aer_growth_stops_only_in_strict_mode(self):
        script = {"stop_reasons": [], "aer_counter_deltas": {"0000:08:00.0/aer_dev_correctable": 1}, "new_kernel_events": []}
        self.assertEqual(self.compare(script, strict=False), [])  # the script alone tolerates a single correctable error
        reasons = self.compare(script, strict=True)
        self.assertTrue(reasons and "aer_dev_correctable +1" in reasons[0])

    def test_a_new_thunderbolt_kernel_event_stops_but_other_pci_noise_does_not(self):
        base = {"stop_reasons": [], "aer_counter_deltas": {}}
        tb = {**base, "new_kernel_events": [{"source": "thunderbolt", "message": "pcieport 0000:08:00.0: AER: Correctable error message received: BadDLLP"}]}
        other = {**base, "new_kernel_events": [{"source": "other-pci", "message": "something on an unrelated device"}]}
        self.assertTrue(any("new Thunderbolt kernel event" in r for r in self.compare(tb, True)))
        self.assertEqual(self.compare(other, True), [])
        self.assertEqual(self.compare(tb, False), [])

    def test_a_clean_comparison_stays_clean(self):
        self.assertEqual(self.compare({"stop_reasons": [], "aer_counter_deltas": {"x/aer_dev_correctable": 0}, "new_kernel_events": []}, True), [])

    def test_monitor_passes_strictness_to_the_default_comparison(self):
        host = two_host_config().hosts["local"]
        script = lambda b, a: {"stop_reasons": [], "aer_counter_deltas": {"d/aer_dev_correctable": 3}, "new_kernel_events": []}  # noqa: E731
        original = health._load_script_module
        health._load_script_module = lambda: type("M", (), {"compare": staticmethod(script)})
        try:
            w = self.wrapper()
            self.assertEqual(HealthMonitor([host], snapshot_fn=lambda h: w, strict=False).compare({"local": w}, {"local": w})[0], [])
            self.assertTrue(HealthMonitor([host], snapshot_fn=lambda h: w, strict=True).compare({"local": w}, {"local": w})[0])
        finally:
            health._load_script_module = original


class LinkObservations(unittest.TestCase):
    def wrap(self, rx, tx, retrans=None, errs=0):
        return {"snapshot": {"interface": {"counters": {"rx_bytes": str(rx), "tx_bytes": str(tx), "rx_errors": str(errs), "tx_errors": "0"}}},
                "system": {"tcp_retransmitted_segments": retrans, "loadavg": [0.5, 0.4, 0.3]}}

    def test_link_delta(self):
        d = health.link_delta(self.wrap(1000, 2000, 10), self.wrap(5000, 9000, 14, errs=0))
        self.assertEqual((d["rx_bytes"], d["tx_bytes"], d["rx_errors"], d["tcp_retransmitted_segments"], d["load_1m_after"]), (4000, 7000, 0, 4, 0.5))
        self.assertNotIn("tcp_retransmitted_segments", health.link_delta(self.wrap(1, 1), self.wrap(2, 2)))

    def test_load_and_competing_processes_are_signals(self):
        host = two_host_config().hosts["local"]
        w = {"interface": None, "gpu": None, "snapshot": {}, "system": {"cpu_count": 8, "loadavg": [7.0, 6.0, 5.0]}, "competing": [4242]}
        sig = health.evaluate(host, w)
        self.assertEqual((sig["system_load"]["status"], sig["competing_processes"]["status"]), ("bad", "bad"))
        w["system"]["loadavg"], w["competing"] = [0.5, 0.5, 0.5], []
        sig = health.evaluate(host, w)
        self.assertEqual((sig["system_load"]["status"], sig["competing_processes"]["status"]), ("ok", "ok"))

    def test_data_path_must_show_the_bytes_on_the_data_interface(self):
        config = two_host_config(workloads=[{"name": "w", "benchmark": "tensor_transfer", "operation": "network_only", "sizes": [1 << 20], "rank_counts": [2], "memory_backend": "host"}])
        runner = run.Runner(config, output_dir="/nonexistent", log=lambda *_: None)
        point = cfg.expand_matrix(config)[0]
        ok = [{"status": {"state": "success"}, "measurements": {"extra": {}}}] * 2
        big = point.estimated_bytes * 2
        self.assertIsNone(runner._data_path_problem(point, ok, {"local": {"rx_bytes": big, "tx_bytes": 0}}))
        self.assertIn("data path not confirmed", runner._data_path_problem(point, ok, {"local": {"rx_bytes": 1000, "tx_bytes": 1000}}))
        self.assertIsNone(runner._data_path_problem(point, [{"status": {"state": "failure"}, "measurements": None}], {"local": {"rx_bytes": 0, "tx_bytes": 0}}))

    def test_tiny_points_are_not_judged_by_counters(self):
        config = two_host_config(workloads=[{"name": "w", "benchmark": "small_message_latency", "operation": "p2p", "sizes": [64], "rank_counts": [2], "memory_backend": "host"}])
        runner = run.Runner(config, output_dir="/nonexistent", log=lambda *_: None)
        point = cfg.expand_matrix(config)[0]
        self.assertIsNone(runner._data_path_problem(point, [{"status": {"state": "success"}, "measurements": {"extra": {}}}], {"local": {"rx_bytes": 0, "tx_bytes": 0}}))


class Routes(unittest.TestCase):
    def test_a_route_through_the_wrong_interface_is_a_problem(self):
        config = two_hosts(workloads=[{"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [4], "memory_backend": "host"}])
        per_host = {"alpha": {"routes": {"192.0.2.20": "wlan0"}}, "beta": {"routes": {"192.0.2.10": "link0"}}}
        problems = provenance.route_problems(config, per_host, {"alpha": ["192.0.2.20"], "beta": ["192.0.2.10"]})
        self.assertEqual(len(problems), 1)
        self.assertIn("alpha", problems[0])
        self.assertIn("'wlan0'", problems[0])
        self.assertEqual(provenance.route_problems(config, {"alpha": {"routes": {}}}, {"alpha": ["192.0.2.20"]})[0].split(":")[0], "alpha")  # unknown route: unproven

    def test_the_helper_reports_routes(self):
        from benchmarks.suite.remote import agent
        self.assertEqual(agent.route_interface("127.0.0.1"), agent.LOOPBACK_INTERFACE)
        self.assertIn("127.0.0.1", agent.probe({}, None, ["127.0.0.1"])["routes"])


class CpuCapture(SuiteTestCase):
    def test_local_and_supervised_ranks_report_cpu_seconds(self):
        wl = [{"name": "a", "benchmark": "fake", "operations": ["ok"], "sizes": [8], "rank_counts": [2], "repetitions": 1}]
        manifest, records = run.Runner(two_host_config(workloads=wl), output_dir=self.dir / "o", log=lambda *_: None).run()
        ranks = {r["rank"]: r for r in records[0]["artifacts"]["ranks"]}
        for rank in (0, 1):
            cpu = ranks[rank]["cpu"]
            self.assertIsNotNone(cpu, rank)
            self.assertGreaterEqual(cpu["child_user_s"] + cpu["child_system_s"], 0)
        self.assertIsNotNone(ranks[1]["cpu"]["supervisor_cpu_s"])  # the remote helper's own cost is reported separately


if __name__ == "__main__":
    unittest.main()
