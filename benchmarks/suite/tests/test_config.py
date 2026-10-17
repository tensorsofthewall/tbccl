import copy
import json
import unittest

from benchmarks.suite import config as cfg
from benchmarks.suite.tests.common import BASE, SuiteTestCase, make_config, requires_yaml


def raw(**changes):
    r = copy.deepcopy(BASE)
    r.update(changes)
    return r


class Sizes(unittest.TestCase):
    def test_parse(self):
        self.assertEqual([cfg.parse_size(v) for v in (4, "4", "4B", "4KiB", "4K", "1MiB", "2KB", "1GiB")], [4, 4, 4, 4096, 4096, 1048576, 2000, 1073741824])

    def test_invalid(self):
        for bad in (0, -4, "abc", "4XB", 1.5, True, "", "4 KiBi"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                cfg.parse_size(bad)

    def test_presets(self):
        self.assertEqual(cfg.resolve_sizes("smoke"), [4, 4096, 65536, 1048576])
        self.assertEqual(len(cfg.resolve_sizes("standard")), 13)
        self.assertEqual(cfg.resolve_sizes("standard")[-1], 64 * 1024 * 1024)


class Validation(unittest.TestCase):
    def errors(self, **changes):
        with self.assertRaises(cfg.ConfigError) as ctx:
            cfg.build_config(raw(**changes))
        return "\n".join(ctx.exception.errors)

    def test_valid(self):
        c = cfg.build_config(raw())
        self.assertEqual(len(cfg.expand_matrix(c)), 2)

    def test_invalid_payload(self):
        self.assertIn("invalid payload size", self.errors(workloads=[{"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": ["bogus"]}]))
        self.assertIn("at least 1 byte", self.errors(workloads=[{"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [0]}]))

    def test_missing_required(self):
        self.assertIn("'benchmark' is required", self.errors(workloads=[{"operation": "pingpong", "sizes": [4]}]))
        self.assertIn("'operation' is required", self.errors(workloads=[{"benchmark": "tcp_pingpong", "sizes": [4]}]))
        self.assertIn("'sizes' is required", self.errors(workloads=[{"benchmark": "tcp_pingpong", "operation": "pingpong"}]))
        self.assertIn("at least one workload", self.errors(workloads=[]))
        self.assertIn("at least one rank", self.errors(ranks=[]))

    def test_unsupported_algorithm(self):
        wl = {"benchmark": "collective_bench", "operation": "all_reduce", "algorithms": ["bogus"], "sizes": [4096]}
        self.assertIn("algorithm 'bogus' is not supported", self.errors(workloads=[wl]))
        wl["algorithms"] = ["ring-pipelined"]
        self.assertIn("needs options.chunk_bytes", self.errors(workloads=[wl]))
        self.assertIn("no selectable algorithm", self.errors(workloads=[{"benchmark": "tcp_pingpong", "operation": "pingpong", "algorithms": ["ring"], "sizes": [4]}]))

    def test_unsupported_memory_backend(self):
        ranks = [{"rank": 0, "host": "local", "memory_backend": "cuda-pinned"}, {"rank": 1, "host": "local", "memory_backend": "host"}]
        self.assertIn("memory backend 'cuda-pinned' (rank 0) is not supported by tcp_pingpong", self.errors(ranks=ranks))

    def test_unsupported_operation_unknown_benchmark_and_integration(self):
        self.assertIn("operation 'bogus' is not supported", self.errors(workloads=[{"benchmark": "tcp_pingpong", "operation": "bogus", "sizes": [4]}]))
        self.assertIn("unknown benchmark", self.errors(workloads=[{"benchmark": "nope", "operation": "x", "sizes": [4]}]))
        self.assertIn("not integrated yet", self.errors(workloads=[{"benchmark": "async_raw_transport", "operation": "raw_transport", "sizes": [4]}]))

    def test_rank_counts(self):
        wl = {"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [4], "rank_counts": [3]}
        self.assertIn("only 2 ranks are configured", self.errors(workloads=[wl]))
        wl["rank_counts"] = [1]
        self.assertIn("supports 2..2 ranks", self.errors(workloads=[wl]))

    def test_port_range_is_validated_and_placeholders_are_reported_not_crashed_on(self):
        for bad in ([5, 10], [40000, 30000], [1], "x", [70000, 80000], [True, 5000]):
            self.assertIn("port_range", self.errors(defaults={"port_range": bad}))
        c = cfg.build_config(raw(defaults={"port_range": ["<START>", "<END>"]}))
        self.assertEqual(cfg.port_range(c), tuple(cfg.BUILTIN_DEFAULTS["port_range"]))
        self.assertTrue(any("port_range" in p for p in cfg.unresolved_placeholders(c)))
        self.assertEqual(cfg.port_range(cfg.build_config(raw(defaults={"port_range": [36000, 36100]}))), (36000, 36100))

    def test_bad_numbers_and_unknown_keys(self):
        self.assertIn("repetitions must be a positive integer", self.errors(defaults={"repetitions": 0}))
        self.assertIn("timeout_s must be a positive number", self.errors(defaults={"timeout_s": -1}))
        self.assertIn("unknown top-level key 'typo'", self.errors(typo=1))
        self.assertIn("unknown key 'sizez'", self.errors(workloads=[{"benchmark": "tcp_pingpong", "operation": "pingpong", "sizez": [4]}]))

    def test_non_loopback_requires_physical_opt_in(self):
        hosts = {"local": {"address": "192.0.2.1"}}
        self.assertIn("topology.physical: true", self.errors(hosts=hosts))

    def test_ranks_need_distinct_numbers_and_known_hosts(self):
        self.assertIn("unknown host", self.errors(ranks=[{"rank": 0, "host": "ghost"}, {"rank": 1, "host": "local"}]))
        self.assertIn("rank numbers", self.errors(ranks=[{"rank": 0, "host": "local"}, {"rank": 0, "host": "local"}]))

    def test_max_processes(self):
        wl = {"benchmark": "collective_bench", "operation": "all_gather", "algorithms": ["ring"], "sizes": [64], "rank_counts": [3]}
        r = raw(workloads=[wl], limits={"max_processes": 2}, ranks=[{"rank": i, "host": "local"} for i in range(3)])
        with self.assertRaises(cfg.ConfigError) as ctx:
            cfg.build_config(r)
        self.assertIn("max_processes", "\n".join(ctx.exception.errors))


class Matrix(unittest.TestCase):
    def test_incompatible_points_are_kept_with_a_reason(self):
        wl = {"benchmark": "collective_bench", "operations": ["all_reduce", "all_gather"], "algorithms": ["ring", "auto"], "sizes": [4, 4096], "datatype": "float32"}
        points = cfg.expand_matrix(cfg.build_config(raw(workloads=[wl])))
        self.assertEqual(len(points), 8)
        skipped = [p for p in points if p.skip_reason]
        self.assertEqual({(p.operation, p.payload_bytes) for p in skipped}, {("all_reduce", 4)})
        self.assertIn("not divisible by 2 ranks", skipped[0].skip_reason)

    def test_dtype_size_and_chunk(self):
        wl = {"benchmark": "collective_bench", "operation": "reduce_scatter", "algorithms": ["ring-pipelined"], "sizes": [6, 64], "datatype": "float64", "options": {"chunk_bytes": 12}}
        points = cfg.expand_matrix(cfg.build_config(raw(workloads=[wl])))
        self.assertIn("not a multiple of the float64 size", points[0].skip_reason)
        self.assertIn("chunk_bytes 12 is not a multiple", points[1].skip_reason)

    def test_required_verification_makes_unverifiable_points_unsupported(self):
        wl = {"benchmark": "small_message_latency", "operations": ["p2p", "allgather"], "sizes": [64], "verification": "required"}
        points = cfg.expand_matrix(cfg.build_config(raw(workloads=[wl])))
        self.assertIsNone(points[0].skip_reason)
        self.assertEqual((points[1].skip_state, bool(points[1].skip_reason)), ("unsupported", True))

    def test_host_only_operations_reject_device_backends(self):
        ranks = [{"rank": 0, "host": "local", "memory_backend": "cuda-pinned"}, {"rank": 1, "host": "local", "memory_backend": "metal-shared"}]
        wl = {"benchmark": "tensor_transfer", "operations": ["latency_floor", "end_to_end"], "sizes": [4096]}
        points = cfg.expand_matrix(cfg.build_config(raw(workloads=[wl], ranks=ranks)))
        self.assertIn("always uses the host backend", points[0].skip_reason)
        self.assertEqual(points[1].backends, ["cuda-pinned", "metal-shared"])

    def test_allreduce_small_message_needs_multiple_of_four(self):
        wl = {"benchmark": "small_message_latency", "operation": "allreduce", "sizes": [2, 6, 8]}
        points = cfg.expand_matrix(cfg.build_config(raw(workloads=[wl])))
        self.assertEqual([bool(p.skip_reason) for p in points], [True, True, False])

    def test_defaults_and_overrides(self):
        wl = {"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [64], "repetitions": 7}
        p = cfg.expand_matrix(cfg.build_config(raw(workloads=[wl])))[0]
        self.assertEqual((p.repetitions, p.warmup, p.iterations), (7, 1, 5))


class Overlays(SuiteTestCase):
    def test_overlay_merges_over_example_and_placeholders_are_found(self):
        base = raw(hosts={"local": {"address": "<LINK_ADDRESS>"}}, topology={"kind": "direct-link", "interconnect": "ethernet", "transport": "tcp", "physical": True})
        (self.dir / "c.json").write_text(json.dumps(base))
        c = cfg.load_config(self.dir / "c.json", auto_overlay=False)
        self.assertTrue(cfg.unresolved_placeholders(c))
        (self.dir / "o.json").write_text(json.dumps({"hosts": {"local": {"address": "198.51.100.7"}}}))
        c2 = cfg.load_config(self.dir / "c.json", [self.dir / "o.json"])
        self.assertEqual(c2.hosts["local"].address, "198.51.100.7")
        self.assertEqual(cfg.unresolved_placeholders(c2), [])

    def test_local_overlay_is_picked_up_automatically(self):
        (self.dir / "c.json").write_text(json.dumps(raw()))
        (self.dir / "c.local.json").write_text(json.dumps({"defaults": {"repetitions": 9}}))
        self.assertEqual(cfg.load_config(self.dir / "c.json").defaults["repetitions"], 9)

    @requires_yaml
    def test_tracked_examples_validate_and_hold_no_private_values(self):
        from benchmarks.suite.tests.common import REPO
        for name in ("smoke.yaml", "core-loopback.yaml", "tb4-core.example.yaml", "b3b-validation.yaml", "b3b-latency.yaml", "b3b-throughput.yaml", "b3b-collectives.yaml"):
            c = cfg.load_config(REPO / "benchmarks/suite/configs" / name, auto_overlay=False)
            self.assertTrue(cfg.expand_matrix(c), name)
        example = (REPO / "benchmarks/suite/configs/tb4-core.example.yaml").read_text()
        self.assertNotRegex(example, r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
        self.assertNotIn("/home/", example)
        self.assertNotIn("/Users/", example)


if __name__ == "__main__":
    unittest.main()
