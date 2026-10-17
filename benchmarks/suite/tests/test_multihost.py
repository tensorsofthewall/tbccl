import copy
import json
import os
import subprocess
import sys
import unittest

from benchmarks.suite import config as cfg
from benchmarks.suite import runner as run
from benchmarks.suite import schema
from benchmarks.suite.adapters import REGISTRY
from benchmarks.suite.tests.common import BASE, REPO, FIXTURES, SuiteTestCase, requires_yaml
from benchmarks.suite.tests.test_provenance_health import fake_build


def two_hosts(**changes):
    raw = copy.deepcopy(BASE)
    raw["topology"] = {"kind": "direct-link", "interconnect": "ethernet", "transport": "tcp", "physical": True}
    raw["hosts"] = {"alpha": {"address": "192.0.2.10", "interfaces": {"link": "192.0.2.10", "mgmt": "198.51.100.10"}, "interface": "link0", "env": {"OMP_NUM_THREADS": "2"}},
                    "beta": {"address": "192.0.2.20", "interfaces": {"link": "192.0.2.20", "mgmt": "198.51.100.20"}, "interface": "link0",
                             "launcher": {"kind": "prefix", "argv": ["remote-shell", "beta"], "shell_join": True}, "executables": {"tb_pingpong": "/opt/x/tb_pingpong"}, "python": "python3.11"}}
    raw["ranks"] = [{"rank": 0, "host": "alpha", "interface": "link", "memory_backend": "cuda-pinned"}, {"rank": 1, "host": "beta", "interface": "link", "memory_backend": "metal-shared"}]
    raw["workloads"] = [{"name": "t", "benchmark": "tensor_transfer", "operations": ["end_to_end"], "sizes": [4096], "orientations": ["forward", "reverse"]}]
    raw.update(changes)
    return cfg.build_config(raw)


class Layout(unittest.TestCase):
    def test_rank_orientations_swap_hosts_and_backends_but_keep_rank_numbers(self):
        points = cfg.expand_matrix(two_hosts())
        fwd, rev = points
        self.assertEqual((fwd.orientation, rev.orientation), ("forward", "reverse"))
        self.assertEqual([(r["rank"], r["host"], r["memory_backend"]) for r in fwd.ranks], [(0, "alpha", "cuda-pinned"), (1, "beta", "metal-shared")])
        self.assertEqual([(r["rank"], r["host"], r["memory_backend"]) for r in rev.ranks], [(0, "beta", "metal-shared"), (1, "alpha", "cuda-pinned")])
        self.assertNotEqual(fwd.point_id, rev.point_id)
        self.assertEqual(fwd.backends, ["cuda-pinned", "metal-shared"])
        self.assertEqual(rev.backends, ["metal-shared", "cuda-pinned"])

    def test_points_that_differ_only_in_backends_have_distinct_ids(self):
        wl = [{"name": "a", "benchmark": "tensor_transfer", "operation": "end_to_end", "sizes": [4096]},
              {"name": "b", "benchmark": "tensor_transfer", "operation": "end_to_end", "sizes": [4096], "memory_backend": ["metal-shared", "host"]}]
        ids = [p.point_id for p in cfg.expand_matrix(two_hosts(workloads=wl, ranks=[{"rank": 0, "host": "alpha", "memory_backend": "cuda-pinned"}, {"rank": 1, "host": "beta", "memory_backend": "host"}]))]
        self.assertEqual(len(ids), len(set(ids)), ids)
        self.assertIn("cuda-pinned+host", ids[0])

    def test_memory_backend_by_host_stays_with_its_machine_when_the_orientation_swaps(self):
        wl = [{"name": "a", "benchmark": "tensor_transfer", "operation": "end_to_end", "sizes": [4096], "orientations": ["forward", "reverse"],
               "memory_backend": {"alpha": "cuda-pageable", "beta": "metal-private-staged"}}]
        fwd, rev = cfg.expand_matrix(two_hosts(workloads=wl))
        self.assertEqual([(r["rank"], r["host"], r["memory_backend"]) for r in fwd.ranks], [(0, "alpha", "cuda-pageable"), (1, "beta", "metal-private-staged")])
        self.assertEqual([(r["rank"], r["host"], r["memory_backend"]) for r in rev.ranks], [(0, "beta", "metal-private-staged"), (1, "alpha", "cuda-pageable")])
        bad = [dict(wl[0], memory_backend={"gamma": "host"})]
        with self.assertRaises(cfg.ConfigError) as ctx:
            two_hosts(workloads=bad)
        self.assertIn("host 'gamma'", str(ctx.exception))

    def test_orientations_are_validated(self):
        raw = copy.deepcopy(BASE)
        wl = {"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [4], "orientations": ["sideways"]}
        raw["workloads"] = [wl]
        with self.assertRaises(cfg.ConfigError) as ctx:
            cfg.build_config(raw)
        self.assertIn("orientation 'sideways'", str(ctx.exception))

    def test_distinct_connection_addresses_per_rank_and_interface(self):
        config = two_hosts()
        ctx = run.RunContext(config)
        self.assertEqual(ctx.rank_address(config.ranks[0]), "192.0.2.10")
        mgmt = {"rank": 0, "host": "alpha", "interface": "mgmt"}
        self.assertEqual(ctx.rank_address(mgmt), "198.51.100.10")
        self.assertEqual(ctx.rank_address({"rank": 0, "host": "alpha", "address": "203.0.113.5"}), "203.0.113.5")
        self.assertEqual(ctx.rank_address({"rank": 0, "host": "alpha"}), "192.0.2.10")

    def test_the_peers_argument_uses_each_ranks_own_address(self):
        ranks = [{"rank": 0, "host": "alpha", "interface": "mgmt"}, {"rank": 1, "host": "beta", "interface": "link"}]
        config = two_hosts(ranks=ranks, workloads=[{"name": "c", "benchmark": "collective_bench", "operation": "all_gather", "algorithms": ["ring"], "sizes": [64]}])
        ctx = run.RunContext(config)
        ctx.executable = lambda name, host: f"/x/{name}"
        group = REGISTRY["collective_bench"].plan(cfg.expand_matrix(config)[0], ctx, [7000, 7001], "/tmp")
        peers = group.invocations[0].argv[group.invocations[0].argv.index("--peers") + 1]
        self.assertEqual(peers, "198.51.100.10:7000,192.0.2.20:7001")

    def test_the_recorded_interface_is_the_device_not_the_address_alias(self):
        config = two_hosts(workloads=[{"name": "t", "benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [64], "memory_backend": "host"}])
        runner = run.Runner(config, output_dir="/nonexistent", log=lambda *_: None)
        point = cfg.expand_matrix(config, runner.ctx)[0]
        env = runner._environment(point)
        self.assertEqual(env["network_interface"], "link0")   # the host's device, not the rank's alias "link"
        self.assertEqual(env["rank_interfaces"], ["link", "link"])

    def test_unknown_rank_interface_is_rejected(self):
        with self.assertRaises(cfg.ConfigError) as ctx:
            two_hosts(ranks=[{"rank": 0, "host": "alpha", "interface": "nope"}, {"rank": 1, "host": "beta"}])
        self.assertIn("has no interface 'nope'", str(ctx.exception))

    def test_per_host_env_executables_python_and_launcher(self):
        config = two_hosts()
        self.assertEqual(config.hosts["alpha"].env, {"OMP_NUM_THREADS": "2"})
        self.assertEqual(config.hosts["beta"].python, "python3.11")
        ctx = run.RunContext(config)
        self.assertEqual(ctx.executable("tb_pingpong", "beta"), "/opt/x/tb_pingpong")
        self.assertEqual(ctx.executable("tbccl_collective_bench", "beta"), "tbccl_collective_bench")  # on the host's PATH
        self.assertFalse(config.hosts["beta"].is_local)

    def test_local_host_executable_override_and_distinct_executable_versions(self):
        config = two_hosts()
        config.hosts["alpha"].executables = {"tb_pingpong": "/definitely/not/there"}
        self.assertIsNone(run.RunContext(config).executable_or_none("tb_pingpong", "alpha"))

    @requires_yaml
    def test_nothing_private_or_fixed_in_the_two_host_examples(self):
        for name in ("two-host.example.yaml", "two-host-reversed.example.yaml", "tb4-core.example.yaml"):
            text = (REPO / "benchmarks/suite/configs" / name).read_text()
            self.assertNotRegex(text, r"\b(?:\d{1,3}\.){3}\d{1,3}\b", name)
            self.assertNotRegex(text, r"/(home|Users|mnt)/", name)
            self.assertNotRegex(text, r'argv:\s*\[\s*"?ssh', name)
            self.assertNotIn("thunderbolt0", text)
            cfg.load_config(REPO / "benchmarks/suite/configs" / name, auto_overlay=False)

    @requires_yaml
    def test_both_orientation_examples_are_valid_and_opposite(self):
        a = cfg.load_config(REPO / "benchmarks/suite/configs/two-host.example.yaml", auto_overlay=False)
        b = cfg.load_config(REPO / "benchmarks/suite/configs/two-host-reversed.example.yaml", auto_overlay=False)
        self.assertEqual([r["host"] for r in a.ranks], list(reversed([r["host"] for r in b.ranks])))
        self.assertEqual([r["memory_backend"] for r in a.ranks], list(reversed([r["memory_backend"] for r in b.ranks])))

    def test_failure_policy_and_lease_settings_are_validated(self):
        raw = copy.deepcopy(BASE)
        raw["defaults"]["failure_policy"] = {"on_failure": "explode"}
        with self.assertRaises(cfg.ConfigError):
            cfg.build_config(raw)
        raw["defaults"] = {"lease_s": 0}
        with self.assertRaises(cfg.ConfigError):
            cfg.build_config(raw)
        raw["defaults"] = {"failure_policy": {"on_failure": "abort_run"}}
        self.assertEqual(cfg.build_config(raw).defaults["failure_policy"]["on_failure"], "abort_run")


class Backends(SuiteTestCase):
    def test_a_backend_missing_from_the_build_makes_the_point_unsupported(self):
        build = self.dir / "b"
        fake_build(build, name="tbccl_tensor_transfer_bench")
        raw = copy.deepcopy(BASE)
        raw["build_dirs"] = [str(build)]
        raw["ranks"] = [{"rank": 0, "host": "local", "memory_backend": "cuda-pinned"}, {"rank": 1, "host": "local", "memory_backend": "metal-shared"}]
        raw["workloads"] = [{"name": "t", "benchmark": "tensor_transfer", "operation": "end_to_end", "sizes": [4096]}]
        config = cfg.build_config(raw)
        points = cfg.expand_matrix(config, run.RunContext(config))
        self.assertEqual(points[0].skip_state, "unsupported")
        self.assertIn("not available in the executable's build", points[0].skip_reason)

    def test_provider_names_match_what_tbccl_info_lists(self):
        from benchmarks.suite.adapters.core import provider_for
        self.assertEqual([provider_for(b) for b in ("host", "cuda-pinned", "cuda-pageable", "metal-shared", "metal-private-staged")], ["host", "cuda", "cuda", "metal_shared", "metal_shared"])
        config = two_hosts(ranks=[{"rank": 0, "host": "alpha", "memory_backend": "metal-shared"}, {"rank": 1, "host": "alpha", "memory_backend": "host"}])
        ctx = run.RunContext(config)
        ctx.providers = lambda host_id: {"host", "metal_shared"}
        self.assertTrue(all(p.skip_reason is None for p in cfg.expand_matrix(config, ctx)))
        ctx.providers = lambda host_id: {"host", "cuda"}
        self.assertTrue(all(p.skip_state == "unsupported" for p in cfg.expand_matrix(config, ctx)))

    def test_unreachable_provider_information_for_a_remote_host_does_not_block_planning(self):
        ranks = [{"rank": 0, "host": "beta", "memory_backend": "cuda-pinned"}, {"rank": 1, "host": "beta", "memory_backend": "metal-shared"}]
        config = two_hosts(ranks=ranks)
        points = cfg.expand_matrix(config, run.RunContext(config))
        self.assertTrue(all(p.skip_reason is None for p in points))

    def test_a_local_build_without_the_requested_device_provider_is_unsupported_not_failed(self):
        config = two_hosts()
        ctx = run.RunContext(config)
        ctx.providers = lambda host_id: {"host"} if host_id == "alpha" else None
        points = cfg.expand_matrix(config, ctx)
        self.assertTrue(all(p.skip_state == "unsupported" for p in points))


class BackwardCompatibility(SuiteTestCase):
    def old_record(self):
        from benchmarks.suite.importers import import_path
        rec = import_path(FIXTURES / "collective_all_reduce_w2.csv").records[0]
        for key in ("repository_commit", "repository_dirty", "build_provenance"):
            rec["benchmark"].pop(key, None)
        rec["artifacts"].pop("ranks", None)
        rec["environment"].pop("rank_interfaces", None)
        return rec

    def test_b0_b2_records_without_the_new_fields_still_validate_and_compare(self):
        from benchmarks.suite.reporting import compare, summary
        old, new = self.old_record(), self.old_record()
        self.assertEqual(schema.validate_record(old), [])
        new["environment"]["hosts"] = old["environment"]["hosts"] = [{"host_id": "h", "ranks": [0, 1], "os": "Linux", "arch": "x86_64", "cpu_model": "X"}]
        new["benchmark"]["build_provenance"] = {"status": "stamped"}
        new["artifacts"]["ranks"] = [{"rank": 0, "host_id": "h"}]
        new["environment"]["hosts"][0]["info_source"] = "probed"
        self.assertEqual(schema.validate_record(new), [])
        result = compare.compare([old], [new])
        self.assertEqual(len(result["comparisons"]), 1)
        self.assertTrue(summary.summarize([old, new]))

    def test_the_schema_version_did_not_change(self):
        self.assertEqual(schema.SCHEMA_VERSION, "tbccl-bench-result/1")

    @requires_yaml
    def test_old_config_files_without_the_new_keys_load(self):
        for name in ("smoke.yaml", "core-loopback.yaml"):
            c = cfg.load_config(REPO / "benchmarks/suite/configs" / name, auto_overlay=False)
            self.assertEqual(c.defaults["lease_s"], 30.0)
            self.assertTrue(c.defaults["provenance"]["require_same_version"])


class Cli(SuiteTestCase):
    def suite(self, *args):
        from benchmarks.suite.tests.test_provenance_health import fake_native_build
        fake_native_build(self.dir / "stub-build")  # no real build is needed to print commands or probe a configuration
        return subprocess.run([sys.executable, "-m", "benchmarks.suite", *args], cwd=REPO, capture_output=True, text=True, timeout=120, env={**os.environ, "TBCCL_BUILD_DIR": str(self.dir / "stub-build")})

    def test_provenance_stamp_command(self):
        from benchmarks.suite.tests.test_provenance_health import repo_with_commit
        build = self.dir / "b"
        fake_build(build)
        commit = repo_with_commit(self.dir / "repo")  # the checkout is named explicitly: this test must not depend on the tree it runs from having Git metadata
        r = self.suite("provenance", "stamp", "--build-dir", str(build), "--repo", str(self.dir / "repo"), "--executable", "tb_pingpong")
        self.assertEqual(r.returncode, 0, r.stderr)
        stamp = json.loads((build / "build-provenance.json").read_text())
        self.assertEqual(list(stamp["executables"]), ["tb_pingpong"])
        self.assertEqual(stamp["repository_commit"], commit)
        self.assertEqual(self.suite("provenance", "stamp", "--build-dir", str(self.dir / "empty"), "--repo", str(self.dir / "repo")).returncode, 2)

    def test_stamping_from_a_source_export_without_git_metadata_is_refused_not_invented(self):
        build = self.dir / "b"
        fake_build(build)
        export = self.dir / "export"
        export.mkdir()
        r = self.suite("provenance", "stamp", "--build-dir", str(build), "--repo", str(export), "--executable", "tb_pingpong")
        self.assertEqual(r.returncode, 2)
        self.assertIn("not a git checkout; cannot record a commit", r.stderr)
        self.assertFalse((build / "build-provenance.json").exists())

    @requires_yaml
    def test_dry_run_of_the_two_host_example_shows_the_supervised_commands(self):
        r = self.suite("run", "benchmarks/suite/configs/two-host.example.yaml", "--dry-run")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("<supervisor helper source>", r.stdout)
        self.assertIn("PHYSICAL", r.stdout)
        self.assertIn("unresolved placeholders", r.stdout)

    @requires_yaml
    def test_doctor_probe_on_a_loopback_config(self):
        r = self.suite("doctor", "benchmarks/suite/configs/smoke.yaml", "--probe")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
