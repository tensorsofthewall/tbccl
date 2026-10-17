"""The provider protocol (offline): description validation, discovery states, the security rules and the provenance a provider adds to result records. Providers here are tiny scripts
written to a temporary directory; no framework is installed or imported."""
import copy
import json
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

from benchmarks.suite import providers
from benchmarks.suite.runner import RunContext
from benchmarks.suite.adapters.base import Point
from benchmarks.suite.tests.common import REPO, SuiteTestCase, make_config

GOOD = {
    "provider_id": "torch-tbccl", "provider_protocol_version": 1, "provider_version": "0.2.0", "supported_workloads": ["p2p"], "supported_platforms": ["linux"],
    "supported_memory_kinds": ["host"], "supported_device_pairings": ["cpu-cpu"], "required_framework_versions": {"torch": ">=2.13,<2.14"},
    "required_runtime_capabilities": [], "optional_dependencies": [], "required_model_configuration": {"required": False}, "benchmark_configuration_schema": {"type": "object"},
    "supported_result_schema_versions": ["tbccl-bench-result/1"], "entry_points": {"communication": "comm_bench.py"}, "entry_point_sha256": {"communication": "a" * 64},
    "availability": {"status": "available", "reasons": [], "detected": {"torch": "2.13.0"}},
}


def description(**changes):
    d = copy.deepcopy(GOOD)
    d.update(changes)
    return d


def point(bench, op, size):
    return Point(workload="t", benchmark=bench, operation=op, algorithm=None, payload_bytes=size, datatype=None, reduce_op=None, rank_count=2,
                 ranks=[{"rank": i, "host": "local", "memory_backend": "host"} for i in range(2)], options={}, warmup=1, iterations=5, repetitions=1, timeout_s=20)


def write_provider(directory, body):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "provider.py").write_text(textwrap.dedent(body))
    return directory


class Validation(unittest.TestCase):
    def test_a_complete_description_is_accepted(self):
        self.assertEqual(providers.validate_description(GOOD, "torch-tbccl"), [])

    def test_every_required_field_is_required(self):
        for key in providers.REQUIRED_FIELDS:
            d = description()
            del d[key]
            self.assertTrue(any(key in p for p in providers.validate_description(d)), key)

    def test_wrong_types_are_reported(self):
        self.assertTrue(providers.validate_description(description(supported_workloads="p2p")))
        self.assertTrue(providers.validate_description(description(provider_protocol_version=True)))
        self.assertTrue(providers.validate_description([]))

    def test_an_unsupported_protocol_version_is_rejected(self):
        problems = providers.validate_description(description(provider_protocol_version=2))
        self.assertTrue(any("protocol 2 is not supported" in p for p in problems))

    def test_a_provider_that_writes_no_readable_result_schema_is_rejected(self):
        self.assertTrue(providers.validate_description(description(supported_result_schema_versions=["tbccl-bench-result/9"])))

    def test_the_provider_id_must_match_what_was_configured(self):
        self.assertTrue(any("expected 'vllm-tbccl'" in p for p in providers.validate_description(GOOD, "vllm-tbccl")))

    def test_an_unavailable_provider_must_say_why(self):
        self.assertTrue(providers.validate_description(description(availability={"status": "environment_missing", "reasons": []})))
        self.assertEqual(providers.validate_description(description(availability={"status": "environment_missing", "reasons": ["torch is not installed"]})), [])
        self.assertTrue(providers.validate_description(description(availability={"status": "fine", "reasons": []})))

    def test_entry_points_cannot_leave_the_provider_directory(self):
        for bad in ("/etc/passwd", "../elsewhere.py"):
            self.assertTrue(providers.validate_description(description(entry_points={"x": bad})), bad)

    def test_entry_point_hashes_are_optional_but_must_be_well_formed(self):
        d = description()
        del d["entry_point_sha256"]
        self.assertEqual(providers.validate_description(d), [])
        self.assertTrue(providers.validate_description(description(entry_point_sha256={"communication": "xyz"})))
        self.assertTrue(providers.validate_description(description(entry_point_sha256={"other": "b" * 64})))
        self.assertTrue(providers.validate_description(description(entry_point_sha256=[])))

    def test_stdout_must_be_one_json_object(self):
        desc, problems = providers.parse_description("INFO starting\n" + json.dumps(GOOD))
        self.assertIsNone(desc)
        self.assertTrue(any("logs must go to stderr" in p for p in problems))
        self.assertEqual(providers.parse_description(json.dumps(GOOD))[1], [])
        self.assertTrue(providers.parse_description("[1]")[1])
        self.assertTrue(providers.parse_description(" " * (providers.MAX_DESCRIPTION_BYTES + 1))[1])


class Describe(SuiteTestCase):
    def run_provider(self, body, expected="torch-tbccl", timeout=10):
        directory = write_provider(self.dir / "p", body)
        return providers.interpret(expected, providers.run_describe(sys.executable, directory, timeout))

    def test_a_working_provider_is_available(self):
        row = self.run_provider(f"import json, sys\nif sys.argv[1:] == ['describe']:\n    print(json.dumps({GOOD!r}))\n")
        self.assertEqual((row["status"], row["reasons"]), ("available", []))

    def test_a_missing_framework_is_a_state_with_a_reason_not_an_error(self):
        d = description(availability={"status": "environment_missing", "reasons": ["Python package 'torch' is not installed in this interpreter"], "detected": {"torch": None}})
        row = self.run_provider(f"import json\nprint(json.dumps({d!r}))\n")
        self.assertEqual(row["status"], "environment_missing")
        self.assertIn("torch", row["reasons"][0])

    def test_logs_on_stdout_make_the_provider_incompatible(self):
        row = self.run_provider(f"import json\nprint('loading torch...')\nprint(json.dumps({GOOD!r}))\n")
        self.assertEqual(row["status"], "incompatible")

    def test_a_provider_that_crashes_or_exits_nonzero_is_reported(self):
        row = self.run_provider("import sys\nsys.stderr.write('ImportError: no torch\\n')\nsys.exit(3)\n")
        self.assertEqual(row["status"], "describe_failed")
        self.assertIn("ImportError", row["reasons"][0])

    def test_a_provider_that_hangs_is_cut_off(self):
        row = self.run_provider("import time\ntime.sleep(60)\n", timeout=1)
        self.assertEqual(row["status"], "describe_failed")
        self.assertIn("Timeout", row["reasons"][0])

    def test_a_directory_without_an_entry_point_is_reported(self):
        (self.dir / "empty").mkdir()
        row = providers.interpret("torch-tbccl", providers.run_describe(sys.executable, self.dir / "empty"))
        self.assertEqual(row["status"], "describe_failed")

    def test_describe_runs_in_the_provider_directory_with_no_shell_and_no_stdin(self):
        row = self.run_provider("import json, os, sys\nd = %r\nd['availability']['detected'] = {'cwd': os.getcwd(), 'stdin_closed': sys.stdin.read() == ''}\nprint(json.dumps(d))\n" % GOOD)
        detected = row["description"]["availability"]["detected"]
        self.assertEqual(Path(detected["cwd"]).resolve(), (self.dir / "p").resolve())
        self.assertTrue(detected["stdin_closed"])


def far_config(hosts):
    return make_config(hosts=hosts, topology={"kind": "direct-link", "interconnect": "ethernet", "transport": "tcp", "physical": True},
                       ranks=[{"rank": 0, "host": "far", "memory_backend": "host"}, {"rank": 1, "host": "far", "memory_backend": "host"}])


class Environment(SuiteTestCase):
    def test_describe_does_not_see_tbccl_variables_from_the_shell(self):
        import os
        body = "import json, os\nd = %r\nd['availability']['detected'] = {'tbccl_vars': sorted(k for k in os.environ if k.startswith('TBCCL_')), 'other': os.environ.get('TBCCL_TEST_KEEP', 'unset')}\nprint(json.dumps(d))\n" % GOOD
        directory = write_provider(self.dir / "p", body)
        os.environ["TBCCL_PROVIDER_PROBE"] = "1"
        self.addCleanup(os.environ.pop, "TBCCL_PROVIDER_PROBE", None)
        row = providers.interpret("torch-tbccl", providers.run_describe(sys.executable, directory))
        self.assertEqual(row["description"]["availability"]["detected"]["tbccl_vars"], [])

    def test_the_remote_helper_strips_them_too(self):
        import os
        from benchmarks.suite.remote import agent
        directory = write_provider(self.dir / "q", "import json, os\nprint(json.dumps({'vars': sorted(k for k in os.environ if k.startswith('TBCCL_'))}))\n")
        os.environ["TBCCL_PROVIDER_PROBE"] = "1"
        self.addCleanup(os.environ.pop, "TBCCL_PROVIDER_PROBE", None)
        out = agent.describe_provider(sys.executable, str(directory))
        self.assertEqual(json.loads(out["stdout"]), {"vars": []})


class Discovery(SuiteTestCase):
    def config(self, executables=None):
        hosts = {"local": {"launcher": {"kind": "local"}, "address": "127.0.0.1", "interface": "lo", "executables": executables or {}}}
        return make_config(hosts=hosts)

    def test_unconfigured_providers_are_listed_with_what_is_missing(self):
        rows = providers.discover(self.config(), None)
        by = {r["provider_id"]: r for r in rows}
        self.assertEqual(by["native"]["status"], "available")
        for pid in providers.PROVIDERS:
            self.assertEqual(by[pid]["status"], "not_configured")
            self.assertIn("executables", by[pid]["reasons"][0])

    def test_nothing_is_searched_on_path_or_in_the_current_directory(self):
        # a provider.py in the working directory must not be picked up
        old = Path.cwd()
        write_provider(self.dir / "cwd", "print('{}')\n")
        try:
            import os
            os.chdir(self.dir / "cwd")
            rows = providers.discover(self.config(), None)
        finally:
            os.chdir(old)
        self.assertTrue(all(r["status"] == "not_configured" for r in rows if r["provider_id"] != "native"))

    def test_a_configured_provider_is_described_and_a_broken_one_does_not_hide_the_others(self):
        good = write_provider(self.dir / "good", f"import json\nprint(json.dumps({GOOD!r}))\n")
        bad = write_provider(self.dir / "bad", "raise SystemExit(5)\n")
        rows = providers.discover(self.config({"torch_python": sys.executable, "torch_bench_dir": str(good), "vllm_python": sys.executable, "vllm_bench_dir": str(bad)}), None)
        by = {r["provider_id"]: r for r in rows}
        self.assertEqual(by["torch-tbccl"]["status"], "available")
        self.assertEqual(by["vllm-tbccl"]["status"], "describe_failed")
        self.assertEqual(by["exo-tbccl"]["status"], "not_configured")
        self.assertEqual(by["native"]["status"], "available")

    def test_no_probe_runs_nothing(self):
        marker = self.dir / "ran"
        d = write_provider(self.dir / "p", f"open({str(marker)!r}, 'w').close()\n")
        rows = providers.discover(self.config({"torch_python": sys.executable, "torch_bench_dir": str(d)}), None, probe=False)
        self.assertFalse(marker.exists())
        self.assertEqual({r["provider_id"]: r["status"] for r in rows}["torch-tbccl"], "configured")

    def test_a_remote_host_without_a_helper_is_reported_unprobed(self):
        hosts = {"far": {"launcher": {"kind": "prefix", "argv": ["ssh", "far"]}, "address": "192.0.2.9", "interface": "eth0",
                         "executables": {"torch_python": "/venv/bin/python", "torch_bench_dir": "/provider"}}}
        rows = providers.discover(far_config(hosts), None)
        self.assertEqual({(r["provider_id"], r["status"]) for r in rows if r.get("host") == "far"}, {("torch-tbccl", "configured"), ("vllm-tbccl", "not_configured"), ("exo-tbccl", "not_configured")})

    def test_a_remote_host_is_described_through_the_injected_helper(self):
        hosts = {"far": {"launcher": {"kind": "prefix", "argv": ["ssh", "far"]}, "address": "192.0.2.9", "interface": "eth0",
                         "executables": {"torch_python": "/venv/bin/python", "torch_bench_dir": "/provider"}}}
        seen = []

        def helper(host, python, directory, timeout):
            seen.append((host.id, python, directory))
            return {"returncode": 0, "stdout": json.dumps(GOOD), "stderr_tail": ""}

        rows = providers.discover(far_config(hosts), None, remote_describe=helper)
        self.assertEqual(seen, [("far", "/venv/bin/python", "/provider")])
        self.assertEqual({r["provider_id"]: r["status"] for r in rows if r.get("host") == "far"}["torch-tbccl"], "available")

    def test_adapters_map_to_exactly_one_provider(self):
        from benchmarks.suite.adapters import REGISTRY
        owned = [a for spec in providers.PROVIDERS.values() for a in spec["adapters"]]
        self.assertEqual(len(owned), len(set(owned)))
        self.assertTrue(set(owned) <= set(REGISTRY), set(owned) - set(REGISTRY))
        self.assertIsNone(providers.provider_of("tcp_pingpong"))
        self.assertEqual(providers.provider_of("torch_ddp"), "torch-tbccl")
        self.assertEqual(providers.provider_of("vllm_pipeline"), "vllm-tbccl")
        self.assertEqual(providers.provider_of("exo_transport_ab"), "exo-tbccl")


class Provenance(SuiteTestCase):
    def test_a_configured_provider_is_recorded_with_what_it_reported(self):
        d = write_provider(self.dir / "p", f"import json\nprint(json.dumps({GOOD!r}))\n")
        hosts = {"local": {"launcher": {"kind": "local"}, "address": "127.0.0.1", "interface": "lo", "executables": {"torch_python": sys.executable, "torch_bench_dir": str(d)}}}
        ctx = RunContext(make_config(hosts=hosts))
        entries = ctx.provider_entries(point("torch_p2p", "p2p_ack", 4096))
        self.assertEqual(len(entries), 1)
        e = entries[0]
        self.assertEqual((e["provider_id"], e["protocol_version"], e["provider_version"], e["status"]), ("torch-tbccl", 1, "0.2.0", "available"))
        self.assertEqual(e["framework_versions"], {"torch": "2.13.0"})
        self.assertEqual(e["entry_point_sha256"], {"communication": "a" * 64})
        self.assertEqual(e["interpreter"], sys.executable)
        self.assertIs(ctx.provider_entries(point("torch_p2p", "p2p_rtt", 64))[0], entries[0])  # described once per host

    def test_the_compiled_extension_of_an_adapter_wheel_is_recorded_by_hash_apart_from_the_versions(self):
        d = description(availability={"status": "available", "reasons": [], "detected": {"torch": "2.13.0", "native_modules": {"torch_tbccl/_C.so": {"sha256": "ab" * 32, "bytes": 7}}}})
        p = write_provider(self.dir / "p", f"import json\nprint(json.dumps({d!r}))\n")
        hosts = {"local": {"launcher": {"kind": "local"}, "address": "127.0.0.1", "interface": "lo", "executables": {"torch_python": sys.executable, "torch_bench_dir": str(p)}}}
        e = RunContext(make_config(hosts=hosts)).provider_entries(point("torch_p2p", "p2p_ack", 4096))[0]
        self.assertEqual(e["framework_versions"], {"torch": "2.13.0"})
        self.assertEqual(e["native_modules"], {"torch_tbccl/_C.so": {"sha256": "ab" * 32, "bytes": 7}})

    def test_a_provider_that_could_not_report_native_modules_records_none_not_an_empty_set(self):
        d = description(availability={"status": "available", "reasons": [], "detected": {"torch": "2.13.0", "native_modules": None}})
        p = write_provider(self.dir / "p", f"import json\nprint(json.dumps({d!r}))\n")
        hosts = {"local": {"launcher": {"kind": "local"}, "address": "127.0.0.1", "interface": "lo", "executables": {"torch_python": sys.executable, "torch_bench_dir": str(p)}}}
        e = RunContext(make_config(hosts=hosts)).provider_entries(point("torch_p2p", "p2p_ack", 4096))[0]
        self.assertIsNone(e["native_modules"])

    def test_the_bundled_copy_is_recorded_as_such_and_versions_stay_unavailable(self):
        ctx = RunContext(make_config())
        e = ctx.provider_entries(point("torch_p2p", "p2p_ack", 4096))[0]
        self.assertEqual((e["status"], e["provider_version"], e["framework_versions"]), ("bundled_compat", None, {}))
        self.assertIn("unavailable", e["note"])
        self.assertIn("comm_bench.py", e["entry_point_sha256"])  # the bundled scripts are named by hash
        self.assertTrue(all(len(h) == 64 for h in e["entry_point_sha256"].values()))

    def test_a_provider_that_cannot_be_described_is_recorded_not_invented(self):
        d = write_provider(self.dir / "p", "raise SystemExit(2)\n")
        hosts = {"local": {"launcher": {"kind": "local"}, "address": "127.0.0.1", "interface": "lo", "executables": {"torch_python": sys.executable, "torch_bench_dir": str(d)}}}
        e = RunContext(make_config(hosts=hosts)).provider_entries(point("torch_p2p", "p2p_ack", 4096))[0]
        self.assertEqual((e["status"], e["provider_version"], e["framework_versions"]), ("describe_failed", None, {}))

    def test_native_benchmarks_carry_no_provider_entry(self):
        self.assertEqual(RunContext(make_config()).provider_entries(point("tcp_pingpong", "pingpong", 64)), [])


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "providers"


class Contract(unittest.TestCase):
    """The shared suite against the recorded descriptions of the three framework providers. The fixtures are content-pinned (the hash of each provider.py), never a branch or a commit, so
    this test does not follow the moving head of another repository. TBCCL_PROVIDER_CHECKOUTS (provider directories joined by the path separator) additionally compares live checkouts with the pins."""

    def fixture(self, provider_id):
        return json.loads((FIXTURES / f"{provider_id}.describe.json").read_text())

    def test_the_description_schema_accepts_the_valid_example_and_rejects_broken_ones(self):
        from benchmarks.suite import schema
        doc = json.loads((Path(providers.__file__).resolve().parent / "schemas" / "provider-description-v1.json").read_text())
        self.assertEqual(schema.validate_against(GOOD, doc), [])
        broken = description()
        del broken["availability"]
        self.assertTrue(schema.validate_against(broken, doc))
        self.assertTrue(schema.validate_against(description(availability={"status": "fine", "reasons": []}), doc))

    def test_each_recorded_provider_is_accepted_by_the_suite_and_by_the_schema(self):
        from benchmarks.suite import schema
        doc = json.loads((Path(providers.__file__).resolve().parent / "schemas" / "provider-description-v1.json").read_text())
        for pid in providers.PROVIDERS:
            d = self.fixture(pid)
            self.assertEqual(schema.validate_against(d, doc), [], pid)
            self.assertEqual(providers.validate_description(d, pid), [], pid)
            self.assertEqual(providers.interpret(pid, {"returncode": 0, "stdout": json.dumps(d), "stderr_tail": ""})["status"], d["availability"]["status"])

    def test_a_framework_free_environment_makes_every_provider_unavailable_with_reasons_not_fatal(self):
        for pid in providers.PROVIDERS:
            d = self.fixture(pid)
            self.assertEqual(d["availability"]["status"], "environment_missing", pid)
            self.assertTrue(d["availability"]["reasons"], pid)

    def test_providers_speak_the_protocol_and_schema_this_suite_reads(self):
        for pid in providers.PROVIDERS:
            d = self.fixture(pid)
            self.assertIn(d["provider_protocol_version"], providers.SUPPORTED_PROTOCOLS)
            self.assertTrue(set(d["supported_result_schema_versions"]) & set(providers.SUPPORTED_RESULT_SCHEMAS))

    def test_workloads_and_the_pins_are_consistent(self):
        pins = json.loads((FIXTURES / "PINNED.json").read_text())["providers"]
        self.assertEqual(set(pins), set(providers.PROVIDERS))
        for pid, pin in pins.items():
            d = self.fixture(pid)
            self.assertEqual(pin["entry_point_sha256"], d["entry_point_sha256"])
            self.assertEqual(len(pin["provider_py_sha256"]), 64)
            self.assertEqual(set(d["entry_point_sha256"]), set(d["entry_points"]))

    def test_the_suite_reads_the_scripts_every_provider_ships_from_one_directory(self):
        # the runner finds comm_bench.py, client.py and so on beside provider.py; a provider that moved them would break the adapters
        needed = {"torch-tbccl": {"comm_bench.py", "ddp_bench.py"}, "vllm-tbccl": {"client.py", "sampler.py", "fault.py", "pg_pingpong.py"}, "exo-tbccl": {"client.py", "sampler.py"}}
        for pid, names in needed.items():
            shipped = {Path(rel).name for rel in self.fixture(pid)["entry_points"].values()}
            self.assertTrue(names <= shipped, (pid, names - shipped))

    def test_the_compatibility_copies_equal_the_scripts_the_providers_ship(self):
        """Parity of the move: every workload script the suite still carries (`torch_bench`, `vllm_bench`, `exo_bench`) is byte-identical to the one the owning provider ships, except the two
        Exo modules whose only edit is the lines that locate the framework-neutral modules the provider now carries itself."""
        import hashlib
        suite = Path(providers.__file__).resolve().parent
        pins = json.loads((FIXTURES / "PINNED.json").read_text())["parity"]

        def exo_text(name, text):
            old, new = ('"vllm_bench" / "metrics.py"', '"stream_metrics.py"') if name == "metrics.py" else ('"vllm_bench" / "sampler.py"', '"process_sampler.py"')
            text = text.replace("Path(__file__).resolve().parent.parent / " + old, "Path(__file__).resolve().parent / " + new)
            return (text.replace("``vllm_bench.metrics``", "``stream_metrics.py``").replace("(``vllm_bench/sampler.py:", "(``process_sampler.py``:").replace("The vLLM benchmark's metrics module", "The shared stream metrics module")
                    .replace("the vLLM sampler (``vllm_bench/sampler.py``:", "the shared process sampler (``process_sampler.py``:"))

        checked = 0
        for repo, files in pins.items():
            for suite_path, pin in files.items():
                data = (suite / suite_path).read_bytes()
                if repo == "exo-tbccl" and suite_path in ("exo_bench/metrics.py", "exo_bench/sampler.py"):
                    data = exo_text(Path(suite_path).name, data.decode()).encode()
                self.assertEqual(hashlib.sha256(data).hexdigest(), pin["provider_sha256"], f"{suite_path} differs from {repo}/{pin['provider_file']}")
                checked += 1
        self.assertGreaterEqual(checked, 19)

    @unittest.skipUnless(__import__("os").environ.get("TBCCL_PROVIDER_CHECKOUTS"), "set TBCCL_PROVIDER_CHECKOUTS to compare live provider checkouts with the pins")
    def test_live_checkouts_match_the_pins(self):
        import hashlib
        import os
        pins = json.loads((FIXTURES / "PINNED.json").read_text())["providers"]
        found = {}
        for directory in os.environ["TBCCL_PROVIDER_CHECKOUTS"].split(os.pathsep):
            script = Path(directory) / providers.ENTRY_POINT
            raw = providers.run_describe(sys.executable, directory)
            d = json.loads(raw["stdout"])
            found[d["provider_id"]] = hashlib.sha256(script.read_bytes()).hexdigest()
        for pid, digest in found.items():
            self.assertEqual(digest, pins[pid]["provider_py_sha256"], f"{pid} changed since the compatibility fixtures were recorded; re-record them after reviewing the change")


class Preflight(SuiteTestCase):
    """A run refuses to start when a framework provider is missing or unusable on a participating host; native-only runs never mention providers."""

    def planned(self, executables, benchmark="torch_p2p", operation="p2p_ack", size=4096):
        from benchmarks.suite.runner import Runner
        hosts = {"local": {"launcher": {"kind": "local"}, "address": "127.0.0.1", "interface": "lo", "executables": executables}}
        config = make_config(hosts=hosts, workloads=[{"name": "w", "benchmark": benchmark, "operations": [operation], "sizes": [size]}])
        ctx = RunContext(config)
        return config, ctx, Runner(config, ctx).points()

    def test_an_unconfigured_provider_fails_the_preflight_and_names_the_keys(self):
        from benchmarks.suite.preflight import check_config, failed
        config, ctx, points = self.planned({})
        checks = [c for c in check_config(config, points, ctx) if c.name.startswith("provider ")]
        self.assertEqual([(c.name, c.status) for c in checks], [("provider torch-tbccl@local", "fail")])
        self.assertIn("torch_python and torch_bench_dir", checks[0].detail)
        self.assertTrue(failed(checks))

    def test_a_broken_provider_fails_with_its_reason_and_a_good_one_passes_with_its_versions(self):
        from benchmarks.suite.preflight import check_config
        bad = write_provider(self.dir / "bad", "import sys\nsys.stderr.write('no torch here\\n')\nsys.exit(4)\n")
        config, ctx, points = self.planned({"torch_python": sys.executable, "torch_bench_dir": str(bad)})
        (c,) = [c for c in check_config(config, points, ctx) if c.name.startswith("provider ")]
        self.assertEqual(c.status, "fail")
        self.assertIn("no torch here", c.detail)
        good = write_provider(self.dir / "good", f"import json\nprint(json.dumps({GOOD!r}))\n")
        config, ctx, points = self.planned({"torch_python": sys.executable, "torch_bench_dir": str(good)})
        (c,) = [c for c in check_config(config, points, ctx) if c.name.startswith("provider ")]
        self.assertEqual(c.status, "ok")
        self.assertIn("torch 2.13.0", c.detail)

    def test_an_environment_missing_provider_fails_the_preflight(self):
        from benchmarks.suite.preflight import check_config
        d = description(availability={"status": "environment_missing", "reasons": ["Python package 'torch' is not installed in this interpreter"]})
        missing = write_provider(self.dir / "m", f"import json\nprint(json.dumps({d!r}))\n")
        config, ctx, points = self.planned({"torch_python": sys.executable, "torch_bench_dir": str(missing)})
        (c,) = [c for c in check_config(config, points, ctx) if c.name.startswith("provider ")]
        self.assertEqual(c.status, "fail")
        self.assertIn("environment missing", c.detail)

    def test_a_dry_run_never_runs_a_provider(self):
        from benchmarks.suite.preflight import check_config
        marker = self.dir / "ran"
        d = write_provider(self.dir / "p", f"open({str(marker)!r}, 'w').close()\n")
        config, ctx, points = self.planned({"torch_python": sys.executable, "torch_bench_dir": str(d)})
        (c,) = [c for c in check_config(config, points, ctx, dry_run=True) if c.name.startswith("provider ")]
        self.assertEqual(c.status, "skip")
        self.assertFalse(marker.exists())

    def test_native_workloads_have_no_provider_checks(self):
        from benchmarks.suite.preflight import check_config
        config, ctx, points = self.planned({}, benchmark="tcp_pingpong", operation="pingpong", size=64)
        self.assertEqual([c for c in check_config(config, points, ctx) if c.name.startswith("provider ")], [])

    def test_doctor_probe_reports_the_provider(self):
        good = write_provider(self.dir / "good", f"import json\nprint(json.dumps({GOOD!r}))\n")
        import yaml
        raw = {"schema": "tbccl-bench-config/1", "name": "t", "defaults": {"warmup": 1, "iterations": 5, "repetitions": 1, "timeout_s": 20},
               "topology": {"kind": "loopback", "interconnect": "loopback", "transport": "tcp", "physical": False},
               "hosts": {"local": {"launcher": {"kind": "local"}, "address": "127.0.0.1", "interface": "lo", "executables": {"torch_python": sys.executable, "torch_bench_dir": str(good)}}},
               "ranks": [{"rank": 0, "host": "local", "memory_backend": "host"}, {"rank": 1, "host": "local", "memory_backend": "host"}],
               "workloads": [{"name": "w", "benchmark": "torch_p2p", "operations": ["p2p_ack"], "sizes": [4096]}]}
        (self.dir / "c.yaml").write_text(yaml.safe_dump(raw))
        out = subprocess.run([sys.executable, "-m", "benchmarks.suite", "doctor", str(self.dir / "c.yaml"), "--probe"], capture_output=True, text=True, cwd=str(REPO))
        self.assertIn("provider torch-tbccl@local", out.stdout)
        self.assertIn("torch-tbccl 0.2.0 (protocol 1)", out.stdout)


class ValidatePreflight(SuiteTestCase):
    """Provider `validate` in the central preflight: run in the provider's own interpreter, on the staged configuration, before anything launches."""

    def provider(self, validate_body):
        return write_provider(self.dir / "p", f"import json, sys\nGOOD = {GOOD!r}\nif sys.argv[1] == 'describe':\n    print(json.dumps(GOOD)); sys.exit(0)\n" + validate_body)

    def planned(self, directory, benchmark="torch_p2p", operation="p2p_ack", size=4096, options=None):
        from benchmarks.suite.runner import Runner
        hosts = {"local": {"launcher": {"kind": "local"}, "address": "127.0.0.1", "interface": "lo", "executables": {"torch_python": sys.executable, "torch_bench_dir": str(directory)}}}
        wl = {"name": "w", "benchmark": benchmark, "operations": [operation], "sizes": [size]}
        if options:
            wl["options"] = options
        config = make_config(hosts=hosts, workloads=[wl])
        ctx = RunContext(config)
        return config, ctx, Runner(config, ctx).points()

    def checks(self, directory, **kw):
        from benchmarks.suite.preflight import check_config
        config, ctx, points = self.planned(directory, **kw)
        return [c for c in check_config(config, points, ctx) if c.name.startswith("validate ")]

    def test_an_accepted_configuration_passes(self):
        d = self.provider("print(json.dumps({'ok': True, 'errors': []}))\n")
        (c,) = self.checks(d)
        self.assertEqual((c.status, c.name), ("ok", "validate torch-tbccl@local"))

    def test_a_rejected_configuration_fails_with_the_providers_reasons_before_anything_launches(self):
        from benchmarks.suite.preflight import failed
        d = self.provider("print(json.dumps({'ok': False, 'errors': ['bytes must be a multiple of 4', 'no such device']})); sys.exit(1)\n")
        cs = self.checks(d)
        self.assertEqual(cs[0].status, "fail")
        self.assertIn("bytes must be a multiple of 4; no such device", cs[0].detail)
        self.assertIn("rejects 1 of 1", cs[0].detail)
        self.assertTrue(failed(cs))

    def test_the_configuration_is_staged_as_a_file_and_describes_the_planned_point(self):
        d = self.provider("cfg = json.load(open(sys.argv[2]))\nprint(json.dumps({'ok': False, 'errors': [json.dumps(cfg, sort_keys=True)]})); sys.exit(1)\n")
        (c,) = self.checks(d, size=65536)
        self.assertIn('"workload": "p2p"', c.detail)
        self.assertIn('"bytes": 65536', c.detail)
        self.assertIn('"devices": "cpu,cpu"', c.detail)

    def test_it_runs_in_the_providers_own_interpreter(self):
        d = self.provider("print(json.dumps({'ok': False, 'errors': [sys.prefix]})); sys.exit(1)\n")  # the environment, not the executable path (which a venv may resolve differently)
        (c,) = self.checks(d)
        self.assertIn(sys.prefix, c.detail)

    def test_a_provider_without_validate_warns_and_a_crash_fails(self):
        d = self.provider("sys.exit(2)\n")
        self.assertEqual(self.checks(d)[0].status, "warn")
        d2 = write_provider(self.dir / "q", f"import json, sys\nif sys.argv[1] == 'describe':\n    print(json.dumps({GOOD!r})); sys.exit(0)\nsys.stderr.write('boom\\n'); sys.exit(7)\n")
        c = self.checks(d2)[0]
        self.assertEqual(c.status, "fail")
        self.assertIn("boom", c.detail)

    def test_identical_points_are_validated_once(self):
        marker = self.dir / "calls"
        d = self.provider(f"open({str(marker)!r}, 'a').write('x')\nprint(json.dumps({{'ok': True, 'errors': []}}))\n")
        self.checks(d)
        self.assertEqual(marker.read_text(), "x")

    def test_a_dry_run_never_runs_validate(self):
        from benchmarks.suite.preflight import check_config
        marker = self.dir / "ran"
        d = self.provider(f"open({str(marker)!r}, 'w').close()\nprint(json.dumps({{'ok': True, 'errors': []}}))\n")
        config, ctx, points = self.planned(d)
        names = [c.name for c in check_config(config, points, ctx, dry_run=True)]
        self.assertFalse([n for n in names if n.startswith("validate ")])
        self.assertFalse(marker.exists())

    def test_the_remote_helper_stages_and_validates_on_its_own_host(self):
        from benchmarks.suite.remote import agent
        d = self.provider("cfg = json.load(open(sys.argv[2]))\nprint(json.dumps({'ok': cfg.get('workload') == 'p2p', 'errors': ['unexpected ' + str(cfg.get('workload'))]}))\n")
        out = agent.validate_provider(sys.executable, str(d), [{"workload": "p2p"}, {"workload": "nope"}])
        self.assertEqual([providers.interpret_validation(o)["state"] for o in out], ["ok", "invalid"])
        self.assertEqual(providers.interpret_validation(out[1])["errors"], ["unexpected nope"])

    def test_a_ddp_point_maps_to_model_settings_not_a_tensor_payload(self):
        config, ctx, points = self.planned(self.dir, benchmark="torch_ddp", operation="train", size=526336, options={"layers": 2, "width": 256, "batch": 16, "steps": 8, "tag": "tiny"})
        cfg = providers.workload_config(points[0])
        self.assertEqual((cfg["workload"], cfg["layers"], cfg["steps"], "bytes" in cfg), ("ddp", 2, 8, False))

    def test_exo_activity_policy_maps_to_the_experimental_key(self):
        pt = point("exo_latency", "single_request", 1)
        pt.options = {"backend": "MlxTbccl", "env": {"EXO_TBCCL_ACTIVITY_MODE": "step"}, "tier": "fastpath"}
        self.assertEqual(providers.workload_config(pt), {"workload": "latency", "backend": "MlxTbccl", "tier": "fastpath", "activity_policy": "step"})
        pt.options = {"backend": "MlxTbccl"}
        self.assertEqual(providers.workload_config(pt), {"workload": "latency", "backend": "MlxTbccl", "tier": "baseline"})
        self.assertIsNone(providers.workload_config(point("tcp_pingpong", "pingpong", 64)))


class Isolation(unittest.TestCase):
    def test_the_core_imports_no_framework(self):
        code = "import sys, benchmarks.suite.cli, benchmarks.suite.providers; print(sorted(m for m in ('torch','vllm','exo','mlx','torch_tbccl','vllm_tbccl','exo_tbccl') if m in sys.modules))"
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(REPO))
        self.assertEqual(out.stdout.strip(), "[]", out.stderr)

    def test_the_providers_command_works_without_any_framework(self):
        out = subprocess.run([sys.executable, "-m", "benchmarks.suite", "providers", "--json"], capture_output=True, text=True, cwd=str(REPO))
        self.assertEqual(out.returncode, 0, out.stderr)
        rows = json.loads(out.stdout)
        self.assertEqual(rows[0]["provider_id"], "native")
        self.assertTrue(all(r["status"] == "not_configured" for r in rows[1:]))


if __name__ == "__main__":
    unittest.main()
