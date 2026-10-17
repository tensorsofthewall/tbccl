import copy
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import unittest
from pathlib import Path

from benchmarks.suite import config as cfg
from benchmarks.suite import health, provenance, schema
from benchmarks.suite import runner as run
from benchmarks.suite.preflight import HealthMonitor
from benchmarks.suite.remote import agent
from benchmarks.suite.tests.common import BASE, LOOPBACK_IF, REPO, SuiteTestCase, make_config
from benchmarks.suite.tests.test_remote import two_host_config


def fake_build(directory, version="0.6.0", wire="4", abi="1", name="tb_pingpong", build_type="Release", home=None):
    """A directory that looks like a TBCCL build: an executable, tbccl-info and a CMakeCache."""
    directory.mkdir(parents=True, exist_ok=True)
    exe = directory / name
    exe.write_text("#!/bin/sh\nexit 0\n")
    info = directory / "tbccl-info"
    info.write_text(f"#!/bin/sh\ncat <<'EOF'\nTBCCL {version}\nPackage version:   {version}\nC ABI version:     {abi}\nWire protocol:     {wire}\nMemory providers:  host\nEOF\n")
    for f in (exe, info):
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    (directory / "CMakeCache.txt").write_text(f"CMAKE_BUILD_TYPE:STRING={build_type}\nCMAKE_GENERATOR:INTERNAL=Ninja\nCMAKE_CXX_COMPILER:FILEPATH={shutil.which('c++') or '/usr/bin/c++'}\n"
                                              "TBCCL_ENABLE_CUDA:BOOL=OFF\nTBCCL_ENABLE_METAL:BOOL=OFF\n" + (f"CMAKE_HOME_DIRECTORY:INTERNAL={home}\n" if home else ""))
    return exe


def fake_native_build(directory):
    """A build directory with a stand-in for every native executable the suite drives, so tests that print commands or probe a configuration need no real build."""
    from benchmarks.suite.adapters import REGISTRY
    last = None
    for name in sorted({a.caps.executable for a in REGISTRY.values() if a.caps.executable_in_build}):
        last = fake_build(directory, name=name)
    return last


def repo_with_commit(path):
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "f").write_text("x")
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid"}
    subprocess.run(["git", "-C", str(path), "add", "f"], check=True, env=env)
    subprocess.run(["git", "-C", str(path), "commit", "-q", "-m", "m"], check=True, env=env)
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


class BuildProvenance(SuiteTestCase):
    def probe(self, exe, declared=None):
        entry = agent.probe({"tb_pingpong": str(exe)})["executables"]["tb_pingpong"]
        return provenance.assess("tb_pingpong", entry, declared, controller_head="HEADCOMMIT")

    def test_without_a_stamp_the_build_is_unverified_but_still_identified(self):
        exe = fake_build(self.dir / "b")
        p = self.probe(exe)
        self.assertEqual((p["status"], p["build_commit"]), ("unverified", None))
        self.assertEqual((p["build_type"], p["tbccl_package_version"], p["c_abi"], p["wire_protocol"]), ("Release", "0.6.0", "1", "4"))
        self.assertEqual(len(p["executable_sha256"]), 64)
        self.assertIn("provenance stamp", p["reason"])
        self.assertTrue(p["compiler"])

    def built_from(self, repo, **kw):
        """A fake build configured from `repo`, executables newer than every file in it."""
        exe = fake_build(self.dir / "b", home=repo.resolve(), **kw)
        for f in repo.rglob("*"):
            if f.is_file() and ".git" not in f.parts:
                os.utime(f, (time.time() - 100, time.time() - 100))
        return exe

    def commit_src(self, repo):
        (repo / "src").mkdir(parents=True, exist_ok=True)
        (repo / "src" / "a.cpp").write_text("x")
        subprocess.run(["git", "-C", str(repo), "add", "src/a.cpp"], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "s"], check=True)

    def test_stamp_attributes_the_executable_to_the_stamped_commit_not_to_head(self):
        repo = self.dir / "repo"
        repo.mkdir()
        commit = repo_with_commit(repo)
        exe = self.built_from(repo)
        path, stamp = provenance.stamp_build(self.dir / "b", repo, ["tb_pingpong", "tbccl-info"])
        self.assertEqual(stamp["repository_commit"], commit)
        self.assertTrue(stamp["cmake_source_dir_matches_repository"])
        p = self.probe(exe)
        self.assertEqual((p["status"], p["build_commit"], p["source"]), ("stamped", commit, "build stamp"))
        self.assertIn("not a cryptographic proof", p["reason"])
        self.assertFalse(p["matches_controller_head"])  # the controller's HEAD is something else: reported, not assumed equal

    def test_a_stamp_from_another_checkout_does_not_attribute_the_build(self):
        build_repo, other = self.dir / "build_repo", self.dir / "other"
        build_repo.mkdir()
        other.mkdir()
        repo_with_commit(build_repo)
        repo_with_commit(other)
        exe = self.built_from(build_repo)
        provenance.stamp_build(self.dir / "b", other, ["tb_pingpong"])  # stamping from the wrong checkout is recorded, not trusted
        p = self.probe(exe)
        self.assertEqual((p["status"], p["build_commit"]), ("unverified", None))
        self.assertIn("different source directory", p["reason"])

    def test_a_build_not_shown_to_come_from_the_checkout_is_unverified(self):
        repo = self.dir / "repo"
        repo.mkdir()
        repo_with_commit(repo)
        exe = fake_build(self.dir / "b")  # no CMAKE_HOME_DIRECTORY at all
        provenance.stamp_build(self.dir / "b", repo, ["tb_pingpong"])
        p = self.probe(exe)
        self.assertEqual(p["status"], "unverified")
        self.assertIn("not shown to be configured from the stamped checkout", p["reason"])

    def test_sources_newer_than_the_executable_make_the_stamp_unacceptable(self):
        repo = self.dir / "repo"
        repo.mkdir()
        repo_with_commit(repo)
        self.commit_src(repo)
        exe = fake_build(self.dir / "b", home=repo.resolve())
        for f in (exe, self.dir / "b" / "tbccl-info"):
            os.utime(f, (time.time() - 1000, time.time() - 1000))
        (repo / "src" / "a.cpp").touch()  # a build input changed after the executables were produced
        provenance.stamp_build(self.dir / "b", repo, ["tb_pingpong", "tbccl-info"])
        p = self.probe(exe)
        self.assertEqual(p["status"], "unverified")
        self.assertIn("newer than the executables", p["reason"])

    def test_dirty_build_inputs_make_the_stamp_unacceptable(self):
        repo = self.dir / "repo"
        repo.mkdir()
        repo_with_commit(repo)
        self.commit_src(repo)
        exe = self.built_from(repo)
        (repo / "src" / "a.cpp").write_text("modified")
        os.utime(repo / "src" / "a.cpp", (time.time() - 500, time.time() - 500))
        provenance.stamp_build(self.dir / "b", repo, ["tb_pingpong", "tbccl-info"])
        p = self.probe(exe)
        self.assertEqual(p["status"], "unverified")
        self.assertIn("uncommitted changes", p["reason"])

    def test_changes_outside_the_build_inputs_do_not_matter(self):
        repo = self.dir / "repo"
        repo.mkdir()
        repo_with_commit(repo)
        exe = self.built_from(repo)
        (repo / "f").write_text("changed outside the build inputs")
        provenance.stamp_build(self.dir / "b", repo, ["tb_pingpong", "tbccl-info"])
        self.assertEqual(self.probe(exe)["status"], "stamped")

    def test_a_rebuilt_executable_no_longer_matches_its_stamp(self):
        repo = self.dir / "repo"
        repo.mkdir()
        repo_with_commit(repo)
        exe = self.built_from(repo)
        provenance.stamp_build(self.dir / "b", repo, ["tb_pingpong"])
        exe.write_text("#!/bin/sh\nexit 1\n")
        p = self.probe(exe)
        self.assertEqual((p["status"], p["build_commit"]), ("unverified", None))
        self.assertIn("does not match", p["reason"])

    def test_declared_commit_is_labelled_declared(self):
        p = self.probe(fake_build(self.dir / "b"), {"commit": "abc123"})
        self.assertEqual((p["status"], p["build_commit"], p["source"]), ("declared", "abc123", "host configuration"))
        self.assertIn("not checked against the binary", p["reason"])

    def test_stamp_needs_a_git_checkout_and_executables(self):
        with self.assertRaises(ValueError):
            provenance.stamp_build(self.dir, self.dir, ["tb_pingpong"])
        repo = self.dir / "repo"
        repo.mkdir()
        repo_with_commit(repo)
        with self.assertRaises(ValueError):
            provenance.stamp_build(self.dir, repo, ["tb_pingpong"])

    def test_missing_executable(self):
        entry = agent.probe({"tb_pingpong": str(self.dir / "nope")})["executables"]["tb_pingpong"]
        self.assertEqual(provenance.assess("tb_pingpong", entry)["reason"], "executable not found")

    def test_cmake_cache_and_info_parsing(self):
        fake_build(self.dir / "b", version="1.2.3", wire="9")
        info = agent.cmake_cache_info(str(self.dir / "b"))
        self.assertEqual((info["build_type"], info["generator"], info["cuda_enabled"]), ("Release", "Ninja", "OFF"))
        self.assertEqual(agent.tbccl_info(str(self.dir / "b" / "tbccl-info"))["wire_protocol"], "9")


class HostConsistency(SuiteTestCase):
    def per_host(self, a, b):
        def entry(v):
            return {"executables": {"tb_pingpong": {"tbccl_package_version": v[0], "c_abi": v[1], "wire_protocol": v[2], "status": v[3], "reason": "r", "build_commit": v[4]}}, "error": None}
        return {"h1": entry(a), "h2": entry(b)}

    def test_different_versions_on_two_hosts_are_a_problem(self):
        problems = provenance.consistency(self.per_host(("0.6.0", "1", "4", "stamped", "a"), ("0.6.0", "1", "5", "stamped", "a")), {"require_same_version": True})
        self.assertEqual(len(problems), 1)
        self.assertIn("wire_protocol differs", problems[0])
        self.assertEqual(provenance.consistency(self.per_host(("0.6.0", "1", "4", "stamped", "a"), ("0.6.1", "1", "4", "stamped", "a")), {"require_same_version": False}), [])

    def test_different_commits_are_noted_not_rejected(self):
        ph = self.per_host(("0.6.0", "1", "4", "stamped", "aaaaaaaaaaaa"), ("0.6.0", "1", "4", "stamped", "bbbbbbbbbbbb"))
        self.assertEqual(provenance.consistency(ph, {"require_same_version": True}), [])
        self.assertIn("build commits differ", provenance.commit_notes(ph)[0])

    def test_provenance_mismatch_when_verification_is_required(self):
        problems = provenance.consistency(self.per_host(("0.6.0", "1", "4", "stamped", "a"), ("0.6.0", "1", "4", "unverified", None)), {"require_stamped": True})
        self.assertEqual(len(problems), 1)
        self.assertIn("h2:tb_pingpong", problems[0])

    def test_the_older_policy_name_still_works(self):
        ph = self.per_host(("0.6.0", "1", "4", "stamped", "a"), ("0.6.0", "1", "4", "unverified", None))
        self.assertEqual(len(provenance.consistency(ph, {"require_verified": True})), 1)

    def test_an_unprobeable_host_is_a_problem(self):
        self.assertTrue(provenance.consistency({"h": {"error": "ssh: no route", "executables": {}}}, {}))

    def test_runner_refuses_a_run_whose_hosts_report_different_wire_protocols(self):
        wl = {"name": "p", "benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [64], "repetitions": 1}
        far_dir, near_dir = self.dir / "far", self.dir / "near"
        fake_build(far_dir, wire="5")
        fake_build(near_dir, wire="4")
        config = two_host_config(workloads=[wl])
        config.hosts["far"].executables_dir = str(far_dir)
        config.hosts["local"].executables_dir = str(near_dir)
        runner = run.Runner(config, output_dir=self.dir / "o", log=lambda *_: None)
        manifest, records = runner.run()
        self.assertEqual((manifest["status"], manifest["reason"]), ("failure", "preflight failed"))
        self.assertTrue(any("wire_protocol differs" in c["detail"] for c in manifest["preflight"]))
        self.assertEqual(records, [])

    def test_results_record_per_host_build_information(self):
        wl = {"name": "f", "benchmark": "fake", "operations": ["ok"], "sizes": [8], "rank_counts": [2], "repetitions": 1}
        manifest, records = run.Runner(two_host_config(workloads=[wl]), output_dir=self.dir / "o", log=lambda *_: None).run()
        self.assertEqual(schema.validate_record(records[0]), [])
        self.assertIn("repository_commit", records[0]["benchmark"])
        self.assertIsNone(records[0]["benchmark"]["source_commit"])  # fake adapter has no executable in a build tree, so nothing is attributed


class Signals(unittest.TestCase):
    def host(self, **expect):
        raw = copy.deepcopy(BASE)
        raw["hosts"] = {"local": {"address": "127.0.0.1", "interface": "eth0", "expect": expect}}
        raw["ranks"] = [{"rank": 0, "host": "local"}, {"rank": 1, "host": "local"}]
        return cfg.build_config(raw).hosts["local"]

    def wrapper(self, **kw):
        iface = {"name": "eth0", "present": True, "operstate": "up", "mtu": 9000, "addresses": ["127.0.0.1/8"]}
        iface.update(kw.pop("interface", {}))
        snap = {"boot_id": "boot-1", "kernel_log_available": True, "interface": {"carrier": "1", "counters": {"rx_errors": "0", "tx_errors": "0"}}}
        snap.update(kw.pop("snapshot", {}))
        return {"interface": iface, "gpu": kw.pop("gpu", {"available": False, "reason": "none"}), "snapshot": snap, "snapshot_error": None}

    def test_healthy_baseline(self):
        host = self.host(mtu=9000)
        events = [{"message": "Command line: BOOT_IMAGE=/vmlinuz root=/dev/sda1 rw quiet"}]
        sig = health.evaluate(host, self.wrapper(snapshot={"kernel_events": events, "pci_devices": [{"aer": {"aer_dev_correctable": "RxErr 0"}}]}))
        self.assertEqual({k: v["status"] for k, v in sig.items()}, {"interface_present": "ok", "link_state": "ok", "ip_configured": "ok", "mtu": "ok", "boot_identity": "ok",
                                                                    "error_counters": "ok", "aer_observability": "ok", "thunderbolt_pcie_errors": "ok"})
        self.assertEqual(health.preflight_problems(host, sig), ([], []))

    def test_unknown_aer_observability_is_a_warning_not_a_failure(self):
        host = self.host(mtu=9000)
        sig = health.evaluate(host, self.wrapper())
        self.assertEqual(sig["aer_observability"]["status"], "unavailable")
        failures, warnings = health.preflight_problems(host, sig)
        self.assertEqual(failures, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("aer_observability", warnings[0])

    def test_bad_signals_are_reported_with_their_reason(self):
        host = self.host(mtu=9000)
        sig = health.evaluate(host, self.wrapper(interface={"operstate": "down", "mtu": 1500, "addresses": []}, snapshot={"interface": {"carrier": "0"}}))
        failures, _ = health.preflight_problems(host, sig)
        text = " ".join(failures)
        self.assertIn("link_state", text)
        self.assertIn("mtu 1500, expected 9000", text)
        self.assertIn("ip_configured", text)

    def test_missing_interface_and_missing_snapshot_are_never_assumed_healthy(self):
        host = self.host()
        sig = health.evaluate(host, self.wrapper(interface={"present": False}))
        self.assertEqual(sig["interface_present"]["status"], "bad")
        sig = health.evaluate(host, {"interface": self.wrapper()["interface"], "gpu": None, "snapshot": None, "snapshot_error": "boom"})
        self.assertEqual(sig["boot_identity"]["status"], "unavailable")
        self.assertEqual(sig["thunderbolt_pcie_errors"]["status"], "unavailable")
        failures, warnings = health.preflight_problems(host, sig)
        self.assertTrue(any("boot_identity unavailable" in f for f in failures))  # critical and not allowed
        self.assertTrue(any("thunderbolt_pcie_errors" in w for w in warnings))
        allowed = self.host(allow_unavailable=["boot_identity"])
        failures, warnings = health.preflight_problems(allowed, health.evaluate(allowed, {"interface": self.wrapper()["interface"], "gpu": None, "snapshot": None, "snapshot_error": "x"}))
        self.assertFalse(any("boot_identity" in f for f in failures))

    def test_no_configured_interface_is_unavailable_not_ok(self):
        raw = copy.deepcopy(BASE)
        host = cfg.build_config(raw).hosts["local"]
        host.interface = None
        sig = health.evaluate(host, self.wrapper())
        self.assertEqual(sig["interface_present"]["status"], "unavailable")

    def test_gpu_availability_and_thermal(self):
        host = self.host(gpu_max_temp_c=80)
        cool = self.wrapper(gpu={"available": True, "gpus": [{"name": "G", "temperature_c": 60.0, "throttle_reasons": "Not Active"}]})
        hot = self.wrapper(gpu={"available": True, "gpus": [{"name": "G", "temperature_c": 91.0, "throttle_reasons": "0x0000000000000004"}]})
        self.assertEqual(health.evaluate(host, cool, True)["gpu_thermal"]["status"], "ok")
        self.assertEqual(health.evaluate(host, hot, True)["gpu_thermal"]["status"], "bad")
        self.assertEqual(health.evaluate(host, self.wrapper(), True)["gpu_available"]["status"], "unavailable")
        self.assertNotIn("gpu_thermal", health.evaluate(host, hot, False))

    def test_throttle_bitmask_decoding_matches_what_nvidia_smi_reports(self):
        self.assertEqual(health.throttle_findings("0x0000000000000001"), ([], []))  # GPU idle: normal
        self.assertEqual(health.throttle_findings("Not Active"), ([], []))
        self.assertEqual(health.throttle_findings("0x0000000000000004"), ([], ["software power cap"]))
        self.assertEqual(health.throttle_findings("0x0000000000000041")[0], ["hardware thermal slowdown"])
        self.assertEqual(sorted(health.throttle_findings("0x00000000000000a8")[0]), ["hardware power-brake slowdown", "hardware slowdown", "software thermal slowdown"])
        self.assertTrue(health.throttle_findings("[N/A]")[0])  # unreadable is not assumed healthy

    def test_an_idle_gpu_is_not_a_thermal_problem(self):
        host = self.host(gpu_max_temp_c=80)
        idle = self.wrapper(gpu={"available": True, "gpus": [{"name": "G", "temperature_c": 46.0, "throttle_reasons": "0x0000000000000001"}]})
        sig = health.evaluate(host, idle, True)["gpu_thermal"]
        self.assertEqual(sig["status"], "ok")
        self.assertIn("46 C", sig["detail"])

    def test_regressions_after_a_run(self):
        host = self.host()
        before = self.wrapper()
        cases = {"interface disappeared": self.wrapper(interface={"present": False}), "link state changed": self.wrapper(interface={"operstate": "down"}),
                 "MTU changed": self.wrapper(interface={"mtu": 1500}), "address was removed": self.wrapper(interface={"addresses": []}),
                 "boot identity changed": self.wrapper(snapshot={"boot_id": "boot-2"}), "rx_errors increased": self.wrapper(snapshot={"interface": {"counters": {"rx_errors": "3", "tx_errors": "0"}}})}
        for expected, after in cases.items():
            reasons = health.regressions(host, before, after)
            self.assertTrue(any(expected in r for r in reasons), (expected, reasons))
        self.assertEqual(health.regressions(host, before, self.wrapper()), [])

    def test_gpu_throttling_after_a_run_invalidates(self):
        host = self.host()
        ok = self.wrapper(gpu={"available": True, "gpus": [{"name": "G", "temperature_c": 50.0, "throttle_reasons": "Not Active"}]})
        bad = self.wrapper(gpu={"available": True, "gpus": [{"name": "G", "temperature_c": 95.0, "throttle_reasons": "Not Active"}]})
        self.assertTrue(health.regressions(host, ok, bad, needs_gpu=True))
        self.assertEqual(health.regressions(host, ok, bad, needs_gpu=False), [])


class MonitorPolicy(SuiteTestCase):
    def test_missing_baseline_snapshot_blocks_the_run(self):
        host = make_config().hosts["local"]

        def snap(h):
            raise RuntimeError("controller gone")

        monitor = HealthMonitor([host], snapshot_fn=snap)
        failures, _ = monitor.baseline_problems(monitor.snapshot())
        self.assertIn("baseline snapshot unavailable", failures[0])

    def test_default_comparison_uses_the_script_policy_and_own_checks(self):
        host = make_config().hosts["local"]
        host.interface = "eth0"
        wrappers = iter([{"interface": {"present": True, "operstate": "up", "mtu": 9000, "addresses": []}, "gpu": None, "snapshot": {"boot_id": "1"}, "snapshot_error": None},
                         {"interface": {"present": False, "operstate": None, "mtu": None, "addresses": []}, "gpu": None, "snapshot": {"boot_id": "1"}, "snapshot_error": None}])
        monitor = HealthMonitor([host], snapshot_fn=lambda h: next(wrappers))
        before, after = monitor.snapshot(), monitor.snapshot()
        # inject the script comparison so the test does not depend on a real snapshot's shape
        monitor_compare = lambda b, a: {"stop_reasons": []}  # noqa: E731
        reasons, _ = HealthMonitor([host], snapshot_fn=lambda h: {}, compare_fn=monitor_compare).compare(before, after)
        self.assertEqual(reasons, [])
        reasons = health.regressions(host, before["local"], after["local"])
        self.assertIn("interface disappeared", reasons)

    def test_local_snapshot_end_to_end_is_read_only_and_shaped(self):
        host = make_config().hosts["local"]
        host.interface = LOOPBACK_IF
        wrapper = health.snapshot_host(host)
        self.assertIn("interface", wrapper)
        self.assertTrue(wrapper["interface"]["present"])
        sig = health.evaluate(host, wrapper)
        self.assertEqual(sig["interface_present"]["status"], "ok")

    def test_remote_host_without_health_source_cannot_be_monitored(self):
        host = two_host_config().hosts["far"]
        with self.assertRaises(RuntimeError) as ctx:
            health.snapshot_host(host)
        self.assertIn("cannot be monitored", str(ctx.exception))

    def test_remote_health_through_the_supervisor_helper(self):
        host = two_host_config().hosts["far"]
        host.interface, host.health_script = LOOPBACK_IF, str(REPO / "scripts" / "tb4_health_snapshot.py")
        wrapper = health.snapshot_host(host)
        self.assertTrue(wrapper["interface"]["present"])
        self.assertIn("gpu", wrapper)

    def test_physical_preflight_requires_health_monitoring_on_every_host(self):
        from benchmarks.suite.preflight import check_config
        raw = copy.deepcopy(BASE)
        raw["topology"] = {"kind": "direct-link", "interconnect": "x", "transport": "tcp", "physical": True}
        raw["hosts"] = {"a": {"address": "192.0.2.1"}, "b": {"address": "192.0.2.2", "launcher": {"kind": "prefix", "argv": ["x"]}}}
        raw["ranks"] = [{"rank": 0, "host": "a"}, {"rank": 1, "host": "b"}]
        config = cfg.build_config(raw)
        points = cfg.expand_matrix(config)
        checks = check_config(config, points, run.RunContext(config), allow_physical=True)
        by = {c.name: c for c in checks}
        self.assertEqual(by["health@a"].status, "ok")
        self.assertEqual(by["health@b"].status, "fail")


if __name__ == "__main__":
    unittest.main()
