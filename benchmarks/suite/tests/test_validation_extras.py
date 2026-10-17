"""B3A.1: token scan on hosts without /proc, honest cleanup claims, control-path warnings, timing isolation, per-host reporting, shared probes."""
import json
import os
import subprocess
import sys
import time
import unittest

from benchmarks.suite import health, inventory
from benchmarks.suite import runner as run
from benchmarks.suite.preflight import HealthMonitor, check_config
from benchmarks.suite.remote import agent, session
from benchmarks.suite.reporting import summary
from benchmarks.suite.tests.common import SuiteTestCase
from benchmarks.suite.tests.test_remote import LAUNCHER, two_host_config


class TagScanWithoutProc(unittest.TestCase):
    """The macOS path (`ps -E`) driven by canned output, plus its real behaviour on Linux where `ps` exists. The canned part is a simulation, not macOS validation."""

    def fake_ps(self, text):
        original = subprocess.run

        def run_(argv, *a, **kw):
            if argv[:2] == ["ps", "-axEww"]:
                return subprocess.CompletedProcess(argv, 0, stdout=text, stderr="")
            return original(argv, *a, **kw)

        return run_

    def test_exact_token_entry_is_matched_and_lookalikes_are_not(self):
        token = "run1.1.abc"
        text = (f"  101 /usr/bin/python3 -c x HOME=/h {agent.TAG_ENV}={token} PATH=/bin\n"
                f"  102 /usr/bin/python3 -c y {agent.TAG_ENV}={token}x PATH=/bin\n"
                f"  103 /usr/bin/python3 -c z --token {token}\n"
                f"  104 /bin/sleep 5 {agent.TAG_ENV}={token}\n")
        original = subprocess.run
        subprocess.run = self.fake_ps(text)
        try:
            self.assertEqual(agent._tagged_ps(token), [101, 104])
        finally:
            subprocess.run = original

    def test_unreadable_ps_means_the_scan_is_unavailable_not_empty(self):
        original = subprocess.run
        subprocess.run = self.fake_ps("")
        try:
            self.assertIsNone(agent._tagged_ps("t"))
        finally:
            subprocess.run = original

    def test_the_scan_is_only_trusted_after_it_finds_a_process_that_it_started(self):
        self.assertTrue(agent.tag_scan_proven())  # Linux, real
        original, agent._TAG_SCAN_PROVEN = agent._TAG_SCAN_PROVEN, None
        monkey = agent.tagged_members
        agent.tagged_members = lambda token: []  # a platform where the scan silently finds nothing
        try:
            self.assertFalse(agent.tag_scan_proven())
        finally:
            agent.tagged_members, agent._TAG_SCAN_PROVEN = monkey, original


class CleanupClaims(SuiteTestCase):
    def test_without_a_working_token_scan_cleanup_is_never_confirmed(self):
        host = two_host_config().hosts["far"]
        p = session.RemoteProcess(host, [sys.executable, "-c", "print('x')"], {}, 1, self.dir / "o", self.dir / "e", "r", 2.0, 30.0, python=sys.executable)
        self.assertTrue(p.wait_for(lambda: p.returncode is not None, 10))
        p.cleanup = dict(p.cleanup, tag_scan=False)  # as reported by a host where the scan cannot be proven
        rep = p.report()
        self.assertIsNone(rep["cleanup_confirmed"])
        self.assertFalse(rep["escaped_descendants_detectable"])
        self.assertIn("escaped descendants are not ruled out", rep["cleanup_note"])

    def test_local_handles_make_the_same_claim(self):
        h = run.LocalHandle([sys.executable, "-c", "pass"], dict(os.environ), self.dir / "o", self.dir / "e", str(self.dir), "tok-1", "local")
        h.proc.wait()
        original, agent._TAG_SCAN_PROVEN = agent._TAG_SCAN_PROVEN, False
        try:
            rep = h.report()
        finally:
            agent._TAG_SCAN_PROVEN = original
            h.stop(False)
        self.assertIsNone(rep["cleanup_confirmed"])
        self.assertIn("cannot be detected", rep["cleanup_note"])


class ControlPath(unittest.TestCase):
    def test_launcher_over_a_data_address_is_flagged(self):
        from benchmarks.suite import config as cfg
        import copy
        from benchmarks.suite.tests.common import BASE
        raw = copy.deepcopy(BASE)
        raw["topology"] = {"kind": "direct-link", "interconnect": "x", "transport": "tcp", "physical": True}
        raw["hosts"] = {"a": {"address": "192.0.2.1"}, "b": {"address": "192.0.2.2", "launcher": {"kind": "prefix", "argv": ["remote-shell", "user@192.0.2.2"]}, "health_script": "/x", "health_command": ["x"]},
                        "c": {"address": "192.0.2.3", "launcher": {"kind": "prefix", "argv": ["remote-shell", "mgmt-name"]}, "health_command": ["x"]}}
        raw["ranks"] = [{"rank": 0, "host": "a"}, {"rank": 1, "host": "b"}]
        raw["workloads"] = [{"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [64]}]
        config = cfg.build_config(raw)
        checks = check_config(config, cfg.expand_matrix(config), run.RunContext(config), allow_physical=True)
        flagged = [c for c in checks if c.name.startswith("control path@")]
        self.assertEqual([c.name for c in flagged], ["control path@b"])
        self.assertIn("share the interface being measured", flagged[0].detail)


class TimingIsolation(SuiteTestCase):
    """Probes, health snapshots and cleanup checks must never overlap a benchmark's own run."""

    def test_snapshots_and_probes_happen_outside_every_measured_group(self):
        spans, events = [], []
        code = ("import os,time,sys; t0=time.time(); time.sleep(0.4); "
                f"open({str(self.dir / 'spans')!r}, 'a').write(f'{{t0}} {{time.time()}}\\n'); print('{{\"median_us\": 5.0, \"n\": 3}}')")
        from benchmarks.suite.adapters import REGISTRY
        adapter = REGISTRY["fake"]
        original = adapter.SCRIPTS["ok"]
        adapter.SCRIPTS["ok"] = code

        def snapshot(host):
            events.append(time.time())
            return {"n": len(events)}

        wl = [{"name": "a", "benchmark": "fake", "operations": ["ok"], "sizes": [8], "rank_counts": [2], "repetitions": 2}]
        config = two_host_config(workloads=wl)
        monitor = HealthMonitor([config.hosts["far"]], snapshot_fn=snapshot, compare_fn=lambda b, a: {"stop_reasons": []})
        try:
            run.Runner(config, output_dir=self.dir / "o", log=lambda *_: None, monitor=monitor).run()
        finally:
            adapter.SCRIPTS["ok"] = original
        spans = [tuple(map(float, line.split())) for line in (self.dir / "spans").read_text().splitlines()]
        self.assertEqual(len(spans), 2)
        self.assertEqual(len(events), 3)  # the baseline, then one before and one after the point
        for t in events:
            self.assertFalse(any(a <= t <= b for a, b in spans), f"a health snapshot at {t} overlapped a measured run {spans}")

    def test_the_helper_sends_only_two_byte_heartbeats_at_a_quarter_of_the_lease(self):
        host = two_host_config().hosts["far"]
        p = session.RemoteProcess(host, [sys.executable, "-c", "import time; time.sleep(1.2)"], {}, 1, self.dir / "o", self.dir / "e", "r", 4.0, 30.0, python=sys.executable)
        sent = []
        original = p.proc.stdin.write
        p.proc.stdin.write = lambda b: (sent.append(b), original(b))[1]
        p.wait_for(lambda: p.returncode is not None, 10)
        self.assertTrue(all(b == b"H\n" for b in sent))
        self.assertLessEqual(len(sent), 2)  # lease 4 s -> one heartbeat per second: at most a couple in a 1.2 s run


class HostsInReports(SuiteTestCase):
    def test_multi_host_report_lists_each_host_build_and_unconfirmed_cleanup(self):
        os.environ["FAKE_LAUNCHER_FAIL_CHECK"] = "1"
        self.addCleanup(os.environ.pop, "FAKE_LAUNCHER_FAIL_CHECK", None)
        wl = [{"name": "a", "benchmark": "fake", "operations": ["ok", "fail_rank1"], "sizes": [8], "rank_counts": [2], "repetitions": 1}]
        manifest, records = run.Runner(two_host_config(workloads=wl), output_dir=self.dir / "o", log=lambda *_: None).run()
        text = summary.to_markdown(summary.summarize(records), records, "T", manifest)
        self.assertIn("## Hosts", text)
        self.assertIn("| local |", text)
        self.assertIn("| far |", text)
        self.assertIn("Cleanup NOT confirmed", text)
        self.assertIn("rank 1@far: exit 3", text)


class SharedProbes(unittest.TestCase):
    def test_inventory_uses_the_helpers_implementations(self):
        self.assertIs(inventory.host_info, agent.host_info)
        self.assertIs(inventory.interface_info, agent.interface_info)
        self.assertIs(inventory.sha256_file, agent.sha256_file)

    def test_build_info_keeps_its_shape(self):
        exe = inventory.find_executable("tb_pingpong", ["build-release", "build-bench", "build"])
        if exe is None:
            self.skipTest("no build")
        b = inventory.build_info(exe)
        for key in ("build_dir", "build_type", "package_version", "wire_protocol", "c_abi", "providers", "version_text"):
            self.assertIn(key, b)


class CudaRuntimeBanner(unittest.TestCase):
    def test_both_banner_spellings_are_read(self):
        import re
        for banner, want in (("| NVIDIA-SMI 610.43.02   KMD Version: 610.43.02   CUDA UMD Version: 13.3 |", "13.3"), ("| NVIDIA-SMI 550.54  Driver Version: 550.54  CUDA Version: 12.4 |", "12.4")):
            m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", banner)
            self.assertEqual(m.group(1), want)

    def test_host_info_reports_the_runtime_on_this_machine_when_there_is_a_gpu(self):
        info = agent.host_info()
        if not info["gpu_model"] or "NVIDIA" not in info["gpu_model"]:
            self.skipTest("no NVIDIA GPU")
        self.assertRegex(info["cuda_runtime"] or "", r"CUDA \d+")


class HealthExtras(SuiteTestCase):
    def test_counter_reset_is_a_regression(self):
        host = two_host_config().hosts["local"]
        b = {"interface": {"present": True, "operstate": "up", "mtu": 1500, "addresses": []}, "gpu": None, "snapshot": {"boot_id": "1", "interface": {"counters": {"rx_bytes": "1000", "tx_bytes": "1000", "rx_errors": "0", "tx_errors": "0"}}}}
        a = {"interface": {"present": True, "operstate": "up", "mtu": 1500, "addresses": []}, "gpu": None, "snapshot": {"boot_id": "1", "interface": {"counters": {"rx_bytes": "5", "tx_bytes": "1000", "rx_errors": "0", "tx_errors": "0"}}}}
        self.assertTrue(any("went backwards" in r for r in health.regressions(host, b, a)))

    def test_inaccessible_remote_host_is_a_failed_baseline_through_a_real_launcher_failure(self):
        host = two_host_config().hosts["far"]
        host.health_script = "/nonexistent"
        host.launcher = {"kind": "prefix", "argv": [sys.executable, "-c", "import sys; sys.stderr.write('no route to host'); sys.exit(255)", "x"], "shell_join": False}
        monitor = HealthMonitor([host])
        failures, _ = monitor.baseline_problems(monitor.snapshot())
        self.assertIn("snapshot unavailable", failures[0])
        self.assertIn("health snapshot failed", failures[0])

    def test_real_gpu_state_is_read_when_a_gpu_exists(self):
        state = agent.gpu_state()
        if not state["available"]:
            self.skipTest("no nvidia-smi")
        g = state["gpus"][0]
        self.assertIsNotNone(g["temperature_c"])
        sig = health.evaluate(two_host_config().hosts["local"], {"interface": None, "gpu": state, "snapshot": {}}, needs_gpu=True)
        self.assertIn(sig["gpu_available"]["status"], ("ok",))
        self.assertIn(sig["gpu_thermal"]["status"], ("ok", "bad"))  # either is a real reading; a hot GPU is reported, not hidden


if __name__ == "__main__":
    unittest.main()
