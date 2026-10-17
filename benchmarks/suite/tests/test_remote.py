"""Supervised (remote-style) execution, exercised through a fake launcher that behaves like ssh: a command string run through a shell. Everything here is local
subprocesses; it tests the lifecycle logic, and is not a substitute for a run against a real remote host."""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
import unittest
from pathlib import Path

from benchmarks.suite import config as cfg
from benchmarks.suite import runner as run
from benchmarks.suite import schema
from benchmarks.suite.adapters.base import Group, Invocation
from benchmarks.suite.remote import agent, session
from benchmarks.suite.tests.common import BASE, FIXTURES, SuiteTestCase

LAUNCHER = str(Path(__file__).resolve().parent / "fake_launcher.py")


from benchmarks.suite.tests.common import pid_alive as alive  # noqa: E402


def wait_gone(pids, seconds=6.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not any(alive(p) for p in pids):
            return True
        time.sleep(0.05)
    return False


def two_host_config(**kw):
    raw = json.loads(json.dumps(BASE))
    raw["topology"] = {"kind": "direct-link", "interconnect": "loopback-simulated", "transport": "tcp", "physical": False}
    raw["hosts"] = {"local": {"address": "127.0.0.1"},
                    "far": {"address": "127.0.0.1", "python": sys.executable, "launcher": {"kind": "prefix", "argv": [sys.executable, LAUNCHER, "far"], "shell_join": True}}}
    raw["ranks"] = [{"rank": 0, "host": "local"}, {"rank": 1, "host": "far"}]
    raw["defaults"].update({"lease_s": 2.0, "term_grace_s": 1.0})
    raw.update(kw)
    return cfg.build_config(raw)


class Supervisor(SuiteTestCase):
    def host(self):
        return two_host_config().hosts["far"]

    def start(self, code, **kw):
        out, err = self.dir / "o", self.dir / "e"
        return session.RemoteProcess(self.host(), [sys.executable, "-c", code], kw.pop("env", {}), 1, out, err, "run-x", kw.pop("lease_s", 2.0), kw.pop("max_runtime_s", 30.0),
                                     kw.pop("grace_s", 1.0), python=sys.executable, **kw)

    def test_output_exit_code_and_independent_cleanup_check(self):
        p = self.start("import sys; print('hello'); sys.stderr.write('warn'); sys.exit(0)")
        p.stop() if False else None
        self.assertTrue(p.wait_for(lambda: p.returncode is not None, 10))
        self.assertEqual((p.returncode, (self.dir / "o").read_text().strip(), (self.dir / "e").read_text()), (0, "hello", "warn"))
        rep = p.report()
        self.assertTrue(rep["cleanup_confirmed"], rep)
        self.assertTrue(rep["independent_check"]["confirmed"])
        self.assertIn("confirmed by the host supervisor and by a separate check", rep["cleanup_note"])
        self.assertNotIn("lost_connection", [k for k, v in rep.items() if v is True])

    def test_nonzero_exit_is_reported(self):
        p = self.start("import sys; sys.exit(7)")
        self.assertTrue(p.wait_for(lambda: p.returncode is not None, 10))
        self.assertEqual(p.returncode, 7)

    def test_arguments_with_spaces_and_quotes_survive_the_launcher(self):
        p = self.start("import sys; print(sys.argv[1:])")
        p.stop()
        q = session.RemoteProcess(self.host(), [sys.executable, "-c", "import sys; print(sys.argv[1:])", "a b", "it's"], {}, 1, self.dir / "o2", self.dir / "e2", "r", 2.0, 30.0, python=sys.executable)
        self.assertTrue(q.wait_for(lambda: q.returncode is not None, 10))
        self.assertEqual((self.dir / "o2").read_text().strip(), "['a b', \"it's\"]")

    def test_cancel_kills_child_and_grandchild_and_cleanup_is_confirmed_twice(self):
        pids = self.dir / "pids"
        code = (f"import os,subprocess,sys,time; g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                f"open({str(pids)!r},'w').write(f'{{os.getpid()}} {{g.pid}}'); time.sleep(60)")
        p = self.start(code)
        self.assertTrue(p.wait_for(lambda: pids.exists() and len(pids.read_text().split()) == 2, 10))
        child, grandchild = map(int, pids.read_text().split())
        p.stop()
        self.assertTrue(wait_gone([child, grandchild]))
        rep = p.report()
        self.assertTrue(rep["cleanup_confirmed"], rep)

    def test_a_grandchild_in_a_new_session_is_found_by_its_ownership_token(self):
        pids = self.dir / "pids"
        code = (f"import os,subprocess,sys,time; g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'], start_new_session=True); "
                f"open({str(pids)!r},'w').write(f'{{os.getpid()}} {{g.pid}}'); time.sleep(60)")
        p = self.start(code)
        self.assertTrue(p.wait_for(lambda: pids.exists() and len(pids.read_text().split()) == 2, 10))
        child, escaped = map(int, pids.read_text().split())
        p.stop()
        self.assertTrue(wait_gone([child, escaped]))
        self.assertTrue(p.report()["cleanup_confirmed"])

    def test_unrelated_processes_are_never_signalled(self):
        bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.addCleanup(lambda: (bystander.kill(), bystander.wait()))
        p = self.start("import time; time.sleep(60)")
        time.sleep(0.5)
        p.stop()
        self.assertIsNone(bystander.poll())

    def test_lost_connection_kills_the_remote_process_group(self):
        pids = self.dir / "pids"
        code = f"import os,time; open({str(pids)!r},'w').write(str(os.getpid())); time.sleep(60)"
        p = self.start(code)
        self.assertTrue(p.wait_for(lambda: pids.exists() and pids.read_text(), 10))
        child = int(pids.read_text())
        os.killpg(p.proc.pid, signal.SIGKILL)  # the launcher (ssh) dies: stdin of the helper closes
        self.assertTrue(wait_gone([child]), "the supervised process survived a lost connection")
        self.assertTrue(p.wait_for(lambda: p.returncode is not None, 5))
        self.assertTrue(p.lost)
        self.assertNotEqual(p.returncode, 0)

    def test_controller_crash_does_not_leave_the_benchmark_running(self):
        pids = self.dir / "pids"
        script = textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r})
            from benchmarks.suite.remote import session
            from benchmarks.suite.tests.test_remote import two_host_config
            host = two_host_config().hosts['far']
            code = "import os,time; open({str(pids)!r},'w').write(str(os.getpid())); time.sleep(60)"
            p = session.RemoteProcess(host, [sys.executable, '-c', code], {{}}, 1, {str(self.dir / 'co')!r}, {str(self.dir / 'ce')!r}, 'run-crash', 2.0, 60.0, python=sys.executable)
            print('ready', flush=True)
            time.sleep(60)
        """)
        controller = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: (controller.kill(), controller.wait()))
        self.assertEqual(controller.stdout.readline().strip(), "ready")
        deadline = time.monotonic() + 10
        while not (pids.exists() and pids.read_text()) and time.monotonic() < deadline:
            time.sleep(0.05)
        child = int(pids.read_text())
        controller.kill()
        self.assertTrue(wait_gone([child], 8), "the benchmark outlived a crashed controller")

    def test_lease_expiry_bounds_a_controller_that_stops_heartbeating(self):
        argv = [sys.executable, "-c", agent_source(), "run", "--token", "lease-test", "--lease", "1", "--max-runtime", "60", "--grace", "1", "--", sys.executable, "-c", "import time; time.sleep(60)"]
        a = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)  # stdin stays open, no heartbeat is ever sent
        self.addCleanup(lambda: (a.kill(), a.wait()))
        started = json.loads(a.stdout.readline())
        t = time.monotonic()
        events = [json.loads(line) for line in a.stdout]
        self.assertLess(time.monotonic() - t, 8)
        self.assertIn("lease of 1 s expired", [e for e in events if e["ev"] == "exit"][0]["reason"])
        self.assertTrue(wait_gone([started["pid"]]))

    def test_runtime_cap_bounds_even_a_healthy_controller(self):
        argv = [sys.executable, "-c", agent_source(), "run", "--token", "cap-test", "--lease", "30", "--max-runtime", "1", "--grace", "1", "--", sys.executable, "-c", "import time; time.sleep(60)"]
        a = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: (a.kill(), a.wait()))
        a.stdin.write("H\n")
        a.stdin.flush()
        events = [json.loads(line) for line in a.stdout]
        self.assertIn("runtime cap", [e for e in events if e["ev"] == "exit"][0]["reason"])

    def test_watchdog_terminates_the_group_when_the_supervisor_itself_is_killed(self):
        pids = self.dir / "pids"
        code = (f"import os,subprocess,sys,time; g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                f"open({str(pids)!r},'w').write(f'{{os.getpid()}} {{g.pid}}'); time.sleep(60)")
        argv = [sys.executable, "-c", agent_source(), "run", "--token", "wd-test", "--lease", "30", "--max-runtime", "60", "--grace", "1", "--", sys.executable, "-c", code]
        a = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.addCleanup(lambda: (a.kill(), a.wait()))
        started = json.loads(a.stdout.readline())
        deadline = time.monotonic() + 10
        while not (pids.exists() and len(pids.read_text().split()) == 2) and time.monotonic() < deadline:
            time.sleep(0.05)
        child, grandchild = map(int, pids.read_text().split())
        os.kill(started["supervisor_pid"], signal.SIGKILL)  # no cleanup code of the supervisor runs
        self.assertTrue(wait_gone([child, grandchild], 10), "orphans survived the supervisor being killed")

    def test_cleanup_is_reported_unverified_when_the_host_cannot_be_asked_again(self):
        os.environ["FAKE_LAUNCHER_FAIL_CHECK"] = "1"
        self.addCleanup(os.environ.pop, "FAKE_LAUNCHER_FAIL_CHECK", None)
        p = self.start("print('x')")
        self.assertTrue(p.wait_for(lambda: p.returncode is not None, 10))
        rep = p.report()
        self.assertIsNone(rep["independent_check"]["confirmed"])
        self.assertIsNone(rep["cleanup_confirmed"])
        self.assertIn("NOT confirmed", rep["cleanup_note"])
        self.assertTrue(rep["supervisor_cleanup"]["confirmed"])

    def test_launcher_that_cannot_connect_is_a_failure_with_its_stderr_kept(self):
        raw = json.loads(json.dumps(BASE))
        raw["hosts"] = {"far": {"address": "127.0.0.1", "python": sys.executable, "launcher": {"kind": "prefix", "argv": [sys.executable, "-c", "import sys; sys.stderr.write('no route'); sys.exit(255)", "x"], "shell_join": False}}}
        raw["topology"] = {"kind": "unknown", "interconnect": "x", "transport": "tcp", "physical": False}
        raw["ranks"] = [{"rank": 0, "host": "far"}, {"rank": 1, "host": "far"}]
        host = cfg.build_config(raw).hosts["far"]
        p = session.RemoteProcess(host, ["true"], {}, 0, self.dir / "o", self.dir / "e", "r", 2.0, 5.0, python=sys.executable)
        self.assertTrue(p.wait_for(lambda: p.returncode is not None, 10))
        self.assertTrue(p.lost)
        self.assertEqual(p.returncode, 255)
        self.assertIn("no route", p.launcher_err_path.read_text())


def agent_source():
    code = session.agent_bootstrap()
    return code


class RunnerWithRemoteRanks(SuiteTestCase):
    """execute_group and the Runner across a 'local' and a 'far' (launcher-prefixed, supervised) host."""

    def group(self, *codes, persistent=(), listens=None, waits=None):
        invs = []
        for i, c in enumerate(codes):
            invs.append(Invocation(i, "r", "local" if i == 0 else "far", [sys.executable, "-c", c], persistent=(i in persistent), listens_on=listens if i == 0 else None,
                                   wait_for_port=waits if i == 1 else None))
        return Group(invs, ports=[])

    def go(self, g, timeout=15, **kw):
        return run.execute_group(g, self.dir / "rep", timeout, 5.0, two_host_config(), run_id="t", **kw)

    def test_success_with_per_rank_reports(self):
        o = self.go(self.group("print('a')", "print('b')"))
        self.assertEqual(o.exit_codes, {0: 0, 1: 0})
        self.assertEqual((self.dir / "rep" / "rank1.stdout").read_text().strip(), "b")
        self.assertFalse(o.rank_reports[0]["supervised"])
        self.assertTrue(o.rank_reports[1]["supervised"])
        self.assertTrue(o.rank_reports[1]["cleanup_confirmed"])
        self.assertEqual(o.cleanup_unconfirmed, [])

    def test_failed_remote_rank_invalidates_the_group_and_stops_the_local_rank(self):
        t = time.monotonic()
        o = self.go(self.group("import time; time.sleep(60)", "import sys; sys.stderr.write('boom'); sys.exit(3)"))
        self.assertLess(time.monotonic() - t, 12)
        self.assertEqual(o.failed_rank, 1)
        self.assertIn(0, o.terminated_by_runner)
        self.assertIn("boom", (self.dir / "rep" / "rank1.stderr").read_text())

    def test_failed_local_rank_stops_the_remote_rank_and_confirms_its_cleanup(self):
        pidfile = self.dir / "remote.pid"
        o = self.go(self.group("import sys; sys.exit(2)", f"import os,time; open({str(pidfile)!r},'w').write(str(os.getpid())); time.sleep(60)"))
        self.assertEqual(o.failed_rank, 0)
        if pidfile.exists():
            self.assertTrue(wait_gone([int(pidfile.read_text())]))
        started = o.rank_reports.get(1)
        if started:
            self.assertTrue(started["cleanup_confirmed"])

    def test_remote_timeout(self):
        o = self.go(self.group("import time; time.sleep(60)", "import time; time.sleep(60)"), timeout=1.5)
        self.assertTrue(o.timed_out)
        self.assertEqual(o.terminated_by_runner, {0, 1})
        self.assertTrue(all(r["cleanup_confirmed"] for r in o.rank_reports.values()))

    def test_remote_persistent_server_is_cancelled_cleanly(self):
        raw_cfg = two_host_config(ranks=[{"rank": 0, "host": "far"}, {"rank": 1, "host": "local"}])
        g = Group([Invocation(0, "server", "far", [sys.executable, "-c", "import time; time.sleep(60)"], persistent=True), Invocation(1, "client", "local", [sys.executable, "-c", "print('done')"])], [])
        o = run.execute_group(g, self.dir / "rep", 15, 5.0, raw_cfg, run_id="t")
        self.assertIsNone(o.failed_rank)
        self.assertEqual(o.exit_codes, {0: None, 1: 0})
        self.assertTrue(o.rank_reports[0]["cleanup_confirmed"])

    def test_remote_listener_readiness_is_proven_by_the_host_supervisor(self):
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        server = f"import socket,time; s=socket.socket(); s.bind(('127.0.0.1',{port})); s.listen(1); time.sleep(0.4); open({str(self.dir / 'served')!r},'w').write('x')"
        raw_cfg = two_host_config(ranks=[{"rank": 0, "host": "far"}, {"rank": 1, "host": "local"}])
        g = Group([Invocation(0, "server", "far", [sys.executable, "-c", server], persistent=True, listens_on=port),
                   Invocation(1, "client", "local", [sys.executable, "-c", "print('client ran after the listener was up')"], wait_for_port=port)], [])
        o = run.execute_group(g, self.dir / "rep", 15, 5.0, raw_cfg, run_id="t")
        self.assertIsNone(o.failed_rank, o.reason)
        self.assertIn("after the listener", (self.dir / "rep" / "rank1.stdout").read_text())

    def test_remote_process_that_never_listens_fails_the_group(self):
        g = Group([Invocation(0, "server", "far", [sys.executable, "-c", "import time; time.sleep(60)"], persistent=True, listens_on=45991),
                   Invocation(1, "client", "local", [sys.executable, "-c", "print('must not run')"], wait_for_port=45991)], [])
        o = run.execute_group(g, self.dir / "rep", 15, 1.0, two_host_config(ranks=[{"rank": 0, "host": "far"}, {"rank": 1, "host": "local"}]), run_id="t")
        self.assertEqual(o.failed_rank, 0)
        self.assertIn("did not start listening", o.reason)
        self.assertNotIn(1, o.stdout)

    def test_waiting_for_a_port_nobody_declares_fails_safely(self):
        g = Group([Invocation(0, "a", "local", [sys.executable, "-c", "import time; time.sleep(5)"]), Invocation(1, "b", "local", [sys.executable, "-c", "pass"], wait_for_port=40000)], [])
        o = run.execute_group(g, self.dir / "rep", 10, 1.0, two_host_config(), run_id="t")
        self.assertIsNotNone(o.failed_rank)
        self.assertIn("cannot be proven", o.reason)

    def test_cleanup_unconfirmed_is_surfaced_on_a_successful_measurement(self):
        os.environ["FAKE_LAUNCHER_FAIL_CHECK"] = "1"
        self.addCleanup(os.environ.pop, "FAKE_LAUNCHER_FAIL_CHECK", None)
        o = self.go(self.group("print('a')", "print('b')"))
        self.assertEqual(o.cleanup_unconfirmed, [1])


class MultiHostRunner(SuiteTestCase):
    def config(self, extra_workload=None, **kw):
        wl = {"name": "f", "benchmark": "fake", "operations": ["ok"], "sizes": [8], "rank_counts": [2], "repetitions": 1}
        wl.update(extra_workload or {})
        return two_host_config(workloads=[wl], **kw)

    def test_per_rank_metadata_and_host_information_in_records_and_manifest(self):
        runner = run.Runner(self.config(), output_dir=self.dir / "o", log=lambda *_: None)
        manifest, records = runner.run()
        self.assertEqual(manifest["status"], "success", manifest)
        r = records[0]
        self.assertEqual(schema.validate_record(r), [])
        ranks = {x["rank"]: x for x in r["artifacts"]["ranks"]}
        self.assertEqual((ranks[0]["host_id"], ranks[0]["supervised"]), ("local", False))
        self.assertEqual((ranks[1]["host_id"], ranks[1]["supervised"], ranks[1]["cleanup_confirmed"]), ("far", True, True))
        self.assertTrue(r["environment"]["multi_host"])
        self.assertEqual({h["host_id"] for h in r["environment"]["hosts"]}, {"local", "far"})
        self.assertIn("host_provenance", manifest)
        self.assertEqual(set(manifest["host_provenance"]), {"local", "far"})
        self.assertEqual(manifest["cleanup_unconfirmed"], [])
        for rank in ranks.values():
            self.assertTrue((self.dir / "o" / manifest["run_id"] / rank["stdout"]).exists())

    def test_unconfirmed_cleanup_is_listed_in_the_manifest(self):
        os.environ["FAKE_LAUNCHER_FAIL_CHECK"] = "1"
        self.addCleanup(os.environ.pop, "FAKE_LAUNCHER_FAIL_CHECK", None)
        manifest, records = run.Runner(self.config(), output_dir=self.dir / "o", log=lambda *_: None).run()
        self.assertEqual(manifest["cleanup_unconfirmed"][0]["host_id"], "far")
        self.assertTrue(any("could not be independently confirmed" in w for w in records[0]["measurements"]["warnings"]))

    def test_failure_policy_aborts_the_run(self):
        wl = [{"name": "bad", "benchmark": "fake", "operations": ["fail_rank1"], "sizes": [8], "rank_counts": [2], "repetitions": 1},
              {"name": "after", "benchmark": "fake", "operations": ["ok"], "sizes": [8], "rank_counts": [2], "repetitions": 1}]
        config = two_host_config(workloads=wl, defaults={"warmup": 1, "iterations": 5, "repetitions": 1, "timeout_s": 10, "failure_policy": {"on_failure": "abort_run"}})
        manifest, records = run.Runner(config, output_dir=self.dir / "o", log=lambda *_: None).run()
        self.assertEqual([r["status"]["state"] for r in records], ["failure", "skipped"])
        self.assertEqual(manifest["status"], "aborted")
        self.assertIn("failure policy", manifest["reason"])

    def test_partial_failure_leaves_no_valid_measurement(self):
        wl = {"name": "bad", "benchmark": "fake", "operations": ["fail_rank1"], "sizes": [8], "rank_counts": [2], "repetitions": 2}
        manifest, records = run.Runner(two_host_config(workloads=[wl]), output_dir=self.dir / "o", log=lambda *_: None).run()
        for r in records:
            self.assertEqual(r["status"]["state"], "failure")
            self.assertIsNone(r["measurements"])
            ranks = {x["rank"]: x for x in r["artifacts"]["ranks"]}
            self.assertEqual(ranks[1]["exit_code"], 3)
        self.assertEqual(manifest["status"], "failure")


class HealthPolicyInRuns(SuiteTestCase):
    def config(self):
        wl = [{"name": "a", "benchmark": "fake", "operations": ["ok"], "sizes": [8], "rank_counts": [2], "repetitions": 1}]
        return two_host_config(workloads=wl)

    def test_a_bad_baseline_stops_the_run_before_anything_is_measured(self):
        from benchmarks.suite.preflight import HealthMonitor
        config = self.config()
        down = {"interface": {"present": True, "operstate": "down", "mtu": 9000, "addresses": []}, "gpu": None, "snapshot": {"boot_id": "1"}, "snapshot_error": None}
        config.hosts["local"].interface = "eth0"
        monitor = HealthMonitor([config.hosts["local"]], snapshot_fn=lambda h: down)
        manifest, records = run.Runner(config, output_dir=self.dir / "o", log=lambda *_: None, monitor=monitor).run()
        self.assertEqual(manifest["status"], "aborted")
        self.assertIn("baseline health not acceptable", manifest["reason"])
        self.assertEqual([r["status"]["state"] for r in records], ["skipped"])
        self.assertTrue(manifest["health"]["baseline_signals"])

    def test_a_missing_baseline_snapshot_stops_the_run(self):
        from benchmarks.suite.preflight import HealthMonitor

        def broken(host):
            raise RuntimeError("cannot reach host")

        manifest, records = run.Runner(self.config(), output_dir=self.dir / "o", log=lambda *_: None, monitor=HealthMonitor([self.config().hosts["far"]], snapshot_fn=broken)).run()
        self.assertEqual(manifest["status"], "aborted")
        self.assertIn("baseline snapshot unavailable", manifest["reason"])

    def test_a_missing_after_snapshot_invalidates_the_measurement(self):
        from benchmarks.suite.preflight import HealthMonitor
        calls = []

        def flaky(host):
            calls.append(1)
            if len(calls) > 2:  # baseline and the pre-point snapshot work; the one after the measurement does not
                raise RuntimeError("lost the host")
            return {"n": 1}

        config = self.config()
        monitor = HealthMonitor([config.hosts["far"]], snapshot_fn=flaky, compare_fn=lambda b, a: {"stop_reasons": []})
        manifest, records = run.Runner(config, output_dir=self.dir / "o", log=lambda *_: None, monitor=monitor).run()
        self.assertEqual(records[0]["status"]["state"], "invalid")
        self.assertIsNone(records[0]["measurements"])
        self.assertIn("snapshot unavailable", records[0]["status"]["reason"])

    def test_a_missing_pre_point_snapshot_stops_before_any_traffic(self):
        from benchmarks.suite.preflight import HealthMonitor
        calls = []

        def flaky(host):
            calls.append(1)
            if len(calls) > 1:
                raise RuntimeError("lost the host")
            return {"n": 1}

        config = self.config()
        monitor = HealthMonitor([config.hosts["far"]], snapshot_fn=flaky, compare_fn=lambda b, a: {"stop_reasons": []})
        manifest, records = run.Runner(config, output_dir=self.dir / "o", log=lambda *_: None, monitor=monitor).run()
        self.assertEqual([r["status"]["state"] for r in records], ["skipped"])
        self.assertIn("pre-point health gate", records[0]["status"]["reason"])
        self.assertEqual(manifest["status"], "aborted")
        self.assertEqual(records[0]["artifacts"]["ranks"], [])  # nothing was started

    def test_a_local_rank_can_be_supervised_too(self):
        wl = [{"name": "a", "benchmark": "fake", "operations": ["ok"], "sizes": [8], "rank_counts": [2], "repetitions": 1}]
        raw = two_host_config(workloads=wl)
        raw.hosts["local"].launcher = {"kind": "local", "supervised": True}
        raw.hosts["local"].python = sys.executable
        manifest, records = run.Runner(raw, output_dir=self.dir / "o", log=lambda *_: None).run()
        ranks = {r["rank"]: r for r in records[0]["artifacts"]["ranks"]}
        self.assertTrue(ranks[0]["supervised"] and ranks[1]["supervised"])
        self.assertEqual(records[0]["status"]["state"], "success")


class PortableProcessTable(unittest.TestCase):
    def test_the_ps_based_table_used_where_proc_is_missing_sees_this_process(self):
        rows = agent._ps_table()
        self.assertIn(os.getpid(), {pid for pid, _, _ in rows})
        mine = [r for r in rows if r[0] == os.getpid()][0]
        self.assertEqual(mine[1], os.getpgid(0))




from benchmarks.suite import inventory  # noqa: E402


@unittest.skipUnless(inventory.find_executable("tbccl_collective_bench", ["build-release", "build-bench", "build"]) or os.environ.get("TBCCL_BUILD_DIR"),
                     "no built TBCCL benchmark executables (set TBCCL_BUILD_DIR)")
class NativeBenchmarksThroughTheSupervisor(SuiteTestCase):
    """The real executables with rank 1 started through the supervised launcher path (still all on this machine): a functional check of the supervision,
    readiness and per-rank capture with real software. It does not exercise a network and is not a remote-host validation."""

    def test_every_integrated_benchmark_with_rank_1_supervised(self):
        exe = inventory.find_executable("tbccl_collective_bench", ["build-release", "build-bench", "build"]).parent
        wl = [{"name": "tcp", "benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [64]},
              {"name": "small", "benchmark": "small_message_latency", "operation": "p2p", "sizes": [64]},
              {"name": "tensor", "benchmark": "tensor_transfer", "operation": "latency_floor", "sizes": [64]},
              {"name": "coll", "benchmark": "collective_bench", "operation": "all_gather", "algorithms": ["ring"], "sizes": [64]}]
        config = two_host_config(workloads=[dict(w, repetitions=1) for w in wl], defaults={"warmup": 2, "iterations": 10, "repetitions": 1, "timeout_s": 60, "lease_s": 5.0})
        config.hosts["far"].executables_dir = str(exe)
        manifest, records = run.Runner(config, output_dir=self.dir / "o", log=lambda *_: None).run()
        self.assertEqual([r["status"]["state"] for r in records], ["success"] * 4, [r["status"]["reason"] for r in records])
        for r in records:
            ranks = {x["rank"]: x for x in r["artifacts"]["ranks"]}
            self.assertTrue(ranks[1]["supervised"])
            self.assertTrue(ranks[1]["cleanup_confirmed"], ranks[1])
            self.assertEqual(schema.validate_record(r), [])
        self.assertEqual(manifest["cleanup_unconfirmed"], [])
        self.assertEqual(manifest["host_provenance"]["far"]["executables"]["tbccl_collective_bench"]["build_type"], "Release")

    def test_tcp_server_on_the_supervised_host_is_stopped_after_the_client_finishes(self):
        exe = inventory.find_executable("tb_pingpong", ["build-release", "build-bench", "build"]).parent
        wl = [{"name": "tcp", "benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [64], "repetitions": 1}]
        config = two_host_config(workloads=wl, ranks=[{"rank": 0, "host": "far"}, {"rank": 1, "host": "local"}])
        config.hosts["far"].executables_dir = str(exe)
        manifest, records = run.Runner(config, output_dir=self.dir / "o", log=lambda *_: None).run()
        self.assertEqual(records[0]["status"]["state"], "success", records[0]["status"])
        server = records[0]["artifacts"]["ranks"][0]
        self.assertTrue(server["terminated_by_runner"] and server["cleanup_confirmed"])




class RealLauncherHarnessRehearsal(SuiteTestCase):
    """The harness used against real hosts (validation/real_launcher.py), run against the local stand-in launcher so the harness itself stays correct."""

    def test_every_scenario_passes_against_the_stand_in_launcher(self):
        from benchmarks.suite.validation import real_launcher
        report = self.dir / "report.json"
        code = real_launcher.main(["--launcher-argv", json.dumps([sys.executable, LAUNCHER, "x"]), "--python", sys.executable, "--report", str(report)])
        results = json.loads(report.read_text())["results"]
        self.assertEqual([r["scenario"] for r in results if not r["passed"]], [])
        self.assertGreaterEqual(len(results), 20)
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
