import json
import os
import signal
import socket
import sys
import time
import unittest

from benchmarks.suite import runner as run
from benchmarks.suite import schema
from benchmarks.suite.adapters.base import Group, Invocation
from benchmarks.suite.tests.common import SuiteTestCase, make_config


from benchmarks.suite.tests.common import pid_alive as alive  # noqa: E402


def group(*codes, persistent=()):
    return Group([Invocation(i, "r", "local", [sys.executable, "-c", c], persistent=(i in persistent)) for i, c in enumerate(codes)], ports=[])


class ExecuteGroup(SuiteTestCase):
    def go(self, g, timeout=10):
        return run.execute_group(g, self.dir / "rep", timeout, 1.0, make_config())

    def test_success_preserves_output_and_exit_codes(self):
        o = self.go(group("print('hello')", "import sys; sys.stderr.write('note')"))
        self.assertEqual(o.exit_codes, {0: 0, 1: 0})
        self.assertIsNone(o.failed_rank)
        self.assertEqual((self.dir / "rep" / "rank0.stdout").read_text().strip(), "hello")
        self.assertEqual((self.dir / "rep" / "rank1.stderr").read_text(), "note")

    def test_nonzero_exit_is_a_failure(self):
        o = self.go(group("pass", "import sys; sys.exit(7)"))
        self.assertEqual(o.failed_rank, 1)
        self.assertEqual(o.exit_codes[1], 7)
        self.assertIn("status 7", o.reason)

    def test_one_rank_failing_terminates_the_others(self):
        t = time.monotonic()
        o = self.go(group("import time; time.sleep(60)", "import sys; sys.exit(3)"))
        self.assertLess(time.monotonic() - t, 10)
        self.assertEqual(o.failed_rank, 1)
        self.assertIn(0, o.terminated_by_runner)

    def test_timeout_terminates_everything_it_started(self):
        pids = self.dir / "pids"
        code = f"import os,time; open({str(pids)!r}+str(os.getpid()), 'w').write('x'); time.sleep(60)"
        o = self.go(group(code, code), timeout=1.0)
        self.assertTrue(o.timed_out)
        self.assertEqual(o.terminated_by_runner, {0, 1})
        started = [int(p.name[len("pids"):]) for p in self.dir.glob("pids*")]
        self.assertEqual(len(started), 2)
        time.sleep(0.2)
        self.assertFalse(any(alive(p) for p in started))

    def test_child_of_a_rank_is_cleaned_up_too(self):
        marker = self.dir / "child.pid"
        code = (f"import subprocess,sys,time; c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                f"open({str(marker)!r},'w').write(str(c.pid)); time.sleep(60)")
        self.go(group(code), timeout=1.5)
        deadline = time.monotonic() + 2
        pid = int(marker.read_text())
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertFalse(alive(pid))

    def test_persistent_server_is_terminated_without_counting_as_failure(self):
        o = self.go(group("import time; time.sleep(60)", "print('client done')", persistent={0}))
        self.assertIsNone(o.failed_rank)
        self.assertFalse(o.timed_out)
        self.assertEqual(o.exit_codes, {0: None, 1: 0})
        self.assertEqual(o.terminated_by_runner, {0})

    def test_persistent_server_dying_early_is_a_failure(self):
        o = self.go(group("import sys; sys.exit(9)", "import time; time.sleep(5)", persistent={0}))
        self.assertEqual(o.failed_rank, 0)

    def test_missing_executable_is_a_failure_not_an_exception(self):
        g = Group([Invocation(0, "r", "local", ["/nonexistent/binary"])], ports=[])
        o = self.go(g)
        self.assertEqual(o.failed_rank, 0)
        self.assertIn("could not start", o.reason)

    def test_does_not_signal_unrelated_processes(self):
        import subprocess
        bystander = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(lambda: (bystander.kill(), bystander.wait()))
        self.go(group("import time; time.sleep(60)", "import sys; sys.exit(1)"))
        self.assertIsNone(bystander.poll())

    def test_environment_is_scrubbed_of_tbccl_variables(self):
        os.environ["TBCCL_ALGORITHM"] = "ring"
        self.addCleanup(os.environ.pop, "TBCCL_ALGORITHM", None)
        env = run.child_env({"CUSTOM": "1"})
        self.assertNotIn("TBCCL_ALGORITHM", env)
        self.assertEqual(env["CUSTOM"], "1")
        self.assertIn("TBCCL_ALGORITHM", run.stripped_env_names())


def can_allocate(port):
    """Whether the allocator would accept `port` right now (it and its +1000 partner can be bound)."""
    return run.PortAllocator(1024, 65000)._free(port)


def ports_free_around(port, span):
    """True when the `span - 1` ports after `port` can be allocated now and the +1000 partner of `port` itself is free. Neighbouring ephemeral ports belong to whatever
    else runs on the machine (another copy of this suite, a browser, a benchmark), so a test must not assume they are free."""
    return all(can_allocate(port + i) for i in range(1, span)) and run.PortAllocator(1024, 65000, offsets=(1000,))._free(port)


def hold_ephemeral_port(test, usable=lambda port: True, tries=200):
    """A bound socket on an ephemeral port for which `usable(port)` holds at this moment; skips the test if none is found."""
    for _ in range(tries):
        held = socket.socket()
        held.bind(("127.0.0.1", 0))
        port = held.getsockname()[1]
        if 2100 < port < 63000 and usable(port):
            test.addCleanup(held.close)
            return held, port
        held.close()
    test.skipTest("no suitable ephemeral port was found on this machine")


class Ports(unittest.TestCase):
    def test_allocations_are_distinct_and_free(self):
        alloc = run.PortAllocator(41000, 41200, seed=1)
        a, b = alloc.allocate(4), alloc.allocate(4)
        self.assertEqual(len(set(a + b)), 8)
        held, busy = hold_ephemeral_port(self, lambda port: ports_free_around(port, 3))
        small = run.PortAllocator(busy, busy + 3, seed=0)
        self.assertNotIn(busy, small.allocate(2))

    def test_the_neighbours_of_an_ephemeral_port_are_not_assumed_free(self):
        """Negative control for hold_ephemeral_port: with a neighbour taken, the narrow range has too few free ports (the failure a concurrent process used to cause)."""
        held, busy = hold_ephemeral_port(self, lambda port: ports_free_around(port, 3))
        neighbour = socket.socket()
        self.addCleanup(neighbour.close)
        neighbour.bind(("127.0.0.1", busy + 1))
        self.assertFalse(ports_free_around(busy, 3))
        with self.assertRaises(RuntimeError):
            run.PortAllocator(busy, busy + 3, seed=0).allocate(2)

    def test_released_ports_are_reused_so_a_narrow_range_lasts_a_whole_campaign(self):
        alloc = run.PortAllocator(41500, 41506, seed=1)  # six usable ports
        first = alloc.allocate(2)
        for _ in range(40):  # far more groups than the range holds
            ports = alloc.allocate(2)
            alloc.release(ports)
        self.assertEqual(len(alloc.allocate(2)), 2)
        self.assertEqual(len(set(first)), 2)

    def test_without_release_the_range_is_exhausted(self):
        for base in range(41600, 41800, 10):  # another process may hold a port of any fixed range
            alloc = run.PortAllocator(base, base + 4, seed=1)
            try:
                alloc.allocate(2)
                alloc.allocate(2)
            except RuntimeError:
                continue
            with self.assertRaises(RuntimeError):
                alloc.allocate(2)
            return
        self.skipTest("no free four-port range was found on this machine")

    def test_a_port_in_time_wait_is_usable_but_a_live_listener_is_not(self):
        import socket as sk
        held, busy = hold_ephemeral_port(self, lambda port: ports_free_around(port, 2))
        held.close()
        listener = sk.socket()
        listener.setsockopt(sk.SOL_SOCKET, sk.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", busy))
        listener.listen(1)
        self.addCleanup(listener.close)
        c = sk.create_connection(("127.0.0.1", busy))
        s, _ = listener.accept()
        s.close()
        c.close()  # leaves the connection in TIME_WAIT on one side
        listener.close()
        alloc = run.PortAllocator(busy, busy + 2, seed=0)
        self.assertIn(busy, alloc.allocate(1) + alloc.allocate(1))

    def test_a_busy_data_port_is_avoided(self):
        held, busy = hold_ephemeral_port(self, lambda port: can_allocate(port - 999))
        alloc = run.PortAllocator(busy - 1000, busy - 998, seed=0)
        self.assertNotIn(busy - 1000, alloc.allocate(1))


class RunnerEndToEnd(SuiteTestCase):
    def runner(self, config, **kw):
        return run.Runner(config, output_dir=self.dir / "out", log=lambda *_: None, **kw)

    def records(self, manifest):
        return schema.read_records(self.dir / "out" / manifest["run_id"] / "results.jsonl")

    def test_success_record_validates_and_keeps_raw_output(self):
        manifest, records = self.runner(self.fake_config("ok", repetitions=2)).run()
        self.assertEqual(manifest["status"], "success")
        self.assertEqual([r["status"]["state"] for r in records], ["success"] * 2)
        for r in records:
            self.assertEqual(schema.validate_record(r), [])
            self.assertTrue((self.dir / "out" / manifest["run_id"] / r["artifacts"]["stdout"][0]).exists())
        self.assertEqual([r["configuration"]["repetition_index"] for r in records], [0, 1])
        self.assertEqual(self.records(manifest), records)

    def test_failed_rank_gives_an_invalid_measurement_never_a_valid_one(self):
        manifest, records = self.runner(self.fake_config("fail_rank1")).run()
        r = records[0]
        self.assertEqual(r["status"]["state"], "failure")
        self.assertIsNone(r["measurements"])
        self.assertIn("rank 1 exited with status 3", r["status"]["reason"])
        self.assertIn("boom", r["status"]["reason"])
        self.assertEqual(manifest["status"], "failure")

    def test_timeout_is_recorded_and_stops_repetitions(self):
        manifest, records = self.runner(self.fake_config("timeout", repetitions=3, timeout_s=1)).run()
        self.assertEqual([r["status"]["state"] for r in records], ["timeout"])
        self.assertIsNone(records[0]["measurements"])

    def test_unparseable_output_is_invalid(self):
        _, records = self.runner(self.fake_config("garbage")).run()
        self.assertEqual(records[0]["status"]["state"], "invalid")
        self.assertIsNone(records[0]["measurements"])
        self.assertIn("could not be parsed", records[0]["status"]["reason"])

    def test_skipped_points_are_recorded_not_dropped(self):
        wl = {"name": "s", "benchmark": "collective_bench", "operation": "all_reduce", "algorithms": ["ring"], "sizes": [4], "datatype": "float32"}
        config = self.fake_config("ok", extra_workloads=[wl])
        manifest, records = self.runner(config).run()
        states = sorted(r["status"]["state"] for r in records)
        self.assertEqual(states, ["skipped", "success"])
        self.assertEqual(manifest["status"], "success")

    def test_persistent_role_does_not_fail_the_run(self):
        _, records = self.runner(self.fake_config("persistent")).run()
        self.assertEqual(records[0]["status"]["state"], "invalid")  # no JSON from rank 0 (the persistent one), but not a process failure
        self.assertEqual(records[0]["status"]["exit_codes"]["0"], None)

    def test_dry_run_prints_commands_and_launches_nothing(self):
        pidfile = str(self.dir / "pid")
        config = self.fake_config("ok", options={"pidfile": pidfile})
        runner = self.runner(config)
        matrix, commands = run.format_matrix(runner.points(), config, runner.ctx)
        self.assertIn("fake", matrix)
        self.assertIn("rank 0", commands)
        self.assertEqual(list(self.dir.glob("pid*")), [])
        self.assertFalse((self.dir / "out").exists())

    def test_physical_configuration_needs_opt_in(self):
        config = make_config(hosts={"local": {"address": "192.0.2.5"}}, topology={"kind": "direct-link", "interconnect": "x", "transport": "tcp", "physical": True})
        with self.assertRaises(PermissionError):
            self.runner(config).run()

    def test_link_health_change_invalidates_the_run_and_stops_it(self):
        from benchmarks.suite.preflight import HealthMonitor
        calls = []

        def snapshot(host):
            calls.append(1)
            return {"n": len(calls)}

        def compare(before, after):
            return {"stop_reasons": ["fatal/uncorrectable Thunderbolt AER counter increased"] if after["n"] >= 5 else []}  # baseline, then pre and after for each point

        config = self.fake_config("ok", extra_workloads=[{"name": "g", "benchmark": "fake", "operation": "ok", "sizes": [16], "rank_counts": [2], "repetitions": 1},
                                                       {"name": "h", "benchmark": "fake", "operation": "ok", "sizes": [32], "rank_counts": [2], "repetitions": 1}])
        monitor = HealthMonitor(list(config.hosts.values()), snapshot_fn=snapshot, compare_fn=compare)
        manifest, records = self.runner(config, monitor=monitor).run()
        states = [(r["configuration"]["payload_bytes"], r["status"]["state"]) for r in records]
        self.assertEqual(states, [(8, "success"), (16, "invalid"), (32, "skipped")])
        self.assertIn("link health changed", records[1]["status"]["reason"])
        self.assertIsNone(records[1]["measurements"])
        self.assertEqual(manifest["status"], "aborted")
        self.assertTrue(manifest["health"]["stop_reasons"])

    def test_unavailable_snapshot_is_a_stop_reason(self):
        from benchmarks.suite.preflight import HealthMonitor

        def snapshot(host):
            raise RuntimeError("controller gone")

        monitor = HealthMonitor([make_config().hosts["local"]], snapshot_fn=snapshot, compare_fn=lambda a, b: {})
        reasons, _ = monitor.compare(monitor.snapshot(), monitor.snapshot())
        self.assertTrue(reasons and "unavailable" in reasons[0])

    def test_no_retry_after_a_failure(self):
        marker = self.dir / "runs"
        config = self.fake_config("fail_rank1", options={"pidfile": str(marker)}, repetitions=1)
        self.runner(config).run()
        self.assertEqual(len(list(self.dir.glob("runs*"))), 2)  # one attempt, two ranks


class Redaction(unittest.TestCase):
    def test_addresses_and_paths_are_replaced(self):
        config = make_config(hosts={"peer": {"address": "192.0.2.77"}, "local": {"address": "127.0.0.1"}}, topology={"kind": "direct-link", "interconnect": "x", "transport": "tcp", "physical": True},
                             ranks=[{"rank": 0, "host": "local"}, {"rank": 1, "host": "peer"}])
        r = run.Redactor(config)
        self.assertEqual(r.text("connect 192.0.2.77:5000 and 127.0.0.1"), "connect {address:peer}:5000 and 127.0.0.1")
        self.assertNotIn(str(run.inventory.REPO_ROOT), r.text(str(run.inventory.REPO_ROOT / "build")))

    def test_nested_records_are_redacted_in_values_and_keys(self):
        config = make_config(hosts={"peer": {"address": "192.0.2.77", "interfaces": {"link": "192.0.2.78"}}, "local": {"address": "127.0.0.1"}},
                             topology={"kind": "direct-link", "interconnect": "x", "transport": "tcp", "physical": True}, ranks=[{"rank": 0, "host": "local"}, {"rank": 1, "host": "peer"}])
        r = run.Redactor(config)
        record = {"extra": {"link_peers": {"value": "192.0.2.77,192.0.2.78", "unit": "addresses"}}, "routes": {"192.0.2.77": "link0"}, "ports": [1, 2], "loopback": "127.0.0.1"}
        out = r.obj(record)
        self.assertEqual(out["extra"]["link_peers"]["value"], "{address:peer},{address:peer}")
        self.assertEqual(list(out["routes"]), ["{address:peer}"])
        self.assertEqual((out["ports"], out["loopback"]), ([1, 2], "127.0.0.1"))
        self.assertNotIn("192.0.2", str(out))


if __name__ == "__main__":
    unittest.main()
