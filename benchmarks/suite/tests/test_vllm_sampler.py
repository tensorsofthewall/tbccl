"""The passive sampler's pure functions: engine-process recognition, address normalisation, the TCP connection census and the liveness decision (no psutil needed)."""
import io
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from benchmarks.suite.vllm_bench import sampler

SCRIPT = Path(sampler.__file__)


class FakePsutil:
    CONN_LISTEN, CONN_ESTABLISHED = "LISTEN", "ESTABLISHED"

    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass


def conn(laddr, raddr, status="ESTABLISHED"):
    return SimpleNamespace(laddr=SimpleNamespace(ip=laddr), raddr=SimpleNamespace(ip=raddr) if raddr else None, status=status)


class Proc:
    def __init__(self, pid, conns, cmd="python worker", name="python"):
        self.pid, self._conns, self._cmd, self._name = pid, conns, cmd, name

    def net_connections(self, kind="tcp"):
        return self._conns

    def cmdline(self):
        return self._cmd.split()

    def name(self):
        return self._name


class Recognition(unittest.TestCase):
    def test_engine_processes_are_recognised_but_a_launcher_carrying_the_command_line_is_not(self):
        self.assertTrue(sampler.is_engine(["/v/bin/python", "/v/bin/vllm", "serve", "/model", "--port", "1"], "python"))
        self.assertTrue(sampler.is_engine(["/v/bin/vllm", "serve", "/model"], "vllm"))
        self.assertTrue(sampler.is_engine([], "VLLM::Worker_PP1"))
        self.assertTrue(sampler.is_engine(["VLLM::EngineCore"], "VLLM::EngineCore"))
        # the controller's ssh process holds the management connection and mentions "vllm serve" in its argument
        self.assertFalse(sampler.is_engine(["ssh", "-o", "BatchMode=yes", "mac", "/v/bin/vllm serve /model"], "ssh"))
        self.assertFalse(sampler.is_engine(["bash", "-c", "vllm serve x"], "bash"))
        self.assertFalse(sampler.is_engine(["/v/bin/python", "client.py", "--spec-b64", "x"], "python"))

    def test_roles(self):
        self.assertEqual(sampler.role_of(Proc(1, [], "VLLM::Worker_PP0", "VLLM::Worker_PP0")), "Worker")
        self.assertEqual(sampler.role_of(Proc(1, [], "VLLM::EngineCore", "VLLM::EngineCore")), "EngineCore")
        self.assertEqual(sampler.role_of(Proc(1, [], "python vllm serve", "python")), "vllm-serve")

    def test_dual_stack_notations_are_normalised(self):
        for raw in ("::ffff:192.0.2.1", "::192.0.2.1", "192.0.2.1"):
            self.assertEqual(sampler.norm_ip(raw), "192.0.2.1")
        self.assertEqual(sampler.norm_ip("2001:db8::1"), "2001:db8::1")


class Census(unittest.TestCase):
    EXPECT = {"192.0.2.1", "192.0.2.2"}

    def census(self, conns):
        return sampler.classify(FakePsutil, {1: Proc(1, conns, "VLLM::Worker_PP0", "VLLM::Worker_PP0")}, self.EXPECT)

    def test_a_connection_between_the_two_link_addresses_is_link(self):
        c, pairs = self.census([conn("192.0.2.1", "192.0.2.2"), conn("::ffff:192.0.2.1", "::ffff:192.0.2.2")])
        self.assertEqual((c["link"], c["other"]), (2, 0))
        self.assertEqual(pairs, [("192.0.2.1", "192.0.2.2", "Worker")])

    def test_loopback_and_an_address_talking_to_itself_are_not_link_traffic(self):
        c, _ = self.census([conn("127.0.0.1", "127.0.0.1"), conn("192.0.2.1", "192.0.2.1"), conn("::1", "::1")])
        self.assertEqual((c["link"], c["loopback"], c["other"]), (0, 3, 0))

    def test_a_peer_on_another_network_is_flagged(self):
        c, pairs = self.census([conn("192.0.2.5", "192.0.2.9"), conn("192.0.2.1", "192.0.2.9")])
        self.assertEqual((c["link"], c["other"]), (0, 2))
        self.assertTrue(all(p[2].endswith(":other") for p in pairs))

    def test_listeners_are_counted_separately_and_unreadable_processes_are_skipped(self):
        c, _ = self.census([conn("192.0.2.1", None, status="LISTEN")])
        self.assertEqual(c["listen"], 1)

        class Denied(Proc):
            def net_connections(self, kind="tcp"):
                raise FakePsutil.AccessDenied()

        c, _ = sampler.classify(FakePsutil, {2: Denied(2, [])}, self.EXPECT)
        self.assertEqual(sum(c.values()), 0)


class Liveness(unittest.TestCase):
    SEEN = {10: "vllm-serve", 11: "EngineCore", 12: "Worker"}

    def test_nothing_lost_is_fine(self):
        self.assertIsNone(sampler.lost_engines(self.SEEN, {10, 11, 12}))
        self.assertIsNone(sampler.lost_engines({}, set()))

    def test_a_dead_worker_or_engine_core_while_the_leader_lives_is_a_loss(self):
        self.assertEqual(sampler.lost_engines(self.SEEN, {10, 11}), {12: "Worker"})
        self.assertEqual(sampler.lost_engines(self.SEEN, {10, 12}), {11: "EngineCore"})

    def test_everything_gone_is_a_loss_but_the_api_process_alone_leaving_is_not(self):
        self.assertEqual(set(sampler.lost_engines(self.SEEN, set())), {10, 11, 12})
        self.assertIsNone(sampler.lost_engines(self.SEEN, {11, 12}))  # the launcher process exited, the engine processes are still being stopped


class Script(unittest.TestCase):
    def test_the_sampler_runs_and_stops_cleanly_when_psutil_is_available(self):
        try:
            import psutil  # noqa: F401
        except ImportError:
            self.skipTest("psutil is not installed")
        proc = subprocess.Popen([sys.executable, str(SCRIPT), "--role", "t", "--interval", "0.2", "--liveness"], stdout=subprocess.PIPE, text=True)
        import time
        time.sleep(1.0)
        proc.terminate()
        out, _ = proc.communicate(timeout=10)
        lines = [json.loads(x.split(" ", 1)[1]) for x in out.splitlines() if x.startswith("TBCCL_VLLM_SAMPLE ")]
        self.assertEqual(lines[0]["event"], "start")
        self.assertTrue(any("cpu_percent" in x for x in lines))
        self.assertEqual(lines[-1]["event"], "stop")
        self.assertEqual(proc.returncode, 0)  # no engine was ever seen, so there is nothing to lose

    def test_exit_status_for_a_lost_engine_is_distinct(self):
        self.assertEqual(sampler.EXIT_ENGINE_LOST, 7)


if __name__ == "__main__":
    unittest.main()
