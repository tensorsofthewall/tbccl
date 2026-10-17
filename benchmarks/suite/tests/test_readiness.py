"""Listener readiness: proven from the kernel's socket table and the owning process, never by connecting to the port."""
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from benchmarks.suite.remote import agent

TAG = "test-token-readiness"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


BINDER = """
import os, socket, sys, time
port, delay, log = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
time.sleep(delay)
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(("127.0.0.1", port)); s.listen(5)
s.settimeout(float(sys.argv[4]))
def note(text):
    try:
        open(log, "w").write(text)
    except OSError:
        pass
try:
    s.accept(); note("connected")
except socket.timeout:
    note("no connection")
"""


class Readiness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.procs = []
        self.addCleanup(self._kill)

    def _kill(self):
        for p in self.procs:
            if p.poll() is None:
                p.kill()
            p.wait()

    def spawn(self, code, *args, tagged=True):
        env = dict(os.environ)
        if tagged:
            env[agent.TAG_ENV] = TAG
        p = subprocess.Popen([sys.executable, "-c", code, *map(str, args)], env=env, start_new_session=True)
        self.procs.append(p)
        return p

    def wait(self, p, port, timeout=5.0):
        return agent.wait_for_listener(port, p.pid, TAG, lambda: p.poll() is None, timeout)

    def test_successful_bind_is_ready_and_nothing_connects(self):
        port, log = free_port(), str(Path(self.tmp.name) / "log")
        p = self.spawn(BINDER, port, 0, log, 1.5)
        outcome, detail = self.wait(p, port)
        self.assertEqual(outcome, "ready")
        self.assertIn(p.pid, detail)
        p.wait(timeout=5)
        self.assertEqual(Path(log).read_text(), "no connection")  # the readiness check never consumed the accept

    def test_delayed_bind_is_waited_for_without_a_fixed_sleep(self):
        port, log = free_port(), str(Path(self.tmp.name) / "log")
        p = self.spawn(BINDER, port, 0.6, log, 0.2)
        t = time.monotonic()
        outcome, _ = self.wait(p, port)
        elapsed = time.monotonic() - t
        self.assertEqual(outcome, "ready")
        self.assertGreater(elapsed, 0.5)
        self.assertLess(elapsed, 2.5)

    def test_process_exiting_before_binding(self):
        port = free_port()
        p = self.spawn("import sys; sys.exit(4)")
        outcome, detail = self.wait(p, port)
        self.assertEqual(outcome, "exited")
        self.assertEqual(detail, [])

    def test_timeout_when_nothing_ever_listens(self):
        port = free_port()
        p = self.spawn("import time; time.sleep(30)")
        t = time.monotonic()
        outcome, detail = self.wait(p, port, timeout=0.6)
        self.assertEqual(outcome, "timeout")
        self.assertLess(time.monotonic() - t, 3)

    def test_port_already_occupied_by_another_process(self):
        port, log = free_port(), str(Path(self.tmp.name) / "log")
        squatter = self.spawn(BINDER, port, 0, log, 10, tagged=False)
        deadline = time.monotonic() + 5
        while not agent.tcp_listeners(port)[1] and time.monotonic() < deadline:
            time.sleep(0.01)
        ours = self.spawn("import time; time.sleep(30)")
        outcome, detail = self.wait(ours, port, timeout=0.5)
        self.assertEqual(outcome, "occupied")
        self.assertEqual(detail, [squatter.pid])

    def test_wrong_process_occupying_while_ours_fails_to_bind_and_exits(self):
        port, log = free_port(), str(Path(self.tmp.name) / "log")
        squatter = self.spawn(BINDER, port, 0, log, 10, tagged=False)
        deadline = time.monotonic() + 5
        while not agent.tcp_listeners(port)[1] and time.monotonic() < deadline:
            time.sleep(0.01)
        ours = self.spawn("import socket,sys; s=socket.socket(); s.bind(('127.0.0.1', int(sys.argv[1])))", port)
        outcome, detail = self.wait(ours, port, timeout=5)
        self.assertEqual(outcome, "exited")
        self.assertEqual(detail, [squatter.pid])  # the report names who held the port; we did not mistake it for ours

    def test_a_listener_in_our_group_counts_even_when_a_child_holds_it(self):
        port, log = free_port(), str(Path(self.tmp.name) / "log")
        code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1],*sys.argv[2:]]); time.sleep(30)"
        p = self.spawn(code, BINDER, port, 0, log, 1.5)
        outcome, detail = self.wait(p, port)
        self.assertEqual(outcome, "ready")
        self.assertNotIn(p.pid, detail)  # the grandchild owns the socket; it is in our process group

    def test_unavailable_tooling_fails_safely(self):
        outcome, detail = agent.wait_for_listener(free_port(), os.getpgid(0), TAG, lambda: True, 0.2, lsof_runner=lambda port: None)
        self.assertEqual(outcome, "unsupported")
        self.assertIn("cannot be established", detail)


class MacOsPathSimulated(unittest.TestCase):
    """The macOS code path (lsof, ps) exercised with canned tool output. This is a simulation: it is NOT validation on a Mac."""

    def test_lsof_output_parsing(self):
        self.assertEqual(agent.parse_lsof_pids("p123\nf5\nn*:5000\np456\nf6\n"), {123, 456})
        self.assertEqual(agent.parse_lsof_pids(""), set())

    def test_ready_when_lsof_reports_a_process_in_our_group(self):
        me = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
        self.addCleanup(lambda: (me.kill(), me.wait()))
        outcome, detail = agent.wait_for_listener(5000, me.pid, TAG, lambda: True, 1.0, lsof_runner=lambda port: f"p{me.pid}\nf9\n")
        self.assertEqual((outcome, detail), ("ready", [me.pid]))

    def test_other_process_in_lsof_output_is_not_ours(self):
        me = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"], start_new_session=True)
        self.addCleanup(lambda: (me.kill(), other.kill(), me.wait(), other.wait()))
        outcome, detail = agent.wait_for_listener(5000, me.pid, TAG, lambda: True, 0.3, lsof_runner=lambda port: f"p{other.pid}\n")
        self.assertEqual((outcome, detail), ("occupied", [other.pid]))


if __name__ == "__main__":
    unittest.main()
