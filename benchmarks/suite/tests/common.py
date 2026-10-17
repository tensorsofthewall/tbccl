"""Helpers shared by the suite tests."""
import copy
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from benchmarks.suite import config as cfg
from benchmarks.suite.adapters import REGISTRY
from benchmarks.suite.adapters.base import BenchmarkAdapter, Capabilities, Group, Invocation, Parsed, timing

FIXTURES = Path(__file__).resolve().parent / "fixtures"
LOOPBACK_IF = "lo0" if sys.platform == "darwin" else "lo"


def pid_alive(pid):
    """True while the process exists and is not a zombie (portable: /proc is not available on macOS)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    import subprocess
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")
REPO = Path(__file__).resolve().parents[3]

BASE = {
    "schema": "tbccl-bench-config/1", "name": "t", "defaults": {"warmup": 1, "iterations": 5, "repetitions": 2, "timeout_s": 20, "startup_wait_s": 2},
    "topology": {"kind": "loopback", "interconnect": "loopback", "transport": "tcp", "physical": False},
    "hosts": {"local": {"launcher": {"kind": "local"}, "address": "127.0.0.1", "interface": LOOPBACK_IF}},
    "ranks": [{"rank": 0, "host": "local", "memory_backend": "host"}, {"rank": 1, "host": "local", "memory_backend": "host"}],
    "workloads": [{"name": "w", "benchmark": "tcp_pingpong", "operations": ["pingpong"], "sizes": [64, "4KiB"]}],
}


def make_config(**changes):
    raw = copy.deepcopy(BASE)
    for key, value in changes.items():
        raw[key] = value
    return cfg.build_config(raw)


class FakeAdapter(BenchmarkAdapter):
    """Runs tiny Python scripts instead of a native benchmark so runner behaviour can be tested without hardware or build output."""
    caps = Capabilities(name="fake", category="other", description="test double", executable="python3", operations=("ok", "fail_rank1", "timeout", "garbage", "zero_exit_no_output", "persistent"),
                        min_ranks=2, max_ranks=3, verification={"ok": "native"}, executable_in_build=False)

    SCRIPTS = {
        "ok": "import json,sys; print(json.dumps({'median_us': 12.5, 'n': 5}))",
        "garbage": "print('nothing useful')",
    }

    def describe(self, point):
        return {"payload_semantics": "test payload", "timing_method": "test timer", "completion_semantics": "test completion", "backend": "fake"}

    def plan(self, point, ctx, ports, rep_dir):
        invs = []
        pidfile = point.options.get("pidfile")
        for rank, spec in enumerate(point.ranks):
            code = "pass"
            if point.operation in ("ok", "garbage"):
                code = self.SCRIPTS[point.operation] if rank == 0 else "pass"
            elif point.operation == "fail_rank1":
                code = "import time; time.sleep(60)" if rank == 0 else "import sys; sys.stderr.write('boom\\n'); sys.exit(3)"
            elif point.operation == "timeout":
                code = "import time; time.sleep(60)"
            elif point.operation == "persistent":
                code = "import time; time.sleep(60)" if rank == 0 else "print('done')"
            if pidfile:
                code = f"import os; open({pidfile!r}+str({rank}), 'w').write(str(os.getpid())); " + code
            invs.append(Invocation(rank, "r", spec["host"], [sys.executable, "-c", code], persistent=(point.operation == "persistent" and rank == 0)))
        return Group(invs, ports=list(ports))

    def parse(self, point, group, outcome):
        import json
        text = Path(outcome.stdout[0]).read_text().strip()
        try:
            obj = json.loads(text)
        except ValueError:
            from benchmarks.suite.adapters.base import ParseError
            raise ParseError("no JSON on stdout")
        tm = timing("fake_time", {"median": obj["median_us"]}, "reported", "median", sample_count=obj["n"])
        return Parsed({"timing": tm, "throughput": [], "extra": {}, "warnings": []}, {"status": "passed", "method": "test", "details": ""})


class SuiteTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        REGISTRY["fake"] = FakeAdapter()
        self.addCleanup(REGISTRY.pop, "fake", None)

    def fake_config(self, operation, ranks=2, repetitions=1, timeout_s=20, options=None, extra_workloads=()):
        wl = {"name": "f", "benchmark": "fake", "operations": [operation], "sizes": [8], "rank_counts": [ranks], "repetitions": repetitions, "timeout_s": timeout_s}
        if options:
            wl["options"] = options
        return make_config(workloads=[wl, *extra_workloads], ranks=[{"rank": i, "host": "local", "memory_backend": "host"} for i in range(3)])


class approx:
    """A small stand-in for pytest.approx so the tests need only the standard library: a float, or a sequence of floats, within a relative and an absolute tolerance."""

    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected, self.rel, self.abs = expected, rel, abs

    def _close(self, a, b):
        return a == b or abs(a - b) <= max(self.rel * max(abs(a), abs(b)), self.abs)

    def __eq__(self, other):
        if isinstance(self.expected, (list, tuple)):
            return isinstance(other, (list, tuple)) and len(other) == len(self.expected) and all(self._close(a, b) for a, b in zip(other, self.expected))
        return isinstance(other, (int, float)) and self._close(other, self.expected)

    def __repr__(self):
        return f"approx({self.expected!r})"


requires_yaml = unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML is not installed (the shipped presets are YAML)")
