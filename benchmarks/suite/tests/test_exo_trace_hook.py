"""The diagnostic trace hook against a stand-in exo_tbccl package (offline): it does nothing without its variable, and with it the logger gets a file handler and the CopyStats of every
communicator are written to a JSON snapshot."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HOOK = str(Path(__file__).resolve().parents[1] / "exo_bench" / "trace_hook")

FAKE_GROUP = '''
import logging
from dataclasses import dataclass, field
log = logging.getLogger("exo_tbccl")

@dataclass
class CopyStats:
    direct_ops: dict = field(default_factory=dict)
    direct_bytes: int = 0
    materialized_copies: int = 0

class TbcclPipelineComm:
    def __init__(self, rank):
        self._rank = rank
        self.stats = CopyStats()
    def send(self, n):
        self.stats.direct_bytes += n
        log.info("rank %d send bytes=%d", self._rank, n)
'''

SCRIPT = '''
import exo_tbccl.group as g
c = g.TbcclPipelineComm(1)
c.send(4096)
print("done")
'''


class Hook(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, self.tmp, True)
        pkg = self.tmp / "pkgs" / "exo_tbccl"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "group.py").write_text(FAKE_GROUP)

    def run_script(self, trace):
        env = {k: v for k, v in os.environ.items() if k not in ("TBCCL_B6_TRACE_DIR", "PYTHONPATH")}
        env["PYTHONPATH"] = os.pathsep.join([HOOK, str(self.tmp / "pkgs")])
        if trace:
            env["TBCCL_B6_TRACE_DIR"] = str(self.tmp / "trace")
        return subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=60)

    def test_nothing_happens_without_the_variable(self):
        out = self.run_script(False)
        self.assertEqual((out.returncode, out.stdout.strip()), (0, "done"), out.stderr)
        self.assertFalse((self.tmp / "trace").exists())

    def test_log_lines_and_copy_stats_are_captured_when_asked(self):
        out = self.run_script(True)
        self.assertEqual((out.returncode, out.stdout.strip()), (0, "done"), out.stderr)
        logs = list((self.tmp / "trace").glob("exo_tbccl-*.log"))
        self.assertEqual(len(logs), 1)
        line = logs[0].read_text().strip()
        self.assertTrue(line.endswith("rank 1 send bytes=4096"))
        self.assertGreater(float(line.split()[0]), 1e9)  # a wall-clock timestamp
        snaps = list((self.tmp / "trace").glob("copystats-*.json"))
        self.assertEqual(len(snaps), 1)
        data = json.loads(snaps[0].read_text())
        self.assertEqual((data["comms"][0]["rank"], data["comms"][0]["stats"]["direct_bytes"], data["comms"][0]["stats"]["materialized_copies"]), (1, 4096, 0))


if __name__ == "__main__":
    unittest.main()
