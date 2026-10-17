"""Validation of supervised execution through a REAL launcher (ssh or any prefix) to a REAL host.

    python3 -m benchmarks.suite.validation.real_launcher --launcher-argv '["ssh","-o","BatchMode=yes","HOSTALIAS"]' --report report.json

Only harmless short-lived `python3 -c` processes run on the host. The helper is shipped inside the command, so nothing is written on the host and nothing
persists after the run. Processes are identified only by the exact pid, process group and ownership token the supervisor reported; nothing is ever matched by
name. The launcher is given on the command line, never stored in the repository. Network traffic is control traffic only (no benchmark data).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .. import config as cfg
from ..adapters.base import Group, Invocation
from ..remote import session
from ..runner import execute_group


def make_host(launcher_argv, python="python3"):
    raw = {"schema": "tbccl-bench-config/1", "name": "real-launcher", "topology": {"kind": "unknown", "interconnect": "control-only", "transport": "tcp", "physical": False},
           "hosts": {"remote": {"address": "127.0.0.1", "python": python, "launcher": {"kind": "prefix", "argv": launcher_argv, "shell_join": True}},
                     "local": {"address": "127.0.0.1"}},
           "ranks": [{"rank": 0, "host": "local"}, {"rank": 1, "host": "remote"}], "defaults": {"lease_s": 3.0, "term_grace_s": 1.0, "startup_wait_s": 10},
           "workloads": [{"benchmark": "tcp_pingpong", "operation": "pingpong", "sizes": [64]}]}
    return cfg.build_config(raw)


class Validation:
    def __init__(self, config, workdir):
        self.config, self.host, self.dir = config, config.hosts["remote"], Path(workdir)
        self.results = []
        self.n = 0
        self.bystander = None

    # -- helpers
    def record(self, name, passed, detail=""):
        self.results.append({"scenario": name, "passed": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}" + (f": {detail}" if detail else ""), flush=True)

    def start(self, code, **kw):
        self.n += 1
        return session.RemoteProcess(self.host, ["python3", "-c", code], kw.pop("env", {}), self.n, self.dir / f"o{self.n}", self.dir / f"e{self.n}", "valid", kw.pop("lease_s", 3.0),
                                     kw.pop("max_runtime_s", 60.0), kw.pop("grace_s", 1.0), python=self.host.python, **kw)

    def remote_alive(self, pids):
        """Ask the host, through a separate connection, which of these exact pids are alive (zombies excluded)."""
        code = ("import os,sys,subprocess; out=subprocess.run(['ps','-axo','pid=,stat='],capture_output=True,text=True).stdout.split('\\n'); "
                "live={int(l.split()[0]) for l in out if l.strip() and not l.split()[1].startswith('Z')}; "
                f"print(sorted(p for p in {list(pids)!r} if p in live))")
        done = subprocess.run(session.launcher_command(self.host, [self.host.python, "-c", code]), capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        return json.loads(done.stdout.strip().replace("'", '"')) if done.returncode == 0 and done.stdout.strip() else None

    def wait_gone(self, pids, seconds=12.0):
        end = time.monotonic() + seconds
        alive = None
        while time.monotonic() < end:
            alive = self.remote_alive(pids)
            if alive == []:
                return True, []
            time.sleep(0.4)
        return False, alive

    def pids_from(self, p, count, timeout=20):
        out = Path(p.out_path)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if out.exists():
                words = out.read_text().split()
                if len(words) >= count:
                    return [int(w) for w in words[:count]]
            time.sleep(0.1)
        return None

    # -- scenarios
    def run_all(self):
        out = subprocess.run(session.launcher_command(self.host, [self.host.python, "-c", "import platform,sys; print(platform.system(), platform.machine(), sys.version.split()[0])"]),
                             capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        self.platform = out.stdout.strip()
        print(f"remote: {self.platform}")
        bystander = subprocess.Popen(session.launcher_command(self.host, [self.host.python, "-c", "import os,time; print(os.getpid(), flush=True); time.sleep(300)"]),
                                     stdout=subprocess.PIPE, text=True, stdin=subprocess.DEVNULL, start_new_session=True)
        self.bystander = (bystander, int(bystander.stdout.readline()))
        for scenario in (self.s1_to_3, self.s4_timeout, self.s5_cancel, self.s6_7_cleanup, self.s7b_escaped, self.s8_connection_loss, self.s9_controller_kill, self.s10_lease,
                         self.s11_watchdog, self.s12_runner_paths, self.s13_probe_and_health):
            try:
                scenario()
            except Exception as error:  # a scenario that cannot run is a failure, never a skip
                self.record(scenario.__name__, False, f"{type(error).__name__}: {error}")
        alive = self.remote_alive([self.bystander[1]])
        self.record("unrelated process untouched by every scenario above", alive == [self.bystander[1]], f"exact pid {self.bystander[1]} alive: {alive}")
        self.kill_bystander()
        return self.results

    def s1_to_3(self):
        p = self.start("import sys; sys.stdout.write('out-line\\n'); sys.stderr.write('err-line\\n'); sys.stdout.write('x'*200000); sys.stdout.write('\\n')")
        p.wait_for(lambda: p.returncode is not None, 60)
        out, err = Path(p.out_path).read_text(), Path(p.err_path).read_text()
        self.record("1 normal remote completion", p.returncode == 0, f"returncode {p.returncode}")
        self.record("2 stdout and stderr forwarded separately and completely", out.startswith("out-line\n") and len(out) == 200000 + 9 + 1 and err == "err-line\n", f"stdout {len(out)} B, stderr {len(err)} B")
        rep = p.report()
        self.record("10 independent cleanup confirmation after a normal exit", rep["cleanup_confirmed"] in (True, None) and rep["independent_check"]["confirmed"] is True,
                    f"cleanup_confirmed={rep['cleanup_confirmed']}; escaped_descendants_detectable={rep['escaped_descendants_detectable']}; {rep['cleanup_note']}")
        q = self.start("import sys; sys.stderr.write('bad\\n'); sys.exit(7)")
        q.wait_for(lambda: q.returncode is not None, 60)
        self.record("3 nonzero remote exit code propagated", q.returncode == 7 and Path(q.err_path).read_text() == "bad\n", f"returncode {q.returncode}")

    def s4_timeout(self):
        t = time.monotonic()
        p = self.start("import time; time.sleep(120)", max_runtime_s=2.0)
        p.wait_for(lambda: p.returncode is not None, 60)
        self.record("4a runtime cap terminates a remote process", p.returncode not in (0, None) and "runtime cap" in str(p.exit_event.get("reason")), f"{time.monotonic() - t:.1f} s, reason: {p.exit_event.get('reason')}")
        group = Group([Invocation(0, "r", "local", [sys.executable, "-c", "import time; time.sleep(60)"]), Invocation(1, "r", "remote", ["python3", "-c", "import time; time.sleep(60)"])], [])
        t = time.monotonic()
        o = execute_group(group, self.dir / "g", 3.0, 10.0, self.config, run_id="valid")
        self.record("4b runner timeout terminates local and remote ranks", o.timed_out and o.terminated_by_runner == {0, 1} and time.monotonic() - t < 25,
                    f"{time.monotonic() - t:.1f} s; cleanup_confirmed={o.rank_reports[1]['cleanup_confirmed']}")

    def s5_cancel(self):
        p = self.start("import os,time; print(os.getpid(), flush=True); time.sleep(120)")
        pid = self.pids_from(p, 1)[0]
        t = time.monotonic()
        p.stop()
        gone, alive = self.wait_gone([pid])
        self.record("5 explicit cancellation stops the remote process", gone and p.returncode not in (0, None), f"{time.monotonic() - t:.1f} s; still alive: {alive}; reason: {p.exit_event.get('reason')}")

    def s6_7_cleanup(self):
        p = self.start("import os,subprocess,sys,time; g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); print(os.getpid(), g.pid, flush=True); time.sleep(120)")
        child, grand = self.pids_from(p, 2)
        p.stop()
        gone, alive = self.wait_gone([child, grand])
        rep = p.report()
        self.record("6/7a remote process and its grandchild (same group) are gone", gone, f"alive: {alive}")
        self.record("10 cleanup confirmation twice (supervisor + separate check)", rep["supervisor_cleanup"]["confirmed"] is True and rep["independent_check"]["confirmed"] is True,
                    f"cleanup_confirmed={rep['cleanup_confirmed']}: {rep['cleanup_note']}")

    def s7b_escaped(self):
        p = self.start("import os,subprocess,sys,time; g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)'], start_new_session=True); print(os.getpid(), g.pid, flush=True); time.sleep(120)")
        child, escaped = self.pids_from(p, 2)
        pgids = self.remote_pgids([child, escaped])
        p.stop()
        gone, alive = self.wait_gone([child, escaped])
        rep = p.report()
        self.record("7b a grandchild that left the process group (new session) is found by its ownership token and terminated", gone and pgids and pgids[escaped] != pgids[child],
                    f"escaped pid had its own group: {pgids}; alive afterwards: {alive}")
        self.record("7c cleanup claim is consistent with what the host can detect", (rep["cleanup_confirmed"] is True and rep["escaped_descendants_detectable"]) or (rep["cleanup_confirmed"] is None and not rep["escaped_descendants_detectable"]),
                    f"cleanup_confirmed={rep['cleanup_confirmed']}, detectable={rep['escaped_descendants_detectable']}")

    def remote_pgids(self, pids):
        code = f"import os,json; print(json.dumps({{p: os.getpgid(p) for p in {list(pids)!r}}}))"
        done = subprocess.run(session.launcher_command(self.host, [self.host.python, "-c", code]), capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL)
        return {int(k): v for k, v in json.loads(done.stdout).items()} if done.returncode == 0 else None

    def s8_connection_loss(self):
        p = self.start("import os,subprocess,sys,time; g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); print(os.getpid(), g.pid, flush=True); time.sleep(120)")
        child, grand = self.pids_from(p, 2)
        t = time.monotonic()
        os.killpg(p.proc.pid, signal.SIGKILL)  # the local ssh process dies, as a dropped connection would
        gone, alive = self.wait_gone([child, grand])
        p.wait_for(lambda: p.returncode is not None, 10)
        self.record("8 ssh connection interruption terminates the remote group", gone and p.lost, f"{time.monotonic() - t:.1f} s; alive: {alive}; lost_connection={p.lost}")

    def s9_controller_kill(self):
        marker = self.dir / "ctl.pids"
        code = (f"import sys,time; sys.path.insert(0, {str(Path(__file__).resolve().parents[3])!r}); from benchmarks.suite.validation.real_launcher import make_host; from benchmarks.suite.remote import session; "
                f"h = make_host({self.host.launcher['argv']!r}).hosts['remote']; "
                f"p = session.RemoteProcess(h, ['python3','-c','import os,time; print(os.getpid(), flush=True); time.sleep(120)'], {{}}, 1, {str(self.dir / 'co')!r}, {str(self.dir / 'ce')!r}, 'ctl', 3.0, 60.0, python=h.python); "
                f"print('up', flush=True); time.sleep(300)")
        ctl = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, start_new_session=True)
        assert ctl.stdout.readline().strip() == "up"
        end = time.monotonic() + 20
        pid = None
        while time.monotonic() < end and pid is None:
            f = self.dir / "co"
            if f.exists() and f.read_text().split():
                pid = int(f.read_text().split()[0])
            time.sleep(0.1)
        t = time.monotonic()
        os.killpg(ctl.pid, signal.SIGKILL)  # the controller and its ssh die abruptly
        ctl.wait()
        gone, alive = self.wait_gone([pid])
        self.record("9 controller killed with SIGKILL leaves no remote process", gone, f"{time.monotonic() - t:.1f} s; alive: {alive}")

    def s10_lease(self):
        argv = session.launcher_command(self.host, [self.host.python, "-c", session.agent_bootstrap(), "run", "--token", "lease-check", "--lease", "2", "--max-runtime", "120", "--grace", "1", "--",
                                                    "python3", "-c", "import os,time; print(os.getpid(), flush=True); time.sleep(120)"])
        a = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)
        self.addclean(a)
        started = json.loads(a.stdout.readline())
        t = time.monotonic()
        events = [json.loads(line) for line in a.stdout]  # stdin stays open; no heartbeat is ever sent
        reason = [e for e in events if e["ev"] == "exit"][0]["reason"]
        gone, alive = self.wait_gone([started["pid"]])
        self.record("lease expiry: no heartbeat for the lease terminates the remote group", "lease of 2 s expired" in reason and gone, f"{time.monotonic() - t:.1f} s; {reason}; alive: {alive}")

    def addclean(self, proc):
        self._procs = getattr(self, "_procs", []) + [proc]

    def s11_watchdog(self):
        argv = session.launcher_command(self.host, [self.host.python, "-c", session.agent_bootstrap(), "run", "--token", "wd-check", "--lease", "60", "--max-runtime", "120", "--grace", "1", "--",
                                                    "python3", "-c", "import os,subprocess,sys,time; g=subprocess.Popen([sys.executable,'-c','import time; time.sleep(120)']); print(os.getpid(), g.pid, flush=True); time.sleep(120)"])
        a = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True)
        self.addclean(a)
        started = json.loads(a.stdout.readline())
        time.sleep(2.0)
        # kill the supervisor itself (exact pid it reported), from a separate connection: none of its cleanup code runs
        code = f"import os,signal; os.kill({started['supervisor_pid']}, signal.SIGKILL)"
        subprocess.run(session.launcher_command(self.host, [self.host.python, "-c", code]), capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
        t = time.monotonic()
        # the child's pid and its grandchild were printed on its stdout, relayed as events
        pids = []
        for line in a.stdout:
            ev = json.loads(line)
            if ev["ev"] == "out":
                import base64
                pids = [int(x) for x in base64.b64decode(ev["d"]).split()]
                break
        gone, alive = self.wait_gone(pids or [started["pid"]], 15)
        self.record("watchdog: terminating the supervisor itself (SIGKILL) still ends the benchmark group", gone, f"{time.monotonic() - t:.1f} s after the kill; alive: {alive}")

    def s12_runner_paths(self):
        g = Group([Invocation(0, "r", "local", [sys.executable, "-c", "import sys; sys.exit(2)"]), Invocation(1, "r", "remote", ["python3", "-c", "import time; time.sleep(60)"])], [])
        o = execute_group(g, self.dir / "g2", 30, 10.0, self.config, run_id="valid")
        self.record("failed local rank stops the remote rank and cleanup is confirmed or honestly unverified", o.failed_rank == 0 and o.rank_reports[1]["cleanup_confirmed"] in (True, None) and o.rank_reports[1]["independent_check"]["confirmed"] is not False,
                    f"cleanup_confirmed={o.rank_reports[1]['cleanup_confirmed']}")
        g = Group([Invocation(0, "r", "local", [sys.executable, "-c", "import time; time.sleep(60)"]), Invocation(1, "r", "remote", ["python3", "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])], [])
        o = execute_group(g, self.dir / "g3", 30, 10.0, self.config, run_id="valid")
        self.record("failed remote rank invalidates the group and stops the local rank", o.failed_rank == 1 and 0 in o.terminated_by_runner and "boom" in Path(o.stderr[1]).read_text())

    def s13_probe_and_health(self):
        data, error = session.run_agent_json(self.host, ["probe"], 60, python=self.host.python)
        self.record("helper runs on the host's Python (probe)", data is not None and data["host"]["os"], error or f"{data['host']['os']} | {data['host']['cpu_model']} | python {data['python']}")
        data, error = session.run_agent_json(self.host, ["check-clean", "--token", "nothing-runs-with-this-token"], 60, python=self.host.python)
        self.record("check-clean on a token nothing uses reports no survivors", data is not None and not data["survivors_tagged"] and not data["survivors_in_group"], error or f"tag_scan={data['tag_scan']}")

    def kill_bystander(self):
        """Remove the one process this harness started on the host outside the supervisor, by its exact pid (a plain ssh command is not stopped when its connection drops)."""
        proc, pid = self.bystander
        code = f"import os,signal; os.kill({pid}, signal.SIGKILL)"
        subprocess.run(session.launcher_command(self.host, [self.host.python, "-c", code]), capture_output=True, timeout=60, stdin=subprocess.DEVNULL)
        gone, alive = self.wait_gone([pid], 10)
        try:
            os.killpg(proc.pid, signal.SIGKILL)  # only the launcher group this harness started; it may already be gone (EPERM on macOS for a zombie leader)
        except (ProcessLookupError, PermissionError):
            pass
        proc.wait()
        self.record("harness cleanup: its own bystander process removed by exact pid", gone, f"alive: {alive}")

    def close(self):
        for p in getattr(self, "_procs", []):
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                p.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launcher-argv", required=True, help="JSON list: the command prefix that runs a command on the host, e.g. '[\"ssh\",\"-o\",\"BatchMode=yes\",\"ALIAS\"]'")
    parser.add_argument("--python", default="python3", help="Python on the host")
    parser.add_argument("--report", help="write the results as JSON here")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        v = Validation(make_host(json.loads(args.launcher_argv), args.python), tmp)
        try:
            results = v.run_all()
        finally:
            v.close()
    failed = [r for r in results if not r["passed"]]
    print(f"\n{len(results) - len(failed)} passed, {len(failed)} failed")
    if args.report:
        Path(args.report).write_text(json.dumps({"platform": v.platform, "results": results}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
