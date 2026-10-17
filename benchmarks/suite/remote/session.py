"""Controller side of supervised execution: starts the host helper (`agent.py`) through a launcher, feeds it heartbeats, turns its events into per-rank output
files and a Popen-like handle, and independently checks afterwards that nothing it started is still running.

Remote cleanup is only ever *claimed* when it has been confirmed twice: by the supervisor on the host, and by a separate `check-clean` run through the launcher.
If the host cannot be reached for the second check the report says 'unverified'; for an abrupt host failure or a network partition nothing can be done from
here, and the bound is the lease and the watchdog on the host (see docs/development/benchmark-suite.md).
"""
from __future__ import annotations

import base64
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

AGENT_PATH = Path(__file__).resolve().parent / "agent.py"
_SOURCE_B64 = None


def agent_bootstrap():
    """Python source that, given to `python3 -c`, runs the helper without any file on the host."""
    global _SOURCE_B64
    if _SOURCE_B64 is None:
        _SOURCE_B64 = base64.b64encode(AGENT_PATH.read_bytes()).decode()
    return ("import base64,sys; exec(compile(base64.b64decode('%s'), 'bench-suite-agent', 'exec'), {'__name__': '__main__'})" % _SOURCE_B64)


def new_token(run_id="", rank=0):
    return f"{run_id[:24]}.{rank}.{uuid.uuid4().hex[:12]}".replace(" ", "_")


def launcher_command(host, remote_argv):
    """The argument array that runs `remote_argv` on `host` (directly for local hosts, behind the user's prefix otherwise). Never a shell string for local hosts."""
    if host.is_local:
        return list(remote_argv)
    prefix = list(host.launcher["argv"])
    return prefix + ([shlex.join(remote_argv)] if host.launcher.get("shell_join", True) else list(remote_argv))


def agent_command(host, subcommand_args, python=None):
    return launcher_command(host, [python or getattr(host, "python", None) or "python3", "-c", agent_bootstrap(), *subcommand_args])


def run_agent_json(host, subcommand_args, timeout=60, python=None):
    """One-shot helper call (probe, check-clean, health). Returns (parsed JSON or None, error text or None)."""
    try:
        done = subprocess.run(agent_command(host, subcommand_args, python), capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, f"{type(error).__name__}: {error}"
    if done.returncode != 0:
        return None, f"exit {done.returncode}: {done.stderr.strip()[-300:]}"
    try:
        return json.loads(done.stdout), None
    except ValueError:
        return None, "the helper's output was not JSON"


class RemoteProcess:
    """Popen-like handle for one supervised rank."""

    def __init__(self, host, argv, env, rank, out_path, err_path, run_id, lease_s, max_runtime_s, grace_s=3.0, await_port=None, await_timeout=10.0, python=None):
        self.host, self.rank = host, rank
        self.token = new_token(run_id, rank)
        self.out_path, self.err_path = Path(out_path), Path(err_path)
        self.launcher_err_path = self.out_path.with_name(f"rank{rank}.launcher.stderr")
        self.lease_s, self.grace_s = lease_s, grace_s
        self.events, self.returncode = [], None
        self.started = self.listening = self.cleanup = self.exit_event = None
        self.listen_failed = None
        self.lost = False
        self.pgid = self.pid = None
        self._lock, self._changed = threading.Lock(), threading.Condition()
        self._stop_heartbeat = threading.Event()
        args = ["run", "--token", self.token, "--rank", str(rank), "--lease", str(lease_s), "--max-runtime", str(max_runtime_s), "--grace", str(grace_s)]
        if await_port:
            args += ["--await-port", str(await_port), "--await-timeout", str(await_timeout)]
        for k, v in (env or {}).items():
            args += ["--env", f"{k}={v}"]
        command = agent_command(host, args + ["--", *argv], python)
        self._out, self._err = open(self.out_path, "wb"), open(self.err_path, "wb")
        self._launcher_err = open(self.launcher_err_path, "wb")
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._launcher_err, start_new_session=True)
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._beat = threading.Thread(target=self._heartbeat, daemon=True)
        self._beat.start()

    # -- events
    def _signal(self):
        with self._changed:
            self._changed.notify_all()

    def _read(self):
        try:
            for raw in self.proc.stdout:
                try:
                    ev = json.loads(raw)
                except ValueError:
                    continue
                kind = ev.get("ev")
                if kind == "out":
                    self._out.write(base64.b64decode(ev["d"]))
                    self._out.flush()
                elif kind == "err":
                    self._err.write(base64.b64decode(ev["d"]))
                    self._err.flush()
                else:
                    self.events.append(ev)
                    if kind == "started":
                        self.started, self.pid, self.pgid = ev, ev.get("pid"), ev.get("pgid")
                    elif kind == "listening":
                        self.listening = ev
                    elif kind == "listen_failed":
                        self.listen_failed = ev
                    elif kind == "cleanup":
                        self.cleanup = ev
                    elif kind == "exit":
                        self.exit_event = ev
                self._signal()
        except (OSError, ValueError):
            pass
        finally:
            code = self.proc.wait()
            if self.exit_event is None:
                self.lost = True
                self.returncode = code if code not in (0, None) else -1
            else:
                e = self.exit_event
                self.returncode = e["code"] if e.get("code") is not None else (-e["signal"] if e.get("signal") else -1)
                if e.get("reason") and self.returncode == 0:
                    self.returncode = -1
            self._stop_heartbeat.set()
            for stream in (self.proc.stdin, self.proc.stdout):
                try:
                    stream.close()
                except (OSError, ValueError):
                    pass
            for handle in (self._out, self._err, self._launcher_err):
                handle.close()
            self._signal()

    def _heartbeat(self):
        while not self._stop_heartbeat.wait(max(0.05, self.lease_s / 4.0)):
            try:
                self.proc.stdin.write(b"H\n")
                self.proc.stdin.flush()
            except (OSError, ValueError):
                return

    # -- Popen-like surface
    def poll(self):
        return self.returncode

    def wait_for(self, predicate, timeout):
        deadline = time.monotonic() + timeout
        with self._changed:
            while not predicate():
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self._changed.wait(min(left, 0.05))
        return True

    def await_listening(self, timeout):
        """('ready', detail) | ('failed', reason). Proven by the host's supervisor from the socket table and the owning process, never by connecting."""
        self.wait_for(lambda: self.listening or self.listen_failed or self.returncode is not None, timeout)
        if self.listening:
            return "ready", self.listening.get("detail")
        if self.listen_failed:
            return "failed", f"{self.listen_failed.get('outcome')}: {self.listen_failed.get('detail')}"
        if self.returncode is not None:
            return "failed", "the process ended before listening"
        return "failed", "readiness could not be proven within the timeout"

    def cancel(self):
        try:
            self.proc.stdin.write(b"C\n")
            self.proc.stdin.flush()
        except (OSError, ValueError):
            pass

    def stop(self, wait_s=None):
        """Cancel and wait for the helper's cleanup report; if the launcher does not answer, kill the launcher (which closes the helper's stdin: a disconnect)."""
        wait_s = (self.grace_s * 2 + 6.0) if wait_s is None else wait_s
        if self.returncode is None:
            self.cancel()
            if not self.wait_for(lambda: self.returncode is not None, wait_s):
                self._kill_launcher()
                self.wait_for(lambda: self.returncode is not None, 5.0)
        self._reader.join(timeout=5)

    def _kill_launcher(self):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)  # only the launcher process group this object started
        except (ProcessLookupError, PermissionError):
            pass

    def verify_clean(self, timeout=30):
        """Independent confirmation through a separate helper run. confirmed is True/False, or None when the host could not be asked."""
        data, error = run_agent_json(self.host, ["check-clean", "--token", self.token] + (["--pgid", str(self.pgid)] if self.pgid else []), timeout)
        if data is None:
            return {"confirmed": None, "error": error}
        survivors = list(data.get("survivors_in_group") or []) + list(data.get("survivors_tagged") or [])
        return {"confirmed": not survivors, "survivors": survivors, "tag_scan": data.get("tag_scan"), "group_checked": data.get("group_checked")}

    def report(self, verify=True):
        rep = {"supervised": True, "token": self.token, "host_pid": self.pid, "host_pgid": self.pgid, "lost_connection": self.lost, "lease_s": self.lease_s,
               "supervisor_cleanup": ({"confirmed": self.cleanup.get("confirmed"), "swept": self.cleanup.get("swept"), "leftover": self.cleanup.get("leftover")} if self.cleanup else None),
               "exit_reason": (self.exit_event or {}).get("reason"), "cpu": (self.exit_event or {}).get("cpu")}
        if verify:
            rep["independent_check"] = self.verify_clean()
        independent = (rep.get("independent_check") or {}).get("confirmed")
        sup = (rep["supervisor_cleanup"] or {}).get("confirmed")
        scan = bool((self.cleanup or {}).get("tag_scan")) and (not verify or bool((rep.get("independent_check") or {}).get("tag_scan")))
        rep["escaped_descendants_detectable"] = scan
        if independent is False or sup is False:
            rep["cleanup_confirmed"], rep["cleanup_note"] = False, "NOT clean: processes owned by this run were still present"
        elif independent is None:
            rep["cleanup_confirmed"] = None
            rep["cleanup_note"] = "NOT confirmed: the host could not be checked independently; processes are bounded by the lease and the host watchdog"
        elif not scan:
            rep["cleanup_confirmed"] = None
            rep["cleanup_note"] = ("NOT confirmed: the process group is empty, but this host cannot prove it can see descendants that left the group (ownership-token scan unavailable), "
                                   "so escaped descendants are not ruled out")
        else:
            rep["cleanup_confirmed"] = bool(sup and independent)
            rep["cleanup_note"] = "confirmed by the host supervisor and by a separate check, including descendants that left the process group (ownership-token scan proven on this host)"
        return rep
