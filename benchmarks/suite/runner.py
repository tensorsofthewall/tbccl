"""Subprocess runner: allocates ports, starts the ranks of one measurement as a process group, enforces timeouts and turns the outcome into result records.

Rules the runner enforces:
  * argument arrays only, never a shell string; every child starts in its own session so only processes this run started are ever signalled;
  * one measurement group at a time (concurrent measurements would disturb each other); at most `limits.max_processes` processes per group;
  * a failure, a nonzero exit or a timeout of any rank invalidates the whole group and every process the runner started is terminated;
  * a failed, timed-out, skipped or unparseable run is recorded without measurements, with its raw output kept;
  * durations measured here use time.monotonic(); timestamps from different hosts are never subtracted.
"""
from __future__ import annotations

import json
import os
import random
import re
import shlex
import signal
import socket
import subprocess
import time
from pathlib import Path

from . import SUITE_VERSION, health, inventory, providers, provenance, schema
from .adapters import REGISTRY
from .adapters.base import Group, Outcome, ParseError, memory_domain_label
from .config import expand_matrix, port_range
from .preflight import Check, check_config, failed, format_checks
from .remote import agent
from .remote import session as remote_session
from .remote.session import RemoteProcess


class MissingExecutable(Exception):
    pass


class RunContext:
    """What adapters may ask about the configured topology and the local installation."""

    def __init__(self, config, repo_root=inventory.REPO_ROOT):
        self.config = config
        self.repo_root = repo_root
        self._builds = {}
        self._host_info = {}
        self.host_provenance = {}
        self._providers = {}
        self._shared = {}

    def address(self, host_id):
        return self.config.hosts[host_id].address

    def rank_address(self, spec):
        """The address peers use to reach this rank: the rank's own `address`, else its host's interface of that name, else the host address."""
        host = self.config.hosts[spec["host"]]
        if spec.get("address"):
            return spec["address"]
        if spec.get("interface") and spec["interface"] in host.interfaces:
            return host.interfaces[spec["interface"]]
        return host.address

    def is_local(self, host_id):
        return self.config.hosts[host_id].is_local

    def executable_or_none(self, name, host_id):
        host = self.config.hosts[host_id]
        if not host.is_local:
            return None
        if name in host.executables:
            path = Path(host.executables[name])
            return path if path.is_file() and os.access(path, os.X_OK) else None
        dirs = ([host.executables_dir] if host.executables_dir else []) + list(self.config.build_dirs)
        return inventory.find_executable(name, dirs, self.repo_root)

    def executable(self, name, host_id):
        host = self.config.hosts[host_id]
        if not host.is_local:
            if name in host.executables:
                return host.executables[name]
            return str(Path(host.executables_dir) / name) if host.executables_dir else name
        path = self.executable_or_none(name, host_id)
        if path is None:
            raise MissingExecutable(f"executable '{name}' not found for host '{host_id}'")
        return str(path)

    def tool(self, name, host_id):
        """A tool that is not part of a TBCCL build (iperf3): the host's configured path, else PATH (local) or the bare name (remote)."""
        host = self.config.hosts[host_id]
        if name in host.executables:
            return host.executables[name]
        if host.is_local:
            import shutil
            found = shutil.which(name)
            if not found:
                raise MissingExecutable(f"tool '{name}' not found on PATH for host '{host_id}' (set hosts.{host_id}.executables.{name})")
            return found
        return name

    def build(self, path):
        if path not in self._builds:
            self._builds[path] = inventory.build_info(Path(path))
        return self._builds[path]

    def providers(self, host_id):
        probed = (self.host_provenance.get(host_id) or {}).get("executables") or {}
        for entry in probed.values():
            if entry.get("providers"):
                return set(entry["providers"])
        if not self.is_local(host_id):
            return None
        for adapter in REGISTRY.values():
            path = self.executable_or_none(adapter.caps.executable, host_id)
            if path:
                providers = self.build(path)["providers"]
                if providers:
                    return set(providers)
        return None

    def host_description(self, host_id):
        if host_id not in self._host_info:
            host = self.config.hosts[host_id]
            declared = {"os": host.os, "arch": host.arch, "cpu_model": host.cpu, "gpu_model": host.gpu, "driver": host.driver}
            probed = (self.host_provenance.get(host_id) or {}).get("host")
            if probed:
                merged = {k: declared.get(k) or probed.get(k) for k in declared}
                merged.update({k: probed.get(k) for k in ("cuda_runtime", "metal") if probed.get(k)})
                source = "detected" if host.is_local else "probed"
            elif host.is_local:
                detected = inventory.host_info()
                merged = {k: declared.get(k) or detected.get(k) for k in declared}
                merged.update({k: detected.get(k) for k in ("cuda_runtime", "metal") if detected.get(k)})
                source = "detected"
            else:
                merged = declared
                source = "declared" if any(declared.values()) else "unknown"
            self._host_info[host_id] = {**merged, "info_source": source}
        return self._host_info[host_id]

    def provider_entries(self, point):
        """What each external provider used by this point reported about itself on the hosts taking part (cached per run). A provider that is not configured, or whose description could not be obtained, is recorded as such."""
        pid = providers.provider_of(point.benchmark)
        if pid is None:
            return []
        out = []
        for host_id in sorted({r["host"] for r in point.ranks}):
            key = (pid, host_id)
            if key not in self._providers:
                host = self.config.hosts[host_id]
                python, directory = providers.configured(host, pid)
                if not python or not directory:
                    self._providers[key] = {"provider_id": pid, "host": host_id, "status": "bundled_compat", "provider_version": None, "protocol_version": None, "interpreter": python,
                                            "directory": None, "framework_versions": {}, "entry_point_sha256": providers.bundled_hashes(self.repo_root, pid) if host.is_local else {},
                                            "note": "no provider directory configured: the compatibility copy bundled with the suite was used; framework and adapter versions are unavailable"}
                else:
                    raw = providers.run_describe(python, directory) if host.is_local else providers.remote_describe(host, python, directory)
                    row = {"host": host_id, "python": python, "directory": directory, **providers.interpret(pid, raw)}
                    entry = (providers.record_entries([row], {host_id}) or [None])[0]
                    self._providers[key] = entry or {"provider_id": pid, "host": host_id, "status": row["status"], "provider_version": None, "protocol_version": None, "interpreter": python,
                                                     "directory": directory, "framework_versions": {}, "note": "; ".join(row["reasons"])[:300]}
            out.append(self._shared.setdefault(key, providers.shareable_entry(self._providers[key])))
        return out

    def executable_provenance(self, host_id, name):
        return ((self.host_provenance.get(host_id) or {}).get("executables") or {}).get(name)


class PortAllocator:
    """Distinct ports for every measurement. Both p and p+1000 are checked because other TBCCL tools place a data plane at +1000 of a rank's endpoint."""

    def __init__(self, low=30000, high=40000, offsets=(0, 1000), seed=None):
        self.low, self.high, self.offsets = low, high, offsets
        self._cursor = random.Random(seed if seed is not None else os.getpid()).randrange(low, high - 1)
        self._used = set()

    def _free(self, port):
        for offset in self.offsets:
            for addr in ("127.0.0.1", "0.0.0.0"):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # as the native listeners do: only a live listener makes a port unavailable, not TIME_WAIT
                    try:
                        sock.bind((addr, port + offset))
                    except OSError:
                        return False
        return True

    def release(self, ports):
        """Return ports once the processes that used them are gone, so a narrow range (a firewall allowance) lasts for a whole campaign. The cursor keeps moving forward,
        so a port is not handed out again until the rest of the range has been used."""
        self._used.difference_update(ports)

    def allocate(self, count):
        ports = []
        for _ in range(self.high - self.low):
            port = self._cursor
            self._cursor = self.low if self._cursor + 1 >= self.high else self._cursor + 1
            if port in self._used or port + max(self.offsets) >= 65536:
                continue
            if self._free(port):
                self._used.add(port)
                ports.append(port)
                if len(ports) == count:
                    return ports
        raise RuntimeError(f"could not find {count} free ports in {self.low}-{self.high}")


def child_env(extra=None):
    """The suite's own environment minus every TBCCL_* variable, so a user's shell cannot change algorithm selection or tracing; explicit options.env is added."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("TBCCL_")}
    env.update({str(k): str(v) for k, v in (extra or {}).items()})
    return env


def stripped_env_names():
    return sorted(k for k in os.environ if k.startswith("TBCCL_"))


def wrap_for_host(host, argv, env_extra):
    """Local hosts run the argument array directly; a 'prefix' launcher (ssh-like, supplied by the user) gets the command and its environment appended."""
    if host.is_local:
        return list(argv)
    prefix = list(host.launcher["argv"])
    remote = ["env"] + [f"{k}={v}" for k, v in env_extra.items()] + list(argv)
    return prefix + ([shlex.join(remote)] if host.launcher.get("shell_join", True) else remote)


def _sweep(pgid):
    """Kill whatever is left in a process group this runner created."""
    if agent.group_members(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def terminate(proc, grace=3.0):
    """Terminate one process group this runner started. Refuses to signal a process that is not its own group leader."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    if pgid != proc.pid:
        proc.terminate()
    else:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(pgid, signal.SIGKILL) if pgid == proc.pid else proc.kill()
        except ProcessLookupError:
            pass
        proc.wait()


class LocalHandle:
    """A rank started directly on this machine, in its own session, carrying a unique ownership token in its environment."""
    supervised = False

    def __init__(self, argv, env, out_path, err_path, cwd, token, host_id):
        env = dict(env)
        env[agent.TAG_ENV] = token
        self.token, self.host_id = token, host_id
        self.rusage, self._started = None, time.monotonic()
        self._out, self._err = open(out_path, "wb"), open(err_path, "wb")
        try:
            self.proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=self._out, stderr=self._err, env=env, start_new_session=True, cwd=cwd)
        except OSError:
            self._out.close()
            self._err.close()
            raise
        self.pgid = self.proc.pid

    def poll(self):
        proc = self.proc
        if proc.returncode is None:
            try:
                pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
            except ChildProcessError:
                return proc.returncode
            if pid:
                proc.returncode = os.waitstatus_to_exitcode(status)
                self.rusage = usage
        return proc.returncode

    @property
    def returncode(self):
        return self.proc.returncode

    def await_listening(self, port, timeout):
        outcome, detail = agent.wait_for_listener(port, self.pgid, self.token, lambda: self.poll() is None, timeout)
        return ("ready", detail) if outcome == "ready" else ("failed", f"{outcome}: {detail}")

    def stop(self, failure):
        """Returns True when the process had to be terminated by the runner."""
        terminated = False
        if self.proc.poll() is None:
            terminated = True
            terminate(self.proc)
        if failure or terminated:
            _sweep(self.pgid)
        self._out.close()
        self._err.close()
        return terminated

    def report(self, verify=True):
        left = sorted(agent.owned_pids(self.pgid, self.token)) if self.proc.poll() is not None else None
        scan = agent.tag_scan_proven()
        confirmed = ((not left) if scan else (None if not left else False)) if left is not None else None
        cpu = None
        if self.rusage is not None:
            cpu = {"child_user_s": round(self.rusage.ru_utime, 4), "child_system_s": round(self.rusage.ru_stime, 4), "child_max_rss_kb": self.rusage.ru_maxrss,
                   "supervisor_cpu_s": None, "wall_s": round(time.monotonic() - self._started, 4)}
        return {"supervised": False, "token": self.token, "host_pid": self.proc.pid, "host_pgid": self.pgid, "cleanup_confirmed": confirmed, "survivors": left or [], "cpu": cpu,
                "escaped_descendants_detectable": scan,
                "cleanup_note": "checked locally against the process group and the ownership token" if scan else
                "the process group is empty, but descendants that left the group cannot be detected on this host (ownership-token scan unavailable)"}


def _handle_for(inv, host, rank_dir, env, run_id, defaults, listens, cwd, deadline_s):
    out_path, err_path = rank_dir / f"rank{inv.rank}.stdout", rank_dir / f"rank{inv.rank}.stderr"
    supervised = (not host.is_local) or host.launcher.get("supervised")
    if supervised:
        merged = {**host.env, **inv.env, **(env or {})}
        return RemoteProcess(host, inv.argv, merged, inv.rank, out_path, err_path, run_id, lease_s=defaults.get("lease_s", 30.0), max_runtime_s=deadline_s,
                             grace_s=defaults.get("term_grace_s", 3.0), await_port=listens, await_timeout=defaults.get("startup_wait_s", 10.0), python=host.python)
    full = child_env({**host.env, **(env or {})})
    full.update(inv.env)
    return LocalHandle(inv.argv, full, out_path, err_path, cwd, remote_session.new_token(run_id, inv.rank), host.id)


def execute_group(group: Group, rep_dir, timeout_s, startup_wait_s, config, env_extra=None, poll_s=0.02, run_id="run", verify_cleanup=True) -> Outcome:
    """Start the group's processes in order, wait for them, and clean up everything started. All durations use the monotonic clock.

    A listener is considered ready only when a process this run started holds a LISTEN socket on the port (no connection is ever made to it); if that cannot be
    shown within the startup wait, or the process exits first, the group fails."""
    rep_dir = Path(rep_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + timeout_s
    handles, stdout, stderr = {}, {}, {}
    failed_rank, reason, timed_out = None, None, False
    persistent = {inv.rank for inv in group.invocations if inv.persistent}
    listeners = {inv.listens_on: inv.rank for inv in group.invocations if inv.listens_on}
    defaults = {**config.defaults, "startup_wait_s": startup_wait_s}

    def early_failure():
        for rank, h in handles.items():
            code = h.poll()
            if code not in (None, 0):
                return rank, f"rank {rank} exited with status {code}" + (" (connection to its host was lost)" if getattr(h, "lost", False) else "")
        return None, None

    try:
        for inv in group.invocations:
            host = config.hosts[inv.host_id]
            if inv.wait_for_port is not None:
                owner = handles.get(listeners.get(inv.wait_for_port))
                if owner is None:
                    failed_rank, reason = inv.rank, f"readiness of port {inv.wait_for_port} cannot be proven: no started process is declared to listen on it"
                    break
                state, detail = owner.await_listening(inv.wait_for_port, max(0.0, min(deadline - time.monotonic(), startup_wait_s))) if not isinstance(owner, RemoteProcess) \
                    else owner.await_listening(max(0.0, min(deadline - time.monotonic(), startup_wait_s + 2.0)))
                if state != "ready":
                    failed_rank, reason = listeners[inv.wait_for_port], f"rank {listeners[inv.wait_for_port]} did not start listening on port {inv.wait_for_port}: {detail}"
                    break
            failed_rank, reason = early_failure()
            if failed_rank is not None:
                break
            try:
                handles[inv.rank] = _handle_for(inv, host, rep_dir, (env_extra or {}), run_id, defaults, inv.listens_on, str(rep_dir), timeout_s + 10.0)
            except OSError as error:
                failed_rank, reason = inv.rank, f"could not start rank {inv.rank}: {error}"
                break
            stdout[inv.rank], stderr[inv.rank] = str(rep_dir / f"rank{inv.rank}.stdout"), str(rep_dir / f"rank{inv.rank}.stderr")
        while failed_rank is None:
            failed_rank, reason = early_failure()
            if failed_rank is not None:
                break
            if all(h.poll() == 0 for r, h in handles.items() if r not in persistent):
                break
            if time.monotonic() >= deadline:
                timed_out, reason = True, f"timed out after {timeout_s:g} s"
                break
            time.sleep(poll_s)
    finally:
        failure = failed_rank is not None or timed_out
        terminated, stop_s = set(), {}
        first = {inv.rank for inv in group.invocations if inv.stop_first}
        for rank, h in sorted(handles.items(), key=lambda kv: kv[0] not in first):
            t_stop = time.monotonic()
            if h.stop(failure) if isinstance(h, LocalHandle) else _stop_remote(h):
                terminated.add(rank)
            stop_s[rank] = round(time.monotonic() - t_stop, 3)
    reports = {}
    for rank, h in handles.items():
        rep = h.report(verify=verify_cleanup) if isinstance(h, RemoteProcess) else h.report()
        rep["host_id"] = group.invocations[rank].host_id
        rep["stop_s"] = stop_s.get(rank)
        reports[rank] = rep
    exit_codes = {rank: (None if rank in terminated and rank in persistent and not (failed_rank is not None or timed_out) else h.returncode) for rank, h in handles.items()}
    unconfirmed = [r for r, rep in reports.items() if rep.get("supervised") and rep.get("cleanup_confirmed") is not True]
    return Outcome(exit_codes=exit_codes, terminated_by_runner=terminated, timed_out=timed_out, failed_rank=failed_rank, stdout=stdout, stderr=stderr,
                   wall_s=time.monotonic() - started, reason=reason, rank_reports=reports, cleanup_unconfirmed=unconfirmed)


def _stop_remote(handle):
    was_running = handle.poll() is None
    handle.stop()
    return was_running


class Redactor:
    """Replaces non-loopback addresses and local paths in strings that go into tracked-looking records: configured addresses, the checkout and the home directory by name, every other absolute path by its last two components. Raw stdout/stderr files are not rewritten."""

    def __init__(self, config, repo_root=inventory.REPO_ROOT):
        self.pairs = [(h.address, f"{{address:{h.id}}}") for h in config.hosts.values() if not h.is_loopback and h.address]
        for h in config.hosts.values():
            if not h.is_loopback:
                self.pairs += [(a, f"{{address:{h.id}}}") for a in dict.fromkeys(h.interfaces.values()) if a and a != h.address]
        self.pairs.append((str(repo_root), "<repo>"))
        home = str(Path.home())
        if home not in ("/", ""):
            self.pairs.append((home, "~"))

    def text(self, value):
        for old, new in self.pairs:
            value = value.replace(old, new)
        return providers.shareable_path(value)  # any other absolute path (another host's home, a mount point, a venv) keeps only its last two components

    def argv(self, argv):
        return [self.text(a) for a in argv]

    def obj(self, value):
        """The same replacement applied to every string, and every key, of a nested record or manifest."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {self.text(k) if isinstance(k, str) else k: self.obj(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.obj(v) for v in value]
        return value


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "run"


def new_run_id(label, name):
    return f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{slug(label or name)}-{os.urandom(2).hex()}"


def display_command(host, inv):
    """The command as it will be started. Supervised hosts show the launcher prefix, the helper's arguments and the benchmark command after `--`."""
    if host.is_local and not host.launcher.get("supervised"):
        return shlex.join(wrap_for_host(host, inv.argv, inv.env))
    args = ["run", "--token", "<token>", "--rank", str(inv.rank), "--lease", "<lease>", "--max-runtime", "<cap>"]
    if inv.listens_on:
        args += ["--await-port", str(inv.listens_on)]
    args += [x for k, v in {**host.env, **inv.env}.items() for x in ("--env", f"{k}={v}")] + ["--", *inv.argv]
    return shlex.join(remote_session.launcher_command(host, [host.python or "python3", "-c", "<supervisor helper source>", *args]))


def format_matrix(points, config, ctx):
    ports = PortAllocator(*port_range(config), seed=0)
    lines = [f"{'#':>3}  {'benchmark':<21} {'operation':<14} {'algorithm':<14} {'dtype':<8} {'ranks':>5} {'backends':<22} {'payload':>12}  warmup/iters/reps  status"]
    commands = []
    for i, p in enumerate(points):
        status = "run" if not p.skip_reason else f"{p.skip_state.upper()}: {p.skip_reason}"
        lines.append(f"{i:>3}  {p.benchmark:<21} {p.operation:<14} {(p.algorithm or '-'):<14} {(p.datatype or '-'):<8} {p.rank_count:>5} {','.join(p.backends):<22} {p.payload_bytes:>10} B  "
                     f"{p.warmup}/{p.iterations}/{p.repetitions:<6}  {status}")
        if not p.skip_reason:
            planned_ports = ports.allocate(p.rank_count if p.benchmark != "tcp_pingpong" else 1)
            try:
                group = REGISTRY[p.benchmark].plan(p, ctx, planned_ports, Path("<run-dir>/raw") / p.point_id / "rep0")
            except MissingExecutable as error:
                commands.append(f"[{i}] cannot plan: {error}")
                continue
            finally:
                ports.release(planned_ports)  # nothing was started, so a long matrix needs no more ports than one point
            for inv in group.invocations:
                commands.append(f"[{i}] rank {inv.rank} ({inv.role}) on {inv.host_id}{' (terminated by the runner when the others finish)' if inv.persistent else ''}: {display_command(config.hosts[inv.host_id], inv)}")
    return "\n".join(lines), "\n".join(commands)


class Runner:
    def __init__(self, config, ctx=None, output_dir=None, label=None, allow_physical=False, monitor=None, log=print, ports=None):
        self.config = config
        self.ctx = ctx or RunContext(config)
        self.output_root = Path(output_dir or config.output_dir)
        if not self.output_root.is_absolute() and output_dir is None:
            self.output_root = inventory.REPO_ROOT / self.output_root
        self.label = label
        self.allow_physical = allow_physical
        self.monitor = monitor
        self.log = log
        self.ports = ports or PortAllocator(*port_range(config))
        self.redactor = Redactor(config)
        self.git = inventory.git_info(self.ctx.repo_root)
        self.provenance_problems, self.provenance_notes = [], []
        self.route_problems, self.management_notes = [], []

    def points(self):
        return expand_matrix(self.config, self.ctx)

    def prepare(self, points):
        """Probe every host the run will use once (hardware, executables, build provenance) and check that the hosts are consistent with each other."""
        used = sorted({r["host"] for p in points if not p.skip_reason for r in p.ranks})
        exes = sorted({REGISTRY[p.benchmark].caps.executable for p in points if not p.skip_reason and REGISTRY[p.benchmark].caps.executable_in_build})
        ifaces = {}
        for p in points:
            for r in p.ranks:
                ifaces.setdefault(r["host"], r.get("interface") or self.config.hosts[r["host"]].interface)
        peers = {}
        for p in points:
            if p.skip_reason:
                continue
            for r in p.ranks:
                for o in p.ranks:
                    if o["host"] != r["host"]:
                        peers.setdefault(r["host"], set()).add(self.ctx.rank_address(o))
        peers = {k: sorted(v) for k, v in peers.items()}
        self.ctx.host_provenance = provenance.collect(self.config, self.ctx, used, exes, ifaces, self.git["commit"], routes_of=peers)
        self.route_problems = provenance.route_problems(self.config, self.ctx.host_provenance, peers) if self.config.physical else []
        self.management_notes = []
        for host in self.config.hosts.values():
            if host.management_address and host.id in used:
                via = agent.route_interface(host.management_address)
                data_ifaces = {h.interface for h in self.config.hosts.values() if h.is_local and h.interface}
                if via is None or via in data_ifaces:
                    self.route_problems.append(f"{host.id}: the management path to {{management address}} uses '{via}', which is a data interface")
                else:
                    self.management_notes.append(f"{host.id}: management traffic (launcher, heartbeats, output relay) leaves the controller through '{via}', separate from the data interface(s) {sorted(data_ifaces)}")
        self.ctx._host_info.clear()
        self.provenance_problems = provenance.consistency(self.ctx.host_provenance, self.config.defaults["provenance"])
        self.provenance_notes = provenance.commit_notes(self.ctx.host_provenance)

    def _environment(self, point):
        hosts = {}
        for rank in point.ranks:
            hosts.setdefault(rank["host"], []).append(rank["rank"])
        exe_name = REGISTRY[point.benchmark].caps.executable
        host_entries = []
        for hid, ranks in hosts.items():
            entry = {"host_id": hid, "ranks": ranks, **self.ctx.host_description(hid)}
            prov = self.ctx.executable_provenance(hid, exe_name)
            if prov:
                entry["build"] = {k: prov.get(k) for k in ("executable_sha256", "build_type", "compiler", "generator", "cuda_enabled", "metal_enabled", "cuda_compiler_version", "tbccl_package_version",
                                                           "c_abi", "wire_protocol", "status", "source", "build_commit", "build_dirty", "reason")}
            host_entries.append(entry)
        first_spec = point.ranks[0]
        first = self.config.hosts[first_spec["host"]]
        alias = first_spec.get("interface")  # a rank's `interface` names an entry of its host's `interfaces` (an address alias); the device is the host's `interface`
        iface = (first.interface if alias in first.interfaces else alias) or first.interface or (agent.LOOPBACK_INTERFACE if first.is_loopback and first.is_local else None)
        probed_iface = (self.ctx.host_provenance.get(first.id) or {}).get("interface") or {}
        info = probed_iface if probed_iface.get("name") == iface and probed_iface.get("present") else (inventory.interface_info(iface) if first.is_local else {"mtu": first.mtu})
        prov0 = self.ctx.executable_provenance(first.id, exe_name)
        build = self.ctx.build(self.ctx.executable_or_none(exe_name, first.id)) if (not prov0 and self.ctx.executable_or_none(exe_name, first.id)) else {}
        versions = {k: v for k, v in (("tbccl", (prov0 or {}).get("tbccl_package_version") or build.get("package_version")), ("wire_protocol", (prov0 or {}).get("wire_protocol") or build.get("wire_protocol")),
                                      ("c_abi", (prov0 or {}).get("c_abi") or build.get("c_abi"))) if v}
        topo = self.config.topology
        return {"hosts": host_entries, "memory_backend": point.backends,
                "memory_domain": {"label": memory_domain_label(point.backends), "per_rank": point.backends},
                "topology": {"kind": topo["kind"], "interconnect": topo["interconnect"], "physical": bool(topo.get("physical"))},
                "network_interface": iface, "transport": topo.get("transport", "tcp"), "mtu": info.get("mtu") or first.mtu, "framework_versions": versions,
                "multi_host": len(hosts) > 1, "rank_interfaces": [r.get("interface") for r in point.ranks],
                **({"providers": self.ctx.provider_entries(point)} if providers.provider_of(point.benchmark) else {})}

    def _benchmark(self, point, adapter, description):
        caps = adapter.caps
        host_id = point.ranks[0]["host"]
        prov = self.ctx.executable_provenance(host_id, caps.executable)
        local_exe = self.ctx.executable_or_none(caps.executable, host_id)
        build = self.ctx.build(local_exe) if (local_exe and not prov) else {}
        verified = prov and prov["status"] in ("stamped", "declared")
        return {"name": caps.name, "adapter": caps.name, "category": caps.category, "operation": point.operation, "source_repository": inventory.TRACKED_REPOSITORY,
                "source_commit": prov["build_commit"] if verified else None, "source_dirty": prov["build_dirty"] if verified else None,
                "repository_commit": self.git["commit"], "repository_dirty": self.git["dirty"], "build_provenance": prov and {k: prov.get(k) for k in ("status", "source", "reason", "build_commit", "matches_controller_head")},
                "executable": caps.executable, "executable_sha256": (prov or {}).get("executable_sha256") or (inventory.sha256_file(local_exe) if local_exe else None),
                "executable_version": build.get("version_text") or (prov and prov.get("tbccl_package_version") and f"TBCCL {prov['tbccl_package_version']}"), "build_type": (prov or {}).get("build_type") or build.get("build_type"),
                "algorithm_requested": point.algorithm, "backend": description.get("backend"), "provenance": "native"}

    def _data_direction(self, point):
        d = dict(REGISTRY[point.benchmark].data_direction(point))
        if d.get("mode") == "directional":
            src = d.get("source_rank") or 0
            dst = 1 - src if point.rank_count == 2 else None
            d.update(source_rank=src, source_host=point.ranks[src]["host"], destination_rank=dst, destination_host=point.ranks[dst]["host"] if dst is not None else None)
        return d

    def _configuration(self, point, description):
        return {"data_direction": self._data_direction(point), "estimated_bytes_per_repetition": point.estimated_bytes, "orientation": point.orientation,
                **self._configuration_core(point, description)}

    def _configuration_core(self, point, description):
        return {"payload_bytes": point.payload_bytes, "payload_semantics": description["payload_semantics"], "datatype": point.datatype, "reduce_op": point.reduce_op,
                "rank_count": point.rank_count, "rank_order": [r["host"] for r in point.ranks], "warmup": point.warmup, "iterations": point.iterations,
                "repetitions": point.repetitions, "timeout_s": point.timeout_s, "timing_method": description["timing_method"],
                "completion_semantics": description["completion_semantics"], "options": point.options}

    def _stderr_tail(self, outcome, rank=None, lines=3):
        ranks = [rank] if rank is not None else sorted(outcome.stderr)
        tail = []
        for r in ranks:
            try:
                text = Path(outcome.stderr[r]).read_text(errors="replace").strip().splitlines()
            except (KeyError, OSError):
                continue
            tail += [f"rank {r}: {line}" for line in text[-lines:]]
        return self.redactor.text(" | ".join(tail))[:400]

    def _record(self, run_id, run_dir, point, rep, state, reason, *, outcome=None, parsed=None, verification=None, group=None, adapter=None):
        adapter = adapter or REGISTRY[point.benchmark]
        description = adapter.describe(point)
        configuration = self._configuration(point, description)
        benchmark = self._benchmark(point, adapter, description)
        measurements = None
        if parsed is not None and state == "success":
            measurements = self.redactor.obj(parsed.measurements)
            benchmark["algorithm_resolved"] = parsed.algorithm_resolved
            benchmark["selection_reason"] = parsed.selection_reason
            configuration.update(parsed.configuration_extra)
            build = benchmark.get("build_type")
            if build and build not in ("Release", "RelWithDebInfo", "MinSizeRel"):
                measurements["warnings"].append(f"the executable was built as '{build}', not an optimized build")
        artifacts = {"commands": [self.redactor.argv(i.argv) for i in group.invocations] if group else [], "ranks": []}
        if outcome:
            rel = lambda p: os.path.relpath(p, run_dir)  # noqa: E731
            artifacts["wall_s"] = round(outcome.wall_s, 3)
            artifacts["stdout"] = [rel(p) for p in outcome.stdout.values()]
            artifacts["stderr"] = [rel(p) for p in outcome.stderr.values()]
            artifacts["raw"] = [rel(i.output_file) for i in (group.invocations if group else []) if i.output_file and Path(i.output_file).exists()]
            for inv in (group.invocations if group else []):
                rpt = outcome.rank_reports.get(inv.rank, {})
                spec = point.ranks[inv.rank] if inv.rank < len(point.ranks) else {"memory_backend": None}  # helper invocations (samplers, clients) follow the ranks
                artifacts["ranks"].append({"rank": inv.rank, "host_id": inv.host_id, "role": inv.role, "memory_backend": spec["memory_backend"], "interface": spec.get("interface"),
                                           "exit_code": outcome.exit_codes.get(inv.rank), "terminated_by_runner": inv.rank in outcome.terminated_by_runner,
                                           "stdout": rel(outcome.stdout[inv.rank]) if inv.rank in outcome.stdout else None, "stderr": rel(outcome.stderr[inv.rank]) if inv.rank in outcome.stderr else None,
                                           "supervised": rpt.get("supervised", False), "lost_connection": rpt.get("lost_connection", False), "cleanup_confirmed": rpt.get("cleanup_confirmed"),
                                           "cleanup_note": rpt.get("cleanup_note"), "started": inv.rank in outcome.stdout, "cpu": rpt.get("cpu"), "stop_s": rpt.get("stop_s")})
            if outcome.cleanup_unconfirmed:
                note = f"cleanup on rank(s) {outcome.cleanup_unconfirmed} could not be independently confirmed"
                if measurements is not None:
                    measurements["warnings"].append(note)
                elif reason:
                    reason = f"{reason} ({note})"
        record = schema.new_record(
            run_id=run_id, record_id=f"{run_id}/{point.point_id}/rep{rep}", benchmark=benchmark, environment=self._environment(point), configuration={**configuration, "repetition_index": rep},
            status=state, reason=reason, exit_codes={str(k): v for k, v in (outcome.exit_codes if outcome else {}).items()}, measurements=measurements,
            verification=verification or {"status": "not_run", "method": None, "details": reason or ""}, artifacts=artifacts)
        errors = schema.validate_record(record)
        if errors and state == "success":
            record = schema.new_record(run_id=run_id, record_id=record["record_id"], benchmark=benchmark, environment=record["environment"],
                                       configuration=record["configuration"], status="invalid", reason="the result failed schema validation: " + "; ".join(errors)[:300],
                                       exit_codes=record["status"]["exit_codes"], verification=record["verification"], artifacts=artifacts)
        return record

    def _repetition(self, run_id, run_dir, point, rep):
        adapter = REGISTRY[point.benchmark]
        rep_dir = run_dir / "raw" / point.point_id / f"rep{rep}"
        try:
            ports = self.ports.allocate(1 if point.benchmark == "tcp_pingpong" else point.rank_count)
            group = adapter.plan(point, self.ctx, ports, rep_dir)
        except (MissingExecutable, RuntimeError) as error:
            return self._record(run_id, run_dir, point, rep, "failure", f"could not plan the run: {error}")
        rep_dir.mkdir(parents=True, exist_ok=True)
        (rep_dir / "commands.json").write_text(json.dumps([{"rank": i.rank, "host": i.host_id, "command": display_command(self.config.hosts[i.host_id], i)} for i in group.invocations], indent=2))
        env_extra = point.options.get("env") or {}
        try:
            outcome = execute_group(group, rep_dir, point.timeout_s, self.config.defaults["startup_wait_s"], self.config, env_extra, run_id=run_id,
                                    verify_cleanup=self.config.defaults.get("verify_remote_cleanup", True))
        finally:
            self.ports.release(ports)  # the group's processes are gone (or were terminated and confirmed), so the ports can be reused later in the range
        if outcome.timed_out:
            return self._record(run_id, run_dir, point, rep, "timeout", outcome.reason, outcome=outcome, group=group)
        if outcome.failed_rank is not None:
            tail = self._stderr_tail(outcome, outcome.failed_rank)
            verification = None
            if re.search(r"VERIFY FAILED|correctness gate|failed verification|verification", tail, re.I):
                verification = {"status": "failed", "method": "native", "details": tail}
            return self._record(run_id, run_dir, point, rep, "failure", f"{outcome.reason}" + (f" ({tail})" if tail else ""), outcome=outcome, group=group, verification=verification)
        try:
            parsed = adapter.parse(point, group, outcome)
        except (ParseError, OSError, ValueError, KeyError) as error:
            return self._record(run_id, run_dir, point, rep, "invalid", f"output could not be parsed: {error}", outcome=outcome, group=group)
        return self._record(run_id, run_dir, point, rep, "success", None, outcome=outcome, parsed=parsed, verification=parsed.verification, group=group)

    def _limit_reason(self, limits, started, volume):
        if limits.get("max_campaign_s") and time.monotonic() - started > limits["max_campaign_s"]:
            return f"limit: the campaign duration limit ({limits['max_campaign_s']:g} s) was reached"
        if limits.get("max_total_bytes") and volume >= limits["max_total_bytes"]:
            return f"limit: the total volume limit ({limits['max_total_bytes'] / 2**30:.1f} GiB) was reached"
        return None

    def _volume_of(self, point, batch):
        """Bytes this point moved over the link, for the volume budget: the measured figure where a tool reports it (iperf3), else the repetitions run times the estimate."""
        total = 0
        for rec in batch:
            m = rec.get("measurements") or {}
            measured = ((m.get("extra") or {}).get("bytes_sent") or {}).get("value")
            total += measured or (point.estimated_bytes or 0)
        return total

    def _data_path_problem(self, point, batch, delta):
        """Interface counters must show the payload crossed the data interface; otherwise the benchmark may have used another path. Only checked where the volume is large
        enough to be unambiguous."""
        ok = [r for r in batch if r["status"]["state"] == "success"]
        if not ok or not delta:
            return None
        expected = 0
        for rec in ok:
            m = rec["measurements"]
            measured = ((m.get("extra") or {}).get("bytes_sent") or {}).get("value")
            expected += measured or (point.estimated_bytes or 0)
        if expected < 8 * 2**20:
            return None
        host0 = point.ranks[0]["host"]
        d = delta.get(host0)
        if d is None or "rx_bytes" not in d:
            return None
        seen = d.get("rx_bytes", 0) + d.get("tx_bytes", 0)
        if seen < 0.5 * expected:
            return f"data path not confirmed: the data interface on '{host0}' counted {seen} bytes but at least {int(0.5 * expected)} were expected"
        return None

    def run(self, points=None):
        points = points if points is not None else self.points()
        if self.config.physical and not self.allow_physical:
            raise PermissionError("this configuration uses physical hosts; physical runs are opt-in, pass --allow-physical")
        self.prepare(points)
        checks = check_config(self.config, points, self.ctx, self.allow_physical)
        checks += [Check("host consistency", "fail", p) for p in self.provenance_problems]
        checks += [Check("build commits", "warn", n) for n in self.provenance_notes]
        checks += [Check("data path", "fail", p) for p in self.route_problems]
        checks += [Check("management path", "ok", n) for n in self.management_notes]
        run_id = new_run_id(self.label, self.config.name)
        run_dir = self.output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "raw").mkdir()
        self.log(f"run {run_id}\n{format_checks(checks)}")
        started = time.monotonic()
        manifest = {"schema_version": schema.MANIFEST_VERSION, "run_id": run_id, "label": self.label, "suite_version": SUITE_VERSION, "started_utc": schema.utc_now(),
                    "configuration_name": self.config.name, "configuration_source": self.config.source and os.path.basename(self.config.source), "overlays_used": len(self.config.overlays),
                    "repository": self.git, "preflight": [{"name": c.name, "status": c.status, "detail": self.redactor.text(c.detail)} for c in checks],
                    "tbccl_environment_variables_stripped": stripped_env_names(), "topology": self.config.topology,
                    "hosts": {hid: {k: v for k, v in {"os": h.os, "arch": h.arch, "cpu": h.cpu, "gpu": h.gpu, "launcher": h.launcher.get("kind"), "interface": h.interface,
                                                      "address": h.address if h.is_loopback else "<redacted>"}.items()} for hid, h in self.config.hosts.items()},
                    "ranks": [{k: v for k, v in r.items() if k != "address"} for r in self.config.ranks],
                    "health": {"monitored": bool(self.monitor), "snapshots": [], "stop_reasons": [], "baseline_signals": {}, "baseline_warnings": []},
                    "host_provenance": self.ctx.host_provenance, "cleanup_unconfirmed": [],
                    "cleanup": "only processes this run started were signalled (by process group and ownership token); supervised remote ranks are bounded by a lease and a host watchdog; "
                               "cleanup is reported per rank as confirmed or not confirmed, never assumed"}
        if failed(checks):
            manifest.update({"status": "failure", "reason": "preflight failed", "counts": {}, "finished_utc": schema.utc_now()})
            (run_dir / "manifest.json").write_text(json.dumps(self.redactor.obj(manifest), indent=2))
            return manifest, []
        records, counts = [], {}
        aborted = None
        baseline = last = None
        volume = 0
        limits = self.config.limits
        hcfg = self.config.defaults.get("health") or {}
        results_path = run_dir / "results.jsonl"
        previous_handler = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            if self.monitor:
                baseline = self.monitor.snapshot()
                (run_dir / "health").mkdir()
                problems, warnings = self.monitor.baseline_problems(baseline)
                manifest["health"]["baseline_signals"] = self.monitor.signals(baseline)
                manifest["health"]["baseline_warnings"] = [self.redactor.text(w) for w in warnings]
                if problems:
                    aborted = "baseline health not acceptable: " + "; ".join(problems)
                    manifest["health"]["stop_reasons"] = problems
                    self.log(f"STOP: {aborted}")
                (run_dir / "health" / "baseline.json").write_text(json.dumps(baseline, indent=2))
                manifest["health"]["snapshots"].append("health/baseline.json")
                last = baseline
            with open(results_path, "w") as sink:
                for index, point in enumerate(points):
                    batch = []
                    if point.skip_reason:
                        batch.append(self._record(run_id, run_dir, point, 0, point.skip_state, point.skip_reason))
                    elif aborted:
                        batch.append(self._record(run_id, run_dir, point, 0, "skipped", f"run stopped earlier: {aborted}"))
                    else:
                        gate = self._limit_reason(limits, started, volume)
                        pre = None
                        if not gate and self.monitor and hcfg.get("snapshot_each_point", True):
                            pre = self.monitor.snapshot()
                            name = f"health/pre-{index:04d}.json"
                            (run_dir / name).write_text(json.dumps(pre, indent=2))
                            manifest["health"]["snapshots"].append(name)
                            failures, _ = self.monitor.baseline_problems(pre)
                            drift, _ = self.monitor.compare(last, pre)
                            if failures or drift:
                                gate = "pre-point health gate: " + "; ".join(failures + drift)
                                manifest["health"]["stop_reasons"] = failures + drift
                        if gate:
                            aborted = gate
                            self.log(f"STOP before running {point.point_id}: {gate}")
                            batch.append(self._record(run_id, run_dir, point, 0, "skipped", f"stopped before running: {gate}"))
                        else:
                            self.log(f"[{index + 1}/{len(points)}] {point.point_id} x{point.repetitions}" + (f"  (~{point.estimated_bytes / 2**20:.0f} MiB per repetition)" if point.estimated_bytes else ""))
                            for rep in range(point.repetitions):
                                record = self._repetition(run_id, run_dir, point, rep)
                                batch.append(record)
                                self.log(f"    rep {rep}: {record['status']['state']}" + (f" ({record['status']['reason']})" if record["status"]["reason"] else ""))
                                if record["status"]["state"] in ("timeout",) or (record["status"]["state"] == "failure" and self.config.physical):
                                    break
                            volume += self._volume_of(point, batch)
                            if self.monitor:
                                after = self.monitor.snapshot()
                                name = f"health/after-{index:04d}.json"
                                (run_dir / name).write_text(json.dumps(after, indent=2))
                                manifest["health"]["snapshots"].append(name)
                                reasons, _ = self.monitor.compare(pre or last, after)
                                cumulative, _ = self.monitor.compare(baseline, after)
                                reasons = list(dict.fromkeys(reasons + cumulative))
                                delta = {h.id: health.link_delta((pre or last).get(h.id) or {}, after.get(h.id) or {}) for h in self.monitor.hosts if "error" not in (after.get(h.id) or {"error": 1}) and "error" not in ((pre or last).get(h.id) or {"error": 1})}
                                for rec in batch:
                                    rec["artifacts"]["link_delta"] = delta
                                path_problem = self._data_path_problem(point, batch, delta)
                                if path_problem and not reasons:
                                    reasons = [path_problem]
                                last = after
                                if reasons:
                                    aborted = "link health changed: " + "; ".join(reasons)
                                    manifest["health"]["stop_reasons"] = reasons
                                    for i, rec in enumerate(batch):
                                        batch[i] = schema.new_record(run_id=run_id, record_id=rec["record_id"], benchmark=rec["benchmark"], environment=rec["environment"],
                                                                     configuration=rec["configuration"], status="invalid", reason=aborted, exit_codes=rec["status"]["exit_codes"],
                                                                     verification=rec["verification"], artifacts=rec["artifacts"])
                                    self.log(f"STOP: {aborted}")
                    for record in batch:
                        manifest["cleanup_unconfirmed"] += [{"record": record["record_id"], "rank": r["rank"], "host_id": r["host_id"]} for r in record["artifacts"].get("ranks", [])
                                                           if r.get("supervised") and r.get("cleanup_confirmed") is not True]
                        sink.write(json.dumps(record, sort_keys=True) + "\n")
                        sink.flush()
                        counts[record["status"]["state"]] = counts.get(record["status"]["state"], 0) + 1
                    records.extend(batch)
                    policy = self.config.defaults.get("failure_policy") or {}
                    bad = sum(counts.get(k, 0) for k in ("failure", "timeout", "invalid"))
                    if not aborted and bad and (policy.get("on_failure") == "abort_run" or (policy.get("max_failures") is not None and bad > policy["max_failures"])):
                        aborted = f"failure policy: {bad} failed measurement(s)"
                        self.log(f"STOP: {aborted}")
        except KeyboardInterrupt:
            aborted = "interrupted"
            manifest["interrupted"] = True
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
        ok = counts.get("success", 0)
        total = sum(counts.values())
        manifest.update({"counts": counts, "finished_utc": schema.utc_now(), "duration_s": round(time.monotonic() - started, 3), "results": "results.jsonl",
                         "status": "aborted" if aborted else ("failure" if not ok else ("partial" if any(counts.get(k) for k in ("failure", "timeout", "invalid")) else "success")),
                         "reason": aborted})
        (run_dir / "manifest.json").write_text(json.dumps(self.redactor.obj(manifest), indent=2))
        return manifest, records
