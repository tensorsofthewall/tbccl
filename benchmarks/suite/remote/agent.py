"""Self-contained host helper (standard library only). It is the only code the suite runs on a remote host, and it is shipped as `python3 -c` text, so nothing
has to be installed there and no daemon is left behind.

Subcommands:
  run          supervise one benchmark process: unique token, own process group, bounded lease, cancellation, child and grandchild cleanup, listener readiness
  check-clean  independently look for processes that still carry a token or belong to a process group
  probe        read-only host, interface, executable and build provenance facts
  health       a read-only link-health snapshot (reuses scripts/tb4_health_snapshot.py where present) plus GPU state

Protocol of `run`: JSON lines on stdout (events), single-letter commands on stdin: 'H' heartbeat, 'C' cancel. The supervised process's output travels as base64
events so it can be stored per rank. If the controller goes away (stdin EOF, a broken stdout, no heartbeat within the lease, the runtime cap) the whole process
group is terminated. A watchdog process forked first enforces the same bound even if the supervisor itself is killed. Processes are only ever signalled through
the exact process group the supervisor created or through the unique token in their environment; there is no name matching of any kind.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

PROTOCOL = 1
LOOPBACK_INTERFACE = "lo0" if sys.platform == "darwin" else "lo"  # what this platform calls its loopback interface
TAG_ENV = "BENCH_SUITE_OWNER"
EXIT_ABNORMAL = 70


# --------------------------------------------------------------------------- processes
def _linux():
    return os.path.isdir("/proc/self")


def _ps_table():
    """[(pid, pgid, stat)] from `ps` (macOS and anywhere /proc is missing)."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2]))
    return rows


def process_table():
    if not _linux():
        return _ps_table()
    rows = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat") as handle:
                rest = handle.read().rsplit(")", 1)[1].split()
            rows.append((int(name), int(rest[2]), rest[0]))
        except (OSError, IndexError, ValueError):
            continue
    return rows


def group_members(pgid):
    return [pid for pid, g, state in process_table() if g == pgid and not state.startswith("Z")]


def start_time(pid):
    """An opaque start-time string used to detect pid reuse."""
    if _linux():
        try:
            with open(f"/proc/{pid}/stat") as handle:
                return handle.read().rsplit(")", 1)[1].split()[19]
        except (OSError, IndexError):
            return None
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=5).stdout.strip()
        return out or None
    except (OSError, subprocess.TimeoutExpired):
        return None


def _tagged_ps(token):
    """Token scan where /proc does not exist (macOS): `ps -E` appends a process's environment to its command line, for the user's own processes, with no
    privileges. Matches only the exact `NAME=token` entry."""
    try:
        out = subprocess.run(["ps", "-axEww", "-o", "pid=,command="], capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    pattern = re.compile(r"(?:^|\s)" + re.escape(f"{TAG_ENV}={token}") + r"(?:\s|$)")
    found = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) != os.getpid() and pattern.search(parts[1]):
            found.append(int(parts[0]))
    return found if out else None


def tagged_members(token):
    """Processes whose environment carries the token: /proc on Linux, `ps -E` elsewhere. None when it cannot be checked at all."""
    if not token:
        return None
    if not _linux():
        return _tagged_ps(token)
    needle = f"{TAG_ENV}={token}".encode()
    found = []
    for pid, _, state in process_table():
        if state.startswith("Z") or pid == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/environ", "rb") as handle:
                if needle in handle.read().split(b"\0"):
                    found.append(pid)
        except OSError:
            continue
    return found


_TAG_SCAN_PROVEN = None


def tag_scan_proven():
    """True only if, on this host, a process started with a token in its environment is actually found by the scan. Without that proof escaped descendants
    (a grandchild that left the process group) cannot be detected, and no cleanup may be called confirmed."""
    global _TAG_SCAN_PROVEN
    if _TAG_SCAN_PROVEN is None:
        token = f"selftest-{os.getpid()}-{time.monotonic_ns()}"
        # The probe ends by itself within a few seconds, so it cannot outlive a supervisor that is killed while this self-test runs.
        probe = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(4)"], env={**os.environ, TAG_ENV: token}, start_new_session=True,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 2.5
            found = False
            while time.monotonic() < deadline and not found:
                found = probe.pid in (tagged_members(token) or [])
                time.sleep(0.02)
            _TAG_SCAN_PROVEN = found
        finally:
            probe.kill()
            probe.wait()
    return _TAG_SCAN_PROVEN


def owned_pids(pgid, token):
    pids = set(group_members(pgid))
    tagged = tagged_members(token)
    if tagged:
        pids.update(tagged)
    pids.discard(os.getpid())
    return pids


def terminate_owned(pgid, token, grace):
    """SIGTERM, wait, SIGKILL for exactly the owned processes (the group, plus anything carrying the token). Returns the pids still alive afterwards."""
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 2.0)):
        pids = owned_pids(pgid, token)
        if not pids:
            return []
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        for pid in pids - set(group_members(pgid)):
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline and owned_pids(pgid, token):
            time.sleep(0.02)
    return sorted(owned_pids(pgid, token))


# --------------------------------------------------------------------------- listener readiness
def parse_lsof_pids(text):
    """pids from `lsof -F p` output."""
    return {int(line[1:]) for line in text.splitlines() if line.startswith("p") and line[1:].isdigit()}


def _linux_listen_inodes(port):
    inodes = set()
    for table in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            lines = _slurp(table).splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) > 9 and fields[3] == "0A" and int(fields[1].rsplit(":", 1)[1], 16) == port:
                inodes.add(fields[9])
    return inodes


def _linux_pids_for_inodes(inodes):
    wanted = {f"socket:[{i}]" for i in inodes}
    pids = set()
    for pid, _, _ in process_table():
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                try:
                    if os.readlink(f"/proc/{pid}/fd/{fd}") in wanted:
                        pids.add(pid)
                        break
                except OSError:
                    continue
        except OSError:
            continue
    return pids


def tcp_listeners(port, lsof_runner=None):
    """(supported, pids, unresolved). `unresolved` is True when a listener exists whose owner could not be identified (another user's process)."""
    if _linux() and lsof_runner is None:
        inodes = _linux_listen_inodes(port)
        pids = _linux_pids_for_inodes(inodes) if inodes else set()
        return True, pids, bool(inodes) and not pids
    runner = lsof_runner or _run_lsof
    text = runner(port)
    if text is None:
        return False, set(), False
    return True, parse_lsof_pids(text), False


def _run_lsof(port):
    if not shutil.which("lsof"):
        return None
    try:
        done = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fp"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return done.stdout if done.returncode in (0, 1) else None


def wait_for_listener(port, pgid, token, alive, timeout, lsof_runner=None, poll_min=0.002, poll_max=0.05):
    """Wait until a process owned by this supervisor (its group or token) holds a LISTEN socket on `port`. Never connects to the port.

    Returns (outcome, detail): ready | exited | occupied | timeout | unsupported. 'occupied': something else listens on the port and ours does not;
    'exited': the supervised process ended before listening (detail says what else held the port, if anything)."""
    deadline = time.monotonic() + timeout
    interval = poll_min
    foreign = set()
    unresolved = False
    while True:
        supported, pids, unresolved = tcp_listeners(port, lsof_runner)
        if not supported:
            return "unsupported", "neither /proc nor lsof is available, so listener ownership cannot be established"
        ours = {p for p in pids if pgid_of(p) == pgid or p in (tagged_members(token) or [])}
        if ours:
            return "ready", sorted(ours)
        foreign = pids
        if not alive():
            return "exited", sorted(foreign)
        if time.monotonic() >= deadline:
            return ("occupied", sorted(foreign)) if (foreign or unresolved) else ("timeout", [])
        time.sleep(interval)
        interval = min(poll_max, interval * 1.6)


def pgid_of(pid):
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


# --------------------------------------------------------------------------- supervised run
class Emitter:
    def __init__(self, stream):
        self.stream, self.lock, self.broken = stream, threading.Lock(), False

    def __call__(self, **event):
        if self.broken:
            return
        with self.lock:
            try:
                self.stream.write(json.dumps(event, separators=(",", ":")) + "\n")
                self.stream.flush()
            except (OSError, ValueError):
                self.broken = True


def _pump(stream, kind, emit):
    try:
        while True:
            chunk = os.read(stream.fileno(), 16384)
            if not chunk:
                return
            emit(ev=kind, d=base64.b64encode(chunk).decode())
    except OSError:
        return


def _pdeathsig():
    if not sys.platform.startswith("linux"):
        return None
    try:
        import ctypes
        libc = ctypes.CDLL(None, use_errno=True)
        return lambda: libc.prctl(1, signal.SIGKILL, 0, 0, 0)
    except (OSError, AttributeError):
        return None


def _watchdog(read_fd, supervisor_pid, hard_deadline, grace):
    """Runs in its own session after the fork. Terminates the supervised group when the supervisor disappears or the hard deadline passes, whichever is first."""
    os.setsid()
    devnull = os.open(os.devnull, os.O_RDWR)
    for fd in (0, 1, 2):
        os.dup2(devnull, fd)
    try:
        info = json.loads(os.fdopen(read_fd).readline() or "null")
    except (OSError, ValueError):
        info = None
    if not info:
        os._exit(0)
    pgid, token = info["pgid"], info["token"]
    while True:
        gone = os.getppid() != supervisor_pid
        if gone or time.monotonic() >= hard_deadline:
            if group_members(pgid) or tagged_members(token):
                terminate_owned(pgid, token, grace)
            os._exit(0)
        time.sleep(0.5)


def poll_with_rusage(proc):
    """Like Popen.poll(), but keeps the child's resource usage (os.wait4) the first time it is reaped. Returns the exit code or None."""
    if proc.returncode is not None:
        return proc.returncode
    try:
        pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
    except ChildProcessError:
        return proc.returncode
    if pid:
        proc.returncode = os.waitstatus_to_exitcode(status)
        proc.rusage = usage
    return proc.returncode


def cpu_report(usage, started_monotonic):
    """CPU seconds of the supervised process (user + system) over its whole life, and the supervisor's own. Whole-process figures, not the timed loop."""
    import resource
    own = resource.getrusage(resource.RUSAGE_SELF)
    out = {"supervisor_cpu_s": round(own.ru_utime + own.ru_stime, 4), "wall_s": round(time.monotonic() - started_monotonic, 4)}
    if usage is not None:
        out.update(child_user_s=round(usage.ru_utime, 4), child_system_s=round(usage.ru_stime, 4), child_max_rss_kb=usage.ru_maxrss)
    return out


def run_supervised(args, argv, stdin=None, stdout=None):
    emit = Emitter(stdout or sys.stdout)
    token = args.token
    started = time.monotonic()
    hard = started + args.max_runtime + args.grace + 5.0
    read_fd, write_fd = os.pipe()
    supervisor_pid = os.getpid()
    if os.fork() == 0:
        os.close(write_fd)
        _watchdog(read_fd, supervisor_pid, hard, args.grace)
        os._exit(0)
    os.close(read_fd)
    env = dict(os.environ)
    env.update(dict(item.split("=", 1) for item in args.env))
    env[TAG_ENV] = token
    try:
        child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, start_new_session=True,
                                 preexec_fn=_pdeathsig())
    except OSError as error:
        emit(ev="exit", code=None, signal=None, reason=f"could not start: {error}")
        return EXIT_ABNORMAL
    pgid = os.getpgid(child.pid)
    os.write(write_fd, (json.dumps({"pgid": pgid, "token": token}) + "\n").encode())
    os.close(write_fd)
    emit(ev="started", pid=child.pid, pgid=pgid, start_time=start_time(child.pid), supervisor_pid=supervisor_pid, token=token, rank=args.rank, protocol=PROTOCOL,
         platform=sys.platform, tag_scan=tag_scan_proven())
    threads = [threading.Thread(target=_pump, args=(child.stdout, "out", emit), daemon=True), threading.Thread(target=_pump, args=(child.stderr, "err", emit), daemon=True)]
    state = {"hb": time.monotonic(), "cancel": False, "eof": False}

    def control():
        stream = stdin or sys.stdin
        try:
            for line in stream:
                line = line.strip()
                if line == "H":
                    state["hb"] = time.monotonic()
                elif line == "C":
                    state["cancel"] = True
        except (OSError, ValueError):
            pass
        state["eof"] = True

    threads.append(threading.Thread(target=control, daemon=True))
    if args.await_port:
        def watch():
            outcome, detail = wait_for_listener(args.await_port, pgid, token, lambda: child.returncode is None, args.await_timeout)
            emit(ev="listening" if outcome == "ready" else "listen_failed", port=args.await_port, outcome=outcome, detail=detail)
        threads.append(threading.Thread(target=watch, daemon=True))
    for t in threads:
        t.start()
    reason = None
    child.rusage = None
    while poll_with_rusage(child) is None:
        now = time.monotonic()
        if state["cancel"]:
            reason = "cancelled by the controller"
        elif state["eof"]:
            reason = "controller connection closed"
        elif emit.broken:
            reason = "controller output closed"
        elif now - state["hb"] > args.lease:
            reason = f"lease of {args.lease:g} s expired without a heartbeat"
        elif now - started > args.max_runtime:
            reason = f"runtime cap of {args.max_runtime:g} s reached"
        if reason:
            break
        time.sleep(0.02)
    leftovers = []
    if reason:
        leftovers = terminate_owned(pgid, token, args.grace)
    code = poll_with_rusage(child)
    if code is None:
        end = time.monotonic() + 2
        while code is None and time.monotonic() < end:
            time.sleep(0.02)
            code = poll_with_rusage(child)
    for t in threads[:2]:
        t.join(timeout=2)
    swept = sorted(owned_pids(pgid, token))
    if swept:
        leftovers = terminate_owned(pgid, token, args.grace)
    emit(ev="cleanup", confirmed=not owned_pids(pgid, token), swept=swept, leftover=leftovers, tag_scan=tag_scan_proven())
    emit(ev="exit", code=code if code is None or code >= 0 else None, signal=(-code if code is not None and code < 0 else None), reason=reason, cpu=cpu_report(child.rusage, started))
    return EXIT_ABNORMAL if reason else 0


# --------------------------------------------------------------------------- probes
def _slurp(path):
    with open(path, errors="replace") as handle:
        return handle.read()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(argv, timeout=10):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return done.stdout if done.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def host_info():
    system = platform.system()
    info = {"os": None, "arch": platform.machine(), "cpu_model": None, "gpu_model": None, "driver": None, "cuda_runtime": None, "metal": None}
    if system == "Linux":
        release = None
        try:
            for line in _slurp("/etc/os-release").splitlines():
                if line.startswith("PRETTY_NAME="):
                    release = line.split("=", 1)[1].strip('"')
        except OSError:
            pass
        info["os"] = f"{release or 'Linux'} (kernel {platform.release()})"
        try:
            for line in _slurp("/proc/cpuinfo").splitlines():
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
        smi = shutil.which("nvidia-smi")
        if smi:
            out = _run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"])
            if out:
                rows = [[f.strip() for f in line.split(",", 1)] for line in out.strip().splitlines() if "," in line]
                info["gpu_model"] = "; ".join(r[0] for r in rows)
                info["driver"] = "; ".join(sorted({r[1] for r in rows}))
            banner = _run([smi])
            m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", banner or "")
            info["cuda_runtime"] = f"driver supports CUDA {m.group(1)}" if m else None
    elif system == "Darwin":
        version = _run(["sw_vers", "-productVersion"])
        info["os"] = f"macOS {version.strip()}" if version else "macOS"
        info["cpu_model"] = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "").strip() or None
        out = _run(["system_profiler", "SPDisplaysDataType"], timeout=20)
        if out:
            m = re.search(r"Chipset Model:\s*(.+)", out)
            info["gpu_model"] = m.group(1).strip() if m else None
            m = re.search(r"Metal(?: Support| Family)?:\s*(.+)", out)
            info["metal"] = m.group(1).strip() if m else None
    else:
        info["os"] = f"{system} {platform.release()}"
    return info


def interface_info(name):
    if not name:
        return {"name": None, "present": None, "mtu": None, "operstate": None, "addresses": []}
    out = {"name": name, "present": False, "mtu": None, "operstate": None, "addresses": []}
    base = f"/sys/class/net/{name}"
    if os.path.isdir(base):
        out["present"] = True
        for key, rel in (("mtu", "mtu"), ("operstate", "operstate")):
            try:
                value = _slurp(f"{base}/{rel}").strip()
                out[key] = int(value) if key == "mtu" and value.isdigit() else value
            except OSError:
                pass
        text = _run(["ip", "-o", "addr", "show", "dev", name]) or ""
        out["addresses"] = re.findall(r"inet6?\s+(\S+)", text)
    elif platform.system() == "Darwin":
        text = _run(["ifconfig", name])
        if text:
            out["present"] = True
            m = re.search(r"mtu\s+(\d+)", text)
            out["mtu"] = int(m.group(1)) if m else None
            m = re.search(r"status:\s*(\w+)", text)
            out["operstate"] = m.group(1) if m else None
            out["addresses"] = re.findall(r"inet6?\s+(\S+)", text)
    return out


def cmake_cache_info(build_dir):
    info = {"build_dir_name": os.path.basename(os.path.abspath(build_dir)), "build_type": None, "generator": None, "compiler": None, "compiler_version": None, "cxx_flags": None,
            "cuda_enabled": None, "metal_enabled": None, "cuda_compiler_version": None, "home_directory": None}
    try:
        text = _slurp(os.path.join(build_dir, "CMakeCache.txt"))
    except OSError:
        return info

    def get(key):
        m = re.search(rf"^{key}:\w+=(.*)$", text, re.M)
        return m.group(1).strip() if m else None

    info.update(build_type=get("CMAKE_BUILD_TYPE") or "(none)", generator=get("CMAKE_GENERATOR"), compiler=get("CMAKE_CXX_COMPILER"), cxx_flags=get("CMAKE_CXX_FLAGS"),
                cuda_enabled=get("TBCCL_ENABLE_CUDA"), metal_enabled=get("TBCCL_ENABLE_METAL"), home_directory=get("CMAKE_HOME_DIRECTORY"))
    if info["compiler"]:
        out = _run([info["compiler"], "--version"])
        info["compiler_version"] = out.splitlines()[0].strip() if out else None
    nvcc = get("CMAKE_CUDA_COMPILER")
    if nvcc and nvcc != "NOTFOUND":
        out = _run([nvcc, "--version"])
        m = re.search(r"release ([\d.]+)", out or "")
        info["cuda_compiler_version"] = m.group(1) if m else None
    return info


def tbccl_info(path):
    out = _run([path])
    if not out:
        return {}
    info = {"version_text": out.splitlines()[0].strip()}
    for key, label in (("package_version", "Package version"), ("wire_protocol", "Wire protocol"), ("c_abi", "C ABI version")):
        m = re.search(rf"^{label}:\s*(.+)$", out, re.M)
        info[key] = m.group(1).strip() if m else None
    m = re.search(r"^Memory providers:\s*(.+)$", out, re.M)
    info["providers"] = sorted(p.strip() for p in m.group(1).split(",")) if m else None
    return info


STAMP_NAME = "build-provenance.json"


def read_stamp(build_dir):
    try:
        return json.loads(_slurp(os.path.join(build_dir, STAMP_NAME)))
    except (OSError, ValueError):
        return None


def probe(executables, interface=None, routes=()):
    out = {"agent_protocol": PROTOCOL, "python": platform.python_version(), "host": host_info(), "interface": interface_info(interface), "executables": {},
           "routes": {address: route_interface(address) for address in routes}}
    for name, path in executables.items():
        entry = {"path_name": os.path.basename(path), "found": os.path.isfile(path) and os.access(path, os.X_OK)}
        if entry["found"]:
            entry["sha256"] = sha256_file(path)
            build_dir = os.path.dirname(os.path.abspath(path))
            entry["cmake"] = cmake_cache_info(build_dir)
            tool = os.path.join(build_dir, "tbccl-info")
            entry["tbccl"] = tbccl_info(tool) if os.path.isfile(tool) else {}
            entry["stamp"] = read_stamp(build_dir)
        out["executables"][name] = entry
    return out


def system_state():
    """Cheap read-only observations taken between measurements: load, CPU count, TCP retransmissions."""
    out = {"cpu_count": os.cpu_count(), "loadavg": None, "tcp_retransmitted_segments": None}
    try:
        out["loadavg"] = list(os.getloadavg())
    except OSError:
        pass
    if _linux():
        try:
            lines = [l.split() for l in _slurp("/proc/net/snmp").splitlines() if l.startswith("Tcp:")]
            out["tcp_retransmitted_segments"] = int(dict(zip(lines[0], lines[1]))["RetransSegs"])
        except (OSError, IndexError, KeyError, ValueError):
            pass
    else:
        text = _run(["netstat", "-s", "-p", "tcp"], timeout=15) or ""
        m = re.search(r"(\d+) data packets? \([\d]+ bytes?\) retransmitted", text)
        out["tcp_retransmitted_segments"] = int(m.group(1)) if m else None
    return out


def route_interface(address):
    """The interface this host would use to reach `address` (read-only): ip route get on Linux, route get on macOS. None when it cannot be determined."""
    if _linux():
        text = _run(["ip", "-o", "route", "get", address]) or ""
        m = re.search(r"\bdev\s+(\S+)", text)
    else:
        text = _run(["route", "-n", "get", address]) or ""
        m = re.search(r"interface:\s*(\S+)", text)
    return m.group(1) if m else None


def competing_processes(names):
    """pids of processes that run one of these executables (matched by the executable's file name; observation only, nothing is ever signalled by name)."""
    wanted = set(names)
    found = []
    out = _run(["ps", "-axo", "pid=,comm="], timeout=15) or ""
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[0].isdigit() and os.path.basename(parts[1]) in wanted and int(parts[0]) != os.getpid():
            found.append(int(parts[0]))
    return found


def gpu_state():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return {"available": False, "reason": "nvidia-smi not found"}
    out = _run([smi, "--query-gpu=name,temperature.gpu,clocks_throttle_reasons.active,power.draw,utilization.gpu", "--format=csv,noheader,nounits"])
    if not out:
        return {"available": False, "reason": "nvidia-smi query failed"}
    gpus = []
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 5:
            gpus.append({"name": f[0], "temperature_c": float(f[1]) if re.fullmatch(r"[\d.]+", f[1]) else None, "throttle_reasons": f[2], "power_w": f[3], "utilization_percent": f[4]})
    return {"available": True, "gpus": gpus}


def describe_provider(python, bench_dir, timeout=30.0):
    """Run a benchmark provider's `describe` on this host with the provider's own interpreter. Returns the raw outcome; the controller validates it."""
    script = os.path.join(bench_dir, "provider.py")
    if not os.path.isfile(script):
        return {"error": f"{script} is not a file"}
    try:
        env = {k: v for k, v in os.environ.items() if not k.startswith("TBCCL_")}
        done = subprocess.run([python, script, "describe"], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, cwd=bench_dir, env=env)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": f"{type(error).__name__}: {error}"}
    return {"returncode": done.returncode, "stdout": done.stdout, "stderr_tail": done.stderr.strip()[-300:]}


def validate_provider(python, bench_dir, configs, timeout=30.0):
    """Stage each workload configuration as a file on THIS host and run the provider's `validate` on it with the provider's own interpreter. One raw outcome per configuration."""
    import tempfile
    script = os.path.join(bench_dir, "provider.py")
    if not os.path.isfile(script):
        return [{"error": f"{script} is not a file"} for _ in configs]
    env = {k: v for k, v in os.environ.items() if not k.startswith("TBCCL_")}
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i, config in enumerate(configs):
            path = os.path.join(tmp, f"config-{i}.json")
            with open(path, "w") as handle:
                json.dump(config, handle)
            try:
                done = subprocess.run([python, script, "validate", path], capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL, cwd=bench_dir, env=env)
                out.append({"returncode": done.returncode, "stdout": done.stdout, "stderr_tail": done.stderr.strip()[-300:]})
            except (OSError, subprocess.TimeoutExpired) as error:
                out.append({"error": f"{type(error).__name__}: {error}"})
    return out


def health(script, interface, busy=()):
    result = {"agent_protocol": PROTOCOL, "gpu": gpu_state(), "interface": interface_info(interface), "snapshot": None, "snapshot_error": None, "system": system_state(),
              "competing": competing_processes(busy) if busy else []}
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("tb4_health_snapshot", script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if platform.system() == "Linux" and interface:
            snap = {"timestamp_utc": None, "system": "Linux", "hostname": "", "uname": []}
            snap.update(module.linux_snapshot(interface=interface))
        else:
            snap = module.snapshot()
        snap.pop("hostname", None)
        snap.pop("uname", None)
        result["snapshot"] = snap
    except Exception as error:  # a snapshot that cannot be taken is reported, never assumed healthy
        result["snapshot_error"] = f"{type(error).__name__}: {error}"
    return result


# --------------------------------------------------------------------------- entry point
def main(argv=None):
    parser = argparse.ArgumentParser(prog="agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--token", required=True)
    r.add_argument("--rank", type=int, default=0)
    r.add_argument("--lease", type=float, default=30.0)
    r.add_argument("--max-runtime", type=float, default=600.0)
    r.add_argument("--grace", type=float, default=3.0)
    r.add_argument("--await-port", type=int)
    r.add_argument("--await-timeout", type=float, default=10.0)
    r.add_argument("--env", action="append", default=[])
    c = sub.add_parser("check-clean")
    c.add_argument("--token", required=True)
    c.add_argument("--pgid", type=int)
    p = sub.add_parser("probe")
    p.add_argument("--exe", action="append", default=[], help="NAME=PATH")
    p.add_argument("--interface")
    p.add_argument("--route", action="append", default=[], help="an address whose route to report")
    d = sub.add_parser("describe-provider")
    d.add_argument("--python", required=True)
    d.add_argument("--dir", required=True)
    d.add_argument("--timeout", type=float, default=30.0)
    v = sub.add_parser("validate-provider")
    v.add_argument("--python", required=True)
    v.add_argument("--dir", required=True)
    v.add_argument("--configs-b64", required=True, help="base64 JSON list of workload configurations to stage on this host and validate")
    v.add_argument("--timeout", type=float, default=30.0)
    h = sub.add_parser("health")
    h.add_argument("--script", required=True)
    h.add_argument("--interface")
    h.add_argument("--busy", action="append", default=[], help="an executable name that must not be running")
    argv = list(sys.argv[1:] if argv is None else argv)
    command = []
    if "--" in argv:
        i = argv.index("--")
        argv, command = argv[:i], argv[i + 1:]
    args = parser.parse_args(argv)
    if args.cmd == "run":
        if not command:
            parser.error("run needs a command after --")
        return run_supervised(args, command)
    if args.cmd == "check-clean":
        members = sorted(group_members(args.pgid)) if args.pgid else []
        tagged = tagged_members(args.token)
        print(json.dumps({"survivors_in_group": members, "survivors_tagged": tagged, "tag_scan": tag_scan_proven() and tagged is not None, "group_checked": args.pgid is not None}))
        return 0
    if args.cmd == "probe":
        print(json.dumps(probe(dict(e.split("=", 1) for e in args.exe), args.interface, args.route)))
        return 0
    if args.cmd == "validate-provider":
        import base64
        print(json.dumps(validate_provider(args.python, args.dir, json.loads(base64.b64decode(args.configs_b64)), args.timeout)))
        return 0
    if args.cmd == "describe-provider":
        print(json.dumps(describe_provider(args.python, args.dir, args.timeout)))
        return 0
    print(json.dumps(health(args.script, args.interface, args.busy)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
