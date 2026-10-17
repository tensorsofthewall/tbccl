"""Passive per-host sampler and engine-liveness supervisor for the vLLM benchmarks.

    python sampler.py --role NAME --interval 1.0 --link-interface IFACE --expect-addresses A,B [--gpu] [--thermal]

Prints one ``TBCCL_VLLM_SAMPLE <json>`` line per interval until it is terminated: CPU and memory of the vLLM process tree, the data interface's counters, the TCP connection
census of the vLLM processes (classified as link, loopback or other), and, where available, GPU utilisation, memory, temperature and throttle reasons (nvidia-smi) and the
CPU's thermal zones and throttle counter. Everything is read-only; nothing is configured, and no temperature ever stops a run.

Liveness: once an engine process (``VLLM::`` title or ``vllm serve``) has been seen, the sampler exits with status 7 if the whole set of engine processes disappears or if a
process it saw as a worker or engine core disappears while the others live. The suite's runner treats a nonzero exit of any member as a failure of the whole group, so a dead
worker on a headless node cannot leave the controller waiting: the leader itself would never notice (an upstream vLLM limitation).
"""
import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time

PREFIX = "TBCCL_VLLM_SAMPLE"
EXIT_ENGINE_LOST = 7


def emit(obj):
    print(f"{PREFIX} {json.dumps(obj)}", flush=True)


def norm_ip(ip):
    """Dual-stack sockets report IPv4 peers as ::ffff:a.b.c.d (macOS: ::a.b.c.d); compare them as plain IPv4 addresses."""
    if ip.startswith("::ffff:"):
        return ip[7:]
    if ip.startswith("::") and ip[2:].count(".") == 3:
        return ip[2:]
    return ip


def is_engine(cmdline, name):
    """A vLLM engine process: a Python (or the ``vllm`` launcher) running ``serve``, or a process titled VLLM::*. A launcher such as ssh that merely carries the command line is not one."""
    if "VLLM::" in name or (cmdline and "VLLM::" in cmdline[0]):
        return True
    if not cmdline:
        return False
    exe = os.path.basename(cmdline[0])
    if not (exe.startswith("python") or exe == "vllm"):
        return False
    joined = " ".join(cmdline)
    return " serve " in joined and "vllm" in joined


def engine_processes(psutil):
    """The vLLM process tree of this host (read-only census, never signalled)."""
    found = {}
    for p in psutil.process_iter(["pid", "cmdline", "name"]):
        try:
            if is_engine(p.info["cmdline"] or [], p.info["name"] or ""):
                found[p.pid] = p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def role_of(p):
    try:
        text = " ".join(p.cmdline()) + " " + p.name()
    except Exception:
        return "unknown"
    for tag in ("EngineCore", "Worker", "APIServer"):
        if "VLLM::" + tag in text or tag in text:
            return tag
    return "vllm-serve" if "vllm" in text else "other"


def classify(psutil, procs, expect):
    census = {"link": 0, "loopback": 0, "other": 0, "listen": 0}  # link: between the two different link addresses; loopback: loopback or an address talking to itself
    pairs = set()
    for p in procs.values():
        try:
            conns = p.net_connections(kind="tcp") if hasattr(p, "net_connections") else p.connections(kind="tcp")
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        for c in conns:
            if c.status == psutil.CONN_LISTEN:
                census["listen"] += 1
                continue
            if c.status != psutil.CONN_ESTABLISHED or not c.raddr:
                continue
            r, l = norm_ip(c.raddr.ip), norm_ip(c.laddr.ip)
            if r.startswith("127.") or r == "::1" or (r == l and r in expect):
                census["loopback"] += 1
            elif r in expect and l in expect:
                census["link"] += 1
                pairs.add((l, r, role_of(p)))
            else:
                census["other"] += 1
                pairs.add((l, r, role_of(p) + ":other"))
    return census, sorted(pairs)


def lost_engines(seen, alive):
    """The engine processes that were seen and are gone, when that means the engine is broken: everything disappeared, or a worker / engine core disappeared while others live.
    ``seen`` is {pid: role}, ``alive`` the set of pids present now. None while all is well (the API process alone going away after the others is an orderly stop)."""
    lost = {pid: r for pid, r in seen.items() if pid not in alive and r in ("EngineCore", "Worker", "APIServer", "vllm-serve")}
    if lost and (not alive or any(r in ("EngineCore", "Worker") for r in lost.values())):
        return lost
    return None


def read_int(path):
    try:
        with open(path) as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def thermal_linux():
    zones = {}
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            with open(z + "/type") as fh:
                kind = fh.read().strip()
            with open(z + "/temp") as fh:
                zones[kind] = int(fh.read()) / 1000.0
        except (OSError, ValueError):
            continue
    freqs = [read_int(f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_cur_freq") for i in range(os.cpu_count() or 0)]
    freqs = [f for f in freqs if f]
    return {"zones_c": zones, "package_throttle_count": read_int("/sys/devices/system/cpu/cpu0/thermal_throttle/package_throttle_count"),
            "core_throttle_count": read_int("/sys/devices/system/cpu/cpu0/thermal_throttle/core_throttle_count"),
            "cpu_freq_khz_mean": (sum(freqs) / len(freqs)) if freqs else None, "cpu_freq_khz_max": max(freqs) if freqs else None, "cpu_freq_khz_min": min(freqs) if freqs else None}


GPU_FIELDS = "temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total,clocks.sm,clocks.mem,power.draw,clocks_throttle_reasons.active"


def gpu_sample():
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    if not out:
        return None
    v = [x.strip() for x in out[0].split(",")]
    keys = ("temperature_c", "utilization_gpu_percent", "utilization_memory_percent", "memory_used_mib", "memory_total_mib", "clock_sm_mhz", "clock_mem_mhz", "power_w", "throttle_reasons")
    return dict(zip(keys, v))


def gpu_processes():
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return None
    res = {}
    for line in out:
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            res[int(parts[0])] = float(parts[1]) if parts[1].replace(".", "").isdigit() else None
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--link-interface", default=None)
    ap.add_argument("--expect-addresses", default="")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--thermal", action="store_true")
    ap.add_argument("--liveness", action="store_true", help="exit with status 7 when an engine process that was seen disappears")
    a = ap.parse_args()
    import psutil
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    expect = set(x for x in a.expect_addresses.split(",") if x)
    seen = {}  # pid -> role, for the liveness decision
    prev_net = None
    started = time.monotonic()
    emit({"event": "start", "role": a.role, "interval_s": a.interval, "psutil": psutil.__version__, "platform": sys.platform, "cpu_count": psutil.cpu_count(), "pid": os.getpid()})
    psutil.cpu_percent(None)
    while not stop["flag"]:
        t0 = time.monotonic()
        procs = engine_processes(psutil)
        roles = {pid: role_of(p) for pid, p in procs.items()}
        seen.update(roles)
        sample = {"role": a.role, "t_mono_s": round(t0 - started, 3), "mono_s": round(t0, 3), "wall_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cpu_percent": psutil.cpu_percent(None),
                  "load1": os.getloadavg()[0], "mem_used_percent": psutil.virtual_memory().percent, "mem_available_mib": psutil.virtual_memory().available // 2**20}
        sample["processes"] = []
        for pid, p in procs.items():
            try:
                with p.oneshot():
                    sample["processes"].append({"pid": pid, "role": roles[pid], "cpu_percent": p.cpu_percent(None), "rss_mib": p.memory_info().rss // 2**20, "threads": p.num_threads(),
                                                "fds": p.num_fds() if hasattr(p, "num_fds") else None})
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if a.link_interface:
            nets = psutil.net_io_counters(pernic=True).get(a.link_interface)
            if nets:
                cur = {"rx_bytes": nets.bytes_recv, "tx_bytes": nets.bytes_sent, "rx_errors": nets.errin, "tx_errors": nets.errout, "rx_dropped": nets.dropin, "tx_dropped": nets.dropout}
                sample["link"] = cur
        census, pairs = classify(psutil, procs, expect)
        sample["connections"] = census
        sample["connection_pairs"] = [list(x) for x in pairs]
        if a.gpu:
            sample["gpu"] = gpu_sample()
            sample["gpu_processes_mib"] = gpu_processes()
        if a.thermal and sys.platform.startswith("linux"):
            sample["thermal"] = thermal_linux()
        emit(sample)
        if a.liveness and seen:
            alive = set(procs)
            lost = lost_engines(seen, alive)
            if lost:
                emit({"event": "engine_lost", "role": a.role, "mono_s": round(time.monotonic(), 3), "lost": {str(k): v for k, v in lost.items()}, "alive": sorted(alive)})
                return EXIT_ENGINE_LOST
        end = t0 + a.interval
        while not stop["flag"]:
            remaining = end - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(0.1, remaining))
    emit({"event": "stop", "role": a.role, "samples_duration_s": round(time.monotonic() - started, 3)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
