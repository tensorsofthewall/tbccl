"""Failure injection for the lifecycle validation tier (never used in the performance campaign).

    python fault.py --api HOST:PORT --title VLLM::Worker_PP1 [--delay-s 3] [--wait-s 600] [--signal KILL]

Waits until the API reports a running request, then waits ``--delay-s`` and sends one signal to exactly one process whose title is exactly ``--title`` on this host. If zero or
several processes carry that title, nothing is signalled and the helper fails. It prints one ``TBCCL_VLLM_FAULT`` line with the pid, the signal and the monotonic times, and
exits 0 after the injection. No pattern kill is ever used.
"""
import argparse
import http.client
import json
import os
import signal
import sys
import time

import_error = None
try:
    import psutil
except ImportError as error:  # the vLLM environments always have psutil; the offline tests may not
    psutil, import_error = None, error


def running_requests(host, port):
    try:
        c = http.client.HTTPConnection(host, port, timeout=3)
        c.request("GET", "/metrics")
        body = c.getresponse().read().decode()
        c.close()
    except OSError:
        return None
    total = 0.0
    for line in body.splitlines():
        if line.startswith("vllm:num_requests_running"):
            total += float(line.rsplit(" ", 1)[1])
    return total


def matching(title):
    pids = []
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            if p.info["name"] == title or (p.info["cmdline"] and p.info["cmdline"][0] == title):
                pids.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return pids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--delay-s", type=float, default=3.0)
    ap.add_argument("--wait-s", type=float, default=600.0)
    ap.add_argument("--signal", default="KILL", choices=["KILL", "TERM"])
    a = ap.parse_args()
    if psutil is None:
        print(f"psutil is required: {import_error}", file=sys.stderr)
        return 2
    host, port = a.api.rsplit(":", 1)
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.wait_s:
        r = running_requests(host, int(port))
        if r and r >= 1:
            break
        time.sleep(0.2)
    else:
        print("no running request was observed in time; nothing was signalled", file=sys.stderr)
        return 3
    seen = time.monotonic()
    time.sleep(a.delay_s)
    pids = matching(a.title)
    if len(pids) != 1:
        print(f"{len(pids)} processes titled {a.title!r}; nothing was signalled", file=sys.stderr)
        return 4
    sig = getattr(signal, "SIG" + a.signal)
    t_inject = time.monotonic()
    os.kill(pids[0], sig)
    print("TBCCL_VLLM_FAULT " + json.dumps({"pid": pids[0], "title": a.title, "signal": a.signal, "running_seen_after_s": seen - t0, "delay_s": a.delay_s,
                                             "injected_at_wall": time.time(), "t_mono": t_inject}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
