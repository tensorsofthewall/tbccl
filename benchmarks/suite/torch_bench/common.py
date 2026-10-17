"""Helpers shared by the PyTorch benchmark wrappers (run with the torch environment's interpreter; never imported by the suite itself).

The wrappers print one line per measurement, ``<PREFIX> {json}``, so the adapter never parses free-form console text. Every timestamp is one process's own
``time.perf_counter_ns``; nothing here compares clocks of different hosts.
"""
from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys
import time

now_ns = time.perf_counter_ns


def device_sync(dev):
    """Measurement synchronization: wait for all queued device work (kept apart from whatever synchronization the API does itself)."""
    import torch
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps":
        torch.mps.synchronize()


def make_device(name):
    import torch
    dev = torch.device(name)
    if dev.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("device cuda requested but torch.cuda.is_available() is False")
        torch.cuda.set_device(0)
        dev = torch.device("cuda", 0)
    elif dev.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("device mps requested but torch.backends.mps.is_available() is False")
    return dev


def environment(backend_name):
    import torch
    env = {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}", "torch": torch.__version__, "backend": backend_name,
           "cuda_available": bool(torch.cuda.is_available()), "mps_available": bool(torch.backends.mps.is_available())}
    if torch.cuda.is_available():
        env["cuda_device"] = torch.cuda.get_device_name(0)
    try:
        import torch_tbccl
        env["torch_tbccl"] = torch_tbccl.__version__
        info = torch_tbccl.info() if callable(getattr(torch_tbccl, "info", None)) else {}
        for key in ("libtbccl", "c_abi", "wire_protocol"):
            if isinstance(info, dict) and key in info:
                env[key] = str(info[key])
    except Exception as error:  # noqa: BLE001 - recorded, not hidden
        env["torch_tbccl_error"] = f"{type(error).__name__}: {error}"
    return env


def own_tcp_connections():
    """Established TCP connections of this process as [local, remote] strings. Used to prove which interface the traffic uses (a LAN fallback must not pass)."""
    pid = os.getpid()
    out = []
    try:
        if sys.platform == "darwin":
            text = subprocess.run(["lsof", "-nP", "-a", "-p", str(pid), "-iTCP", "-sTCP:ESTABLISHED", "-Fn"], capture_output=True, text=True, timeout=20).stdout
            for line in text.splitlines():
                if line.startswith("n") and "->" in line:
                    local, remote = line[1:].split("->", 1)
                    out.append([local, remote])
        else:
            text = subprocess.run(["ss", "-H", "-tnp", "state", "established"], capture_output=True, text=True, timeout=20).stdout
            for line in text.splitlines():
                if f"pid={pid}," in line:
                    cols = line.split()
                    out.append([cols[2], cols[3]])
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out


def emit(prefix, payload):
    sys.stdout.write(prefix + " " + json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def pattern(nelem, seed, dtype=None):
    import torch
    dtype = dtype or torch.float32
    return ((torch.arange(nelem, dtype=torch.int64) * 7 + seed * 13) % 101).to(dtype)


def bits_equal(a, b):
    return bool(a.cpu().contiguous().view(-1).view(__import__("torch").uint8).equal(b.cpu().contiguous().view(-1).view(__import__("torch").uint8)))


HEX = re.compile(r"^[0-9a-fA-F]+$")
