"""What is installed and available here: native executables, their build metadata, host hardware and optional tools. Read-only; nothing is started or changed."""
from __future__ import annotations

import importlib.util
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

from .adapters import REGISTRY
from .adapters.network import Iperf3Baseline
from .remote import agent

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACKED_REPOSITORY = "tbccl"


def _run(argv, timeout=10):
    try:
        done = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return done.stdout if done.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def candidate_dirs(build_dirs, repo_root=REPO_ROOT):
    env = os.environ.get("TBCCL_BUILD_DIR")
    dirs = [Path(env).resolve()] if env else []
    for d in build_dirs:
        p = Path(d)
        dirs.append(p if p.is_absolute() else repo_root / p)
    return dirs


def find_executable(name, build_dirs, repo_root=REPO_ROOT):
    for d in candidate_dirs(build_dirs, repo_root):
        for rel in (name, f"bin/{name}"):
            path = d / rel
            if path.is_file() and os.access(path, os.X_OK):
                return path
    return None


sha256_file = agent.sha256_file  # one implementation, shared with the helper that runs on remote hosts


def git_info(repo_root=REPO_ROOT):
    commit = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    if commit is None:
        return {"commit": None, "dirty": None, "branch": None}
    status = _run(["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"])
    branch = _run(["git", "-C", str(repo_root), "rev-parse", "--abbrev-ref", "HEAD"])
    return {"commit": commit.strip(), "dirty": bool(status.strip()) if status is not None else None, "branch": branch.strip() if branch else None}


def build_info(exe: Path):
    """tbccl-info output and CMake build configuration of the build directory an executable came from (the helper's probes, in this shape)."""
    cmake = agent.cmake_cache_info(str(exe.parent))
    tool = exe.parent / "tbccl-info"
    tb = agent.tbccl_info(str(tool)) if tool.exists() else {}
    return {"build_dir": str(exe.parent), "build_type": cmake["build_type"], "package_version": tb.get("package_version"), "wire_protocol": tb.get("wire_protocol"), "c_abi": tb.get("c_abi"),
            "providers": tb.get("providers"), "version_text": tb.get("version_text"), "compiler": cmake["compiler_version"], "generator": cmake["generator"]}


host_info = agent.host_info            # hardware and OS of this machine
interface_info = agent.interface_info  # MTU, link state and addresses of a network interface, read-only


def python_modules():
    return {name: importlib.util.find_spec(name) is not None for name in ("yaml", "jsonschema", "matplotlib", "numpy")}


def inventory(build_dirs=("build-release", "build-bench", "build"), repo_root=REPO_ROOT):
    out = {"python": sys.version.split()[0], "platform": platform.platform(), "host": host_info(), "repository": git_info(repo_root), "optional_modules": python_modules(),
           "iperf3": Iperf3Baseline.detect(), "executables": {}, "build_dirs_searched": [str(d) for d in candidate_dirs(build_dirs, repo_root)]}
    seen = set()
    for adapter in REGISTRY.values():
        exe_name = adapter.caps.executable
        if exe_name in seen:
            continue
        seen.add(exe_name)
        owner = next(a for a in REGISTRY.values() if a.caps.executable == exe_name)
        path = find_executable(exe_name, build_dirs, repo_root) if owner.caps.executable_in_build else (Path(shutil.which(exe_name)) if shutil.which(exe_name) else None)
        entry = {"found": path is not None, "path": str(path) if path else None, "used_by": [a.caps.name for a in REGISTRY.values() if a.caps.executable == exe_name]}
        if path and owner.caps.executable_in_build:
            entry["build"] = build_info(path)
        out["executables"][exe_name] = entry
    health = repo_root / "scripts" / "tb4_health_snapshot.py"
    out["health_snapshot_script"] = str(health) if health.exists() else None
    return out


def format_inventory(inv: dict) -> str:
    lines = [f"python {inv['python']} on {inv['platform']}"]
    h = inv["host"]
    lines.append(f"host: {h['os']} | {h['arch']} | cpu: {h['cpu_model'] or 'unknown'} | gpu: {h['gpu_model'] or 'none detected'}" + (f" | driver {h['driver']}" if h["driver"] else ""))
    r = inv["repository"]
    lines.append(f"repository: {r['commit'] or 'unknown'}{' (uncommitted changes)' if r['dirty'] else ''} on {r['branch']}")
    lines.append("optional modules: " + ", ".join(f"{k}={'yes' if v else 'no'}" for k, v in inv["optional_modules"].items()))
    ip = inv["iperf3"]
    lines.append(f"iperf3: {ip['version'] if ip['available'] else 'not installed'} (optional baseline tool; the `iperf3` benchmark runs it when a configuration asks for it)")
    lines.append("executables:")
    for name, e in inv["executables"].items():
        if e["found"] and "build" not in e:
            lines.append(f"  {name:32} {e['path']}  [tool found on PATH]")
        elif e["found"]:
            b = e["build"]
            lines.append(f"  {name:32} {e['path']}  [{b.get('version_text') or 'version unknown'}; build type {b.get('build_type') or 'unknown'}; providers {','.join(b['providers'] or []) or 'unknown'}]")
        else:
            lines.append(f"  {name:32} not found (searched: {', '.join(inv['build_dirs_searched'])})")
    return "\n".join(lines)
