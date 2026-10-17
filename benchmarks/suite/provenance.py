"""Build and host provenance for every host a run uses.

The repository HEAD is *not* assumed to be the commit a binary was built from: a binary is attributed to a commit only through a build stamp
(`python3 -m benchmarks.suite provenance stamp`, run on the host that built it, which records the commit, the executables' SHA-256 and checks the build was configured from that checkout with no build input modified or newer than the executables) whose hashes still match,
or through an explicit declaration in the host's configuration. Anything else is reported as `unverified`, and the result still records the executable hash,
compiler, build type and versions so two runs can be told apart.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import inventory, schema
from .remote import agent, session

STATUSES = ("stamped", "declared", "unverified")
# What a build is made from. benchmarks/suite (this tooling) is not an input of any executable.
BUILD_INPUTS = ["src", "include", "cmake", "CMakeLists.txt", "benchmarks", ":(exclude)benchmarks/suite", "tools/tbccl_info.cpp"]



def _git(repo_root, *args):
    import subprocess
    done = subprocess.run(["git", "-C", str(repo_root), *args], capture_output=True, text=True, timeout=60)
    return done.stdout if done.returncode == 0 else None


def build_input_state(repo_root, newest_allowed_mtime):
    """(dirty, newer_files): are the tracked build inputs modified, and are any newer than the executables? Both are needed to say a binary matches a commit."""
    dirty = _git(repo_root, "status", "--porcelain", "--untracked-files=no", "--", *BUILD_INPUTS)
    listing = _git(repo_root, "ls-files", "-z", "--", *BUILD_INPUTS) or ""
    newer = []
    for name in filter(None, listing.split("\0")):
        try:
            if os.path.getmtime(Path(repo_root) / name) > newest_allowed_mtime:
                newer.append(name)
        except OSError:
            continue
    return (bool(dirty.strip()) if dirty is not None else None), newer


def stamp_build(build_dir, repo_root, executables=None):
    """Write build-provenance.json next to the executables. Run it on the host that built them, right after building.

    The stamp is an attestation, not a proof: it records the commit of `repo_root`, the executables' hashes, whether the build inputs were modified or newer than
    the executables, and whether the build was configured from `repo_root` at all. A stamp made long after a build, or from another checkout, therefore says so
    and is not accepted as provenance when read back."""
    build_dir, repo_root = Path(build_dir).resolve(), Path(repo_root).resolve()
    git = inventory.git_info(repo_root)
    if git["commit"] is None:
        raise ValueError(f"{repo_root} is not a git checkout; cannot record a commit")
    names = list(executables or [])
    if not names:
        names = sorted({a.caps.executable for a in __import__("benchmarks.suite.adapters", fromlist=["REGISTRY"]).REGISTRY.values()} | {"tbccl-info"})
    present = [n for n in names if (build_dir / n).is_file()]
    exes = {n: agent.sha256_file(str(build_dir / n)) for n in present}
    if not exes:
        raise ValueError(f"no known executables found in {build_dir}")
    home = agent.cmake_cache_info(str(build_dir))["home_directory"]
    exe_mtime = min(os.path.getmtime(build_dir / n) for n in present)
    inputs_dirty, newer = build_input_state(repo_root, exe_mtime)
    stamp = {"schema": "tbccl-build-provenance/2", "created_utc": schema.utc_now(), "repository_commit": git["commit"], "repository_dirty": git["dirty"], "branch": git["branch"],
             "build_inputs_dirty": inputs_dirty, "build_inputs_newer_than_executables": newer[:10], "build_inputs_newer_count": len(newer),
             "cmake_source_dir_matches_repository": (Path(home).resolve() == repo_root) if home else None, "source_dir_name": repo_root.name, "executables": exes}
    path = build_dir / agent.STAMP_NAME
    path.write_text(json.dumps(stamp, indent=2) + "\n")
    return path, stamp


def stamp_problems(stamp):
    """Reasons a stamp does not attribute the binaries to its commit (empty when it does)."""
    problems = []
    if stamp.get("cmake_source_dir_matches_repository") is not True:
        problems.append("the build was not shown to be configured from the stamped checkout" if stamp.get("cmake_source_dir_matches_repository") is None
                        else "the build was configured from a different source directory than the stamped checkout")
    if stamp.get("build_inputs_dirty") is not False:
        problems.append("build inputs had uncommitted changes (or could not be checked) when stamped")
    if stamp.get("build_inputs_newer_count", 1):
        problems.append(f"{stamp.get('build_inputs_newer_count')} build input file(s) are newer than the executables, so they may predate the current sources")
    return problems


def assess(name, entry, declared=None, controller_head=None):
    """Provenance of one executable on one host from a probe entry. Never upgrades 'unverified' by guessing."""
    cmake = entry.get("cmake") or {}
    tb = entry.get("tbccl") or {}
    out = {"executable": name, "executable_sha256": entry.get("sha256"), "build_type": cmake.get("build_type"), "generator": cmake.get("generator"),
           "compiler": cmake.get("compiler_version") or cmake.get("compiler"), "cuda_enabled": cmake.get("cuda_enabled"), "metal_enabled": cmake.get("metal_enabled"),
           "cuda_compiler_version": cmake.get("cuda_compiler_version"), "tbccl_package_version": tb.get("package_version"), "c_abi": tb.get("c_abi"),
           "wire_protocol": tb.get("wire_protocol"), "providers": tb.get("providers"), "status": "unverified", "source": None, "build_commit": None, "build_dirty": None,
           "matches_controller_head": None, "reason": None}
    if not entry.get("found"):
        out["reason"] = "executable not found"
        return out
    stamp = entry.get("stamp")
    if stamp and (stamp.get("executables") or {}).get(name) == entry.get("sha256"):
        problems = stamp_problems(stamp)
        if problems:
            out["reason"] = "a stamp lists this executable's hash but does not attribute it to a commit: " + "; ".join(problems)
        else:
            out.update(status="stamped", source="build stamp", build_commit=stamp.get("repository_commit"), build_dirty=False,
                       reason="attested by a build stamp: hash matches, built from the stamped checkout, no build input modified or newer than the executable (an attestation, not a cryptographic proof)")
    elif stamp:
        out["reason"] = "a build stamp exists but does not match this executable (rebuilt after stamping?)"
    elif declared and declared.get("commit"):
        out.update(status="declared", source="host configuration", build_commit=declared["commit"], build_dirty=declared.get("dirty"), reason="declared by the operator in the configuration; not checked against the binary")
    else:
        out["reason"] = "no build stamp and nothing declared; run `provenance stamp` after building"
    if out["build_commit"] and controller_head:
        out["matches_controller_head"] = out["build_commit"] == controller_head
    return out


def collect(config, ctx, host_ids, executables, interface_of=None, controller_head=None, timeout=60, routes_of=None):
    """Probe each host once (locally in process, remotely through its launcher). Returns {host_id: {...}} with an `error` key where a host could not be probed."""
    result = {}
    for host_id in host_ids:
        host = config.hosts[host_id]
        paths = {}
        for name in executables:
            paths[name] = ctx.executable(name, host_id) if (not host.is_local or ctx.executable_or_none(name, host_id)) else None
        paths = {k: v for k, v in paths.items() if v}
        iface = (interface_of or {}).get(host_id) or host.interface
        if host.is_local and not host.launcher.get("supervised"):
            data, error = agent.probe(paths, iface, (routes_of or {}).get(host_id, ())), None
        else:
            args = ["probe"] + [a for name, p in paths.items() for a in ("--exe", f"{name}={p}")] + (["--interface", iface] if iface else []) \
                + [a for addr in (routes_of or {}).get(host_id, ()) for a in ("--route", addr)]
            data, error = session.run_agent_json(host, args, timeout=timeout, python=host.python)
        if data is None:
            result[host_id] = {"error": error, "executables": {}}
            continue
        entries = {name: assess(name, e, host.build, controller_head) for name, e in data.get("executables", {}).items()}
        result[host_id] = {"host": data.get("host"), "interface": data.get("interface"), "python": data.get("python"), "executables": entries, "routes": data.get("routes") or {}, "error": None}
    return result


def consistency(per_host, policy):
    """Problems between hosts of one run. A different TBCCL package, C ABI or wire protocol on two hosts is never acceptable when require_same_version is set."""
    problems = []
    keys = ("tbccl_package_version", "c_abi", "wire_protocol")
    seen = {}
    for host_id, info in per_host.items():
        for name, e in (info.get("executables") or {}).items():
            for k in keys:
                seen.setdefault((name, k), {})[host_id] = e.get(k)
    if policy.get("require_same_version", True):
        for (name, k), values in seen.items():
            if len({v for v in values.values()}) > 1:
                problems.append(f"{name}: {k} differs between hosts ({', '.join(f'{h}={v}' for h, v in sorted(values.items()))})")
    if policy.get("require_stamped", policy.get("require_verified", False)):
        for host_id, info in per_host.items():
            for name, e in (info.get("executables") or {}).items():
                if e["status"] != "stamped":
                    problems.append(f"{host_id}:{name}: build provenance is {e['status']} ({e['reason']})")
    for host_id, info in per_host.items():
        if info.get("error"):
            problems.append(f"{host_id}: host could not be probed ({info['error']})")
    return problems


def commit_notes(per_host):
    """Informational: build commits that differ between hosts (not an error: heterogeneous builds are normal)."""
    commits = {}
    for host_id, info in per_host.items():
        for name, e in (info.get("executables") or {}).items():
            commits.setdefault(name, {})[host_id] = e.get("build_commit")
    notes = []
    for name, per_host_commits in commits.items():
        if len(set(per_host_commits.values())) > 1:
            listing = ", ".join("{}={}".format(h, (c or "unknown")[:10]) for h, c in sorted(per_host_commits.items()))
            notes.append(f"{name}: build commits differ ({listing})")
    return notes


def route_problems(config, per_host, peer_addresses):
    """The route from each host to its peers' benchmark addresses must leave through that host's data interface: otherwise the benchmark would silently use another
    path (for example the LAN). `peer_addresses` is {host_id: [addresses of ranks on other hosts]}."""
    problems = []
    for host_id, info in per_host.items():
        host = config.hosts[host_id]
        if info.get("error") or not host.interface:
            continue
        for addr in peer_addresses.get(host_id, ()):
            via = (info.get("routes") or {}).get(addr)
            if via is None:
                problems.append(f"{host_id}: the route to {{peer address}} could not be determined, so the data path is unproven")
            elif via != host.interface:
                problems.append(f"{host_id}: traffic to the peer's benchmark address would leave through '{via}', not the configured data interface '{host.interface}'")
    return problems
