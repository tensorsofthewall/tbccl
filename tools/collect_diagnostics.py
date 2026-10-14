#!/usr/bin/env python3
"""Print a short, non-invasive diagnostics report for a TBCCL bug report. This file is kept identical in the four TBCCL release repositories.

    collect_diagnostics.py                  print to standard output
    collect_diagnostics.py -o report.txt    write to a file you choose
    collect_diagnostics.py --tbccl-info /path/to/tbccl-info

It collects: the operating system and CPU architecture, the Python version, the versions of the TBCCL packages and of the frameworks they work with, the output
of `tbccl-info` when it can be found, network interface names with their MTU and link state, and the NAMES (never the values) of TBCCL-related environment
variables. It does not read prompts, models, files in your home directory, tokens, credentials or the full environment, it does not use the network, and
it uploads nothing. The user name and home directory are replaced in the output. Read the report before you post it.
"""
import argparse
import getpass
import importlib.metadata as md
import os
import platform
import re
import shutil
import subprocess
import sys

PACKAGES = ("tbccl", "torch-tbccl", "vllm-tbccl", "exo-tbccl", "torch", "vllm", "vllm-metal", "mlx", "exo", "numpy")
ENV_PREFIXES = ("TBCCL_", "TORCH_TBCCL_", "VLLM_TBCCL_", "EXO_TBCCL_", "TORCH_DISTRIBUTED_", "MASTER_", "WORLD_SIZE", "RANK", "NCCL_", "GLOO_")
IFACE_HINT = re.compile(r"^(bridge|en|eth|thunderbolt|tb|lo)", re.I)


def sh(cmd, timeout=20):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"(unavailable: {type(e).__name__})"
    return (p.stdout or p.stderr).strip()


def scrub(text):
    home = os.path.expanduser("~")
    if len(home) > 1:
        text = text.replace(home, "<home>")
    user = getpass.getuser()
    return re.sub(rf"\b{re.escape(user)}\b", "<user>", text) if user else text


def package_versions():
    out = []
    for name in PACKAGES:
        try:
            out.append(f"{name} {md.version(name)}")
        except md.PackageNotFoundError:
            out.append(f"{name} (not installed)")
    return out


def interfaces():
    if sys.platform == "darwin":
        text = sh(["ifconfig", "-a"])
        rows, name, mtu, status = [], None, None, ""
        for line in text.splitlines():
            m = re.match(r"^(\S+): flags=\S+<([^>]*)> mtu (\d+)", line)
            if m:
                if name:
                    rows.append(f"{name} mtu {mtu} {status}".strip())
                name, mtu, status = m.group(1), m.group(3), "UP" if "UP" in m.group(2).split(",") else "DOWN"
            elif "status:" in line:
                status += " " + line.split("status:")[1].strip()
        if name:
            rows.append(f"{name} mtu {mtu} {status}".strip())
        return [r for r in rows if IFACE_HINT.match(r)]
    text = sh(["ip", "-o", "link", "show"])
    rows = []
    for line in text.splitlines():
        m = re.match(r"^\d+:\s+(\S+?)(?:@\S+)?:\s+<([^>]*)>\s+mtu\s+(\d+).*?state\s+(\S+)", line)
        if m and IFACE_HINT.match(m.group(1)):
            rows.append(f"{m.group(1)} mtu {m.group(3)} state {m.group(4)}")
    return rows or [text.splitlines()[0] if text else "(none)"]


def find_tbccl_info(explicit):
    if explicit:
        return explicit
    found = shutil.which("tbccl-info")
    if found:
        return found
    for base in sys.path:
        for rel in ("bin/tbccl-info", "tbccl/bin/tbccl-info", "vllm_tbccl/bin/tbccl-info", "torch_tbccl/bin/tbccl-info", "exo_tbccl/bin/tbccl-info"):
            cand = os.path.join(base, rel)
            if base and os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
    return None


def build_report(tbccl_info=None):
    lines = ["# TBCCL diagnostics (review before posting)", "", "## Platform",
             f"system: {platform.system()} {platform.release()}", f"machine: {platform.machine()}", f"python: {platform.python_version()} ({platform.python_implementation()})", "",
             "## Packages", *package_versions(), "", "## tbccl-info"]
    exe = find_tbccl_info(tbccl_info)
    lines.append(sh([exe]) if exe else "(tbccl-info not found; pass --tbccl-info PATH, or paste `tbccl-info` output by hand)")
    lines += ["", "## Network interfaces (name, MTU, state)", *interfaces(), "", "## Environment variable names set (values not collected)"]
    names = sorted(k for k in os.environ if k.startswith(ENV_PREFIXES))
    lines += names or ["(none)"]
    return scrub("\n".join(lines)) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output")
    ap.add_argument("--tbccl-info")
    args = ap.parse_args()
    report = build_report(args.tbccl_info)
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"wrote {args.output}")
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
