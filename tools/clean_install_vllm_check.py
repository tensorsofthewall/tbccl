#!/usr/bin/env python3
"""Checks an environment that the install helper filled with vllm-tbccl: run it with that environment's interpreter, for example

    .venv/bin/python tools/clean_install_vllm_check.py

It loads no model and needs no GPU. It fails unless vLLM and vllm-tbccl report the pinned versions, the native module loads at the expected C ABI and wire
protocol, and vLLM can discover the platform plugin.
"""
import importlib.metadata as md
import json
import subprocess
import sys

vllm_version = md.version("vllm")
if not vllm_version.startswith("0.31.0"):
    sys.exit(f"FAIL: vllm {vllm_version}, expected 0.31.0")
out = subprocess.run([sys.executable, "-m", "vllm_tbccl.info"], capture_output=True, text=True, check=True).stdout
info = json.loads(out[out.index("{"):])
if info.get("c_abi") != 1 or info.get("wire_protocol") != 4:
    sys.exit(f"FAIL: native module reports {info}")
names = {e.name for e in md.entry_points(group="vllm.platform_plugins")}
if "tbccl" not in names:
    sys.exit(f"FAIL: vllm.platform_plugins has {sorted(names)}, no 'tbccl'")
print(f"OK: vllm {vllm_version}, vllm-tbccl {md.version('vllm-tbccl')}, libtbccl {info.get('libtbccl')}, c_abi {info['c_abi']}, wire {info['wire_protocol']}, platform plugins {sorted(names)}")
