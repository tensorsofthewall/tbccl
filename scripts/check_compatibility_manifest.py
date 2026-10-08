#!/usr/bin/env python3
"""Check that compatibility.json agrees with the values defined in the sources.

The manifest is the machine-readable statement of the compatibility surface that TBCCL owns (package version, C ABI version,
wire protocol version). The sources remain the single definition; this check fails when the two drift apart.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def first(pattern: str, path: str) -> str:
    m = re.search(pattern, (ROOT / path).read_text(), re.M | re.S)
    if not m:
        sys.exit(f"cannot find {pattern!r} in {path}")
    return m.group(1)


manifest = json.loads((ROOT / "compatibility.json").read_text())
actual = {
    "package.development_version": first(r"project\(.*?VERSION\s+(\d+\.\d+\.\d+)", "CMakeLists.txt"),
    "package.public_version": first(r'set\(TBCCL_PUBLIC_VERSION\s+"\$\{PROJECT_VERSION\}(rc\d+)"', "CMakeLists.txt"),
    "c_abi.version": int(first(r"#define\s+TBCCL_C_ABI_VERSION\s+(\d+)u", "include/tbccl/tbccl.h")),
    "wire_protocol.version": int(first(r"kWireProtocolVersion\s*=\s*(\d+)", "include/tbccl/rank_directory.hpp")),
}
declared = {
    "package.development_version": manifest["package"]["development_version"],
    "package.public_version": manifest["package"]["public_version"].removeprefix(manifest["package"]["development_version"]),
    "c_abi.version": manifest["c_abi"]["version"],
    "wire_protocol.version": manifest["wire_protocol"]["version"],
}
bad = [k for k in actual if actual[k] != declared[k]]
for k in bad:
    print(f"MISMATCH {k}: manifest {declared[k]!r}, sources {actual[k]!r}")
if manifest["wire_protocol"]["version"] not in manifest["wire_protocol"]["interoperable_with"]:
    bad.append("wire_protocol.interoperable_with")
    print("MISMATCH: a wire protocol version must interoperate with itself")
if bad:
    sys.exit(1)
print("compatibility.json agrees with the sources:", declared)
