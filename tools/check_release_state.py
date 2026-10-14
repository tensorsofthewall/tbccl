#!/usr/bin/env python3
"""The canonical release-state check of a TBCCL repository. This file is kept identical in the four TBCCL release repositories.

    check_release_state.py                       offline consistency of the version, the compatibility manifest, the CHANGELOG and the framework pins
    check_release_state.py --tag v0.2.0          additionally: the tag name must agree with the package version and the release state
    check_release_state.py --require-released    additionally: the manifest must say the version is released (use on main after a release)
    check_release_state.py --tbccl-root DIR      adapters: the TBCCL checkout the artifact will be built against must have the expected C ABI and wire protocol, must
                                                 not be older than the minimum package, and must be a stable version when this adapter version is stable

It reads the manifest and calls the repository's own source-agreement checker and, where one exists, the framework-pin checker; it publishes and changes
nothing. A stable tag with a release-candidate package version, a release candidate marked released, a missing manifest and a missing framework pin all
fail here, so the release workflows refuse to build a wrong-version artifact.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maint_common as mc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCE_CHECKERS = ("tools/check_compatibility_manifest.py", "scripts/check_compatibility_manifest.py")
PIN_CHECKER = "tools/check_framework_pin.py"


def package_version(manifest):
    pkg = manifest["package"]
    return pkg.get("public_version") or pkg["development_version"]


def check_manifest(manifest, rep, tag=None, require_released=False):
    pkg = manifest.get("package", {})
    version = package_version(manifest) if pkg else ""
    parts = mc.split_version(version)
    if not parts:
        rep.add(mc.INTEGRITY, f"package version {version!r} is not MAJOR.MINOR.PATCH[rcN]")
        return None
    base = "{}.{}.{}".format(*parts[:3])
    is_rc = bool(parts[3])
    if pkg.get("release_candidate") is not is_rc:
        rep.add(mc.INTEGRITY, f"release_candidate is {pkg.get('release_candidate')!r} but the version {version} {'has' if is_rc else 'has no'} an rc suffix")
    released = pkg.get("released")
    if not isinstance(released, bool):
        rep.add(mc.INTEGRITY, "package.released must be a boolean")
    elif released and is_rc:
        rep.add(mc.INTEGRITY, "a release candidate cannot be marked released")
    if require_released and released is not True:
        rep.add(mc.INTEGRITY, f"{version} is not marked released in the manifest")
    if pkg.get("target_release") != base:
        rep.add(mc.INTEGRITY, f"target_release {pkg.get('target_release')!r} differs from the base version {base}")
    if tag is not None:
        if tag != f"v{version}":
            rep.add(mc.INTEGRITY, f"tag {tag} does not match the package version {version} (expected v{version})")
        stable_tag = mc.split_version(tag[1:] if tag.startswith("v") else tag)
        if stable_tag and not stable_tag[3] and is_rc:
            rep.add(mc.INTEGRITY, f"stable tag {tag} but the package version {version} is a release candidate")
        if stable_tag and stable_tag[3] and released:
            rep.add(mc.INTEGRITY, f"release-candidate tag {tag} but the manifest says released")
    abi = manifest.get("c_abi", {}).get("version") if manifest.get("project") == "tbccl" else None
    if manifest.get("project") == "tbccl":
        wire = manifest.get("wire_protocol", {}).get("version")
        if abi != 1 or wire != 4:
            rep.add(mc.INTEGRITY, f"TBCCL C ABI {abi!r} / wire protocol {wire!r}, expected 1 / 4")
    else:
        req = manifest.get("requires_tbccl") or manifest.get("requires", {}).get("tbccl")
        if not req:
            rep.add(mc.INTEGRITY, "the manifest does not record the TBCCL line it requires")
        else:
            if req.get("c_abi") != [1]:
                rep.add(mc.INTEGRITY, f"requires C ABI {req.get('c_abi')!r}, expected [1]")
            if "wire_protocol" in req and req["wire_protocol"] != [4]:
                rep.add(mc.INTEGRITY, f"requires wire protocol {req['wire_protocol']!r}, expected [4]")
            minimum = req.get("minimum_package")
            if minimum is not None and (not mc.split_version(minimum) or mc.split_version(minimum)[3]):
                rep.add(mc.INTEGRITY, f"minimum TBCCL package {minimum!r} must be a stable version")
    if released:
        templates = manifest.get("release_assets")
        if not templates or not all("{version}" in t for t in templates):
            rep.add(mc.INTEGRITY, "a released manifest must list release_assets (names with a {version} placeholder)")
    return version


def check_core(manifest, core_root, rep):
    """The TBCCL core that an adapter artifact is about to be built against."""
    core = mc.load_json(Path(core_root) / "compatibility.json")
    if core.get("project") != "tbccl":
        rep.add(mc.INTEGRITY, f"{core_root} is not a TBCCL checkout")
        return
    req = manifest.get("requires_tbccl") or manifest.get("requires", {}).get("tbccl") or {}
    version = package_version(core)
    parts, minimum = mc.split_version(version), mc.split_version(req.get("minimum_package", "0.0.0"))
    if core["c_abi"]["version"] not in req.get("c_abi", []):
        rep.add(mc.INTEGRITY, f"TBCCL core has C ABI {core['c_abi']['version']}, the adapter requires {req.get('c_abi')}")
    if "wire_protocol" in req and core["wire_protocol"]["version"] not in req["wire_protocol"]:
        rep.add(mc.INTEGRITY, f"TBCCL core has wire protocol {core['wire_protocol']['version']}, the adapter requires {req['wire_protocol']}")
    if parts and minimum and parts[:3] < minimum[:3]:
        rep.add(mc.INTEGRITY, f"TBCCL core {version} is older than the required minimum {req.get('minimum_package')}")
    if parts and parts[3] and manifest["package"].get("release_candidate") is False:
        rep.add(mc.INTEGRITY, f"a stable adapter release cannot be built against the TBCCL release candidate {version}")
    rep.ok(f"TBCCL core {version}: C ABI {core['c_abi']['version']}, wire protocol {core['wire_protocol']['version']}")


def check_changelog(root, version, released, rep):
    path = root / "CHANGELOG.md"
    if not path.exists():
        rep.add(mc.INTEGRITY, "CHANGELOG.md is missing")
        return
    headings = re.findall(r"^## \[?([^\]\n]+?)\]?(?:\s+-.*)?$", path.read_text(), re.M)
    if version not in headings:
        if released:
            rep.add(mc.INTEGRITY, f"CHANGELOG.md has no section for {version}")
        elif not any(h.lower() == "unreleased" for h in headings):
            rep.add(mc.INTEGRITY, f"CHANGELOG.md has neither a section for {version} nor an Unreleased section")
    elif released and headings[0] != version:
        rep.add(mc.INTEGRITY, f"the newest CHANGELOG.md section is {headings[0]}, expected {version}")


def run_script(root, rel, rep, label):
    p = subprocess.run([sys.executable, str(root / rel)], capture_output=True, text=True, cwd=root)
    out = (p.stdout + p.stderr).strip().splitlines()
    if p.returncode == 0:
        rep.ok(f"{label}: {out[-1] if out else 'passed'}")
    else:
        rep.add(mc.INTEGRITY if p.returncode == 1 else mc.ERROR, f"{label} failed: {' | '.join(out[-4:])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag")
    ap.add_argument("--require-released", action="store_true")
    ap.add_argument("--tbccl-root")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    rep = mc.Report("release state")
    if not (root / "compatibility.json").exists():
        rep.add(mc.INTEGRITY, "compatibility.json is missing")
        return rep.finish()
    manifest = mc.load_json(root / "compatibility.json")
    version = check_manifest(manifest, rep, args.tag, args.require_released)
    if version:
        check_changelog(root, version, manifest["package"].get("released") is True, rep)
        rep.ok(f"{manifest['project']} {version}: released={manifest['package'].get('released')}")
    if args.tbccl_root:
        check_core(manifest, args.tbccl_root, rep)
    for rel in SOURCE_CHECKERS:
        if (root / rel).exists():
            run_script(root, rel, rep, "source agreement")
            break
    else:
        rep.add(mc.INTEGRITY, "no source-agreement checker found")
    if (root / PIN_CHECKER).exists():
        run_script(root, PIN_CHECKER, rep, "framework pin (offline)")
    return rep.finish()


if __name__ == "__main__":
    mc.main_guard(main)
