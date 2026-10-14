#!/usr/bin/env python3
"""Verify a published GitHub Release of a TBCCL repository from the outside. This file is kept identical in the four TBCCL release repositories.

    verify_public_release.py                       metadata only, for the version in compatibility.json
    verify_public_release.py --tag v0.2.0          a specific tag
    verify_public_release.py --local-dir DIR       check a directory of built artifacts (a release-artifacts run's output) the same way, before anything is published
    verify_public_release.py --download            also download every artifact, check SHA256SUMS and verify the build-provenance attestations (gh)

The metadata check needs no large download: it confirms that the release exists and is not a draft, that the tag resolves (to --commit when given), that
the expected artifacts (release_assets in compatibility.json) are attached, that each has an SPDX SBOM, that SHA256SUMS lists them, and that the digests
GitHub recorded for the assets agree with SHA256SUMS. A missing release or artifact is a RELEASE SURFACE FAILURE; a tag that resolves to another commit is an
INTEGRITY FAILURE; an unreachable GitHub is a WARNING. Nothing is created, changed or uploaded.
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maint_common as mc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_SUFFIXES = (".whl", ".tar.gz")


def parse_sums(text):
    sums = {}
    for line in text.splitlines():
        if line.strip():
            digest, _, name = line.partition("  ")
            sums[name.strip()] = digest.strip().lower()
    return sums


def expected_names(templates, version):
    return [t.format(version=version) for t in templates]


def check_assets(release, templates, version, rep):
    assets = {a["name"]: a for a in release.get("assets", [])}
    for name in expected_names(templates, version):
        if name not in assets:
            rep.add(mc.SURFACE, f"artifact {name} is not attached")
            continue
        if not assets[name].get("size"):
            rep.add(mc.SURFACE, f"artifact {name} is empty")
        if name + ".spdx.json" not in assets:
            rep.add(mc.SURFACE, f"SBOM {name}.spdx.json is not attached")
    if "SHA256SUMS" not in assets:
        rep.add(mc.SURFACE, "SHA256SUMS is not attached")
    extra = [n for n in assets if n.endswith(ARTIFACT_SUFFIXES) and n not in expected_names(templates, version)]
    for n in extra:
        rep.add(mc.WARNING, f"unexpected artifact {n} (not in release_assets)")
    return assets


def check_sums_against_digests(sums, assets, templates, version, rep):
    for name in expected_names(templates, version):
        if name not in sums:
            rep.add(mc.SURFACE, f"SHA256SUMS does not list {name}")
            continue
        digest = (assets.get(name) or {}).get("digest") or ""
        if digest.startswith("sha256:") and digest[7:].lower() != sums[name]:
            rep.add(mc.INTEGRITY, f"{name}: GitHub records {digest[7:19]}..., SHA256SUMS says {sums[name][:12]}...")


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "tbccl-maintenance"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def download_and_verify(repo, release, sums, templates, version, workdir, rep):
    assets = {a["name"]: a for a in release["assets"]}
    have_gh = shutil.which("gh") is not None
    if not have_gh:
        rep.add(mc.WARNING, "gh is not installed: attestations were not verified")
    for name in expected_names(templates, version):
        if name not in assets or name not in sums:
            continue
        path = Path(workdir) / name
        try:
            path.write_bytes(fetch(assets[name]["browser_download_url"]))
        except OSError as e:
            rep.add(mc.WARNING, f"download of {name} failed ({e})")
            continue
        if sha256_bytes(path.read_bytes()) != sums[name]:
            rep.add(mc.INTEGRITY, f"{name}: the downloaded file does not match SHA256SUMS")
            continue
        rep.ok(f"{name}: checksum verified")
        if have_gh:
            p = subprocess.run(["gh", "attestation", "verify", str(path), "--repo", repo], capture_output=True, text=True)
            if p.returncode == 0:
                rep.ok(f"{name}: attestation verified")
            else:
                rep.add(mc.INTEGRITY, f"{name}: attestation did not verify: {(p.stderr or p.stdout).strip().splitlines()[-1:]}")


def check_local_dir(directory, templates, version, rep):
    """The same checks on files on disk: names, SBOMs, SHA256SUMS and the checksum of every file."""
    d = Path(directory)
    files = {p.name: p for p in d.iterdir() if p.is_file()}
    assets = {n: {"name": n, "size": p.stat().st_size} for n, p in files.items()}
    check_assets({"assets": list(assets.values())}, templates, version, rep)
    if "SHA256SUMS" not in files:
        return
    sums = parse_sums(files["SHA256SUMS"].read_text())
    for name in expected_names(templates, version):
        if name not in sums:
            rep.add(mc.SURFACE, f"SHA256SUMS does not list {name}")
        elif name in files and sha256_bytes(files[name].read_bytes()) != sums[name]:
            rep.add(mc.INTEGRITY, f"{name}: the file does not match SHA256SUMS")
        elif name in files:
            rep.ok(f"{name}: checksum verified")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag")
    ap.add_argument("--repo")
    ap.add_argument("--commit", help="the commit the tag must resolve to")
    ap.add_argument("--metadata-only", action="store_true", help="the default; accepted for explicitness")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--dir", help="download directory (default: a temporary directory)")
    ap.add_argument("--local-dir")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    manifest = mc.load_json(Path(args.root) / "compatibility.json")
    pkg = manifest["package"]
    version = pkg.get("public_version") or pkg["development_version"]
    tag = args.tag or f"v{version}"
    version = tag[1:]
    repo = args.repo or f"tensorsofthewall/{manifest['project']}"
    templates = manifest.get("release_assets") or []
    rep = mc.Report(f"public release {repo} {tag}")
    if not templates:
        rep.add(mc.ERROR, "compatibility.json has no release_assets")
        return rep.finish()
    if args.local_dir:
        rep.title = f"built artifacts in {Path(args.local_dir).name} for {tag}"
        check_local_dir(args.local_dir, templates, version, rep)
        return rep.finish()
    try:
        release = mc.github_api(f"repos/{repo}/releases/tags/{tag}")
    except mc.Network as e:
        rep.add(mc.WARNING, f"GitHub API unreachable ({e})")
        return rep.finish()
    if release is None:
        rep.add(mc.SURFACE, f"no GitHub Release for {tag}")
        return rep.finish()
    rc = mc.split_version(version)
    if release.get("draft"):
        rep.add(mc.SURFACE, "the release is a draft")
    if rc and bool(release.get("prerelease")) != bool(rc[3]):
        rep.add(mc.SURFACE, f"prerelease flag is {release.get('prerelease')} for {tag}")
    try:
        got = mc.ls_remote_tag(repo, tag)
    except mc.Network as e:
        rep.add(mc.WARNING, f"tag lookup failed ({e})")
        got = "unknown"
    if got is None:
        rep.add(mc.INTEGRITY, f"tag {tag} does not exist")
    elif args.commit and got not in ("unknown", args.commit):
        rep.add(mc.INTEGRITY, f"tag {tag} resolves to {got}, expected {args.commit}")
    elif got != "unknown":
        rep.ok(f"tag {tag} -> {got[:12]}")
    assets = check_assets(release, templates, version, rep)
    sums = {}
    if "SHA256SUMS" in assets:
        try:
            sums = parse_sums(fetch(assets["SHA256SUMS"]["browser_download_url"]).decode())
        except OSError as e:
            rep.add(mc.WARNING, f"could not read SHA256SUMS ({e})")
        else:
            check_sums_against_digests(sums, assets, templates, version, rep)
    if args.download and sums:
        workdir = args.dir or tempfile.mkdtemp(prefix="tbccl-verify-")
        Path(workdir).mkdir(parents=True, exist_ok=True)
        download_and_verify(repo, release, sums, templates, version, workdir, rep)
        if not args.dir:
            shutil.rmtree(workdir, ignore_errors=True)
    return rep.finish()


if __name__ == "__main__":
    mc.main_guard(main)
