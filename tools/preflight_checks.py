#!/usr/bin/env python3
"""The two release-preflight checks that need GitHub. This file is kept identical in the four TBCCL release repositories.

    preflight_checks.py tag-free TAG      the intended tag must not exist yet (release tags are immutable and are never recreated)
    preflight_checks.py artifacts TAG     a successful release-artifacts run must exist for exactly this commit (GITHUB_SHA), and the artifacts it built must have the
                                          expected names, SBOMs and checksums for TAG; needs the gh CLI and GITHUB_REPOSITORY

Nothing is created or published; the run's artifacts are downloaded into a temporary directory and deleted.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maint_common as mc  # noqa: E402
import verify_public_release as vp  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def tag_free(tag, repo):
    rep = mc.Report(f"tag {tag} is free")
    try:
        got = mc.ls_remote_tag(repo, tag)
    except mc.Network as e:
        rep.add(mc.WARNING, f"GitHub unreachable ({e})")
        return rep.finish()
    if got:
        rep.add(mc.INTEGRITY, f"{tag} already exists at {got}; a published tag is never moved or recreated")
    else:
        rep.ok(f"{tag} does not exist yet")
    return rep.finish()


def artifacts(tag, repo, sha):
    rep = mc.Report(f"built artifacts for {tag} at {sha[:12]}")
    manifest = mc.load_json(ROOT / "compatibility.json")
    templates = manifest.get("release_assets") or []
    if not shutil.which("gh"):
        rep.add(mc.ERROR, "the gh CLI is required")
        return rep.finish()
    p = mc.run(["gh", "run", "list", "-R", repo, "--workflow", "release-artifacts.yml", "--commit", sha, "--status", "success", "--json", "databaseId,url", "-L", "1"])
    if p.returncode:
        rep.add(mc.ERROR, f"gh run list failed: {p.stderr.strip()[:200]}")
        return rep.finish()
    runs = json.loads(p.stdout or "[]")
    if not runs:
        rep.add(mc.SURFACE, f"no successful release-artifacts run for {sha}; run the release-artifacts workflow on this commit first")
        return rep.finish()
    rep.ok(f"release-artifacts run {runs[0]['databaseId']}: {runs[0]['url']}")
    with tempfile.TemporaryDirectory(prefix="preflight-") as tmp:
        d = mc.run(["gh", "run", "download", str(runs[0]["databaseId"]), "-R", repo, "-D", tmp])
        if d.returncode:
            rep.add(mc.WARNING, f"could not download the run's artifacts: {d.stderr.strip()[:200]}")
            return rep.finish()
        dirs = sorted({p.parent for p in Path(tmp).rglob("SHA256SUMS")})
        if not dirs:
            rep.add(mc.SURFACE, "the run produced no SHA256SUMS")
            return rep.finish()
        vp.check_local_dir(dirs[0], templates, tag[1:], rep)
        for p in sorted(dirs[0].glob("*.spdx.json")):
            if not p.stat().st_size:
                rep.add(mc.SURFACE, f"{p.name} is empty")
    return rep.finish()


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("tag-free", "artifacts"):
        print(__doc__)
        return mc.ERROR
    repo = os.environ.get("GITHUB_REPOSITORY") or f"tensorsofthewall/{mc.load_json(ROOT / 'compatibility.json')['project']}"
    if sys.argv[1] == "tag-free":
        return tag_free(sys.argv[2], repo)
    return artifacts(sys.argv[2], repo, os.environ.get("GITHUB_SHA") or mc.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).stdout.strip())


if __name__ == "__main__":
    mc.main_guard(main)
