#!/usr/bin/env python3
"""Validate release/ecosystem.json, the canonical statement of the current stable TBCCL ecosystem.

    check_ecosystem_manifest.py            offline: structure, versions, tags, ABI and wire protocol, pin formats, agreement with compatibility.json and the docs table
    check_ecosystem_manifest.py --online   additionally: every recorded tag exists and resolves to the recorded commit, and every stable GitHub Release exists

Offline checks never touch the network and are what pull-request CI runs. A tag that resolves to a different commit is an INTEGRITY FAILURE, never a warning; an
unreachable GitHub is a WARNING. The recorded commits are never updated by this script.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maint_common as mc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROJECTS = ("tbccl", "torch_tbccl", "vllm_tbccl", "exo_tbccl")
TABLE_BEGIN, TABLE_END = "<!-- ecosystem-table:begin -->", "<!-- ecosystem-table:end -->"


def render_table(eco):
    t, tt, v, e = (eco[p] for p in PROJECTS)
    rows = [
        "| Project | Current stable | Tag | Commit | Pins |",
        "|---|---|---|---|---|",
        f"| tbccl | {t['version']} | `{t['tag']}` | `{t['commit'][:12]}` | C ABI {eco['c_abi']}, wire protocol {eco['wire_protocol']} |",
        f"| torch-tbccl | {tt['version']} | `{tt['tag']}` | `{tt['commit'][:12]}` | PyTorch {tt['torch']}.x |",
        f"| vllm-tbccl | {v['version']} | `{v['tag']}` | `{v['commit'][:12]}` | vLLM {v['vllm']}; `{v['metal_repo']}@{v['metal_tag']}` (`{v['metal_sha'][:12]}`) |",
        f"| exo-tbccl | {e['version']} | `{e['tag']}` | `{e['commit'][:12]}` | `{e['exo_repo']}@{e['exo_tag']}` (`{e['exo_sha'][:12]}`) |",
    ]
    return "\n".join(rows)


def check_offline(eco, rep, root=ROOT):
    if eco.get("c_abi") != 1:
        rep.add(mc.INTEGRITY, f"c_abi is {eco.get('c_abi')!r}, expected 1")
    if eco.get("wire_protocol") != 4:
        rep.add(mc.INTEGRITY, f"wire_protocol is {eco.get('wire_protocol')!r}, expected 4")
    for name in PROJECTS:
        p = eco.get(name)
        if not isinstance(p, dict):
            rep.add(mc.INTEGRITY, f"{name}: section missing")
            continue
        parts = mc.split_version(str(p.get("version", "")))
        if not parts:
            rep.add(mc.INTEGRITY, f"{name}: version {p.get('version')!r} is not MAJOR.MINOR.PATCH")
        elif parts[3]:
            rep.add(mc.INTEGRITY, f"{name}: version {p['version']} carries a release-candidate suffix; the ecosystem lists stable releases only")
        if p.get("tag") != f"v{p.get('version')}":
            rep.add(mc.INTEGRITY, f"{name}: tag {p.get('tag')!r} does not match version {p.get('version')!r}")
        if not mc.FULL_SHA.match(str(p.get("commit", ""))):
            rep.add(mc.INTEGRITY, f"{name}: commit is not a full 40-character SHA")
        if not mc.REPO_NAME.match(str(p.get("repo", ""))):
            rep.add(mc.INTEGRITY, f"{name}: repo {p.get('repo')!r} is not owner/name")
    v, e = eco.get("vllm_tbccl", {}), eco.get("exo_tbccl", {})
    for who, repo_key, tag_key, sha_key, base_key in (
        ("vllm_tbccl", "metal_repo", "metal_tag", "metal_sha", "metal_upstream_base"),
        ("exo_tbccl", "exo_repo", "exo_tag", "exo_sha", "exo_upstream_base"),
    ):
        p = eco.get(who, {})
        for key in (repo_key, tag_key, sha_key, base_key):
            if not p.get(key):
                rep.add(mc.INTEGRITY, f"{who}: framework pin field {key} is missing")
        if p.get(repo_key) and not mc.REPO_NAME.match(p[repo_key]):
            rep.add(mc.INTEGRITY, f"{who}: {repo_key} {p[repo_key]!r} is not owner/name")
        for key in (sha_key, base_key):
            if p.get(key) and not mc.FULL_SHA.match(p[key]):
                rep.add(mc.INTEGRITY, f"{who}: {key} is not a full 40-character SHA")
        if p.get(tag_key) and re.search(r"(^|/)(main|master|HEAD)$", p[tag_key]):
            rep.add(mc.INTEGRITY, f"{who}: {tag_key} {p[tag_key]!r} is a moving ref; stable releases pin immutable tags")
    manifest = mc.load_json(root / "compatibility.json")
    pkg = manifest["package"]
    if pkg["public_version"] != eco["tbccl"]["version"] and pkg.get("released"):
        rep.add(mc.INTEGRITY, f"compatibility.json public_version {pkg['public_version']} differs from the ecosystem's tbccl {eco['tbccl']['version']}")
    if manifest["c_abi"]["version"] != eco.get("c_abi") or manifest["wire_protocol"]["version"] != eco.get("wire_protocol"):
        rep.add(mc.INTEGRITY, "compatibility.json ABI or wire protocol differs from the ecosystem file")
    page = root / "docs" / "project" / "ecosystem.md"
    if page.exists() and not any(code != mc.OK for code, _ in rep.items):
        text = page.read_text()
        if TABLE_BEGIN not in text or TABLE_END not in text:
            rep.add(mc.INTEGRITY, f"{page.name}: ecosystem table markers are missing")
        elif text.split(TABLE_BEGIN)[1].split(TABLE_END)[0].strip() != render_table(eco):
            rep.add(mc.INTEGRITY, f"{page.name}: the ecosystem table does not match release/ecosystem.json (run with --write-table)")
    if not any(code != mc.OK for code, _ in rep.items):
        rep.ok("ecosystem.json is consistent offline")


def check_online(eco, rep):
    pairs = [(eco[p]["repo"], eco[p]["tag"], eco[p]["commit"]) for p in PROJECTS]
    pairs.append((eco["vllm_tbccl"]["metal_repo"], eco["vllm_tbccl"]["metal_tag"], eco["vllm_tbccl"]["metal_sha"]))
    pairs.append((eco["exo_tbccl"]["exo_repo"], eco["exo_tbccl"]["exo_tag"], eco["exo_tbccl"]["exo_sha"]))
    for repo, tag, sha in pairs:
        try:
            got = mc.ls_remote_tag(repo, tag)
        except mc.Network as ex:
            rep.add(mc.WARNING, f"{repo}@{tag}: GitHub unreachable ({ex})")
            continue
        if got is None:
            rep.add(mc.INTEGRITY, f"{repo}@{tag}: the tag no longer exists")
        elif got != sha:
            rep.add(mc.INTEGRITY, f"{repo}@{tag}: resolves to {got}, recorded {sha}")
        else:
            rep.ok(f"{repo}@{tag} -> {sha[:12]}")
    for name in PROJECTS:
        p = eco[name]
        try:
            rel = mc.github_api(f"repos/{p['repo']}/releases/tags/{p['tag']}")
        except mc.Network as ex:
            rep.add(mc.WARNING, f"{p['repo']} {p['tag']}: GitHub API unreachable ({ex})")
            continue
        if rel is None:
            rep.add(mc.SURFACE, f"{p['repo']} {p['tag']}: GitHub Release does not exist")
        elif rel.get("draft") or rel.get("prerelease"):
            rep.add(mc.SURFACE, f"{p['repo']} {p['tag']}: release is a draft or a pre-release")
        else:
            rep.ok(f"{p['repo']} release {p['tag']} exists")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--file", default=str(ROOT / "release" / "ecosystem.json"))
    ap.add_argument("--write-table", action="store_true", help="rewrite the table in docs/project/ecosystem.md from the JSON (a maintainer action)")
    args = ap.parse_args()
    eco = mc.load_json(args.file)
    if args.write_table:
        page = ROOT / "docs" / "project" / "ecosystem.md"
        text = page.read_text()
        head, rest = text.split(TABLE_BEGIN)
        tail = rest.split(TABLE_END)[1]
        page.write_text(f"{head}{TABLE_BEGIN}\n{render_table(eco)}\n{TABLE_END}{tail}")
        return mc.OK
    rep = mc.Report("ecosystem manifest" + (" (online)" if args.online else " (offline)"))
    check_offline(eco, rep)
    if args.online:
        check_online(eco, rep)
    return rep.finish()


if __name__ == "__main__":
    mc.main_guard(main)
