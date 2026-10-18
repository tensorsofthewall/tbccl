#!/usr/bin/env python3
"""Check the facts a README must keep true. Offline by default; nothing is downloaded or run.

    python3 tools/check_readme.py            # the structural and consistency checks
    python3 tools/check_readme.py --online   # additionally: every documentation and GitHub URL of this ecosystem answers

What is checked (and nothing about wording, heading order or length):

* the lockup is the approved `<picture>` with a dark and a light file that exist, and its `alt` is the project name;
* every relative link, image and `srcset` points at a file that exists;
* every release asset named in the README (a `releases/download/vX.Y.Z/...` URL or an archive or wheel file name) uses the version in `compatibility.json` and an
  asset name that its `release_assets` list allows, so a version bump cannot leave a stale wheel name behind;
* the current package version, C ABI and wire protocol of `compatibility.json` appear;
* every integration tag or fork tag the README names also appears in `compatibility.json` or in the compatibility page;
* the links that every README owes its reader exist (limitations, troubleshooting, compatibility, contributing, license) and the project's own entry points are named
  (`"tbccl"` for torch-tbccl, `VLLM_TBCCL_ENABLE` for vllm-tbccl, `MlxTbccl` for exo-tbccl, `TBCCL::tbccl_c` for tbccl);
* no internal planning wording, plan paths or session identifiers.

The same file is used by all four TBCCL repositories.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITES = ("tbccl", "torch-tbccl", "vllm-tbccl", "exo-tbccl")
OWN_URL = re.compile(r"https://(?:(?P<site>[a-z-]+)\.tensorsofthewall\.com/en/(?P<ver>latest|stable|v[\w.]+)/(?P<path>[^\s)\]>\"'`]*)|github\.com/tensorsofthewall/[^\s)\]>\"'`]+)")
REQUIRED_TOKENS = {
    "tbccl": ["TBCCL::tbccl_c", "tbccl-info"],
    "torch-tbccl": ['"tbccl"', "torch_tbccl", "--no-deps"],
    "vllm-tbccl": ["VLLM_TBCCL_ENABLE", "vllm_tbccl", "v0.31.0-tbccl1"],
    "exo-tbccl": ["MlxTbccl", "exo_tbccl", "tbccl-integration-v2"],
}
REQUIRED_LINKS = {
    "tbccl": ["docs/project/limitations.md", "docs/guides/troubleshooting.md", "docs/reference/compatibility-matrix.md", "CONTRIBUTING.md", "docs/development/code-map.md", "LICENSE"],
    "torch-tbccl": ["docs/project/limitations.md", "docs/guides/troubleshooting.md", "docs/reference/compatibility.md", "CONTRIBUTING.md", "docs/development/code-map.md", "LICENSE"],
    "vllm-tbccl": ["docs/project/limitations.md", "docs/guides/troubleshooting.md", "docs/reference/compatibility.md", "CONTRIBUTING.md", "docs/development/code-map.md", "LICENSE"],
    "exo-tbccl": ["docs/project/limitations.md", "docs/guides/discovery-troubleshooting.md", "docs/reference/compatibility.md", "CONTRIBUTING.md", "docs/development/code-map.md", "LICENSE"],
}
INTERNAL = re.compile(r"(?i)\bphase [a-f]\b|\.local/plans|claude\.ai|session_[0-9A-Za-z]{8}|Claude-Session")


def load(root):
    compat = json.loads((root / "compatibility.json").read_text(encoding="utf-8"))
    return compat, (root / "README.md").read_text(encoding="utf-8")


def relative_targets(text):
    found = set(re.findall(r"\]\(([^)\s]+)", text))
    found |= set(re.findall(r'(?:src|srcset|href)="([^"]+)"', text))
    for t in found:
        t = t.split("#")[0]
        if t and not re.match(r"[a-z][a-z0-9+.-]*:", t) and not t.startswith("#"):
            yield t


def problems(root=ROOT):
    compat, text = load(root)
    project = compat["project"]
    out = []
    if project not in SITES:
        return [f"compatibility.json names an unknown project {project!r}"]
    pkg = compat["package"]
    version = pkg.get("public_version") or pkg["development_version"]

    # the lockup
    m = re.search(r"<h1[^>]*>\s*<picture>\s*<source[^>]*media=\"\(prefers-color-scheme: dark\)\"[^>]*srcset=\"([^\"]+)\"[^>]*>\s*<img[^>]*src=\"([^\"]+)\"[^>]*alt=\"([^\"]+)\"", text)
    if not m:
        out.append("the README does not open with the approved <h1><picture> lockup (dark source, light image, alt text)")
    else:
        dark, light, alt = m.groups()
        for f in (dark, light):
            if not (root / f).is_file():
                out.append(f"the lockup file {f} does not exist")
        if alt != project:
            out.append(f"the lockup alt text is {alt!r}, expected {project!r}")
        if not (dark.endswith("-dark.svg") and light.endswith("-light.svg") and project in dark and project in light):
            out.append("the lockup files are not the project's dark and light horizontal lockups")

    # relative targets
    for t in sorted(set(relative_targets(text))):
        if not (root / t).exists():
            out.append(f"the README links to {t}, which does not exist")

    # release assets and versions
    templates = compat.get("release_assets", [])
    allowed = [re.compile("^" + re.escape(t).replace(r"\{version\}", re.escape(version)) + "$") for t in templates]
    own = re.escape(f"github.com/tensorsofthewall/{project}/releases/download/v")
    for v in sorted(set(re.findall(rf"{own}(\d+\.\d+\.\d+)", text))):
        if v != version:
            out.append(f"a release URL names v{v}, but the package version is {version}")
    for v, asset in re.findall(rf"{own}(\d+\.\d+\.\d+)/([^\s)\"'`]+)", text):
        if v == version and not any(a.match(asset) for a in allowed):
            out.append(f"the release asset {asset} is not one of the release_assets of compatibility.json")
    for v, tail in re.findall(r"\b(?:tbccl|torch_tbccl|vllm_tbccl|exo_tbccl)-(\d+\.\d+\.\d+)-([\w.-]+\.(?:whl|tar\.gz))", text):
        name = re.search(rf"(?:tbccl|torch_tbccl|vllm_tbccl|exo_tbccl)-{re.escape(v)}-{re.escape(tail)}", text).group(0)
        if project == "tbccl" and not name.startswith("tbccl-"):
            continue
        if project != "tbccl" and not name.startswith(project.replace("-", "_") + "-"):
            continue
        if v != version:
            out.append(f"{name} names version {v}, but the package version is {version}")
        elif not any(a.match(name) for a in allowed):
            out.append(f"{name} is not one of the release_assets of compatibility.json")
    if not re.search(rf"\*\*{re.escape(version)}\*\*", text):
        out.append(f"the status line does not state the current version {version} in bold")

    # protocol values
    abi = compat.get("c_abi", {}).get("version")
    wire = compat.get("wire_protocol", {}).get("version")
    req = compat.get("requires_tbccl") or (compat.get("requires") or {}).get("tbccl") or {}  # exo-tbccl records only the C ABI: its wire protocol is not checked here
    abi = abi if abi is not None else (req.get("c_abi") or [None])[0]
    wire = wire if wire is not None else (req.get("wire_protocol") or [None])[0]
    if abi is not None and not re.search(rf"C ABI\W{{0,4}}(?:version )?\*{{0,2}}{abi}\b", text):
        out.append(f"the README does not state C ABI {abi}")
    if wire is not None and not re.search(rf"wire protocol\W{{0,4}}(?:version )?\*{{0,2}}{wire}\b", text):
        out.append(f"the README does not state wire protocol {wire}")

    # tags and pins named in the README exist in the compatibility facts
    facts = json.dumps(compat)
    page = root / "docs" / "reference" / ("compatibility-matrix.md" if project == "tbccl" else "compatibility.md")
    facts += page.read_text(encoding="utf-8") if page.is_file() else ""
    for tag in sorted(set(re.findall(r"\btbccl-integration-v\d+\b|\bv\d+\.\d+\.\d+-tbccl\d+\b", text))):
        if tag not in facts:
            out.append(f"the README names {tag}, which neither compatibility.json nor the compatibility page records")

    # what every README owes its reader
    for tok in REQUIRED_TOKENS[project]:
        if tok not in text:
            out.append(f"the README no longer mentions {tok}")
    for link in REQUIRED_LINKS[project]:
        if not re.search(rf"\]\({re.escape(link)}[#)]|`{re.escape(link)}`|\b{re.escape(link)}\b", text):
            out.append(f"the README does not link to {link}")
    if f"https://{project}.tensorsofthewall.com/" not in text:
        out.append("the README does not link to the hosted documentation")
    if not re.search(r"(?m)^## Independence\s*$", text) or not re.search(r"(?m)^## License\s*$", text):
        out.append("the README lacks its License or Independence section")

    # own URLs: right host, no stray version, no .md pages
    for m_ in OWN_URL.finditer(text):
        if m_.group("site") and m_.group("site") not in SITES:
            out.append(f"{m_.group(0)} names an unknown documentation site")
        if m_.group("site") and m_.group("path") and m_.group("path").endswith(".md"):
            out.append(f"{m_.group(0)} points at a Markdown source, not a built page")

    for m_ in INTERNAL.finditer(text):
        out.append(f"internal wording in the README: {m_.group(0)!r}")
    return out


def online(root=ROOT):
    _, text = load(root)
    bad = []
    for u in sorted({m.group(0).rstrip(".,;:*") for m in OWN_URL.finditer(text)}):
        base = u.split("#")[0]
        if re.search(r"/releases/download/v[\d.]+$", base):  # the prefix of a shell variable, not a page
            continue
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-L", "-w", "%{http_code}", "--max-time", "25", base], capture_output=True, text=True)
        if r.stdout != "200":
            bad.append(f"{u} answers {r.stdout or 'nothing'}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--online", action="store_true", help="also check that every own URL in the README answers")
    args = ap.parse_args(argv)
    found = problems()
    if args.online:
        found += online()
    for p in found:
        print(f"README: {p}", file=sys.stderr)
    print(f"README check: {'FAILED' if found else 'OK'} ({len(found)} problem(s))")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
