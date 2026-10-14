#!/usr/bin/env python3
"""Link checker for the documentation of a TBCCL repository. This file is kept identical in the four TBCCL release repositories.

    check_links.py             offline: every relative link in the Markdown files must point at a file that exists (blocking)
    check_links.py --online    additionally: the links to this project's GitHub ecosystem and documentation hosts must resolve; other external links are retried
                               and only ever produce warnings

Three classes of link: repository-internal links (a broken one is an INTEGRITY FAILURE), critical ecosystem links (the tensorsofthewall GitHub repositories,
releases and documentation sites; a 404 is a RELEASE SURFACE FAILURE, an unreachable host a WARNING) and arbitrary external links (always at most a WARNING,
so one slow unrelated website never turns a build red). Fenced code blocks are ignored.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maint_common as mc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LINK = re.compile(r"\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)|<(https?://[^>\s]+)>|(?<![\(\[<\"'`])(https?://[^\s<>)\]\"'`]+)")
CRITICAL = re.compile(r"^https://(github\.com/tensorsofthewall/|[a-z-]*\.?tensorsofthewall\.com)")
SKIP_DIRS = {".git", "_build", ".venv-docs", "node_modules", "build", "third_party"}
EXAMPLE_HOSTS = ("example.com", "example.org", "localhost", "127.0.0.1", "0.0.0.0")


def markdown_files(root):
    for p in sorted(root.rglob("*.md")):
        if not (set(p.relative_to(root).parts) & SKIP_DIRS):
            yield p


def links_in(path):
    in_fence = False
    for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for m in LINK.finditer(line):
            target = next(g for g in m.groups() if g)
            yield n, target.rstrip(".,;:")


def classify(target):
    if target.startswith(("mailto:", "#", "{", "tel:")):
        return "skip"
    if re.match(r"^https?://", target):
        host = target.split("/")[2].split(":")[0]
        if host in EXAMPLE_HOSTS or "<" in target or "{" in target or "$" in target:
            return "skip"
        return "critical" if CRITICAL.match(target) else "external"
    return "internal"


def check_internal(path, target, root):
    rel = target.split("#")[0].split("?")[0]
    if not rel:
        return True
    resolved = (root / rel.lstrip("/")) if rel.startswith("/") else (path.parent / rel)
    return resolved.exists()


def probe(url, retries=2):
    last = None
    for _ in range(retries + 1):
        try:
            status = mc.http_status(url)
        except mc.Network as e:
            last = str(e)
            continue
        if status in (429,) or status >= 500:
            last = f"HTTP {status}"
            continue
        return status, None
    return None, last


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--online", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    rep = mc.Report("links" + (" (online)" if args.online else " (offline)"))
    seen = {}
    files = 0
    for path in markdown_files(root):
        files += 1
        for n, target in links_in(path):
            kind = classify(target)
            where = f"{path.relative_to(root)}:{n}"
            if kind == "internal" and not check_internal(path, target, root):
                rep.add(mc.INTEGRITY, f"{where}: broken relative link {target}")
            elif kind in ("critical", "external") and args.online:
                url = target.split("#")[0]
                if url not in seen:
                    seen[url] = probe(url)
                status, err = seen[url]
                if err:
                    rep.add(mc.WARNING, f"{where}: {url} unreachable ({err})")
                elif status >= 400:
                    code = mc.SURFACE if kind == "critical" else mc.WARNING
                    rep.add(code, f"{where}: {url} returned HTTP {status}")
    if all(code == mc.OK for code, _ in rep.items):
        rep.ok(f"{files} Markdown files checked, {len(seen)} external URLs probed")
    return rep.finish()


if __name__ == "__main__":
    mc.main_guard(main)
