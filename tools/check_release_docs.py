#!/usr/bin/env python3
"""Checks that the release-specific statements of the documentation describe the release the source reports. This file is kept identical in the four TBCCL
release repositories; what differs between them is only tools/release_docs.json.

    check_release_docs.py                  the current branch: the documentation must describe the package version of the source and compatibility.json
    check_release_docs.py --tag v0.2.1     the intended release: additionally, the tag must be the package version, so a version bump whose README, homepage or
                                           installation commands still describe the previous release fails before anything is tagged
    check_release_docs.py --tbccl-root DIR adapters: the TBCCL that the wheel bundles (versions.bundled_tbccl) must be the version of that TBCCL checkout

The intended version comes from compatibility.json (which the source-agreement checker verifies against the source), never from release/ecosystem.json, which is
only updated after a release is published. The check needs no network, no release and no asset.

It does not reject every old version number. It looks at the places where a version is a promise about *this* release:
  * the download, tag and source-archive URLs of this repository, and every archive or wheel name built from release_assets in compatibility.json;
  * the statements listed in tools/release_docs.json (the status line, the homepage, the expected output of a version command), each of which must be found
    and must carry the expected version: the current release, or a named other version such as the TBCCL a wheel bundles or the wheel an immutable tag installs.
Everything else (changelog sections, benchmark and validation records, comparisons, warnings about superseded releases) is left alone. A deliberate older
reference inside a scanned file is declared in "allow" with the reason, so it is visible in review.
"""
import argparse
import fnmatch
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maint_common as mc  # noqa: E402
import check_release_state as crs  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CONFIG = "tools/release_docs.json"
VERSION = r"\d+\.\d+\.\d+(?:rc\d+)?"
WHY = {
    "url": "an installation link that names another release downloads or shows the wrong release",
    "asset": "an archive or wheel name with another version is not an asset of this release, so the command fails or installs the wrong thing",
    "statement": "users read this statement as the description of the current release",
}


def asset_patterns(templates):
    """Regexes for the artifact names of a release; a .tar.gz name also matches without the extension (the extracted directory)."""
    patterns = []
    for t in templates:
        variants = [t]
        if t.endswith(".tar.gz"):
            variants.append(t[: -len(".tar.gz")])
        for v in variants:
            parts = [re.escape(p) for p in v.split("{version}")]
            patterns.append(re.compile(f"(?P<v>{VERSION})".join(parts)))
    return patterns


def line_of(text, offset):
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    return text.count("\n", 0, offset) + 1, text[start: end if end >= 0 else len(text)].strip()


class Checker:
    def __init__(self, root, manifest, config, rep, version):
        self.root, self.manifest, self.config, self.rep, self.version = root, manifest, config, rep, version
        self.repo_name = config["repo"].split("/")[-1]
        self.named = config.get("versions", {})
        self.findings = []

    def fail(self, rel, lineno, line, what, found, expected, why):
        self.findings.append((rel, lineno, f"{self.repo_name}: {rel}:{lineno}: {what} says {found}; expected {expected}. {why}. Line: {line[:160]}"))

    def report(self):
        for _, _, message in sorted(self.findings):
            self.rep.add(mc.INTEGRITY, message)

    def allowed(self, rel, line, found):
        for entry in self.config.get("allow", []):
            if fnmatch.fnmatch(rel, entry["file"]) and re.search(entry["line"], line) and found in entry["versions"]:
                return True
        return False

    def files(self):
        out = []
        for pattern in self.config["scan"]:
            out += [p for p in sorted(self.root.glob(pattern)) if p.is_file()]
        seen, result = set(), []
        for p in out:
            rel = p.relative_to(self.root).as_posix()
            if rel in seen or any(fnmatch.fnmatch(rel, x) for x in self.config.get("exclude", [])):
                continue
            seen.add(rel)
            result.append((rel, p.read_text()))
        return result

    def check_links_and_assets(self, rel, text):
        repo = re.escape(self.config["repo"])
        url = re.compile(rf"github\.com/{repo}/(?:releases/(?:download|tag)|archive/refs/tags)/v(?P<v>{VERSION})")
        patterns = [("url", url)] + [("asset", p) for p in asset_patterns(self.manifest.get("release_assets", []))]
        seen = set()
        for kind, rx in patterns:
            for m in rx.finditer(text):
                found = m.group("v")
                if (m.start("v"), found) in seen:
                    continue
                seen.add((m.start("v"), found))
                lineno, line = line_of(text, m.start())
                if found != self.version and not self.allowed(rel, line, found):
                    what = "download URL" if kind == "url" else "archive or wheel name"
                    self.fail(rel, lineno, line, what, found, self.version, WHY[kind])

    def expected_for(self, spec):
        if spec == "current":
            return self.version
        if spec.startswith("version:") and spec[8:] in self.named:
            return self.named[spec[8:]]
        raise ValueError(f"unknown expectation {spec!r}")

    def check_statements(self):
        for st in self.config.get("statements", []):
            path = self.root / st["file"]
            if not path.exists():
                self.fail(st["file"], 0, "", "statement file", "missing", "present", "the release statement cannot be checked")
                continue
            text = path.read_text()
            expected = self.expected_for(st["expect"])
            matches = list(re.finditer(st["find"], text, re.M))
            if len(matches) < st.get("min", 1):
                self.fail(st["file"], 0, "", f"statement '{st['describe']}'", "not found", expected,
                          "the wording changed or the statement was removed; update tools/release_docs.json with the new wording so it stays checked")
            for m in matches:
                found = m.group("v")
                if found != expected:
                    lineno, line = line_of(text, m.start("v"))
                    self.fail(st["file"], lineno, line, st["describe"], found, expected, st.get("why", WHY["statement"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag")
    ap.add_argument("--tbccl-root")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args()
    root = Path(args.root)
    rep = mc.Report("release documentation")
    if not (root / "compatibility.json").exists() or not (root / CONFIG).exists():
        rep.add(mc.INTEGRITY, f"compatibility.json or {CONFIG} is missing")
        return rep.finish()
    manifest, config = mc.load_json(root / "compatibility.json"), mc.load_json(root / CONFIG)
    version = crs.check_manifest(manifest, rep, args.tag)
    if not version:
        return rep.finish()
    for rel in crs.SOURCE_CHECKERS:
        if (root / rel).exists():
            crs.run_script(root, rel, rep, "source agreement")
            break
    if args.tbccl_root and "bundled_tbccl" in config.get("versions", {}):
        core = crs.package_version(mc.load_json(Path(args.tbccl_root) / "compatibility.json"))
        if core != config["versions"]["bundled_tbccl"]:
            rep.add(mc.INTEGRITY, f"{CONFIG} says the wheel bundles TBCCL {config['versions']['bundled_tbccl']} but the TBCCL checkout is {core}")
    checker = Checker(root, manifest, config, rep, version)
    scanned = checker.files()
    for rel, text in scanned:
        checker.check_links_and_assets(rel, text)
    checker.check_statements()
    checker.report()
    if not checker.findings:
        rep.ok(f"{checker.repo_name} {version}: {len(scanned)} files and {len(config.get('statements', []))} release statements describe the release"
               + (f" (intended tag {args.tag})" if args.tag else ""))
    return rep.finish()


if __name__ == "__main__":
    mc.main_guard(main)
