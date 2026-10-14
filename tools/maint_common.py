"""Shared helpers of the maintenance tooling. This file is kept identical in the four TBCCL release repositories.

Every maintenance script exits with one of the codes below, so a scheduled run says what kind of problem it found instead of only failing:

    0  OK
    1  INTEGRITY FAILURE        an immutable pin or tag no longer resolves as recorded, or the repository state is inconsistent
    2  RELEASE SURFACE FAILURE  an expected stable release or artifact is missing or does not verify
    3  WARNING                  a network or documentation-host problem that is probably transient; never a mismatch
    4  ERROR                    the automation itself failed (bad usage, missing tool, unreadable file)
    5  DRIFT                    an upstream moved (used by the fork monitors)

Nothing here writes to a repository, creates a tag or a release, or needs a credential; GITHUB_TOKEN is only used, when present, to raise the API rate limit.
"""
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

OK, INTEGRITY, SURFACE, WARNING, ERROR, DRIFT = 0, 1, 2, 3, 4, 5
CATEGORY = {OK: "OK", INTEGRITY: "INTEGRITY FAILURE", SURFACE: "RELEASE SURFACE FAILURE", WARNING: "WARNING", ERROR: "ERROR", DRIFT: "DRIFT"}

SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(rc(0|[1-9]\d*))?$")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
REPO_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9._-]+$")


class Network(Exception):
    """A request failed in a way that says nothing about the data (timeout, DNS, rate limit, 5xx)."""


class Report:
    """Collects findings by category and turns them into one exit code, a console log and a job summary."""

    def __init__(self, title):
        self.title = title
        self.items = []

    def add(self, code, message):
        self.items.append((code, message))
        prefix = CATEGORY[code]
        if os.environ.get("GITHUB_ACTIONS") and code != OK:
            level = "warning" if code in (WARNING, DRIFT) else "error"
            print(f"::{level} title={prefix}::{message}")
        print(f"{prefix}: {message}" if code != OK else f"ok: {message}")

    def ok(self, message):
        self.add(OK, message)

    def worst(self):
        failures = [c for c, _ in self.items if c in (INTEGRITY, SURFACE, ERROR)]
        if failures:
            return min(failures)
        for c in (DRIFT, WARNING):
            if any(code == c for code, _ in self.items):
                return c
        return OK

    def summary(self):
        path = os.environ.get("GITHUB_STEP_SUMMARY")
        if not path:
            return
        with open(path, "a") as f:
            f.write(f"### {self.title}: {CATEGORY[self.worst()]}\n\n")
            for code, message in self.items:
                f.write(f"- **{CATEGORY[code]}**: {message}\n")
            f.write("\n")

    def finish(self):
        self.summary()
        code = self.worst()
        print(f"{self.title}: {CATEGORY[code]}")
        return code


def split_version(text):
    m = SEMVER.match(text)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4) or ""


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def ls_remote_tag(repo, tag):
    """The commit a tag resolves to on GitHub (the peeled commit for an annotated tag), or None when the tag does not exist. Raises Network."""
    url = f"https://github.com/{repo}.git"
    try:
        p = run(["git", "ls-remote", "--tags", url, f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"], timeout=60)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise Network(str(e))
    if p.returncode != 0:
        raise Network(p.stderr.strip() or f"git ls-remote exited {p.returncode}")
    refs = {}
    for line in p.stdout.splitlines():
        sha, ref = line.split("\t")
        refs[ref] = sha
    return refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")


def ls_remote_head(repo, branch):
    url = f"https://github.com/{repo}.git"
    try:
        p = run(["git", "ls-remote", url, f"refs/heads/{branch}"], timeout=60)
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise Network(str(e))
    if p.returncode != 0:
        raise Network(p.stderr.strip() or f"git ls-remote exited {p.returncode}")
    return p.stdout.split("\t")[0] if p.stdout.strip() else None


def github_api(path, retries=2):
    """Parsed JSON of a GitHub REST path, or None for 404. Raises Network for anything that is not a definite answer."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "tbccl-maintenance"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last = None
    for _ in range(retries + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(f"https://api.github.com/{path}", headers=headers), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = f"HTTP {e.code} for {path}"
            if e.code not in (403, 429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = f"{type(e).__name__}: {e}"
    raise Network(last)


def http_status(url, timeout=30):
    """HTTP status of a GET, or raises Network."""
    req = urllib.request.Request(url, headers={"User-Agent": "tbccl-maintenance"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise Network(f"{type(e).__name__}: {e}")


ECOSYSTEM_URL = "https://raw.githubusercontent.com/tensorsofthewall/tbccl/{ref}/release/ecosystem.json"


def load_ecosystem(file=None, ref="main"):
    """The canonical CURRENT-ecosystem record (tbccl release/ecosystem.json): a local file, or the given ref of tensorsofthewall/tbccl. Raises Network when unavailable."""
    try:
        if file:
            with open(file) as f:
                return json.load(f)
        req = urllib.request.Request(ECOSYSTEM_URL.format(ref=ref), headers={"User-Agent": "tbccl-maintenance"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except (OSError, ValueError) as e:
        raise Network(f"{type(e).__name__}: {e}") from None


def compare_current_ecosystem(rep, eco, key, mine, version, released):
    """Compare this repository's recorded facts with its entry in the current-ecosystem record. This is CURRENT-ECOSYSTEM CONSISTENCY, never release-pin integrity.

    The entry describes the currently published stable version. A disagreement is an INTEGRITY FAILURE only when this repository IS that published version
    (released and the same version): the public ecosystem is then inconsistent. For an unreleased candidate, or an older release, the ecosystem record simply has not been
    (or is no longer) updated for this version, which is a WARNING; the repository's own immutable pin stays authoritative.
    """
    entry = eco.get(key) if isinstance(eco, dict) else None
    if not isinstance(entry, dict):
        rep.add(WARNING, f"current ecosystem record has no {key!r} entry")
        return
    diffs = [f"{k}: ecosystem says {entry.get(k)!r}, this repository records {v!r}" for k, v in mine.items() if entry.get(k) != v]
    if not diffs:
        rep.ok(f"current-ecosystem consistency: the {key} entry agrees with this repository")
    elif released and entry.get("version") == version:
        for d in diffs:
            rep.add(INTEGRITY, f"current-ecosystem inconsistency for the published {version}: {d}")
    else:
        why = "this repository is not the currently published stable version" if released else "this version is an unreleased candidate"
        rep.add(WARNING, f"current ecosystem manifest has not been updated for this candidate ({why}); the repository's own immutable pin remains authoritative: " + "; ".join(diffs))


def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit(f"{CATEGORY[ERROR]}: cannot read {path}: {e}") from None


def main_guard(fn):
    """Run a script's main(); a crash is the automation's fault (ERROR), never an integrity verdict."""
    try:
        sys.exit(fn())
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"{CATEGORY[ERROR]}: {type(e).__name__}: {e}")
        sys.exit(ERROR)
