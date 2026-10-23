#!/usr/bin/env python3
"""Trust-store dependency policy of a TBCCL repository. This file is kept identical in the four TBCCL release repositories.

After the October 2026 eMudhra/emSign incident the emSign roots were removed from the CA bundles that Python tooling uses. The minimum fixed versions are
certifi 2026.10.19, pip 26.2.2 and uv 0.13.4 (Python Security Response Team announcement of 2026-10-20). This check keeps an affected version from coming back
into an ACTIVE dependency manifest or build path:

  1. certifi in requirements, constraints, pyproject and uv.lock files is at least the minimum (an exact pin, or a lower bound that is not below it);
  2. uv pins in workflows, build scripts and Read the Docs files (`UV_VERSION`, `uv==`, `astral.sh/uv/<version>/install.sh`) are at least the minimum;
  3. every workflow, CI script, Makefile and Read the Docs file that runs `pip install` also bootstraps the hash-pinned pip (`bootstrap_pip.sh`, or the wheel
     URL itself with its sha256), and the Read the Docs file always does, because pip trusts its own vendored certifi and an older pip cannot be repaired
     from the outside;
  4. every CI script that runs `dnf` first runs `distrust_emsign.sh` (the manylinux system bundle still lists the emSign roots);
  5. no container image digest known to be affected is used again;
  6. no TLS-verification bypass (`--trusted-host`, `curl -k`, `--insecure`, `verify=False` on an HTTP client, `PYTHONHTTPSVERIFY=0`, ...) appears in a workflow,
     script or build file.

Only active files are read. Validation pages, the changelog, benchmark records, release records and documentation are historical or explanatory and are never
scanned, so a record that names an older version stays as it was.

    check_trust_dependencies.py [--root DIR]      exits 0 when clean, 1 (INTEGRITY FAILURE) when a rule is broken
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import maint_common as mc  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIN_CERTIFI = (2026, 10, 19)
MIN_UV = (0, 13, 4)
PIP_WHEEL_MARK = "pip-26.2.2-py3-none-any.whl#sha256=fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"

SKIP_DIRS = {".git", "_build", "build", "node_modules", "third_party", "__pycache__"}
HISTORICAL = ("docs/validation/", "docs/benchmarks/", "benchmarks/results/", "release/", "results/", "CHANGELOG.md")
AFFECTED_IMAGE_DIGESTS = {
    "39df0042d5cc900b085aa25a0659368b42a0006c54c474299b785b44c1b4ff82": "manylinux_2_28 of 2026-10-03 (pip 26.2.1, certifi 2026.7.22 in its tools, uv 0.12.22)",
    "bf44cdfcb76cd3b41e879bc058fc37ec5872002ccfde7fcb765e218cde0cd79c": "python 3.13.16-slim (pip 26.2.1 with certifi 2026.06.17)",
}

REQ_NAME = re.compile(r"^(?:requirements|constraints)[\w.-]*\.(?:txt|in)$")
CERTIFI_REQ = re.compile(r"""(?<![\w-])certifi\s*(==|>=|~=|===)\s*([0-9][0-9A-Za-z.]*)""")
UV_PIN = [
    re.compile(r"""\bUV_VERSION\s*[:=]\s*["']?([0-9][0-9.]*)"""),
    re.compile(r"""(?<![\w-])uv==([0-9][0-9.]*)"""),
    re.compile(r"""astral\.sh/uv/([0-9][0-9.]*)/install"""),
]
PIP_INSTALL = re.compile(r"(?<!uv )pip3? install|-m pip\b")
BYPASS = [
    (re.compile(r"--trusted-host"), "pip --trusted-host"),
    (re.compile(r"\bcurl\b[^\n|;&]*\s(?:-[A-Za-z]*k[A-Za-z]*|--insecure)(?=\s|$)"), "curl -k / --insecure"),
    (re.compile(r"\bwget\b[^\n|;&]*--no-check-certificate"), "wget --no-check-certificate"),
    (re.compile(r"PYTHONHTTPSVERIFY\s*=\s*0"), "PYTHONHTTPSVERIFY=0"),
    (re.compile(r"GIT_SSL_NO_VERIFY|http\.sslVerify\s*=?\s*false"), "git TLS verification off"),
    (re.compile(r"NODE_TLS_REJECT_UNAUTHORIZED\s*=\s*0"), "NODE_TLS_REJECT_UNAUTHORIZED=0"),
    (re.compile(r"--allow-insecure-host|UV_INSECURE_HOST|UV_NATIVE_TLS\s*=\s*0"), "uv insecure host"),
    (re.compile(r"(?:requests|httpx|aiohttp|urllib3?|session|client|Client)\b[^\n]*\bverify\s*=\s*False"), "HTTP client verify=False"),
    (re.compile(r"_create_unverified_context|ssl\.CERT_NONE|check_hostname\s*=\s*False"), "ssl verification off"),
]


def version_tuple(text):
    parts = []
    for piece in text.split("."):
        m = re.match(r"\d+", piece)
        if not m:
            break
        parts.append(int(m.group(0)))
    return tuple(parts)


def files(root):
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if set(rel.parts) & SKIP_DIRS or any(part.startswith(".venv") for part in rel.parts):
            continue
        r = rel.as_posix()
        if r.startswith(HISTORICAL):
            continue
        yield p, r


def text_of(path):
    try:
        return path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None


def lock_certifi(text):
    """The version of the certifi package recorded in a uv.lock, or None."""
    m = re.search(r'\[\[package\]\]\s*\nname = "certifi"\s*\nversion = "([^"]+)"', text)
    return m.group(1) if m else None


def is_ci_file(rel):
    return (
        rel.startswith(".github/workflows/")
        or rel == "Makefile"
        or rel.startswith((".readthedocs.", "readthedocs."))
        or re.match(r"^(?:tools|scripts)/ci/[^/]+\.sh$", rel) is not None
    )


def is_build_file(rel):
    return is_ci_file(rel) or (rel.startswith(("packaging/", "scripts/", "tools/")) and rel.endswith((".sh", ".yml", ".yaml")))


def problems(root):
    found = []
    selfs = {"tools/check_trust_dependencies.py", "tests/trust_dependencies_test.py", "tests/test_trust_dependencies.py"}
    for path, rel in files(root):
        if rel in selfs:
            continue
        name = path.name
        text = text_of(path)
        if text is None:
            continue
        if REQ_NAME.match(name) or name == "pyproject.toml":
            for n, line in enumerate(text.splitlines(), 1):
                code = line.split("#", 1)[0]
                for op, ver in CERTIFI_REQ.findall(code):
                    if version_tuple(ver) < MIN_CERTIFI:
                        found.append(f"{rel}:{n}: certifi{op}{ver} is older than the minimum {'.'.join(map(str, MIN_CERTIFI))} (it still trusts the revoked emSign roots)")
        if name == "uv.lock":
            ver = lock_certifi(text)
            if ver and version_tuple(ver) < MIN_CERTIFI:
                found.append(f"{rel}: the locked certifi {ver} is older than the minimum {'.'.join(map(str, MIN_CERTIFI))}")
        if is_build_file(rel):
            for n, line in enumerate(text.splitlines(), 1):
                for pat in UV_PIN:
                    for ver in pat.findall(line):
                        if version_tuple(ver) < MIN_UV:
                            found.append(f"{rel}:{n}: uv {ver} is older than the minimum {'.'.join(map(str, MIN_UV))}")
                for digest, why in AFFECTED_IMAGE_DIGESTS.items():
                    if digest in line:
                        found.append(f"{rel}:{n}: container image {digest[:12]} is affected: {why}")
                for pat, what in BYPASS:
                    if pat.search(line):
                        found.append(f"{rel}:{n}: TLS-verification bypass ({what})")
        if is_ci_file(rel):
            if PIP_INSTALL.search(text) and "bootstrap_pip.sh" not in text and PIP_WHEEL_MARK not in text:
                found.append(f"{rel}: runs pip but does not bootstrap the hash-pinned pip 26.2.2 (bootstrap_pip.sh)")
            if rel.startswith(".readthedocs.") and PIP_WHEEL_MARK not in text:
                found.append(f"{rel}: the Read the Docs build does not install the hash-pinned pip 26.2.2 before the requirements")
            if rel.endswith(".sh") and re.search(r"^\s*dnf\b", text, re.M) and "distrust_emsign.sh" not in text:
                found.append(f"{rel}: runs dnf in the build container without distrust_emsign.sh first")
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args(argv)
    rep = mc.Report("trust-store dependencies")
    issues = problems(args.root)
    for line in issues:
        rep.add(mc.INTEGRITY, line)
    if not issues:
        rep.ok("no affected certifi, uv or container pins, pip is bootstrapped from its hash-pinned wheel, no TLS-verification bypass")
    return rep.worst()


if __name__ == "__main__":
    sys.exit(main())
