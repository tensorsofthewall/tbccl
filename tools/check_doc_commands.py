#!/usr/bin/env python3
"""Static checks of the shell commands in the documentation. Nothing is executed and nothing is downloaded.

    python3 tools/check_doc_commands.py            # check docs/ and the top-level Markdown files
    python3 tools/check_doc_commands.py --list     # also print the inventory of command blocks

For every fenced `sh`, `bash` or `console` block it checks that

* the block is valid `bash -n` syntax once placeholders such as `<PORT>` are replaced (the block is parsed, never run);
* every repository path the block names (`tools/...`, `examples/...`, `scripts/...`, `benchmarks/...`, ...) exists;
* every `python -m <module>` that names a module of this repository resolves to a file;
* every `make <target>` names a target of the Makefile;
* a block is not left with a marker such as `TODO` or `XXX`.

A block can opt out with `<!-- doccheck: skip (reason) -->` on the line before it; the reason is mandatory. Blocks that need hardware, a download or
another machine are listed by `--list` so a reviewer can see what is only inspected. The same file is used by all four TBCCL repositories.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHELL = {"sh", "bash", "shell", "console", "zsh"}
FENCE = re.compile(r"^(?P<ind>\s*)(?P<fence>`{3,}|~{3,}|:{3,})\{?(?P<lang>[\w+-]*)\}?[^\n]*$")
SKIP = re.compile(r"<!--\s*doccheck:\s*skip\s*\((?P<why>[^)]+)\)\s*-->")
PATH = re.compile(r"""(?<![\w./:@-])(?:\./)?(?P<p>(?:tools|scripts|examples|benchmarks|tests|packaging|cmake|src|include|docs|csrc|release)/[\w./-]*[\w])""")
PLACEHOLDER = re.compile(r"<[A-Za-z_][\w .-]*>")
MODULE = re.compile(r"\bpython3?(?:\.\d+)?\s+(?:-\w+\s+)*-m\s+(?P<m>[A-Za-z_][\w.]*)")
THIRD_PARTY = {"pip", "venv", "unittest", "pytest", "http", "json", "build", "sphinx", "uv", "torch", "vllm", "exo", "mlx", "mlx_lm", "twine", "ensurepip", "cProfile", "timeit", "site"}
MARKERS = re.compile(r"\b(TODO|FIXME|XXX)\b")


def markdown_files():
    files = [p for p in ROOT.glob("*.md")]
    files += sorted((ROOT / "docs").rglob("*.md"))
    return [p for p in files if "_build" not in p.parts and ".venv" not in str(p)]


def blocks(path):
    """Yield (line number, language, text, skip reason) for each fenced block."""
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if m and m.group("fence")[0] in "`~":
            fence, lang, start = m.group("fence"), m.group("lang").lower(), i
            body = []
            i += 1
            while i < len(lines) and not (lines[i].strip().startswith(fence) and set(lines[i].strip()) <= set(fence[0])):
                body.append(lines[i])
                i += 1
            prev = lines[start - 1] if start else ""
            skip = SKIP.search(prev)
            yield start + 1, lang, "\n".join(body), skip.group("why") if skip else None
        i += 1


def make_targets():
    mk = ROOT / "Makefile"
    return set(re.findall(r"^([A-Za-z][\w-]*)\s*:", mk.read_text(), re.M)) if mk.is_file() else set()


def module_exists(name):
    parts = name.split(".")
    if parts[0] in THIRD_PARTY or not (ROOT / parts[0]).exists():
        return True  # not a module of this repository
    base = ROOT.joinpath(*parts)
    return base.with_suffix(".py").is_file() or (base / "__main__.py").is_file() or (base / "__init__.py").is_file() and len(parts) == 1


def syntax_ok(text):
    cleaned = []
    for line in text.splitlines():
        line = re.sub(r"^\s*\$ ", "", line)
        cleaned.append(PLACEHOLDER.sub("PLACEHOLDER", line))
    r = subprocess.run(["bash", "-n"], input="\n".join(cleaned) + "\n", capture_output=True, text=True)
    return r.returncode == 0, r.stderr.strip().splitlines()[-1] if r.stderr.strip() else ""


def created_in_block(text):
    """Files a block creates itself (`cat > file`, `> file`, `touch`, `mkdir`), which may then be named later in the same block."""
    return set(re.findall(r"(?:cat\s*>>?|tee|touch|mkdir(?:\s+-p)?|>)\s+([\w./-]+)", text))


def check(path):
    problems, inventory = [], []
    targets = make_targets()
    for line, lang, text, skip in blocks(path):
        if lang not in SHELL:
            continue
        rel = path.relative_to(ROOT)
        where = f"{rel}:{line}"
        inventory.append((where, text.strip().splitlines()[0][:70] if text.strip() else "", bool(skip)))
        if skip:
            continue
        ok, err = syntax_ok(text)
        if not ok:
            problems.append(f"{where}: shell syntax: {err}")
        made = created_in_block(text)
        for m in PATH.finditer(text):
            p = m.group("p")
            if any(c in p for c in "*$") or "_build" in p:
                continue
            if p in made or any(p.startswith(x + "/") for x in made):
                continue
            if not (ROOT / p).exists():
                problems.append(f"{where}: names {p}, which does not exist in this repository")
        for m in MODULE.finditer(text):
            if not module_exists(m.group("m")):
                problems.append(f"{where}: python -m {m.group('m')} does not resolve to a file")
        for m in re.finditer(r"(?m)^\s*(?:\$ )?make\s+(?:-\w+\s+)*([A-Za-z][\w-]*)", text):
            if targets and m.group(1) not in targets and m.group(1) not in {"install", "all"}:
                problems.append(f"{where}: make {m.group(1)}: no such target in the Makefile")
        if MARKERS.search(text):
            problems.append(f"{where}: an unfinished marker in a command block")
    return problems, inventory


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="print every command block checked or skipped")
    args = ap.parse_args(argv)
    problems, inventory = [], []
    for p in markdown_files():
        a, b = check(p)
        problems += a
        inventory += b
    if args.list:
        for where, first, skipped in inventory:
            print(f"{'skip ' if skipped else '     '}{where}  {first}")
    for p in problems:
        print(p, file=sys.stderr)
    print(f"{len(inventory)} command block(s) checked, {sum(1 for _, _, s in inventory if s)} skipped, {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
