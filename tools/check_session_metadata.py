#!/usr/bin/env python3
"""Reject private AI-assistant session metadata in text that becomes public Git history.

Looks for session trailers (for example a `Claude-Session:` line), links to private assistant conversations or sessions, and session identifiers, in commit messages,
annotated tag messages, pull request titles and descriptions, and release notes. It does not reject the generic product link or ordinary technical text that merely
mentions the word "session".

A finding is reported by category, source and line number only. The matching text is never printed, so a failing check cannot copy the private link into a log.

    tools/check_session_metadata.py --message-file .git/COMMIT_EDITMSG --strip-comments   # commit-msg hook
    tools/check_session_metadata.py --commits BASE..HEAD                                  # every commit message in a range
    tools/check_session_metadata.py --tag v1.2.3                                          # an annotated tag's message
    tools/check_session_metadata.py --message-file pr_body.txt CHANGELOG.md               # any text files (PR text, release notes)

Exit status: 0 nothing found, 1 something found, 2 usage or git error.
"""

import argparse
import re
import subprocess
import sys

RULES = (
    ("session trailer", re.compile(r"^\s*(?:claude|assistant|agent|ai)[-_ ]?session(?:[-_ ]?(?:id|url|link))?\s*:", re.I)),
    ("private session link", re.compile(
        r"claude\.ai/(?:chat|share|c|new)/[\w-]{6,}"
        r"|claude\.ai/code/(?:sessions?|cse)[_/][\w-]+"
        r"|claude\.ai/code/[\w-]*?(?:session|cse)_[\w-]+"
        r"|chatgpt\.com/(?:c|share)/[0-9a-f][0-9a-f-]{7,}"
        r"|gemini\.google\.com/(?:app|share)/[0-9a-f]{8,}", re.I)),
    ("session identifier", re.compile(r"\b(?:session|cse)_(?=[A-Za-z]*\d)[0-9A-Za-z]{16,}\b")),
)


def scan(text, source, strip_comments=False):
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        if strip_comments and line.startswith("#"):
            continue
        for category, pattern in RULES:
            if pattern.search(line):
                findings.append((source, number, category))
    return findings


def git(*args):
    p = subprocess.run(["git", *args], capture_output=True, text=True, errors="replace")
    if p.returncode != 0:
        raise SystemExit("error: git %s failed: %s" % (args[0], p.stderr.strip().splitlines()[-1] if p.stderr.strip() else p.returncode))
    return p.stdout


def commit_messages(rev_range):
    out = git("log", "--format=%H%x00%B%x01", rev_range)
    for record in out.split("\x01"):
        record = record.strip("\n")
        if record:
            sha, _, body = record.partition("\x00")
            yield "commit " + sha[:12], body


def tag_message(tag):
    if git("cat-file", "-t", tag).strip() != "tag":
        raise SystemExit("error: %s is not an annotated tag" % tag)
    body = git("cat-file", "-p", tag)
    return "tag " + tag, body.split("\n\n", 1)[1] if "\n\n" in body else ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commits", metavar="RANGE", help="check every commit message in a git revision range")
    ap.add_argument("--tag", action="append", default=[], metavar="TAG", help="check an annotated tag's message")
    ap.add_argument("--message-file", nargs="+", default=[], metavar="FILE", help="check these text files (commit message, PR text, release notes)")
    ap.add_argument("--strip-comments", action="store_true", help="ignore lines starting with # (the commit-msg hook's template)")
    args = ap.parse_args(argv)
    if not (args.commits or args.tag or args.message_file):
        ap.error("nothing to check")
    findings, checked = [], 0
    try:
        if args.commits:
            for source, body in commit_messages(args.commits):
                findings += scan(body, source)
                checked += 1
        for tag in args.tag:
            source, body = tag_message(tag)
            findings += scan(body, source)
            checked += 1
        for path in args.message_file:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    findings += scan(f.read(), path, args.strip_comments)
            except OSError as e:
                raise SystemExit("error: cannot read %s: %s" % (path, e.strerror))
            checked += 1
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 2
        raise
    for source, number, category in findings:
        print("FAIL: %s, line %d: %s (text withheld)" % (source, number, category))
    if findings:
        print("%d finding(s). Remove private AI session links, trailers and identifiers; describe the technical change instead. "
              "AI assistance may be disclosed without linking a private session (see AGENTS.md)." % len(findings))
        return 1
    print("ok: no private session metadata in %d item(s)" % checked)
    return 0


if __name__ == "__main__":
    sys.exit(main())
