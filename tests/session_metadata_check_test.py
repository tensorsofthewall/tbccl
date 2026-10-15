#!/usr/bin/env python3
"""tools/check_session_metadata.py with synthetic identifiers only: what it rejects, what it must not reject, and that it never prints what it matched."""
import contextlib
import importlib.util
import io
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("check_session_metadata", ROOT / "tools" / "check_session_metadata.py")
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)

# Entirely made up: assembled at run time so no fake or real link is stored as a literal.
FAKE_ID = "session_" + "SYNTHETIC0123456789ZZ"
FAKE_CSE = "cse_" + "SYNTHETIC0123456789ZZ"
HOST = "claude" + ".ai"
FAKE_URL = "https://%s/code/%s" % (HOST, FAKE_ID)


def run(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = chk.main(argv)
    return rc, out.getvalue() + err.getvalue()


def check_text(text, strip=False):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "msg"
        p.write_text(text)
        return run(["--message-file", str(p)] + (["--strip-comments"] if strip else []))


class Rejected(unittest.TestCase):
    def test_trailer_variants(self):
        for line in ("Claude-Session: " + FAKE_URL, "claude-session: x", "CLAUDE-SESSION:x", "  Claude-Session :  x", "Claude_Session: x", "AI-Session: x", "Assistant-Session-Id: x"):
            self.assertEqual(check_text("Fix a thing\n\nBody.\n\n" + line + "\n")[0], 1, line)

    def test_private_links(self):
        for url in (FAKE_URL, "https://%s/code/%s" % (HOST, FAKE_CSE), "https://%s/chat/0a1b2c3d-4e5f" % HOST, "http://www.%s/share/abc123def456" % HOST,
                    "https://chatgpt" + ".com/c/0a1b2c3d-4e5f-6a7b", "https://gemini.google" + ".com/app/0a1b2c3d4e5f"):
            self.assertEqual(check_text("See %s for context\n" % url)[0], 1, url)

    def test_bare_identifiers(self):
        self.assertEqual(check_text("ref %s\n" % FAKE_ID)[0], 1)
        self.assertEqual(check_text("ref %s\n" % FAKE_CSE)[0], 1)

    def test_a_finding_never_echoes_what_it_matched(self):
        rc, text = check_text("Subject\n\nClaude-Session: %s\n%s\n" % (FAKE_URL, FAKE_ID))
        self.assertEqual(rc, 1)
        for secret in (FAKE_ID, FAKE_URL, "SYNTHETIC"):
            self.assertNotIn(secret, text)
        self.assertIn("session trailer", text)
        self.assertIn("private session link", text)


class Accepted(unittest.TestCase):
    def test_ordinary_and_conventional_messages(self):
        for msg in ("Fix CUDA staging\n\nCorrect the host-to-device path.\n",
                    "docs: describe the session lifetime\n\nA session is bound to one communicator.\n",
                    "tools: handle session_manager and session_timeout settings\n",
                    "Add retry\n\nSigned-off-by: A Person <a@example.org>\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>\n",
                    "Add login session handling\n\nSession: expired tokens are refreshed.\n"):
            self.assertEqual(check_text(msg)[0], 0, msg)

    def test_product_home_page_and_disclosure_are_allowed(self):
        msg = "Add feature\n\nGenerated with [Claude Code](https://%s/code)\nAI-assisted: yes, reviewed by the author.\n" % HOST
        self.assertEqual(check_text(msg)[0], 0)

    def test_documentation_about_the_feature_is_allowed(self):
        self.assertEqual(check_text("The hook rejects a Claude-Session trailer; it does not reject the word session.\n")[0], 0)

    def test_identifiers_need_a_digit_and_length(self):
        self.assertEqual(check_text("uses session_initialization_manager here\n")[0], 0)
        self.assertEqual(check_text("uses session_authenticationtoken here\n")[0], 0)
        self.assertEqual(check_text("uses session_" + "Z" * 5 + "1\n")[0], 0)

    def test_commit_msg_template_comments_are_stripped_only_with_the_flag(self):
        text = "Subject\n\n# see %s in the template\n" % FAKE_URL
        self.assertEqual(check_text(text, strip=True)[0], 0)
        self.assertEqual(check_text(text)[0], 1)


class GitIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.env = dict(os.environ, GIT_AUTHOR_NAME="T", GIT_AUTHOR_EMAIL="t@example.org", GIT_COMMITTER_NAME="T", GIT_COMMITTER_EMAIL="t@example.org",
                        GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
        self.git("init", "-q", "-b", "main")
        (self.dir / "a").write_text("a")
        self.git("add", "a")
        self.git("-c", "commit.gpgsign=false", "commit", "-q", "-m", "base")

    def tearDown(self):
        self.tmp.cleanup()

    def git(self, *a, check=True):
        return subprocess.run(["git", *a], cwd=self.dir, env=self.env, capture_output=True, text=True, check=check)

    def commit(self, msg):
        (self.dir / "a").write_text(msg)
        self.git("add", "a")
        return self.git("-c", "commit.gpgsign=false", "commit", "-q", "-m", msg, check=False)

    def run_in_repo(self, argv):
        cwd = os.getcwd()
        os.chdir(self.dir)
        try:
            return run(argv)
        finally:
            os.chdir(cwd)

    def test_range(self):
        self.commit("ordinary change")
        self.assertEqual(self.run_in_repo(["--commits", "HEAD~1..HEAD"])[0], 0)
        self.commit("change\n\nClaude-Session: " + FAKE_URL)
        rc, text = self.run_in_repo(["--commits", "HEAD~2..HEAD"])
        self.assertEqual(rc, 1)
        self.assertNotIn("SYNTHETIC", text)

    def test_annotated_tag_and_lightweight_tag(self):
        self.git("-c", "tag.gpgsign=false", "tag", "-a", "v1", "-m", "Release 1")
        self.git("-c", "tag.gpgsign=false", "tag", "-a", "v2", "-m", "Release 2\n\nSession: " + FAKE_ID + "\nClaude-Session: x")
        self.git("tag", "light")
        self.assertEqual(self.run_in_repo(["--tag", "v1"])[0], 0)
        self.assertEqual(self.run_in_repo(["--tag", "v2"])[0], 1)
        self.assertEqual(self.run_in_repo(["--tag", "light"])[0], 2)

    def test_commit_msg_hook_blocks_a_commit(self):
        hooks = self.dir / ".githooks"
        hooks.mkdir()
        hook = hooks / "commit-msg"
        src = (ROOT / ".githooks" / "commit-msg").read_text().replace('"$(git rev-parse --show-toplevel)/tools/check_session_metadata.py"', '"%s"' % (ROOT / "tools" / "check_session_metadata.py"))
        hook.write_text(src)
        hook.chmod(0o755)
        self.git("config", "core.hooksPath", ".githooks")
        bad = self.commit("change\n\nClaude-Session: " + FAKE_URL)
        self.assertNotEqual(bad.returncode, 0)
        self.assertNotIn("SYNTHETIC", bad.stdout + bad.stderr)
        self.assertEqual(self.commit("an ordinary change\n\nCo-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>").returncode, 0)


class Repository(unittest.TestCase):
    def test_no_literal_private_links_in_the_tracked_tests_and_tools(self):
        for rel in ("tools/check_session_metadata.py", "tests/session_metadata_check_test.py"):
            self.assertEqual(run(["--message-file", str(ROOT / rel)])[0], 0, rel)

    def test_agents_md_states_the_rule(self):
        text = (ROOT / "AGENTS.md").read_text()
        self.assertIn("session URLs", text)
        self.assertIn("check_session_metadata.py", text)


if __name__ == "__main__":
    unittest.main()
