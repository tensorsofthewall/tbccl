"""Regression tests of tools/check_release_docs.py. The file is kept identical in the four TBCCL release repositories.

Each test copies the repository's own README, docs, compatibility.json and tools/release_docs.json into a temporary directory, introduces one inconsistency (or one
legitimate old reference) and runs the checker on the copy. The first scenario is the TBCCL v0.6.1 failure: the version was bumped everywhere except the README, the
homepage and the installation commands. Nothing here needs the network, a release, an artifact or a framework."""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECKER = ROOT / "tools" / "check_release_docs.py"
os.environ.pop("GITHUB_ACTIONS", None)
os.environ.pop("GITHUB_STEP_SUMMARY", None)

MANIFEST = json.loads((ROOT / "compatibility.json").read_text())
CONFIG = json.loads((ROOT / "tools" / "release_docs.json").read_text())
PKG = MANIFEST["package"]
CURRENT = PKG.get("public_version") or PKG["development_version"]
MAJOR, MINOR, PATCH = (int(x) for x in CURRENT.split("rc")[0].split("."))
PREVIOUS = f"{MAJOR}.{MINOR}.{PATCH - 1}" if PATCH else f"{MAJOR}.{MINOR - 1}.0"
IS_EXO = "fork_extra_wheel" in CONFIG.get("versions", {})


class ReleaseDocs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("README.md", "MAINTENANCE.md", "RELEASE.md", "CHANGELOG.md", "compatibility.json"):
            if (ROOT / name).exists():
                shutil.copy(ROOT / name, self.root / name)
        shutil.copytree(ROOT / "docs", self.root / "docs", ignore=shutil.ignore_patterns("_build", "_static", "_templates", "*.png", "*.svg"))
        (self.root / "tools").mkdir()
        shutil.copy(ROOT / "tools" / "release_docs.json", self.root / "tools" / "release_docs.json")

    def run_checker(self, *args):
        p = subprocess.run([sys.executable, str(CHECKER), "--root", str(self.root), *args], capture_output=True, text=True)
        return p.returncode, p.stdout + p.stderr

    def edit(self, rel, old, new, count=None):
        path = self.root / rel
        text = path.read_text()
        self.assertIn(old, text, f"{rel} does not contain {old!r}; the fixture is out of date")
        path.write_text(text.replace(old, new) if count is None else text.replace(old, new, count))

    def assertFails(self, needle, *args):
        code, out = self.run_checker(*args)
        self.assertEqual(code, 1, out)
        self.assertIn(needle, out)
        return out

    # The documentation as it is on this branch.
    def test_the_repository_documentation_passes(self):
        code, out = self.run_checker()
        self.assertEqual(code, 0, out)
        self.assertIn(f"{CURRENT}", out)

    def test_intended_tag_of_the_current_version_passes(self):
        code, out = self.run_checker("--tag", f"v{CURRENT}")
        self.assertEqual(code, 0, out)

    # The v0.6.1 failure: version and manifest are new, the text around the install commands is old.
    def test_readme_status_naming_the_previous_release_fails(self):
        status = next(s for s in CONFIG["statements"] if s["describe"] == "README status line")
        text = (self.root / "README.md").read_text()
        m = re.search(status["find"], text, re.M)
        self.edit("README.md", m.group(0), m.group(0).replace(CURRENT, PREVIOUS))
        out = self.assertFails(f"README.md:{text[:m.start('v')].count(chr(10)) + 1}: README status line says {PREVIOUS}; expected {CURRENT}")
        self.assertIn("users read this statement", out)
        self.assertFails(f"says {PREVIOUS}; expected {CURRENT}", "--tag", f"v{CURRENT}")

    def test_homepage_naming_the_previous_release_fails(self):
        text = (self.root / "docs" / "index.md").read_text()
        m = re.search(rf"The current release is {re.escape(CURRENT)}", text)
        self.assertTrue(m)
        self.edit("docs/index.md", m.group(0), m.group(0).replace(CURRENT, PREVIOUS))
        code, out = self.run_checker()
        self.assertEqual(code, 1, out)
        self.assertRegex(out, rf"docs/index\.md:\d+: documentation homepage release statement says {re.escape(PREVIOUS)}; expected {re.escape(CURRENT)}")

    def test_installation_url_of_the_previous_release_fails(self):
        repo = re.escape(CONFIG["repo"])
        for rel in ("README.md", "docs/getting-started/install.md"):
            text = (self.root / rel).read_text()
            m = re.search(rf"github\.com/{repo}/releases/(?:download|tag)/v{re.escape(CURRENT)}", text)
            if not m:
                continue
            with self.subTest(rel):
                (self.root / rel).write_text(text.replace(m.group(0), m.group(0).replace(CURRENT, PREVIOUS), 1))
                self.assertFails(f"{rel}:", )
                self.assertFails("download URL says " + PREVIOUS)
                (self.root / rel).write_text(text)
                return
        self.fail("no release URL in the README or the installation page")

    def test_artifact_name_with_the_wrong_version_fails(self):
        for template in MANIFEST["release_assets"]:
            right, wrong = template.replace("{version}", CURRENT), template.replace("{version}", PREVIOUS)
            for rel in ("README.md", "docs/getting-started/install.md"):
                text = (self.root / rel).read_text()
                if right in text:
                    (self.root / rel).write_text(text.replace(right, wrong, 1))
                    self.assertFails(f"archive or wheel name says {PREVIOUS}; expected {CURRENT}")
                    return
        self.fail("no full artifact name in the README or the installation page")

    def test_a_removed_release_statement_is_reported_instead_of_ignored(self):
        self.edit("docs/index.md", "The current release is", "The latest release is")
        self.assertFails("documentation homepage release statement")

    # Old versions that are legitimate.
    def test_historical_changelog_sections_pass(self):
        template = MANIFEST["release_assets"][0]
        old = template.replace("{version}", PREVIOUS)
        changelog = self.root / "CHANGELOG.md"
        changelog.write_text(changelog.read_text() + f"\n## {PREVIOUS}\n\nThe archive was named {old}; see https://github.com/{CONFIG['repo']}/releases/tag/v{PREVIOUS}\n")
        page = self.root / "docs" / "project" / "changelog.md"
        if page.exists():
            page.write_text(page.read_text() + f"\n## {PREVIOUS}\n\n{old}\n")
        code, out = self.run_checker()
        self.assertEqual(code, 0, out)

    def test_benchmark_and_validation_records_of_an_older_build_pass(self):
        old = MANIFEST["release_assets"][0].replace("{version}", PREVIOUS)
        record = self.root / "docs" / "validation" / "0.0.0.md"
        record.parent.mkdir(exist_ok=True)
        record.write_text(f"Measured with {old} (version {PREVIOUS}) on one machine pair.\n")
        code, out = self.run_checker()
        self.assertEqual(code, 0, out)

    def test_a_declared_historical_reference_in_a_scanned_page_passes_and_an_undeclared_one_fails(self):
        old = MANIFEST["release_assets"][0].replace("{version}", PREVIOUS)
        page = self.root / "docs" / "project" / "history-note.md"
        page.write_text(f"# Note\n\nThe superseded build {old} had a defect.\n")
        self.assertFails(f"docs/project/history-note.md:3: archive or wheel name says {PREVIOUS}")
        CONFIG_COPY = json.loads((self.root / "tools" / "release_docs.json").read_text())
        CONFIG_COPY["allow"].append({"file": "docs/project/history-note.md", "line": "superseded build", "versions": [PREVIOUS], "why": "a defective build is named on purpose"})
        (self.root / "tools" / "release_docs.json").write_text(json.dumps(CONFIG_COPY))
        code, out = self.run_checker()
        self.assertEqual(code, 0, out)

    # The release tag.
    def test_a_tag_that_disagrees_with_the_package_version_fails(self):
        self.assertFails(f"tag v{PREVIOUS} does not match the package version {CURRENT}", "--tag", f"v{PREVIOUS}")
        self.assertFails("does not match the package version", "--tag", f"v{MAJOR}.{MINOR}.{PATCH + 1}")

    def test_a_release_candidate_tag_of_a_stable_package_fails(self):
        self.assertFails("does not match the package version", "--tag", f"v{CURRENT}rc1")

    def test_a_version_bump_without_the_documentation_is_what_the_check_is_for(self):
        """The manifest moves to the next patch version; every page still describes the current one."""
        manifest = json.loads((self.root / "compatibility.json").read_text())
        nxt = f"{MAJOR}.{MINOR}.{PATCH + 1}"
        manifest["package"]["development_version"] = nxt
        manifest["package"]["target_release"] = nxt
        if "public_version" in manifest["package"]:
            manifest["package"]["public_version"] = nxt
        manifest["package"]["released"] = False
        (self.root / "compatibility.json").write_text(json.dumps(manifest))
        out = self.assertFails(f"says {CURRENT}; expected {nxt}", "--tag", f"v{nxt}")
        self.assertNotIn("INTEGRITY FAILURE: tag", out)

    # The adapters bundle a TBCCL whose version is recorded and cross-checked against the TBCCL checkout.
    @unittest.skipUnless("bundled_tbccl" in CONFIG.get("versions", {}), "TBCCL itself bundles nothing")
    def test_bundled_tbccl_version_must_match_the_tbccl_checkout(self):
        core = self.root / "core"
        core.mkdir()
        bundled = CONFIG["versions"]["bundled_tbccl"]
        (core / "compatibility.json").write_text(json.dumps({"project": "tbccl", "package": {"development_version": bundled, "public_version": bundled}}))
        code, out = self.run_checker("--tbccl-root", str(core))
        self.assertEqual(code, 0, out)
        (core / "compatibility.json").write_text(json.dumps({"project": "tbccl", "package": {"development_version": "9.9.9", "public_version": "9.9.9"}}))
        self.assertFails(f"bundles TBCCL {bundled} but the TBCCL checkout is 9.9.9", "--tbccl-root", str(core))

    @unittest.skipUnless("bundled_tbccl" in CONFIG.get("versions", {}), "TBCCL itself bundles nothing")
    def test_the_bundled_tbccl_version_in_the_expected_output_must_be_the_recorded_one(self):
        bundled = CONFIG["versions"]["bundled_tbccl"]
        statement = next(s for s in CONFIG["statements"] if s["expect"] == "version:bundled_tbccl")
        text = (self.root / statement["file"]).read_text()
        m = re.search(statement["find"], text, re.M)
        self.edit(statement["file"], m.group(0), m.group(0).replace(bundled, "0.0.1"))
        self.assertFails(f"says 0.0.1; expected {bundled}")

    # exo-tbccl: the current wheel and the wheel that the immutable fork tag installs are different on purpose.
    @unittest.skipUnless(IS_EXO, "only exo-tbccl has a pinned older wheel")
    def test_the_fork_installing_the_older_wheel_is_accepted(self):
        pinned = CONFIG["versions"]["fork_extra_wheel"]
        self.assertNotEqual(pinned, CURRENT)
        self.assertIn(pinned, (self.root / "docs" / "getting-started" / "install.md").read_text())
        code, out = self.run_checker("--tag", f"v{CURRENT}")
        self.assertEqual(code, 0, out)

    @unittest.skipUnless(IS_EXO, "only exo-tbccl has a pinned older wheel")
    def test_confusing_the_current_wheel_with_the_fork_installed_wheel_fails(self):
        pinned, current = CONFIG["versions"]["fork_extra_wheel"], CURRENT
        # The tbccl extra is said to install the current wheel, which the immutable tag cannot do.
        self.edit("README.md", f"The `tbccl` extra installs the {pinned} release wheel", f"The `tbccl` extra installs the {current} release wheel")
        self.assertFails(f"says {current}; expected {pinned}")

    @unittest.skipUnless(IS_EXO, "only exo-tbccl has a pinned older wheel")
    def test_the_standalone_wheel_shown_with_the_older_version_fails(self):
        pinned, current = CONFIG["versions"]["fork_extra_wheel"], CURRENT
        self.edit("docs/getting-started/install.md", f"# {current} (True, '') for the {current} wheel", f"# {pinned} (True, '') for the {current} wheel")
        self.assertFails(f"says {pinned}; expected {current}")

    @unittest.skipUnless(IS_EXO, "only exo-tbccl has a pinned older wheel")
    def test_a_manual_wheel_command_with_the_older_wheel_fails(self):
        pinned, current = CONFIG["versions"]["fork_extra_wheel"], CURRENT
        self.edit("README.md", f"exo_tbccl-{current}-cp313-cp313-manylinux", f"exo_tbccl-{pinned}-cp313-cp313-manylinux", 1)
        self.assertFails(f"archive or wheel name says {pinned}; expected {current}")


if __name__ == "__main__":
    unittest.main()
