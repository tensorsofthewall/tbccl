"""The README keeps the facts it owes its reader: lockup, links, release assets, versions, pins and entry points.

Static; no network. The same file is used by all four TBCCL repositories (it is `test_readme.py` in the adapters).
"""
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_readme as cr  # noqa: E402

README = (ROOT / "README.md").read_text(encoding="utf-8")
COMPAT = json.loads((ROOT / "compatibility.json").read_text(encoding="utf-8"))


class Fixture:
    """A scratch repository: the real README (optionally edited), the real manifest, and an empty file for every relative target."""

    def __init__(self, readme=None, compat=None):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        text = readme if readme is not None else README
        (self.root / "README.md").write_text(text, encoding="utf-8")
        (self.root / "compatibility.json").write_text(json.dumps(compat if compat is not None else COMPAT), encoding="utf-8")
        for t in set(cr.relative_targets(README)):
            p = self.root / t.split("#")[0]
            if not p.suffix or p.is_dir() or not (ROOT / t.split("#")[0]).is_file():
                p.mkdir(parents=True, exist_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / t.split("#")[0], p)
        for page in ("docs/reference/compatibility.md", "docs/reference/compatibility-matrix.md"):
            if (ROOT / page).is_file():
                (self.root / page).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / page, self.root / page)

    def problems(self):
        return cr.problems(self.root)

    def close(self):
        self.dir.cleanup()


class ReadmeIntegrity(unittest.TestCase):
    def test_the_real_readme_is_consistent(self):
        self.assertEqual(cr.problems(), [])

    def test_the_fixture_reproduces_the_real_result(self):
        f = Fixture()
        self.addCleanup(f.close)
        self.assertEqual(f.problems(), [])

    def expect(self, mutate, message, compat=None):
        f = Fixture(readme=mutate(README) if mutate else None, compat=compat)
        self.addCleanup(f.close)
        found = f.problems()
        self.assertTrue(any(re.search(message, p) for p in found), f"{message!r} not reported: {found}")

    def test_a_missing_lockup_file_is_caught(self):
        f = Fixture()
        self.addCleanup(f.close)
        next(f.root.glob("branding/lockups/*-dark.svg")).unlink()
        self.assertTrue(any("does not exist" in p for p in f.problems()))

    def test_a_wrong_alt_text_is_caught(self):
        self.expect(lambda t: re.sub(r'alt="[^"]+"', 'alt="wrong"', t, count=1), "alt text")

    def test_a_broken_relative_link_is_caught(self):
        self.expect(lambda t: t + "\n[gone](docs/does/not/exist.md)\n", "does not exist")

    def test_a_stale_release_url_is_caught(self):
        v = (COMPAT["package"].get("public_version") or COMPAT["package"]["development_version"])
        major, minor, patch = v.split("rc")[0].split(".")
        stale = f"{major}.{minor}.{int(patch) + 5}"
        self.expect(lambda t: re.sub(r"(releases/download/v)" + re.escape(v), r"\g<1>" + stale, t), "names v" + re.escape(stale))

    def test_an_unknown_asset_name_is_caught(self):
        pat = r"\b((?:tbccl|torch_tbccl|vllm_tbccl|exo_tbccl)-\d+\.\d+\.\d+(?:rc\d+)?)-[\w.-]+\.(?:whl|tar\.gz)"
        m = re.search(pat, README)
        self.assertTrue(m, "the README names no release asset")
        self.expect(lambda t: t.replace(m.group(0), m.group(1) + "-plan9-surprise.whl", 1), "release_assets")

    def test_a_missing_version_is_caught(self):
        v = COMPAT["package"].get("public_version") or COMPAT["package"]["development_version"]
        self.expect(lambda t: t.replace(f"**{v}**", "**a release**"), "status line")

    def test_an_unrecorded_integration_tag_is_caught(self):
        self.expect(lambda t: t + "\nSee tbccl-integration-v9 and v9.9.9-tbccl9.\n", "neither compatibility.json nor the compatibility page")

    def test_a_missing_entry_point_is_caught(self):
        token = cr.REQUIRED_TOKENS[COMPAT["project"]][0]
        self.expect(lambda t: t.replace(token, "REDACTED"), "no longer mentions")

    def test_missing_required_links_and_sections_are_caught(self):
        link = cr.REQUIRED_LINKS[COMPAT["project"]][0]
        self.expect(lambda t: t.replace(link, "elsewhere"), "does not link to")
        self.expect(lambda t: t.replace("## Independence", "## Notes"), "License or Independence")

    def test_internal_wording_is_caught(self):
        self.expect(lambda t: t + "\nPhase B artwork, see .local/plans/x.md\n", "internal wording")

    def test_unknown_site_and_markdown_sources_are_caught(self):
        self.expect(lambda t: t + "\nhttps://other-tbccl.tensorsofthewall.com/en/latest/x.html\n", "unknown documentation site")
        self.expect(lambda t: t + "\nhttps://tbccl.tensorsofthewall.com/en/latest/x.md\n", "Markdown source")

    def test_protocol_values_must_match_the_manifest(self):
        compat = json.loads(json.dumps(COMPAT))
        holder = compat.get("wire_protocol") and compat["wire_protocol"] or None
        req = compat.get("requires_tbccl") or (compat.get("requires") or {}).get("tbccl") or {}
        if holder:
            holder["version"] = 9
        elif "wire_protocol" in req:
            req["wire_protocol"] = [9]
        else:
            self.skipTest("this manifest records no wire protocol; only the C ABI is checked")
        self.expect(None, "wire protocol 9", compat=compat)

    def test_checks_do_not_pin_prose_or_heading_order(self):
        # reorder two sections and reword a sentence: nothing may be reported
        t = README.replace("## Development", "## Zeta", 1).replace("## Limitations", "## Development", 1).replace("## Zeta", "## Limitations", 1)
        f = Fixture(readme=t)
        self.addCleanup(f.close)
        self.assertEqual(f.problems(), [])


if __name__ == "__main__":
    unittest.main()
