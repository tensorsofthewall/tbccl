"""The wordmark lockups are portable, accessible, unclipped SVG built from the unchanged TBCCL mark, and the README and docs page that show them point at files that exist.

No font and no network is needed: the checks read the committed SVG files and MANIFEST.json. The same file is used by all four TBCCL repositories.
"""
import hashlib
import json
import re
import subprocess
import sys
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCKUPS = ROOT / "branding" / "lockups"
MANIFEST = json.loads((LOCKUPS / "MANIFEST.json").read_text())
PROJECTS = list(MANIFEST["projects"])
SVG_NS = "{http://www.w3.org/2000/svg}"
LAYOUTS = ("horizontal", "compact", "stacked")
THEMES = ("light", "dark", "mono-black", "mono-white")
MARK_SRC = (ROOT / "docs" / "_static" / "brand" / "tbccl-mark.svg").read_text()
STRANDS = re.findall(r'<path d="([^"]+)" fill="url', MARK_SRC)


def lockup_files():
    return sorted(p for p in LOCKUPS.glob("*.svg"))


class LockupFiles(unittest.TestCase):
    def test_expected_set(self):
        want = {f"{p}-{l}-{t}.svg" for p in PROJECTS for l in LAYOUTS for t in THEMES}
        self.assertEqual({p.name for p in lockup_files()}, want)

    def test_manifest_hashes_match_files(self):
        self.assertFalse(MANIFEST["typeface"]["embedded"])
        self.assertFalse(MANIFEST["mark"]["modified"])
        for name, digest in MANIFEST["files"].items():
            data = (LOCKUPS / name).read_bytes()
            self.assertEqual(hashlib.sha256(data).hexdigest(), digest, f"{name} changed without regenerating the manifest")

    def test_mark_is_the_approved_artwork(self):
        self.assertEqual(len(STRANDS), 5)
        for f in lockup_files():
            text = f.read_text()
            for d in STRANDS:
                self.assertIn(d, text, f"{f.name}: strand path differs from docs/_static/brand/tbccl-mark.svg")

    def test_portable_and_accessible(self):
        for f in lockup_files():
            root = ET.fromstring(f.read_text())
            self.assertEqual(root.tag, SVG_NS + "svg", f.name)
            self.assertEqual(root.get("role"), "img", f.name)
            project = next(p for p in sorted(PROJECTS, key=len, reverse=True) if f.name.startswith(p + "-"))
            self.assertEqual(root.find(SVG_NS + "title").text, project, f.name)
            self.assertGreater(len(root.find(SVG_NS + "desc").text), 20, f.name)
            tags = {e.tag.replace(SVG_NS, "") for e in root.iter()}
            self.assertFalse(tags & {"text", "image", "script", "style", "use", "foreignObject", "font"}, f"{f.name}: {tags}")
            raw = f.read_text()
            self.assertNotIn("font-family", raw, f.name)
            self.assertNotIn("href", raw, f.name)
            self.assertFalse(re.findall(r"https?://(?!www\.w3\.org/2000/svg)", raw), f.name)
            x0, y0, w, h = (float(v) for v in root.get("viewBox").split())
            self.assertEqual((x0, y0), (0, 0), f.name)
            self.assertLess(abs(float(root.get("width")) - w), 1, f.name)
            self.assertLess(abs(float(root.get("height")) - h), 1, f.name)
            self.assertLess(len(raw), 12000, f"{f.name} is unexpectedly large")

    def test_connector_uses_the_project_accent(self):
        for project, info in MANIFEST["projects"].items():
            if project == "tbccl":
                continue
            for layout in LAYOUTS:
                for theme, key in (("light", "light"), ("dark", "dark")):
                    raw = (LOCKUPS / f"{project}-{layout}-{theme}.svg").read_text()
                    self.assertIn(f'stroke="{info["accent"][key]}"', raw, f"{project} {layout} {theme}")
                for theme, colour in (("mono-black", "#0F172A"), ("mono-white", "#FFFFFF")):
                    raw = (LOCKUPS / f"{project}-{layout}-{theme}.svg").read_text()
                    self.assertIn(f'stroke="{colour}"', raw)
                    for a in info["accent"].values():
                        self.assertNotIn(a, raw, f"{project} {layout} {theme}: accent colour in a monochrome file")

    def test_renders_unclipped(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed")
        import io
        import shutil

        if not shutil.which("rsvg-convert"):
            self.skipTest("rsvg-convert is not installed")
        for f in lockup_files():
            png = subprocess.run(["rsvg-convert", "-z", "4", str(f)], capture_output=True, check=True).stdout
            im = Image.open(io.BytesIO(png)).convert("RGBA")
            box = im.split()[3].point(lambda v: 255 if v > 8 else 0).getbbox()
            self.assertIsNotNone(box, f"{f.name} renders empty")
            w, h = im.size
            margins = (box[0], box[1], w - box[2], h - box[3])
            self.assertGreater(min(margins), 0, f"{f.name} is clipped at the edge {margins}")
            self.assertLess(max(margins) / 4, 12, f"{f.name} has excess whitespace {margins}")


class Placement(unittest.TestCase):
    def test_docs_copies_match(self):
        docs = ROOT / "docs" / "_static" / "brand"
        own = [p for p in PROJECTS if (docs / f"{p}-horizontal-light.svg").exists() or (docs / f"{p}-horizontal-dark.svg").exists()]
        self.assertTrue(own, "the documentation needs its own project's horizontal lockups in docs/_static/brand")
        for p in own:
            for theme in ("light", "dark"):
                a, b = docs / f"{p}-horizontal-{theme}.svg", LOCKUPS / f"{p}-horizontal-{theme}.svg"
                self.assertTrue(a.is_file(), f"{a} is missing")
                self.assertEqual(a.read_bytes(), b.read_bytes(), f"{a.name} differs from branding/lockups")

    def test_readme_header_files_exist(self):
        readme = (ROOT / "README.md").read_text()
        refs = re.findall(r'(?:src|srcset)="(branding/lockups/[^"]+)"', readme)
        self.assertGreaterEqual(len(refs), 2, "README header should offer a light and a dark lockup")
        for r in refs:
            self.assertTrue((ROOT / r).is_file(), r)
        self.assertIn(f'alt="{PROJECTS[0]}"', readme)

    def test_independence_statement(self):
        page = (ROOT / "docs" / "project" / "branding.md").read_text()
        readme = (ROOT / "README.md").read_text()
        for text in (page, readme):
            self.assertIn("not affiliated with, sponsored by or endorsed by", text)
            self.assertIn("trademarks of their respective owners", text)
        self.assertNotRegex(page + readme, r"(?i)\bofficial (pytorch|vllm|exo)\b|\bpartner of\b|\bcertified\b")
        index = (ROOT / "docs" / "project" / "index.md").read_text()
        self.assertIn("branding", index)


class Regeneration(unittest.TestCase):
    def test_matches_a_fresh_build(self):
        tool = ROOT / "branding" / "tools" / "build_lockups.py"
        if not tool.is_file():
            self.skipTest("the generator lives in the tbccl repository")
        try:
            import fontTools  # noqa: F401
            import uharfbuzz  # noqa: F401
        except ImportError:
            self.skipTest("uharfbuzz and fonttools are not installed")
        r = subprocess.run([sys.executable, str(tool), "--check", "--editable"], capture_output=True, text=True)
        if "not found in" in r.stderr:
            self.skipTest("the Inter font file is not available here")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
