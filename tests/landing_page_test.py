"""The documentation landing page keeps its structure, its ecosystem navigation and its claims.

Static checks of docs/index.md, docs/_static/custom.css and docs/conf.py; no build, no network. The same file is used by all four TBCCL repositories.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
INDEX = (DOCS / "index.md").read_text()
CSS = (DOCS / "_static" / "custom.css").read_text()
CONF = (DOCS / "conf.py").read_text()
SITE = re.search(r'^project = "([^"]+)"', CONF, re.M).group(1).lower()
SITES = ["tbccl", "torch-tbccl", "vllm-tbccl", "exo-tbccl"]
DOCS_URL = "https://{}.tensorsofthewall.com/en/latest/"
GITHUB = "https://github.com/tensorsofthewall/{}"


def nav_block():
    m = re.search(r'<nav class="tb-eco"[^>]*>(.*?)</nav>', INDEX, re.S)
    assert m, "the ecosystem navigation is missing"
    return m.group(0)


class Structure(unittest.TestCase):
    def test_one_h1_named_after_the_project(self):
        h1 = re.findall(r"^# (.+)$", INDEX, re.M)
        self.assertEqual([h.lower() for h in h1], [SITE])

    def test_heading_levels_do_not_skip(self):
        levels = []
        for line in INDEX.splitlines():
            m = re.match(r"^(#{1,6}) ", line) or re.match(r"^<h([1-6])>", line)
            if m:
                levels.append(len(m.group(1)) if line.startswith("#") else int(m.group(1)))
        for a, b in zip(levels, levels[1:]):
            self.assertLessEqual(b - a, 1, f"heading level jumps from {a} to {b}")

    def test_two_actions_in_the_hero(self):
        m = re.search(r":::\{container\} tb-actions\n(.*?)\n:::", INDEX, re.S)
        self.assertTrue(m, "no tb-actions block")
        self.assertEqual(len(re.findall(r"\]\(", m.group(1))), 2)

    def test_release_note_is_kept(self):
        self.assertIn("**Release.** The current release is", INDEX)

    def test_toctrees_are_kept(self):
        blocks = INDEX.split("```{toctree}")[1:]
        self.assertGreaterEqual(len(blocks), 5)
        for block in blocks:
            for line in block.split("```")[0].splitlines():
                name = line.strip()
                if name and not name.startswith(":"):
                    self.assertTrue((DOCS / f"{name}.md").exists() or (DOCS / name / "index.md").exists(), name)

    def test_toctrees_are_hidden_on_the_page(self):
        """Furo lists the tree in the sidebar; repeating it in the article only buries the landing page."""
        for block in INDEX.split("```{toctree}")[1:]:
            self.assertIn(":hidden:", block.split("```")[0])

    def test_internal_links_resolve(self):
        for target in re.findall(r"\]\(([^)#\s]+\.md)(?:#[^)]*)?\)", INDEX):
            if target.startswith("http"):
                continue
            self.assertTrue((DOCS / target).is_file(), f"docs/index.md links to a missing page: {target}")

    def test_no_script_style_or_remote_resource(self):
        self.assertNotRegex(INDEX, r"<(script|link|style|iframe|img)\b")
        self.assertNotRegex(INDEX, r"(src|srcset)=")


class EcosystemNavigation(unittest.TestCase):
    def test_lists_all_four_sites_and_marks_this_one(self):
        nav = nav_block()
        self.assertEqual(nav.count("<li"), 4)
        self.assertEqual(nav.count('aria-current="page"'), 1)
        for site in SITES:
            item = re.search(rf'<li class="tb-\w+"( aria-current="page")?>(?:(?!</li>).)*<span class="tb-eco-name">{re.escape(site)}</span>.*?</li>', nav, re.S)
            self.assertTrue(item, f"{site} is missing from the navigation")
            if site == SITE:
                self.assertTrue(item.group(1), "the current site is not marked")
                self.assertNotIn("<a ", item.group(0))
            else:
                self.assertIn(f'href="{DOCS_URL.format(site)}"', item.group(0))

    def test_has_an_accessible_name(self):
        self.assertRegex(nav_block(), r'<nav class="tb-eco" aria-label="[^"]{10,}"')

    def test_every_project_is_linked_with_docs_and_source_somewhere(self):
        if SITE != "tbccl":
            self.skipTest("the card overview lives on the TBCCL site")
        for site in SITES:
            self.assertIn(GITHUB.format(site), INDEX)
            if site != "tbccl":
                self.assertIn(DOCS_URL.format(site), INDEX)


class Claims(unittest.TestCase):
    def test_rdma_and_dma_are_only_mentioned_as_not_supported(self):
        unsupported = INDEX.find("**Not supported or not claimed**")
        end = INDEX.find("\n## ", unsupported) if unsupported >= 0 else -1
        for m in re.finditer(r"\bR?DMA\b", INDEX):
            before, after = INDEX[max(0, m.start() - 45):m.start()], INDEX[m.end():m.end() + 30]
            in_unsupported = unsupported >= 0 and unsupported < m.start() < (end if end > 0 else len(INDEX))
            ok = re.search(r"\b(not|no|never|neither)\b[^.]*$", before, re.I) or re.search(r"does not|cannot", after) or in_unsupported
            self.assertTrue(ok, f"RDMA or DMA mentioned without a negation: ...{before}{m.group(0)}{after}")

    def test_validated_is_separated_from_unsupported(self):
        self.assertRegex(INDEX, r"\*\*(Demonstrated|Validated)\*\*")
        self.assertIn("**Not supported or not claimed**", INDEX)

    def test_authoritative_pages_are_linked(self):
        self.assertRegex(INDEX, r"validated[ -]configurations")
        self.assertRegex(INDEX, r"limitations")


class Figure(unittest.TestCase):
    def test_svg_is_accessible(self):
        svgs = re.findall(r"<svg\b.*?</svg>", INDEX, re.S)
        if SITE != "tbccl":
            self.assertFalse(svgs)
            return
        self.assertEqual(len(svgs), 2, "a wide and a narrow variant")
        for svg in svgs:
            self.assertIn('role="img"', svg)
            ids = re.search(r'aria-labelledby="(\S+) (\S+)"', svg)
            self.assertTrue(ids)
            for i in ids.groups():
                self.assertIn(f'id="{i}"', svg)
            self.assertNotIn("<image", svg)
            desc = re.search(r"<desc[^>]*>(.*?)</desc>", svg, re.S).group(1)
            self.assertRegex(desc, r"C and C\+\+ programs call the runtime[^.]*directly", "C and C++ programs must be shown using the runtime without an adapter")
            self.assertNotRegex(svg, r"#[0-9a-fA-F]{3,8}\b", "colours come from the page, not from the figure")


class Styles(unittest.TestCase):
    def test_every_class_used_is_defined(self):
        used = set()
        for value in re.findall(r'class="([^"]+)"', INDEX):
            used |= {c for c in value.split() if c.startswith("tb-")}
        for value in re.findall(r"\{container\} ([^\n]+)", INDEX):
            used |= {c for c in value.split() if c.startswith("tb-")}
        self.assertTrue(used)
        for cls in used:
            self.assertRegex(CSS, rf"\.{re.escape(cls)}\b", f"{cls} has no CSS rule")

    def test_landing_rules_name_no_colours(self):
        rules = CSS[CSS.index("/* Landing pages"):]
        self.assertNotRegex(rules, r"#[0-9a-fA-F]{3,8}\b")
        self.assertNotRegex(rules, r"\brgba?\(")

    def test_accent_tokens_exist_for_all_projects(self):
        for key in ("tbccl", "torch", "vllm", "exo"):
            self.assertIn(f'"{key}":', CONF.split("_ECO")[1].split("\n")[0])
        self.assertIn("tb-on-accent", CONF)

    def test_focus_ring_covers_links(self):
        self.assertIn("a:focus-visible", CSS)


if __name__ == "__main__":
    unittest.main()
