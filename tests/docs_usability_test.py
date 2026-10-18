"""The onboarding, troubleshooting and contributor pages keep the structure that makes them usable, and the commands they name still exist.

Static checks of docs/ and the repository tree; no build, no network, no download, no hardware. The same file is used by all four TBCCL repositories.
"""
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import check_doc_commands as cdc  # noqa: E402

DOCS = ROOT / "docs"
CONF = (DOCS / "conf.py").read_text()
SITE = re.search(r'^project = "([^"]+)"', CONF, re.M).group(1).lower()
ADAPTER = SITE != "tbccl"


def read(rel):
    return (DOCS / rel).read_text(encoding="utf-8")


def slug(heading):
    """The anchor MyST gives a heading."""
    h = re.sub(r"[`*_]", "", heading).strip().lower()
    return re.sub(r"\s+", "-", re.sub(r"[^\w\s-]", "", h))


def headings(text):
    return {slug(m.group(2)) for m in re.finditer(r"^(#{1,6})\s+(.+?)\s*$", text, re.M)}


def section(text, title):
    m = re.search(rf"^(#+)\s+{re.escape(title)}\s*$", text, re.M)
    assert m, f"no section {title!r}"
    end = re.search(rf"^#{{1,{len(m.group(1))}}}\s", text[m.end():], re.M)
    return text[m.end(): m.end() + end.start()] if end else text[m.end():]


class CommandChecker(unittest.TestCase):
    def test_every_command_block_in_the_docs_is_sound(self):
        problems, inventory = [], []
        for p in cdc.markdown_files():
            a, b = cdc.check(p)
            problems += a
            inventory += b
        self.assertGreater(len(inventory), 20, "the checker found almost no command blocks")
        self.assertEqual(problems, [])

    def test_checker_catches_each_kind_of_fault(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "docs").mkdir()
            (root / "tools").mkdir()
            (root / "tools" / "real.py").write_text("")
            (root / "pkg").mkdir()
            (root / "pkg" / "__init__.py").write_text("")
            (root / "pkg" / "info.py").write_text("")
            (root / "Makefile").write_text("docs:\n\ttrue\n")
            page = root / "docs" / "p.md"
            cases = {
                "shell syntax": "```sh\nif then fi\n```\n",
                "does not exist": "```sh\npython3 tools/missing.py\n```\n",
                "does not resolve": "```sh\npython -m pkg.nothing\n```\n",
                "no such target": "```sh\nmake nothing\n```\n",
                "unfinished marker": "```sh\necho TODO\n```\n",
            }
            old = cdc.ROOT
            cdc.ROOT = root
            try:
                for expected, body in cases.items():
                    page.write_text(body)
                    problems, _ = cdc.check(page)
                    self.assertTrue(any(expected in p for p in problems), f"{expected!r} was not reported: {problems}")
                ok = "```sh\npython3 tools/real.py --port <PORT>\npython -m pkg.info\nmake docs\nls docs/_build/html\n```\n"
                page.write_text(ok)
                self.assertEqual(cdc.check(page)[0], [], "placeholders, real paths, modules, targets and generated output are fine")
                page.write_text("<!-- doccheck: skip (needs a second machine) -->\n```sh\npython3 tools/missing.py\n```\n")
                self.assertEqual(cdc.check(page)[0], [])
                page.write_text("<!-- doccheck: skip -->\n```sh\npython3 tools/missing.py\n```\n")
                self.assertTrue(cdc.check(page)[0], "a skip without a reason must not skip")
            finally:
                cdc.ROOT = old


class Troubleshooting(unittest.TestCase):
    PAGE = "guides/troubleshooting.md"

    def test_has_the_common_structure(self):
        t = read(self.PAGE)
        for title in ("Start here", "Work through it in this order", "Find your symptom", "Which layer owns the error", "When to ask for help"):
            self.assertIn(slug(title), headings(t), title)
        self.assertRegex(t, r"(?m)^## Ping works but")
        order = section(t, "Work through it in this order")
        steps = re.findall(r"(?m)^\| (\d) \|", order)
        self.assertEqual(steps, [str(i) for i in range(1, 9)], "eight diagnostic steps, in order")

    def test_every_in_page_link_in_start_here_resolves(self):
        t = read(self.PAGE)
        start = section(t, "Start here")
        anchors = headings(t)
        for target in re.findall(r"\]\(#([^)]+)\)", start):
            self.assertIn(target, anchors, f"the symptom index links to #{target}, which is not a heading")

    def test_symptoms_a_user_searches_for_are_headings_or_index_entries(self):
        t = read(self.PAGE).lower()
        for phrase in ("ping works", "wire protocol", "timeout"):
            self.assertIn(phrase, t)
        self.assertIn("protocol_mismatch", t)

    def test_ownership_names_upstream_and_the_operating_system(self):
        owner = section(read(self.PAGE), "Which layer owns the error").lower()
        for layer in ("operating system", "network configuration", "external dependencies"):
            self.assertIn(layer, owner)
        self.assertRegex(owner, r"native tbccl")

    def test_checks_are_read_only_and_say_not_to_change_settings(self):
        t = read(self.PAGE)
        self.assertRegex(t, r"(?i)do not\*\* change firewall rules, mtu or interface settings")
        self.assertRegex(t, r"(?i)read-only")

    def test_no_guide_block_recovers_hardware_or_changes_the_system(self):
        risky = re.compile(r"\b(rmmod|modprobe|setpci|ufw|iptables|firewall-cmd|pfctl|nmcli\s+\S+\s+modify|ip\s+link\s+set|ifconfig\s+\S+\s+mtu|sudo)\b|/sys/bus/pci|pci=noaer|iommu=off")
        for page in sorted((DOCS / "guides").glob("*.md")):
            for line, lang, text, _ in cdc.blocks(page):
                if lang in cdc.SHELL:
                    self.assertIsNone(risky.search(text), f"{page.name}:{line} runs a command that changes the system or the hardware")

    def test_the_link_doctor_can_be_obtained_where_it_is_used(self):
        for page in sorted(DOCS.rglob("*.md")):
            if "_build" in page.parts or "validation" in page.parts:
                continue
            text = page.read_text(encoding="utf-8")
            used = any("thunderbolt_link_doctor.py" in b for _, lang, b, _ in cdc.blocks(page) if lang in cdc.SHELL)
            if not used:
                continue
            fetched = "curl -fsSLO" in text and "thunderbolt_link_doctor.py" in text
            linked = re.search(r"\]\([^)]*(thunderbolt-troubleshooting|two-machine-thunderbolt)", text)
            self.assertTrue(fetched or linked, f"{page.relative_to(DOCS)} runs the link doctor without saying where to get it")

    def test_escalation_names_what_to_collect(self):
        t = read(self.PAGE)
        self.assertIn("collect_diagnostics.py", t)
        self.assertRegex(t, r"(?i)getting.help")


class Onboarding(unittest.TestCase):
    CHOOSER = {"tbccl": "Which path do I need?", "torch-tbccl": "Which commands do I run?", "vllm-tbccl": "Which pairing do I need?", "exo-tbccl": "Which integration do I have, and which do I want?"}

    def test_the_install_page_starts_with_a_decision(self):
        t = read("getting-started/install.md")
        self.assertIn(slug(self.CHOOSER[SITE]), headings(t))
        self.assertLess(t.index(f"## {self.CHOOSER[SITE]}"), t.index("## Requirements") if "## Requirements" in t else len(t))

    def test_the_success_block_has_all_five_parts(self):
        pages = [p for p in (DOCS / "getting-started").glob("*.md") if "## What success looks like" in p.read_text(encoding="utf-8")]
        self.assertTrue(pages, "no getting-started page has a success block")
        if ADAPTER:
            self.assertIn("quickstart.md", {p.name for p in pages})
        for p in pages:
            block = section(p.read_text(encoding="utf-8"), "What success looks like")
            for part in (r"showed", r"did not show", r"[Rr]ead next|Next:", r"[Ii]f it fails"):
                self.assertRegex(block, part, f"{p.name}: the success block lacks {part!r}")

    def test_success_blocks_do_not_promise_unverified_output(self):
        for p in (DOCS / "getting-started").glob("*.md"):
            t = p.read_text(encoding="utf-8")
            if "## What success looks like" in t:
                self.assertNotRegex(section(t, "What success looks like"), r"(?i)\b(always|guaranteed|will be faster)\b")

    def test_immutable_tags_and_pins_are_unchanged(self):
        facts = {
            "exo-tbccl": ["tbccl-integration-v2", "tbccl-integration-v1", "d1497b28b89618172d273e70e9ac7992811635f5", "0.3.0", "0.3.1"],
            "vllm-tbccl": ["v0.31.0-tbccl1", "fa4f934586bc27ac974a54183237594d1dc28fc5", "0.31.0"],
            "torch-tbccl": ["--no-deps", "2.13.0", "cu130"],
            "tbccl": ["tbccl-0.6.1-linux-x86_64.tar.gz", "SHA256SUMS", "TBCCL::tbccl_c"],
        }
        text = read("getting-started/install.md")
        for fact in facts[SITE]:
            self.assertIn(fact, text, f"{SITE}: install.md no longer states {fact!r}")

    def test_exo_keeps_the_two_integration_paths_distinct(self):
        if SITE != "exo-tbccl":
            self.skipTest("exo only")
        t = read("getting-started/install.md")
        self.assertRegex(t, r"`tbccl-integration-v2`[^\n]*\*\*0\.3\.0\*\*")
        self.assertRegex(t, r"`tbccl-integration-v1`[^\n]*\*\*0\.3\.1\*\*")
        ladder = read("guides/verify-active.md")
        self.assertIn("discovery", section(ladder, "The verification order").lower())
        self.assertRegex(section(ladder, "The verification order"), r"(?s)Exo.*IPv6.*TBCCL.*IPv4|IPv6 link-local multicast, the second IPv4 TCP")

    def test_vllm_keeps_the_pairings_distinct(self):
        if SITE != "vllm-tbccl":
            self.skipTest("vllm only")
        t = section(read("getting-started/install.md"), "Which pairing do I need?")
        for fact in ("CUDA and CPU", "CUDA and Metal", "v0.31.0-tbccl1", "VLLM_TBCCL_BACKEND"):
            self.assertIn(fact, t)
        q = read("getting-started/quickstart.md")
        self.assertIn("must be equal", section(q, "Settings that must agree, and those that must differ"))


class Contributors(unittest.TestCase):
    def test_code_map_paths_exist(self):
        text = read("development/code-map.md")
        paths = set(re.findall(r"`([\w./-]+(?:/|\.(?:py|cpp|hpp|h|c|cu|mm|sh|json|patch|md|toml)))`", text))
        self.assertGreaterEqual(len(paths), 8)
        for p in sorted(paths):
            self.assertTrue((ROOT / p).exists(), f"code-map.md names {p}, which does not exist")

    def test_code_map_is_linked_and_points_at_the_runtime_or_the_adapters(self):
        self.assertIn("code-map", read("development/index.md"))
        text = read("development/code-map.md")
        self.assertIn("where-should-this-change-go.md", text)
        if ADAPTER:
            self.assertIn("tbccl.tensorsofthewall.com/en/latest/development/code-map.html", text)

    def test_testing_page_says_what_hardware_each_tier_needs(self):
        t = section(read("development/testing.md"), "What needs what")
        self.assertRegex(t, r"(?i)do not need a gpu")
        for tier in ("Documentation only", "Physical link" if SITE != "exo-tbccl" else "Two hosts"):
            self.assertIn(tier, t)
        self.assertIn("check_doc_commands.py", t)

    def test_ownership_page_has_examples_independence_and_counterexamples(self):
        t = read("development/where-should-this-change-go.md")
        for title in ("Examples", "What does not belong in the core runtime"):
            self.assertIn(slug(title), headings(t))
        self.assertRegex(t, r"(?m)^## How .*without depending on")
        self.assertIn("adapters must not require other adapters", t.lower())

    def test_contributing_points_at_the_tiers_and_keeps_its_policies(self):
        t = (ROOT / "CONTRIBUTING.md").read_text(encoding="utf-8")
        self.assertIn("What needs what", t)
        self.assertIn("code-map.md", t)


class CrossLinks(unittest.TestCase):
    def test_architecture_points_to_the_other_side_of_the_boundary(self):
        arch = read("concepts/architecture.md")
        if ADAPTER:
            self.assertIn("tbccl.tensorsofthewall.com/en/latest/concepts/architecture.html", arch)
        else:
            for a in ("torch-tbccl", "vllm-tbccl", "exo-tbccl"):
                self.assertIn(f"{a}.tensorsofthewall.com/en/latest/concepts/architecture.html", arch)


if __name__ == "__main__":
    unittest.main()
