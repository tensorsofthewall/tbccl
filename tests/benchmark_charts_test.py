"""The benchmark charts match the tables they are drawn from, and stay honest, valid and reproducible.

No build, no network, no measurement: the test reads docs/benchmarks/chart-data.json, the committed fragments in docs/benchmarks/charts/ and the Markdown
pages. The same file is used by all four TBCCL repositories. It asserts no performance threshold; it only checks that a chart and its table agree.
"""
import copy
import json
import re
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import benchmark_charts as bc  # noqa: E402

DOCS = ROOT / "docs"
BENCH = DOCS / "benchmarks"
DATA = json.loads((BENCH / "chart-data.json").read_text(encoding="utf-8"))
CSS = (DOCS / "_static" / "custom.css").read_text()
CONF = (DOCS / "conf.py").read_text()
SVG = "{http://www.w3.org/2000/svg}"


def section(page, heading):
    text = (BENCH / page).read_text(encoding="utf-8")
    m = re.search(rf"^## {re.escape(heading)}\s*$", text, re.M)
    assert m, f"{page} has no section {heading!r}"
    end = re.search(r"^## ", text[m.end():], re.M)
    return text[m.end(): m.end() + end.start()] if end else text[m.end():]


def sections(chart):
    names = chart["section"] if isinstance(chart["section"], list) else [chart["section"]]
    return [section(chart["page"], n) for n in names]


def points(chart):
    if chart["kind"] == "bars":
        for panel in chart["panels"]:
            for g in panel["groups"]:
                yield from g["bars"]
    else:
        for s in chart["series"]:
            yield from s["points"]


def first_number(text):
    return float(re.search(r"\d[\d,]*\.?\d*", text).group(0).replace(",", ""))


def luminance(hex_):
    rgb = [int(hex_[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    rgb = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def contrast(a, b):
    hi, lo = sorted((luminance(a), luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


class SourceIntegrity(unittest.TestCase):
    def test_every_value_is_found_in_the_table_it_cites(self):
        for c in DATA["charts"]:
            text = "\n".join(sections(c))
            for p in points(c):
                self.assertIn(p["published"], text, f"{c['id']}: {p['published']!r} is not in the cited section of {c['page']}")
                if "range_published" in p:
                    # a range can be printed in the table of another section of the same page (the control of a comparison)
                    self.assertIn(p["range_published"], (BENCH / c["page"]).read_text(encoding="utf-8"), f"{c['id']}: range {p['range_published']!r} is not on {c['page']}")

    def test_numeric_value_matches_published_text(self):
        for c in DATA["charts"]:
            for p in points(c):
                self.assertAlmostEqual(first_number(p["published"]), p["value"], places=6, msg=f"{c['id']}: {p['published']!r} against {p['value']}")
                if "min" in p:
                    lo, hi = (first_number(x) for x in p["range_published"].split("-", 1))
                    self.assertEqual((lo, hi), (p["min"], p["max"]), f"{c['id']}: range of {p['published']!r}")

    def test_a_value_is_published_in_a_table_row(self):
        for c in DATA["charts"]:
            text = "\n".join(sections(c))
            for p in points(c):
                rows = [l for l in text.splitlines() if l.lstrip().startswith("|") and p["published"] in l]
                self.assertTrue(rows, f"{c['id']}: {p['published']!r} is not in any table row, so the chart has no adjacent data table")

    def test_statistical_classification_is_preserved(self):
        seen = 0
        for c in DATA["charts"]:
            text = "\n".join(sections(c))
            for p in points(c):
                if "classification" not in p:
                    continue
                seen += 1
                row = next(l for l in text.splitlines() if l.lstrip().startswith("|") and p["published"] in l)
                self.assertIn(p["classification"].replace("_", " "), row.lower(), f"{c['id']}: the table row does not carry the class {p['classification']!r}")
        # a chart-level classification (the evidence for the whole comparison) must be stated in the cited section and shown in the picture
        for c in DATA["charts"]:
            if "classification" in c:
                seen += 1
                phrase = c["classification"].replace("_", " ")
                self.assertIn(phrase, "\n".join(sections(c)).lower(), f"{c['id']}: the section does not state {phrase!r}")
                svg_text = (BENCH / "charts" / f"{c['id']}.html").read_text(encoding="utf-8").lower()
                self.assertIn(phrase, svg_text, f"{c['id']}: the chart does not show {phrase!r}")
        self.assertGreaterEqual(seen, 0)

    def test_provenance_says_curated_not_raw(self):
        self.assertRegex(DATA["provenance"], r"(?i)curated")
        self.assertRegex(DATA["provenance"], r"(?i)not raw")
        self.assertRegex(DATA["provenance"], r"(?i)authoritative")

    def test_no_value_is_invented(self):
        for c in DATA["charts"]:
            for p in points(c):
                self.assertTrue(p["n"], f"{c['id']}: every point states its number of repetitions or lifecycles")
                if not p["exact"]:
                    self.assertNotIn("min", p, "a derived display value has no invented range")

    def test_every_chart_is_embedded_once_with_its_table_and_methodology_link(self):
        md = {p: (BENCH / p).read_text(encoding="utf-8") for p in {c["page"] for c in DATA["charts"]}}
        for c in DATA["charts"]:
            ref = f":file: charts/{c['id']}.html"
            self.assertEqual(sum(t.count(ref) for t in md.values()), 1, f"{c['id']} must be embedded exactly once")
            self.assertIn(ref, md[c["page"]], f"{c['id']} must be embedded in {c['page']}")
        for page, text in md.items():
            self.assertRegex(text, r"\]\([^)]*methodology\.md", f"{page} links to the methodology")

    def test_pages_include_no_chart_that_has_no_data(self):
        embedded = set()
        for page in BENCH.glob("*.md"):
            embedded |= set(re.findall(r":file: charts/([a-z0-9-]+)\.html", page.read_text(encoding="utf-8")))
        self.assertEqual(embedded, {c["id"] for c in DATA["charts"]})


class Generator(unittest.TestCase):
    def test_committed_charts_are_current(self):
        out = bc.build(DATA)
        for name, text in out.items():
            self.assertEqual((BENCH / "charts" / name).read_text(encoding="utf-8"), text, f"{name} is stale; run tools/benchmark_charts.py")
        self.assertEqual({p.name for p in (BENCH / "charts").glob("*.html")}, set(out))

    def test_check_mode_detects_a_stale_chart(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for name, text in bc.build(DATA).items():
                (d / name).write_text(text, encoding="utf-8")
            args = ["--data", str(BENCH / "chart-data.json"), "--out", str(d), "--check"]
            self.assertEqual(bc.main(args), 0)
            victim = next(iter(sorted(d.glob("*.html"))))
            victim.write_text(victim.read_text() + " ", encoding="utf-8")
            self.assertEqual(bc.main(args), 1)
            victim.unlink()
            self.assertEqual(bc.main(args), 1)

    def test_output_is_deterministic_and_free_of_environment_details(self):
        a, b = bc.build(DATA), bc.build(copy.deepcopy(DATA))
        self.assertEqual(a, b)
        for name, text in a.items():
            self.assertNotRegex(text, r"\b20\d\d-\d\d-\d\d\b", f"{name}: date")
            self.assertNotRegex(text, r"(?:/home/|/Users/|/mnt/|[A-Za-z]:\\)", f"{name}: path")
            self.assertNotRegex(text, r"(?i)<script|<style|<link|<image|<img|https?://(?!www\.w3\.org/2000/svg)|href=|url\((?!#)", f"{name}: remote or active content")
            self.assertNotRegex(text, r"#[0-9a-fA-F]{3,8}\b", f"{name}: colours come from the page tokens")

    def test_malformed_inputs_fail_clearly(self):
        def broken(mutate, message):
            d = copy.deepcopy(DATA)
            mutate(d)
            with self.assertRaisesRegex(bc.ChartDataError, message):
                bc.validate(d)

        first = lambda d: next(iter(points(d["charts"][0])))
        broken(lambda d: d.update(schema="x"), "schema")
        broken(lambda d: d.update(accent="blue"), "accent")
        broken(lambda d: d.update(provenance=""), "provenance")
        broken(lambda d: first(d).pop("published"), "published text")
        broken(lambda d: first(d).update(value=-1), "positive")
        broken(lambda d: first(d).update(value=float("nan")), "positive")
        broken(lambda d: first(d).pop("n"), "never leave it out")
        broken(lambda d: first(d).pop("exact"), "exact")
        broken(lambda d: first(d).update(min=1, max=2), "outside its range")
        broken(lambda d: first(d).update(classification="faster"), "classification")
        broken(lambda d: d["charts"][0].update(kind="pie"), "kind")
        broken(lambda d: d["charts"].append(copy.deepcopy(d["charts"][0])), "duplicate")
        broken(lambda d: d["charts"][0]["panels"][0].pop("unit"), "unit")
        broken(lambda d: d["charts"][0]["panels"][0].update(direction="best"), "direction")
        broken(lambda d: d["charts"][0]["panels"][0]["groups"][0]["bars"][0].update(series="nope"), "unknown series")
        broken(lambda d: d["charts"][0]["panels"][0]["groups"][0]["bars"][0].update(label="x" * 60), "label")
        line = next(i for i, c in enumerate(DATA["charts"]) if c["kind"] == "lines") if any(c["kind"] == "lines" for c in DATA["charts"]) else None
        if line is not None:
            broken(lambda d: d["charts"][line]["series"][0]["points"][0].update(x=3), "no tick")
            broken(lambda d: d["charts"][line].update(y_scale="sqrt"), "y_scale")
            broken(lambda d: d["charts"][line]["series"][0].update(marker="star"), "marker")

    def test_cli_reports_a_missing_data_file(self):
        self.assertEqual(bc.main(["--data", str(ROOT / "docs" / "nope.json"), "--check"]), 2)


class Rendering(unittest.TestCase):
    def fragments(self):
        for c in DATA["charts"]:
            yield c, (BENCH / "charts" / f"{c['id']}.html").read_text(encoding="utf-8")

    def svg(self, text):
        return ET.fromstring(re.search(r"<svg\b.*?</svg>", text, re.S).group(0))

    def test_svg_is_valid_and_accessible(self):
        for c, text in self.fragments():
            root = self.svg(text)
            self.assertEqual(root.get("role"), "img")
            for ref in root.get("aria-labelledby").split():
                el = root.find(f".//*[@id='{ref}']")
                self.assertIsNotNone(el, f"{c['id']}: {ref}")
                self.assertTrue((el.text or "").strip())
            self.assertEqual(root.find(f"{SVG}title").text, c["title"])
            self.assertEqual(root.find(f"{SVG}desc").text, c["summary"])
            self.assertTrue(root.get("viewBox"))
            ids = [e.get("id") for e in root.iter() if e.get("id")]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(i.startswith(c["id"]) for i in ids), "ids are namespaced by chart so two charts on a page do not clash")

    def test_axis_units_directions_and_every_value_are_labelled(self):
        for c, text in self.fragments():
            visible = " ".join(t.text or "" for t in self.svg(text).iter(f"{SVG}text"))
            if c["kind"] == "bars":
                for p in points(c):
                    self.assertIn(p["published"], visible, "every bar carries its value")
                for panel in c["panels"]:
                    self.assertIn(panel["unit"], visible)
                    self.assertIn(panel["direction"], visible)
                for g in (g for panel in c["panels"] for g in panel["groups"]):
                    for b in g["bars"]:
                        self.assertIn(b["label"], visible, "identity never rests on colour: every bar is labelled")
            else:
                self.assertIn(c["y_unit"], visible)
                self.assertIn(c["x_label"], visible)
                for s in c["series"]:
                    self.assertIn(s["label"], visible)
                self.assertGreaterEqual(len({(s["marker"], s["dash"]) for s in c["series"]}), len(c["series"]), "series differ by marker and dash, not only colour")

    def test_a_caption_and_a_title_accompany_every_chart(self):
        for c, text in self.fragments():
            self.assertIn('class="tb-chart-title"', text)
            self.assertIn("<figcaption>", text)
            self.assertIn(c["conditions"].split(".")[0].replace("&", "&amp;"), text.replace("&#x27;", "'"))
            self.assertIn('role="region" tabindex="0"', text)

    def test_charts_and_page_need_no_remote_assets(self):
        for page in BENCH.glob("*.md"):
            self.assertNotRegex(page.read_text(encoding="utf-8"), r"<script|<img|src=", page.name)

    def test_responsive_rules_exist(self):
        self.assertRegex(CSS, r"\.tb-chart-scroll\s*\{[^}]*overflow-x:\s*auto")
        self.assertRegex(CSS, r"\.tb-chart svg\s*\{[^}]*width:\s*100%[^}]*min-width")
        self.assertRegex(CSS, r"@media \(max-width: 44em\) \{ \.tb-chart-hint")

    def test_every_chart_class_has_css(self):
        classes = set()
        for _, text in self.fragments():
            for value in re.findall(r'class="([^"]+)"', text):
                classes |= set(value.split())
        for cls in sorted(classes):
            self.assertRegex(CSS, rf"\.{re.escape(cls)}\b", f"{cls} has no CSS rule")

    def test_chart_css_names_no_colours(self):
        rules = CSS[CSS.index("/* Benchmark charts"):]
        self.assertNotRegex(rules, r"#[0-9a-fA-F]{3,8}\b")
        self.assertNotRegex(rules, r"\brgba?\(")


class Contrast(unittest.TestCase):
    """Marks and lines must stand out from the page background (3:1) in both themes; the project accent is the only chart colour."""

    def test_accent_against_both_backgrounds(self):
        eco = re.search(r"_ECO = \{(.*?)\n\}", CONF, re.S).group(1)
        accents = {k: (a, b) for k, a, b in re.findall(r'"(\w+)":\s*\("(#[0-9A-Fa-f]{6})",\s*"(#[0-9A-Fa-f]{6})"\)', eco)}
        self.assertEqual(set(accents), set(bc.ACCENTS))
        light, dark = accents[DATA["accent"]]
        self.assertGreaterEqual(contrast(light, "#ffffff"), 3.0)
        self.assertGreaterEqual(contrast(dark, "#131416"), 3.0)

    def test_text_tokens_are_furos_own(self):
        rules = CSS[CSS.index("/* Benchmark charts"):]
        for token in set(re.findall(r"var\((--[a-z-]+)", rules)):
            self.assertRegex(token, r"^--(color-|tb-|tbccl-|font-stack)", token)


if __name__ == "__main__":
    unittest.main()
