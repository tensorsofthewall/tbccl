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


def tables(body):
    """Pipe tables of a Markdown section as (header cells, rows of cells)."""
    lines, out, i = body.splitlines(), [], 0
    split = lambda l: [c.strip() for c in l.strip().strip("|").split("|")]
    while i < len(lines):
        if lines[i].lstrip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]):
            header, rows, i = split(lines[i]), [], i + 2
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                rows.append(split(lines[i]))
                i += 1
            out.append((header, rows))
        else:
            i += 1
    return out


def read_cell(page, spec):
    """The one cell at (row, column) of the cited section; the row is named by its first cell, the column by its header."""
    found = []
    for header, rows in tables(section(page, spec["section"])):
        if spec["column"] in header:
            j = header.index(spec["column"])
            found += [(header, r) for r in rows if r[0] == spec["row"] and len(r) == len(header)]
    assert len(found) == 1, f"{page}: {spec} matches {len(found)} rows, expected exactly one"
    header, row = found[0]
    return row[header.index(spec["column"])], dict(zip(header, row))


def sections_of(chart):
    return chart["section"] if isinstance(chart["section"], list) else [chart["section"]]


class SourceIntegrity(unittest.TestCase):
    def test_every_value_is_read_from_the_cell_it_names(self):
        for c in DATA["charts"]:
            for p in points(c):
                spec = p["table"]
                self.assertIn(spec["section"], sections_of(c), f"{c['id']}: {p['published']!r} cites a section the chart does not")
                cell, _ = read_cell(c["page"], spec)
                self.assertIn(p["published"], cell, f"{c['id']}: cell ({spec['row']!r}, {spec['column']!r}) is {cell!r}, not {p['published']!r}")

    def test_a_value_attached_to_the_wrong_row_or_column_is_caught(self):
        c = DATA["charts"][0]
        pts = list(points(c))
        a, b = pts[0], pts[1]
        self.assertNotEqual(a["table"], b["table"])
        cell, _ = read_cell(c["page"], b["table"])
        self.assertNotIn(a["published"], cell, "two different measurements must not sit in the same cell")
        with self.assertRaises(AssertionError):
            read_cell(c["page"], dict(a["table"], row="no such row"))
        with self.assertRaises(AssertionError):
            read_cell(c["page"], dict(a["table"], column="no such column"))

    def test_ranges_come_from_the_named_cell(self):
        for c in DATA["charts"]:
            for p in points(c):
                if "range_published" not in p:
                    continue
                spec = dict(p["table"])
                if "range_source" in spec:
                    spec = spec["range_source"]
                elif "range_column" in spec:
                    spec["column"] = spec["range_column"]
                cell, _ = read_cell(c["page"], spec)
                self.assertIn(p["range_published"], cell, f"{c['id']}: range of {p['published']!r}")

    def test_a_line_series_is_read_along_one_column_or_one_row(self):
        for c in DATA["charts"]:
            if c["kind"] != "lines":
                continue
            for s in c["series"]:
                cols, rows = [p["table"]["column"] for p in s["points"]], [p["table"]["row"] for p in s["points"]]
                by_column = len(set(cols)) == 1 and len(set(rows)) == len(rows)
                by_row = len(set(rows)) == 1 and len(set(cols)) == len(cols)
                self.assertTrue(by_column or by_row or len(s["points"]) == 1, f"{c['id']} / {s['label']}: points are not read along one column or one row")

    def test_bars_of_one_chart_never_share_a_cell(self):
        for c in DATA["charts"]:
            cells = [(p["table"]["section"], p["table"]["row"], p["table"]["column"]) for p in points(c)]
            if c["kind"] == "bars":
                self.assertEqual(len(cells), len(set(cells)), f"{c['id']}: two bars read the same cell")

    def test_x_values_agree_with_the_row_labels_of_payload_charts(self):
        sizes = {"1 KiB": 1024, "4 KiB": 4096, "64 KiB": 65536, "1 MiB": 1048576, "4 MiB": 4194304, "16 MiB": 16777216}
        for c in DATA["charts"]:
            if c["kind"] == "lines" and c["x_scale"] == "log":
                for s in c["series"]:
                    for p in s["points"]:
                        self.assertEqual(sizes[p["table"]["row"]], p["x"], f"{c['id']}: x of {p['published']!r} disagrees with its row")
            if c["kind"] == "lines" and c["x_scale"] == "linear":
                for s in c["series"]:
                    for p in s["points"]:
                        self.assertEqual(float(re.sub(r"\D", "", p["table"]["row"] if p["table"]["row"].isdigit() else p["table"]["column"])), float(p["x"]), f"{c['id']}: x of {p['published']!r} disagrees with its row or column")

    def test_statistical_classification_is_preserved(self):
        seen = 0
        for c in DATA["charts"]:
            for p in points(c):
                if "classification" not in p:
                    continue
                seen += 1
                _, row = read_cell(c["page"], p["table"])
                self.assertIn("Class", row, f"{c['id']}: the cited table has no Class column")
                self.assertEqual(row["Class"].lower().replace(" ", "_"), p["classification"], f"{c['id']}: the table row says {row['Class']!r}")
        # a chart-level classification (the evidence for the whole comparison) must be stated in the cited section and shown in the picture
        for c in DATA["charts"]:
            if "classification" in c:
                seen += 1
                phrase = c["classification"].replace("_", " ")
                self.assertIn(phrase, "\n".join(sections(c)).lower(), f"{c['id']}: the section does not state {phrase!r}")
                svg_text = (BENCH / "charts" / f"{c['id']}.html").read_text(encoding="utf-8").lower()
                self.assertIn(phrase, svg_text, f"{c['id']}: the chart does not show {phrase!r}")
        self.assertGreaterEqual(seen, 0)

    def test_numeric_value_matches_published_text(self):
        for c in DATA["charts"]:
            for p in points(c):
                self.assertAlmostEqual(first_number(p["published"]), p["value"], places=6, msg=f"{c['id']}: {p['published']!r} against {p['value']}")
                if "min" in p:
                    lo, hi = (first_number(x) for x in p["range_published"].split("-", 1))
                    self.assertEqual((lo, hi), (p["min"], p["max"]), f"{c['id']}: range of {p['published']!r}")

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


def synthetic(scale, xs=(1, 2, 4, 12)):
    point = lambda x, v: {"x": x, "value": v, "published": f"{v} u", "exact": True, "n": 1, "table": {"section": "s", "row": str(x), "column": "c"}}
    return {"id": "synthetic", "title": "t", "summary": "s", "conditions": "c.", "page": "p.md", "section": "s", "statistic": "x", "kind": "lines",
            "x_label": "X", "y_label": "Y", "y_unit": "u", "y_scale": "linear", "x_scale": scale,
            "x_ticks": [{"x": x, "label": str(x)} for x in xs],
            "series": [{"label": "a", "marker": "circle", "dash": "solid", "color": "accent", "points": [point(x, 10 * (i + 1)) for i, x in enumerate(xs)]}]}


class XAxis(unittest.TestCase):
    """The x scale is stated, validated and drawn as stated."""

    @staticmethod
    def tick_positions(chart):
        svg = ET.fromstring(re.search(r"<svg\b.*?</svg>", bc.render(chart, "tbccl"), re.S).group(0))
        return [float(t.get("x")) for t in svg.iter(f"{SVG}text") if t.get("class") == "tb-chart-tick" and t.get("text-anchor") == "middle"]

    def test_scale_must_be_stated_and_valid(self):
        for bad in (None, "", "sqrt", "Linear"):
            c = synthetic("linear")
            if bad is None:
                del c["x_scale"]
            else:
                c["x_scale"] = bad
            with self.assertRaisesRegex(bc.ChartDataError, "x_scale"):
                bc.validate({"schema": bc.SCHEMA, "accent": "tbccl", "provenance": "p", "charts": [c]})

    def test_log_axis_rejects_zero_and_linear_accepts_it(self):
        ok = synthetic("linear", (0, 2, 4))
        bc.validate({"schema": bc.SCHEMA, "accent": "tbccl", "provenance": "p", "charts": [ok]})
        bad = synthetic("log", (0, 2, 4))
        with self.assertRaises(bc.ChartDataError):
            bc.validate({"schema": bc.SCHEMA, "accent": "tbccl", "provenance": "p", "charts": [bad]})

    def test_linear_ticks_are_evenly_proportional(self):
        pos = self.tick_positions(synthetic("linear"))
        xs = (1, 2, 4, 12)
        for p, x in zip(pos, xs):
            self.assertAlmostEqual((p - pos[0]) / (pos[-1] - pos[0]), (x - xs[0]) / (xs[-1] - xs[0]), places=2)

    def test_log_ticks_are_proportional_to_the_logarithm(self):
        import math
        pos = self.tick_positions(synthetic("log"))
        xs = (1, 2, 4, 12)
        for p, x in zip(pos, xs):
            self.assertAlmostEqual((p - pos[0]) / (pos[-1] - pos[0]), math.log2(x / xs[0]) / math.log2(xs[-1] / xs[0]), places=2)

    def test_the_label_says_logarithmic_only_for_a_log_axis(self):
        self.assertIn("X (logarithmic)", bc.render(synthetic("log"), "tbccl"))
        self.assertNotIn("X (logarithmic)", bc.render(synthetic("linear"), "tbccl"))

    def test_the_published_charts_use_the_scale_their_x_variable_needs(self):
        for c in DATA["charts"]:
            if c["kind"] != "lines":
                continue
            if re.search(r"(?i)size|payload", c["x_label"]):
                self.assertEqual(c["x_scale"], "log", c["id"])
            if re.search(r"(?i)concurren", c["x_label"]):
                self.assertEqual(c["x_scale"], "linear", c["id"])


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
        broken(lambda d: first(d).pop("table"), "table must name")
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
