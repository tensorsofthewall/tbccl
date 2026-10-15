#!/usr/bin/env python3
"""The issue forms, the contact links and the community pages stay consistent: every form is valid and has the fields a report needs, labels used exist in the repository's usual set, and the help and privacy pages are reachable from the documentation index."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORMS = ROOT / ".github" / "ISSUE_TEMPLATE"
KNOWN_LABELS = {"bug", "documentation", "enhancement", "question"}


def parse_form(path):
    """A small reader for the subset of GitHub issue-form YAML these files use: top-level keys and the list of body items with their id, type and required flag."""
    text = path.read_text()
    top = dict(re.findall(r"^(name|description|labels): (.+)$", text, re.M))
    items = []
    for block in re.split(r"\n  - type: ", "\n" + text.split("\nbody:\n", 1)[1]):
        if not block.strip():
            continue
        kind = block.split("\n", 1)[0].strip()
        ident = re.search(r"^    id: (\S+)", block, re.M)
        items.append({"type": kind, "id": ident.group(1) if ident else None, "required": bool(re.search(r"required: true", block))})
    return top, items


class IssueForms(unittest.TestCase):
    def test_every_form_has_a_name_description_and_a_required_field(self):
        forms = sorted(FORMS.glob("*.yml"))
        names = {p.name for p in forms}
        for needed in ("bug_report.yml", "install_problem.yml", "tested_configuration.yml", "docs_feedback.yml"):
            self.assertIn(needed, names)
        for p in forms:
            if p.name == "config.yml":
                continue
            top, items = parse_form(p)
            self.assertIn("name", top, p.name)
            self.assertIn("description", top, p.name)
            ids = [i["id"] for i in items if i["id"]]
            self.assertEqual(len(ids), len(set(ids)), f"{p.name}: duplicate ids")
            self.assertTrue(any(i["required"] for i in items), f"{p.name}: nothing is required")
            for label in re.findall(r"[a-z-]+", top.get("labels", "")):
                self.assertIn(label, KNOWN_LABELS, f"{p.name}: label {label!r} does not exist in the repository")

    def test_forms_ask_people_to_remove_private_data(self):
        for name in ("bug_report.yml", "install_problem.yml", "tested_configuration.yml"):
            self.assertRegex((FORMS / name).read_text().lower(), r"private|remove anything", name)

    def test_config_links_to_help_and_security(self):
        cfg = (FORMS / "config.yml").read_text()
        self.assertIn("blank_issues_enabled: false", cfg)
        self.assertIn("security/policy", cfg)
        self.assertIn("project/getting-help.html", cfg)


class Pages(unittest.TestCase):
    def test_help_and_privacy_are_in_the_project_toctree_and_on_the_landing_page(self):
        index = (ROOT / "docs" / "project" / "index.md").read_text()
        landing = (ROOT / "docs" / "index.md").read_text()
        for page in ("getting-help", "privacy"):
            self.assertIn(page, index)
            self.assertTrue((ROOT / "docs" / "project" / f"{page}.md").exists())
            self.assertIn(f"project/{page}.md", landing)

    def test_privacy_page_states_the_scope_and_date_of_its_check(self):
        text = (ROOT / "docs" / "project" / "privacy.md").read_text()
        self.assertIn("collect no telemetry", text)
        self.assertRegex(text, r"On \d{4}-\d{2}-\d{2} ")
        self.assertIn("outside our control", text.lower())

    def test_help_page_does_not_promise_support(self):
        text = (ROOT / "docs" / "project" / "getting-help.md").read_text().lower()
        self.assertIn("no support contract", text)
        self.assertIn("best-effort", text)


if __name__ == "__main__":
    unittest.main()
