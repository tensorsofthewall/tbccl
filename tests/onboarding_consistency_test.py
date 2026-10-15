#!/usr/bin/env python3
"""The first-time-setup path stays self-consistent: files the pages name exist, the README and the documentation landing page offer the same entry points, example addresses agree, and the pages state what is not validated."""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGES = [ROOT / "docs/getting-started/two-machine-thunderbolt.md", ROOT / "docs/guides/thunderbolt-troubleshooting.md", ROOT / "docs/guides/thunderbolt-link.md"]
README = (ROOT / "README.md").read_text()
INDEX = (ROOT / "docs/index.md").read_text()
GUIDE = PAGES[0].read_text()


class Onboarding(unittest.TestCase):
    def test_repository_paths_named_in_the_pages_exist(self):
        for page in PAGES:
            for rel in re.findall(r"`((?:tools|scripts|tests|docs|examples)/[\w./-]+\.\w+)`", page.read_text()):
                if "<" in rel or "*" in rel:
                    continue
                self.assertTrue((ROOT / rel).exists(), "%s names a missing file: %s" % (page.name, rel))

    def test_relative_links_between_the_pages_resolve(self):
        for page in PAGES:
            for target in re.findall(r"\]\((\.\./[\w./-]+\.md|[\w-]+\.md)(?:#[\w-]+)?\)", page.read_text()):
                self.assertTrue((page.parent / target).resolve().exists(), "%s -> %s" % (page.name, target))

    def test_readme_and_landing_page_offer_the_same_entry_points(self):
        for needle in ("two-machine-thunderbolt", "thunderbolt-troubleshooting", "torch-tbccl.tensorsofthewall.com/en/latest/getting-started/quickstart", "install-adapters"):
            self.assertIn(needle, README, "README lacks " + needle)
            self.assertIn(needle, INDEX, "docs/index.md lacks " + needle)

    def test_example_addresses_agree(self):
        addresses = set(re.findall(r"\b192\.168\.77\.\d+\b", GUIDE))
        self.assertEqual(addresses - {"192.168.77.0"}, {"192.168.77.1", "192.168.77.2"})
        self.assertNotIn("192.168.3.", GUIDE)
        self.assertIn("never-default yes", GUIDE)

    def test_interface_names_come_from_commands_not_assumptions(self):
        self.assertIn("TB_IF=", GUIDE)
        self.assertIn("BR=$(", GUIDE)
        for line in GUIDE.splitlines():
            if line.startswith(("sudo nmcli", "ip -br addr show", "ifconfig ")):
                self.assertNotRegex(line, r"\b(thunderbolt0|bridge0)\b", line)

    def test_existing_profiles_are_checked_before_creating_one(self):
        self.assertLess(GUIDE.index("nmcli -f NAME,UUID,TYPE,DEVICE connection show"), GUIDE.index("sudo nmcli connection add"))
        self.assertIn("Do not create a second profile", GUIDE)

    def test_default_mtu_and_no_gateway_are_stated(self):
        self.assertIn("Do not change the MTU to get started", GUIDE)
        self.assertIn("leave gateway and dns empty", GUIDE.lower().replace("**", ""))

    def test_validation_status_is_stated_not_implied(self):
        self.assertIn("not yet been run from scratch on real hardware", GUIDE)
        self.assertNotIn("independently validated", GUIDE.lower())


if __name__ == "__main__":
    unittest.main()
