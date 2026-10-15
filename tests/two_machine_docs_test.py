#!/usr/bin/env python3
"""The two-machine Thunderbolt pages: every doctor option they show exists, the safety rules hold, and the troubleshooting index resolves."""
import importlib.util
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GUIDE = (ROOT / "docs/getting-started/two-machine-thunderbolt.md").read_text()
TROUBLE = (ROOT / "docs/guides/thunderbolt-troubleshooting.md").read_text()
spec = importlib.util.spec_from_file_location("doctor", ROOT / "tools" / "thunderbolt_link_doctor.py")
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)


class Pages(unittest.TestCase):
    def test_doctor_options_in_docs_exist(self):
        for text in (GUIDE, TROUBLE):
            for line in re.findall(r"thunderbolt_link_doctor\.py([^\n`]*)", text):
                args = re.sub(r"<[^>]*>", "192.168.77.1", line.split("#")[0]).split()
                if args:
                    doctor.parse_args(args)

    def test_example_addresses_are_consistent(self):
        self.assertIn("192.168.77.1", GUIDE)
        self.assertIn("192.168.77.2/24", GUIDE)
        self.assertIn("ipv4.never-default yes", GUIDE)
        self.assertIn("ipv6.method link-local", GUIDE)

    def test_no_instruction_to_disable_protection(self):
        for text in (GUIDE, TROUBLE):
            for bad in ("sudo ufw disable", "systemctl stop firewalld", "setglobalstate off", "pci=noaer", "iommu=off", "echo 1 > /sys/bus/pci"):
                self.assertNotIn(bad, text)
        self.assertIn("Do not disable the firewall", GUIDE)

    def test_troubleshooting_entries_are_all_linked(self):
        for n in range(1, 14):
            self.assertRegex(TROUBLE, r"\(#%d-" % n)
            self.assertRegex(TROUBLE, r"(?m)^## %d\. " % n)

    def test_known_untested_steps_are_labelled(self):
        self.assertIn("was **not** a first-time setup", GUIDE)
        self.assertIn("Nobody who had never seen TBCCL", GUIDE)


if __name__ == "__main__":
    unittest.main()
