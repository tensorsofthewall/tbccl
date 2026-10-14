"""The installation commands in README.md and docs/getting-started/install.md must name the archives and URLs of the release recorded in release/ecosystem.json."""
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARCHIVES = ("linux-x86_64", "linux-x86_64-cuda13", "macos-arm64")


class InstallDocs(unittest.TestCase):
    def setUp(self):
        eco = json.loads((ROOT / "release" / "ecosystem.json").read_text())["tbccl"]
        self.version, self.tag = eco["version"], eco["tag"]
        self.pages = {n: (ROOT / n).read_text() for n in ("README.md", "docs/getting-started/install.md")}

    def test_archive_names_and_urls_match_the_current_release(self):
        for name, text in self.pages.items():
            self.assertIn(f"releases/download/{self.tag}", text, name)
            for found in re.findall(r"tbccl-(\d+\.\d+\.\d+)-(?:linux|macos)", text):
                self.assertEqual(found, self.version, name)
            for arch in re.findall(rf"tbccl-{re.escape(self.version)}-([a-z0-9_-]+?)\.tar\.gz", text):
                self.assertIn(arch, ARCHIVES, f"{name}: unknown archive {arch}")

    def test_both_platforms_have_commands(self):
        text = self.pages["docs/getting-started/install.md"]
        self.assertIn("sha256sum -c SHA256SUMS --ignore-missing", text)
        self.assertIn("shasum -a 256 -c SHA256SUMS --ignore-missing", text)
        self.assertIn(f"tbccl-{self.version}-macos-arm64.tar.gz", text)
        self.assertIn("mkdir tbccl-quickstart", text)

    def test_readme_points_adapter_users_away_from_libtbccl(self):
        text = self.pages["README.md"]
        self.assertIn("install the corresponding adapter instead", text)

    def test_cmake_consumer_in_the_guide_is_self_contained(self):
        text = self.pages["docs/getting-started/install.md"]
        for needle in ("find_package(TBCCL CONFIG REQUIRED)", "TBCCL::tbccl_c", "cat > version_check.c", "cat > CMakeLists.txt", "-DCMAKE_PREFIX_PATH="):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
