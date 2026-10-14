"""Unit tests of the maintenance tooling (tools/maint_common.py, check_release_state.py, verify_public_release.py, check_links.py and, where present,
check_ecosystem_manifest.py and check_framework_pin.py). The file is kept identical in the four TBCCL release repositories; parts that belong to one repository are
skipped elsewhere. Nothing here needs the network: GitHub access is replaced by fakes."""
import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
sys.path.insert(0, str(TOOLS))
import maint_common as mc  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def worst(rep):
    return rep.worst()


class Common(unittest.TestCase):
    def test_versions(self):
        self.assertEqual(mc.split_version("0.6.0"), (0, 6, 0, ""))
        self.assertEqual(mc.split_version("0.2.0rc2"), (0, 2, 0, "rc2"))
        self.assertIsNone(mc.split_version("0.2"))
        self.assertIsNone(mc.split_version("v0.2.0"))

    def test_worst_ordering(self):
        rep = mc.Report("t")
        rep.add(mc.WARNING, "w")
        self.assertEqual(rep.worst(), mc.WARNING)
        rep.add(mc.DRIFT, "d")
        self.assertEqual(rep.worst(), mc.DRIFT)
        rep.add(mc.SURFACE, "s")
        self.assertEqual(rep.worst(), mc.SURFACE)
        rep.add(mc.INTEGRITY, "i")
        self.assertEqual(rep.worst(), mc.INTEGRITY)

    def test_a_crash_is_an_error_not_an_integrity_verdict(self):
        with self.assertRaises(SystemExit) as cm:
            mc.main_guard(lambda: 1 / 0)
        self.assertEqual(cm.exception.code, mc.ERROR)


class ReleaseState(unittest.TestCase):
    def setUp(self):
        self.rs = load("check_release_state")
        self.stable = {
            "project": "torch-tbccl",
            "package": {"development_version": "0.2.0", "release_candidate": False, "target_release": "0.2.0", "released": True},
            "requires_tbccl": {"minimum_package": "0.6.0", "c_abi": [1], "wire_protocol": [4]},
            "release_assets": ["torch_tbccl-{version}-x.whl"],
        }

    def run_state(self, manifest, **kw):
        rep = mc.Report("t")
        with mock.patch.object(mc.Report, "summary"):
            self.rs.check_manifest(manifest, rep, **kw)
        return rep.worst()

    def test_valid_stable(self):
        self.assertEqual(self.run_state(self.stable, tag="v0.2.0", require_released=True), mc.OK)

    def test_rc_where_stable_expected(self):
        m = copy.deepcopy(self.stable)
        m["package"].update(development_version="0.2.0rc2", release_candidate=True, released=False)
        self.assertEqual(self.run_state(m, tag="v0.2.0rc2"), mc.OK)
        self.assertEqual(self.run_state(m, tag="v0.2.0"), mc.INTEGRITY)
        self.assertEqual(self.run_state(m, require_released=True), mc.INTEGRITY)

    def test_tag_version_mismatch(self):
        self.assertEqual(self.run_state(self.stable, tag="v0.2.1"), mc.INTEGRITY)
        self.assertEqual(self.run_state(self.stable, tag="v0.2.0rc1"), mc.INTEGRITY)

    def test_rc_marked_released_is_refused(self):
        m = copy.deepcopy(self.stable)
        m["package"].update(development_version="0.2.0rc1", release_candidate=True)
        self.assertEqual(self.run_state(m), mc.INTEGRITY)

    def test_released_must_be_boolean(self):
        m = copy.deepcopy(self.stable)
        m["package"]["released"] = "yes"
        self.assertEqual(self.run_state(m), mc.INTEGRITY)

    def test_wrong_abi_and_wire(self):
        m = copy.deepcopy(self.stable)
        m["requires_tbccl"]["c_abi"] = [2]
        self.assertEqual(self.run_state(m), mc.INTEGRITY)
        m = copy.deepcopy(self.stable)
        m["requires_tbccl"]["wire_protocol"] = [5]
        self.assertEqual(self.run_state(m), mc.INTEGRITY)
        core = {"project": "tbccl", "package": {"development_version": "0.6.0", "public_version": "0.6.0", "release_candidate": False, "target_release": "0.6.0", "released": True},
                "c_abi": {"version": 1}, "wire_protocol": {"version": 5}, "release_assets": ["a-{version}"]}
        self.assertEqual(self.run_state(core), mc.INTEGRITY)
        core["wire_protocol"]["version"] = 4
        self.assertEqual(self.run_state(core), mc.OK)

    def test_core_line(self):
        with tempfile.TemporaryDirectory() as d:
            core = {"project": "tbccl", "package": {"development_version": "0.6.0", "public_version": "0.6.0"}, "c_abi": {"version": 1}, "wire_protocol": {"version": 4}}
            (Path(d) / "compatibility.json").write_text(json.dumps(core))
            rep = mc.Report("t")
            self.rs.check_core(self.stable, d, rep)
            self.assertEqual(rep.worst(), mc.OK)
            core["package"]["public_version"] = "0.5.0"
            (Path(d) / "compatibility.json").write_text(json.dumps(core))
            rep = mc.Report("t")
            self.rs.check_core(self.stable, d, rep)
            self.assertEqual(rep.worst(), mc.INTEGRITY)
            core["package"]["public_version"] = "0.7.0rc1"
            (Path(d) / "compatibility.json").write_text(json.dumps(core))
            rep = mc.Report("t")
            self.rs.check_core(self.stable, d, rep)
            self.assertEqual(rep.worst(), mc.INTEGRITY)

    def test_missing_requirement_and_assets(self):
        m = copy.deepcopy(self.stable)
        del m["requires_tbccl"]
        self.assertEqual(self.run_state(m), mc.INTEGRITY)
        m = copy.deepcopy(self.stable)
        del m["release_assets"]
        self.assertEqual(self.run_state(m), mc.INTEGRITY)

    def test_changelog(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "CHANGELOG.md").write_text("# Changelog\n\n## 0.2.0\n\nx\n\n## 0.1.0\n")
            rep = mc.Report("t")
            self.rs.check_changelog(Path(d), "0.2.0", True, rep)
            self.assertEqual(rep.worst(), mc.OK)
            rep = mc.Report("t")
            self.rs.check_changelog(Path(d), "0.3.0", True, rep)
            self.assertEqual(rep.worst(), mc.INTEGRITY)

    def test_missing_manifest(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(sys, "argv", ["x", "--root", d]), mock.patch.object(mc.Report, "summary"):
            self.assertEqual(self.rs.main(), mc.INTEGRITY)


class PublicRelease(unittest.TestCase):
    def setUp(self):
        self.vp = load("verify_public_release")
        self.templates = ["pkg-{version}.whl"]
        self.release = {"assets": [{"name": "pkg-1.0.0.whl", "size": 5}, {"name": "pkg-1.0.0.whl.spdx.json", "size": 3}, {"name": "SHA256SUMS", "size": 9}]}

    def test_complete_release(self):
        rep = mc.Report("t")
        self.vp.check_assets(self.release, self.templates, "1.0.0", rep)
        self.assertEqual(rep.worst(), mc.OK)

    def test_missing_release_asset_metadata(self):
        for drop in ("pkg-1.0.0.whl", "pkg-1.0.0.whl.spdx.json", "SHA256SUMS"):
            r = {"assets": [a for a in self.release["assets"] if a["name"] != drop]}
            rep = mc.Report("t")
            self.vp.check_assets(r, self.templates, "1.0.0", rep)
            self.assertEqual(rep.worst(), mc.SURFACE, drop)

    def test_checksum_mismatch_is_integrity(self):
        assets = {"pkg-1.0.0.whl": {"digest": "sha256:" + "a" * 64}}
        rep = mc.Report("t")
        self.vp.check_sums_against_digests({"pkg-1.0.0.whl": "b" * 64}, assets, self.templates, "1.0.0", rep)
        self.assertEqual(rep.worst(), mc.INTEGRITY)
        rep = mc.Report("t")
        self.vp.check_sums_against_digests({}, assets, self.templates, "1.0.0", rep)
        self.assertEqual(rep.worst(), mc.SURFACE)

    def test_local_directory(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "pkg-1.0.0.whl").write_bytes(b"wheel")
            (Path(d) / "pkg-1.0.0.whl.spdx.json").write_text("{}")
            (Path(d) / "SHA256SUMS").write_text(f"{hashlib.sha256(b'wheel').hexdigest()}  pkg-1.0.0.whl\n")
            rep = mc.Report("t")
            self.vp.check_local_dir(d, self.templates, "1.0.0", rep)
            self.assertEqual(rep.worst(), mc.OK)
            (Path(d) / "pkg-1.0.0.whl").write_bytes(b"tampered")
            rep = mc.Report("t")
            self.vp.check_local_dir(d, self.templates, "1.0.0", rep)
            self.assertEqual(rep.worst(), mc.INTEGRITY)

    def test_parse_sums(self):
        self.assertEqual(self.vp.parse_sums("AB  x.whl\n\n"), {"x.whl": "ab"})

    def run_main(self, api, tag_sha="c" * 40):
        manifest = {"project": "p", "package": {"development_version": "1.0.0"}, "release_assets": self.templates}
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "compatibility.json").write_text(json.dumps(manifest))
            with mock.patch.object(sys, "argv", ["x", "--root", d, "--commit", "c" * 40]), mock.patch.object(mc, "github_api", api), \
                    mock.patch.object(mc, "ls_remote_tag", lambda *a: tag_sha), mock.patch.object(self.vp, "fetch", lambda u: b""), mock.patch.object(mc.Report, "summary"):
                return self.vp.main()

    def test_offline_is_a_warning_not_a_failure(self):
        def down(path):
            raise mc.Network("no route")
        self.assertEqual(self.run_main(down), mc.WARNING)

    def test_missing_release_is_surface_failure(self):
        self.assertEqual(self.run_main(lambda p: None), mc.SURFACE)

    def test_tag_resolving_to_another_commit_is_integrity(self):
        self.release["assets"][2]["browser_download_url"] = "u"
        self.assertEqual(self.run_main(lambda p: self.release, tag_sha="d" * 40), mc.INTEGRITY)


class Links(unittest.TestCase):
    def test_internal_links(self):
        cl = load("check_links")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "docs").mkdir()
            (root / "docs" / "a.md").write_text("[ok](b.md#x) [bad](missing.md)\n```\n[fenced](nope.md)\n```\n")
            (root / "docs" / "b.md").write_text("[up](../a.md)\n")
            (root / "a.md").write_text("")
            with mock.patch.object(sys, "argv", ["x", "--root", d]), mock.patch.object(mc.Report, "summary"):
                self.assertEqual(cl.main(), mc.INTEGRITY)
            (root / "docs" / "a.md").write_text("[ok](b.md)\n")
            with mock.patch.object(sys, "argv", ["x", "--root", d]), mock.patch.object(mc.Report, "summary"):
                self.assertEqual(cl.main(), mc.OK)

    def test_classes(self):
        cl = load("check_links")
        self.assertEqual(cl.classify("https://github.com/tensorsofthewall/tbccl"), "critical")
        self.assertEqual(cl.classify("https://tbccl.tensorsofthewall.com/en/latest/"), "critical")
        self.assertEqual(cl.classify("https://example.org/x"), "skip")
        self.assertEqual(cl.classify("https://www.python.org/"), "external")
        self.assertEqual(cl.classify("../x.md"), "internal")


@unittest.skipUnless((TOOLS / "check_ecosystem_manifest.py").exists(), "tbccl only")
class Ecosystem(unittest.TestCase):
    def setUp(self):
        self.em = load("check_ecosystem_manifest")
        self.eco = json.loads((ROOT / "release" / "ecosystem.json").read_text())

    def offline(self, eco):
        rep = mc.Report("t")
        with mock.patch.object(mc.Report, "summary"):
            self.em.check_offline(eco, rep)
        return rep.worst()

    def test_current_file_is_valid(self):
        self.assertEqual(self.offline(self.eco), mc.OK)

    def test_rc_is_refused(self):
        e = copy.deepcopy(self.eco)
        e["vllm_tbccl"]["version"] = "0.2.0rc2"
        e["vllm_tbccl"]["tag"] = "v0.2.0rc2"
        self.assertEqual(self.offline(e), mc.INTEGRITY)

    def test_tag_version_mismatch(self):
        e = copy.deepcopy(self.eco)
        e["torch_tbccl"]["tag"] = "v0.2.1"
        self.assertEqual(self.offline(e), mc.INTEGRITY)

    def test_invalid_sha(self):
        for path in (("tbccl", "commit"), ("vllm_tbccl", "metal_sha"), ("exo_tbccl", "exo_sha")):
            e = copy.deepcopy(self.eco)
            e[path[0]][path[1]] = e[path[0]][path[1]][:39]
            self.assertEqual(self.offline(e), mc.INTEGRITY, path)

    def test_missing_fork_tag_field(self):
        for who, key in (("vllm_tbccl", "metal_tag"), ("exo_tbccl", "exo_tag")):
            e = copy.deepcopy(self.eco)
            del e[who][key]
            self.assertEqual(self.offline(e), mc.INTEGRITY)

    def test_moving_fork_ref_is_refused(self):
        e = copy.deepcopy(self.eco)
        e["exo_tbccl"]["exo_tag"] = "main"
        self.assertEqual(self.offline(e), mc.INTEGRITY)

    def test_wrong_abi_and_wire(self):
        for key, bad in (("c_abi", 2), ("wire_protocol", 5)):
            e = copy.deepcopy(self.eco)
            e[key] = bad
            self.assertEqual(self.offline(e), mc.INTEGRITY, key)

    def test_online_tag_sha_mismatch_and_network(self):
        rep = mc.Report("t")
        with mock.patch.object(mc, "ls_remote_tag", lambda r, t: "0" * 40), mock.patch.object(mc, "github_api", lambda p: {"draft": False, "prerelease": False}), mock.patch.object(mc.Report, "summary"):
            self.em.check_online(self.eco, rep)
        self.assertEqual(rep.worst(), mc.INTEGRITY)
        rep = mc.Report("t")

        def down(*a):
            raise mc.Network("offline")
        with mock.patch.object(mc, "ls_remote_tag", down), mock.patch.object(mc, "github_api", down), mock.patch.object(mc.Report, "summary"):
            self.em.check_online(self.eco, rep)
        self.assertEqual(rep.worst(), mc.WARNING)


@unittest.skipUnless((TOOLS / "check_framework_pin.py").exists(), "adapters with a framework pin only")
class FrameworkPin(unittest.TestCase):
    def setUp(self):
        self.fp = load("check_framework_pin")
        self.manifest = json.loads((ROOT / "compatibility.json").read_text())
        self.key = "vllm_metal" if "vllm_metal" in self.manifest["frameworks"] else "exo"
        self.pin = self.manifest["frameworks"][self.key]
        self.sha = self.pin["fork_revision"]
        self.repo, self.tag = self.pin["fork"], self.pin["fork_tag"]

    def offline(self, manifest):
        rep = mc.Report("t")
        with mock.patch.object(mc.Report, "summary"):
            if self.key == "exo":
                self.fp.check_offline(manifest, rep)
            else:
                self.fp.check_offline(manifest, ROOT, rep)
        return rep.worst()

    def test_current_pin_is_valid(self):
        self.assertEqual(self.offline(self.manifest), mc.OK)

    def test_invalid_sha_and_missing_tag(self):
        m = copy.deepcopy(self.manifest)
        m["frameworks"][self.key]["fork_revision"] = self.sha[:10]
        self.assertEqual(self.offline(m), mc.INTEGRITY)
        m = copy.deepcopy(self.manifest)
        del m["frameworks"][self.key]["fork_tag"]
        self.assertEqual(self.offline(m), mc.INTEGRITY)
        m = copy.deepcopy(self.manifest)
        m["frameworks"][self.key]["fork_tag"] = "main"
        self.assertEqual(self.offline(m), mc.INTEGRITY)

    def online(self, ls, api=lambda p: {"merge_base_commit": {"sha": None}}):
        rep = mc.Report("t")
        with mock.patch.object(mc, "ls_remote_tag", ls), mock.patch.object(mc, "github_api", api), \
                mock.patch.object(self.fp.urllib.request, "urlopen", side_effect=OSError("offline")), mock.patch.object(mc.Report, "summary"):
            self.fp.check_online(self.manifest, rep)
        return [c for c, m in rep.items]

    def test_tag_resolving_elsewhere_fails_hard(self):
        self.assertIn(mc.INTEGRITY, self.online(lambda r, t: "f" * 40))
        self.assertIn(mc.INTEGRITY, self.online(lambda r, t: None))

    def test_matching_tag_passes_and_offline_only_warns(self):
        self.assertNotIn(mc.INTEGRITY, [c for c in self.online(lambda r, t: self.sha, api=lambda p: {"merge_base_commit": {"sha": self.pin.get("upstream_base", self.pin.get("revision"))}, "ahead_by": 1})])

        def down(*a):
            raise mc.Network("offline")
        codes = self.online(down, api=down)
        self.assertNotIn(mc.INTEGRITY, codes)
        self.assertIn(mc.WARNING, codes)

    def test_diff_normalization_ignores_mail_headers(self):
        if self.key != "vllm_metal":
            self.skipTest("vllm-tbccl only")
        body = "diff --git a/x b/x\nindex 1..2 100644\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n"
        mail = "From " + "0" * 40 + " Mon Sep 17 00:00:00 2001\nSubject: [PATCH] x\n---\n x | 2 +-\n\n" + body + "-- \n2.43.0\n\n"
        self.assertEqual(self.fp.normalize_diff(mail), self.fp.normalize_diff(body.replace("index 1..2", "index 9..8")))


if __name__ == "__main__":
    unittest.main()
