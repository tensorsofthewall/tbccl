"""The shipped configuration profiles: each loads and expands, holds placeholders instead of private values, names benchmarks rather than implementing workloads, and maps to the providers it needs."""
import re
import unittest
from pathlib import Path

from benchmarks.suite import config as cfg, profiles, providers
from benchmarks.suite.tests.common import REPO, requires_yaml


@requires_yaml
class Profiles(unittest.TestCase):
    def test_the_expected_profiles_exist(self):
        self.assertEqual([r["name"] for r in profiles.list_profiles()], list(profiles.ORDER))
        self.assertEqual({p.stem for p in profiles.PROFILE_DIR.glob("*.yaml")}, set(profiles.ORDER))

    def test_every_profile_loads_and_expands_to_workload_points(self):
        for name in profiles.ORDER:
            c = cfg.load_config(profiles.PROFILE_DIR / f"{name}.yaml", auto_overlay=False)
            self.assertTrue(cfg.expand_matrix(c), name)

    def test_profiles_hold_no_private_values(self):
        for path in profiles.PROFILE_DIR.glob("*.yaml"):
            text = path.read_text()
            addresses = set(re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)) - {"127.0.0.1"}
            self.assertEqual(addresses, set(), path.name)
            for marker in ("/home/", "/Users/", "/mnt/", ".ssh", "@"):
                self.assertNotIn(marker, text, f"{path.name}: {marker}")

    def test_physical_profiles_declare_it_and_loopback_ones_do_not(self):
        by = {r["name"]: r for r in profiles.list_profiles()}
        for name in ("native", "device", "vllm", "exo"):
            self.assertTrue(by[name]["physical"], name)
        for name in ("quick", "pytorch", "custom"):
            self.assertFalse(by[name]["physical"], name)

    def test_profiles_map_to_providers_and_native_ones_need_none(self):
        by = {r["name"]: r for r in profiles.list_profiles()}
        self.assertEqual(by["pytorch"]["providers"], ["torch-tbccl"])
        self.assertEqual(by["vllm"]["providers"], ["vllm-tbccl"])
        self.assertEqual(by["exo"]["providers"], ["exo-tbccl"])
        for name in ("quick", "native", "device", "custom"):
            self.assertEqual(by[name]["providers"], [], name)

    def test_profiles_ready_to_run_have_no_placeholders_and_others_say_how_many(self):
        by = {r["name"]: r for r in profiles.list_profiles()}
        self.assertEqual((by["quick"]["placeholders"], by["custom"]["placeholders"]), (0, 0))
        for name in ("native", "device", "pytorch", "vllm", "exo"):
            self.assertGreater(by[name]["placeholders"], 0, name)

    def test_a_profile_holds_names_and_settings_not_workload_code(self):
        for path in profiles.PROFILE_DIR.glob("*.yaml"):
            c = cfg.load_config(path, auto_overlay=False)
            from benchmarks.suite.adapters import REGISTRY
            for w in c.workloads:
                self.assertIn(w["benchmark"], REGISTRY, f"{path.name}: {w['benchmark']}")

    def test_the_exo_profile_never_enables_a_fast_path(self):
        from benchmarks.suite.adapters.exo import FORBIDDEN_FASTPATH_ENV
        text = (profiles.PROFILE_DIR / "exo.yaml").read_text()
        for name in FORBIDDEN_FASTPATH_ENV:
            self.assertNotIn(name, text)
        c = cfg.load_config(profiles.PROFILE_DIR / "exo.yaml", auto_overlay=False)
        for w in c.workloads:
            self.assertNotEqual((w.get("options") or {}).get("tier"), "fastpath")

    def test_the_provider_directories_in_profiles_are_the_ones_the_runner_reads(self):
        for name, key in (("pytorch", "torch_bench_dir"), ("vllm", "vllm_bench_dir"), ("exo", "exo_bench_dir")):
            self.assertIn(key, (profiles.PROFILE_DIR / f"{name}.yaml").read_text())
        self.assertEqual({providers.PROVIDERS[p]["dir"] for p in providers.PROVIDERS}, {"torch_bench_dir", "vllm_bench_dir", "exo_bench_dir"})

    def test_cli_lists_them(self):
        import json
        import subprocess
        import sys
        out = subprocess.run([sys.executable, "-m", "benchmarks.suite", "profiles", "--json"], capture_output=True, text=True, cwd=str(REPO))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual([r["name"] for r in json.loads(out.stdout)], list(profiles.ORDER))


if __name__ == "__main__":
    unittest.main()
