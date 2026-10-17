"""Build identity in comparisons: executable hashes, TBCCL versions per host, provider and framework versions, the compiled extension an adapter wheel bundles and the workload scripts.
Two identities are matching, different or unknown, never just equal or not; synthetic mismatches show each case."""
import copy
import unittest

from benchmarks.suite import provenance_compare as pc
from benchmarks.suite.baselines import comparison
from benchmarks.suite.reporting import compare
from benchmarks.suite.tests.test_baselines_comparison import STEADY, TPOT, snap
from benchmarks.suite.tests.test_compare_report import reps

PROVIDER = {"provider_id": "exo-tbccl", "host": "host-a", "status": "available", "provider_version": "0.3.0", "framework_versions": {"exo": "0.3.70", "mlx": "0.32.0", "python": "3.13.0"},
            "native_modules": {"exo_tbccl/_native.cpython-313.so": {"sha256": "aa" * 32, "bytes": 5}}, "entry_point_sha256": {"client": "bb" * 32}}


def run(values, providers=None, exe="1", tbccl="0.6.0", wire=4, name="a"):
    recs = reps(values, run=name, sha=exe)
    for r in recs:
        r["environment"]["hosts"][0]["build"] = {"tbccl_package_version": tbccl, "c_abi": 1, "wire_protocol": wire, "executable_sha256": exe * 8}
        if providers is not None:
            r["environment"]["providers"] = copy.deepcopy(providers)
    return recs


def with_change(**changes):
    p = copy.deepcopy(PROVIDER)
    for k, v in changes.items():
        p[k] = v
    return [p]


class Identity(unittest.TestCase):
    def test_every_recorded_fact_is_extracted(self):
        ident = pc.identity_from_records(run([1, 2, 3], [PROVIDER]))
        self.assertEqual(ident["provider.exo-tbccl.host-a.version"], "0.3.0")
        self.assertEqual(ident["provider.exo-tbccl.host-a.framework.mlx"], "0.32.0")
        self.assertNotIn("provider.exo-tbccl.host-a.framework.python", ident)
        self.assertEqual(ident["provider.exo-tbccl.host-a.native._native"], "aa" * 32)
        self.assertEqual(ident["provider.exo-tbccl.host-a.script.client"], "bb" * 32)
        self.assertEqual(ident["tbccl.h.package_version"], "0.6.0")
        self.assertEqual(ident["tbccl.h.wire_protocol"], "4")
        self.assertEqual(ident["executable.collective_bench.sha256"], "1")

    def test_a_run_that_mixed_builds_is_not_mistaken_for_one_build(self):
        a, b = run([1, 2], exe="1"), run([3, 4], exe="2")
        self.assertEqual(pc.identity_from_records(a + b)["executable.collective_bench.sha256"], "1|2")

    def test_failed_records_do_not_contribute(self):
        recs = run([1, 2])
        recs[0]["status"]["state"] = "failure"
        recs[0]["environment"]["hosts"][0]["build"]["tbccl_package_version"] = "9.9.9"
        self.assertEqual(pc.identity_from_records(recs)["tbccl.h.package_version"], "0.6.0")

    def test_a_provider_that_was_not_described_gives_a_state_and_no_invented_versions(self):
        ident = pc.identity_from_records(run([1, 2], with_change(status="bundled_compat", provider_version=None, framework_versions={}, native_modules=None, entry_point_sha256={})))
        self.assertEqual(ident["provider.exo-tbccl.host-a.state"], "bundled_compat")
        self.assertFalse([k for k in ident if k.endswith((".version", ".native._native"))])


class ThreeStates(unittest.TestCase):
    def test_matching_different_and_unknown_are_distinct(self):
        r = pc.compare_identity({"a": "1", "b": "1", "c": "1"}, {"a": "1", "b": "2", "d": "1"})
        self.assertEqual((r["matching"], r["different"], r["unknown"]), (["a"], {"b": ["1", "2"]}, {"c": "candidate", "d": "baseline"}))

    def test_two_sides_without_any_identity_are_not_known_matching(self):
        r = pc.compare_identity({}, {})
        self.assertEqual((r["matching"], r["known_build_on_both_sides"]), ([], False))
        self.assertIn("cannot be attributed to a build", " ".join(pc.describe(r)))

    def test_one_side_unrecorded_is_unknown_not_equal(self):
        r = pc.compare_identity({"x": "1"}, {})
        self.assertEqual((r["matching"], r["unknown"]), ([], {"x": "candidate"}))
        self.assertFalse(r["known_build_on_both_sides"])


class RunComparison(unittest.TestCase):
    def md(self, a, b):
        r = compare.compare(a, b)
        return r, compare.to_markdown(r)

    def test_identical_builds_are_reported_as_identical_on_every_recorded_fact(self):
        r, md = self.md(run([100, 101, 99], [PROVIDER], name="a"), run([100, 101, 99], [PROVIDER], name="b"))
        p = r["comparisons"][0]["provenance"]
        self.assertTrue(p["matching"] and not p["different"] and not p["unknown"])
        self.assertIn("identical on every recorded fact", md)
        self.assertNotIn("## Build and provider provenance", md)

    def test_a_changed_compiled_extension_framework_adapter_script_executable_and_tbccl_are_each_listed(self):
        cand_provider = with_change(provider_version="0.3.1", framework_versions={"exo": "0.3.71", "mlx": "0.32.0"}, native_modules={"exo_tbccl/_native.cpython-313.so": {"sha256": "cc" * 32, "bytes": 5}},
                                    entry_point_sha256={"client": "dd" * 32})
        r, md = self.md(run([100, 101, 99], [PROVIDER], name="a"), run([100, 101, 99], cand_provider, exe="2", tbccl="0.7.0", wire=5, name="b"))
        different = r["comparisons"][0]["provenance"]["different"]
        for fact in ("provider.exo-tbccl.host-a.version", "provider.exo-tbccl.host-a.framework.exo", "provider.exo-tbccl.host-a.native._native", "provider.exo-tbccl.host-a.script.client",
                     "executable.collective_bench.sha256", "tbccl.h.package_version", "tbccl.h.wire_protocol"):
            self.assertIn(fact, different)
        self.assertNotIn("provider.exo-tbccl.host-a.framework.mlx", different)
        self.assertIn("## Build and provider provenance", md)
        self.assertIn("provider.exo-tbccl.host-a.framework.exo: 0.3.70 -> 0.3.71", md)

    def test_provider_provenance_recorded_on_one_side_only_is_unknown_not_a_match(self):
        r, md = self.md(run([100, 101, 99], [PROVIDER], name="a"), run([100, 101, 99], None, name="b"))
        unknown = r["comparisons"][0]["provenance"]["unknown"]
        self.assertEqual(unknown["provider.exo-tbccl.host-a.version"], "candidate")
        self.assertIn("not recorded for the candidate (unknown, not assumed equal)", md)
        self.assertNotIn("identical on every recorded fact", md)

    def test_builds_that_were_never_identified_are_not_reported_as_the_same_build(self):
        a, b = reps([100, 101, 99], run="a"), reps([100, 101, 99], run="b")
        for r in a + b:
            r["benchmark"]["executable_sha256"] = None
        r, md = self.md(a, b)
        p = r["comparisons"][0]["provenance"]
        self.assertFalse(p["known_build_on_both_sides"])
        self.assertIn("cannot be attributed to a build", md)
        self.assertNotIn("identical on every recorded fact", md)

    def test_a_bundled_compat_run_does_not_match_a_described_provider(self):
        compat = with_change(status="bundled_compat", provider_version=None, framework_versions={}, native_modules=None)
        r, _ = self.md(run([100, 101, 99], [PROVIDER], name="a"), run([100, 101, 99], compat, name="b"))
        p = r["comparisons"][0]["provenance"]
        self.assertEqual(p["unknown"]["provider.exo-tbccl.host-a.version"], "candidate")
        self.assertIn("provider.exo-tbccl.host-a.state", p["unknown"])


class BaselineComparison(unittest.TestCase):
    def cmp(self, base_versions, cand_versions):
        return comparison.compare(snap(TPOT, STEADY, software_versions=base_versions), snap(TPOT, [v * 1.5 for v in STEADY], software_versions=cand_versions))

    def test_a_difference_in_a_recorded_fact_is_reported_with_both_values(self):
        r = self.cmp({"exo": "0.3.70", "native.sha": "aa"}, {"exo": "0.3.71", "native.sha": "aa"})
        self.assertEqual(r["software_version_differences"], {"exo": ["0.3.70", "0.3.71"]})
        self.assertEqual(r["software_provenance"]["matching"], ["native.sha"])

    def test_a_fact_recorded_on_one_side_only_is_unknown_and_labelled(self):
        r = self.cmp({"exo": "0.3.70", "mlx": "0.32.0"}, {"exo": "0.3.70"})
        self.assertEqual(r["software_version_differences"], {})
        self.assertEqual(r["software_provenance"]["unknown"], {"mlx": "candidate"})
        self.assertTrue([l for l in r["labels"] if "only one side" in l])

    def test_unrecorded_builds_cannot_be_attributed_and_the_classification_is_unchanged(self):
        r = self.cmp({}, {})
        self.assertTrue([l for l in r["labels"] if "build identity not recorded" in l])
        self.assertEqual(r["classification"], self.cmp({"x": "1"}, {"x": "1"})["classification"])
        md = comparison.to_markdown([self.cmp({"a": "1"}, {})])
        self.assertIn("recorded for only one side", md)

    def test_matching_known_builds_carry_no_provenance_label(self):
        r = self.cmp({"x": "1"}, {"x": "1"})
        self.assertFalse([l for l in r["labels"] if "build" in l])


class Attribution(unittest.TestCase):
    """What may be concluded about the cause of a difference: one changed component, several at once, or nothing known."""
    def result(self, base, cand):
        return pc.compare_identity(base, cand)

    def test_a_single_changed_component_is_named_and_not_refused(self):
        r = self.result({"executable.x.sha256": "1", "provider.p.h.version": "0.2.0"}, {"executable.x.sha256": "2", "provider.p.h.version": "0.2.0"})
        self.assertEqual(pc.components_changed(r), [pc.COMPONENT_TBCCL])
        line = pc.attribution(r)[0]
        self.assertIn("only the TBCCL build differs", line)
        self.assertNotIn("WARNING", line)

    def test_an_unintended_framework_change_is_named_as_the_software_underneath(self):
        r = self.result({"executable.x.sha256": "1", "provider.p.h.framework.torch": "2.13.0"}, {"executable.x.sha256": "1", "provider.p.h.framework.torch": "2.13.1"})
        self.assertEqual(pc.components_changed(r), [pc.COMPONENT_FRAMEWORK])
        self.assertIn("software underneath the run changed", pc.attribution(r)[0])

    def test_several_changed_components_are_warned_about_together(self):
        r = self.result({"executable.x.sha256": "1", "provider.p.h.version": "0.2.0", "provider.p.h.framework.torch": "2.13.0"},
                        {"executable.x.sha256": "2", "provider.p.h.version": "0.2.1", "provider.p.h.framework.torch": "2.13.1"})
        self.assertEqual(len(pc.components_changed(r)), 3)
        line = pc.attribution(r)[0]
        self.assertTrue(line.startswith("WARNING"))
        self.assertIn("cannot be attributed to one component alone", line)

    def test_an_adapter_wheel_swap_is_one_component_not_three(self):
        """A new adapter wheel changes its version, its compiled extension and its scripts together."""
        r = self.result({"provider.p.h.version": "0.2.0", "provider.p.h.native.ext": "aa", "provider.p.h.script.s": "bb"},
                        {"provider.p.h.version": "0.2.1", "provider.p.h.native.ext": "cc", "provider.p.h.script.s": "dd"})
        self.assertEqual(pc.components_changed(r), [pc.COMPONENT_ADAPTER])

    def test_unknown_facts_do_not_count_as_a_changed_component(self):
        r = self.result({"provider.p.h.version": "0.2.0", "provider.p.h.native.ext": "aa"}, {"provider.p.h.version": "0.2.0"})
        self.assertEqual(pc.components_changed(r), [])
        self.assertEqual(pc.attribution(r), [])
        self.assertIn("provider.p.h.native.ext", r["unknown"])

    def test_the_run_report_and_the_baseline_comparison_both_carry_the_warning(self):
        cand = with_change(provider_version="0.2.1", framework_versions={"exo": "0.3.71", "mlx": "0.32.0"})
        r, md = RunComparison().md(run([100, 101, 99], [PROVIDER], name="a"), run([100, 101, 99], cand, exe="2", name="b"))
        self.assertIn("cannot be attributed to one component alone", md)
        base = comparison.compare(snap(TPOT, STEADY, software_versions={"torch": "2.13.0", "tbccl": "0.6.0"}), snap(TPOT, STEADY, software_versions={"torch": "2.13.1", "tbccl": "0.6.1"}))
        self.assertEqual(base["software_provenance"]["changed_components"], [pc.COMPONENT_TBCCL, pc.COMPONENT_FRAMEWORK])
        self.assertTrue([l for l in base["labels"] if "cannot be attributed to one component alone" in l])
        self.assertEqual(base["classification"], comparison.compare(snap(TPOT, STEADY, software_versions={"x": "1"}), snap(TPOT, STEADY, software_versions={"x": "1"}))["classification"])


if __name__ == "__main__":
    unittest.main()
