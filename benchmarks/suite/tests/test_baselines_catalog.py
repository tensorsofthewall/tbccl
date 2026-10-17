import copy
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.suite.baselines import BASELINE_SCHEMA_VERSION, catalog
from benchmarks.suite.schema import SchemaError


def definition(**over):
    case = {"case_id": "x.case", "campaign_id": "t", "family": "native_p2p", "benchmark_adapter": "tensor_transfer", "metric_name": "completion_confirmed_time", "unit": "us",
            "aggregation": "median_of_repetition_medians", "direction": "lower_is_better", "validity": {"status": "valid"}, "operation": "end_to_end", "payload_bytes": 4096}
    case.update(over)
    return case


def catalog_doc(kind="definitions", *cases):
    return {"schema_version": BASELINE_SCHEMA_VERSION, "kind": kind, "catalog_id": "synthetic", "cases": list(cases)}


def snapshot_case(observations=(10.0, 11.0, 12.0), **over):
    case = definition(run_ids=["r1"], sample_count=90, independent_lifecycle_count=len(observations), value=sorted(observations)[len(observations) // 2],
                      observations=list(observations), source_artifacts=[{"path": "runA/results.jsonl", "sha256": None}], replication_unit="repetition")
    case.update(over)
    return case


class ShippedDefinitions(unittest.TestCase):
    def test_every_shipped_catalog_validates_and_is_public(self):
        docs = catalog.definitions_catalogs()
        self.assertGreaterEqual(len(docs), 4)
        for doc in docs:
            self.assertEqual(doc["kind"], "definitions")
            self.assertEqual(catalog.validate_catalog(doc), [])
            self.assertEqual(catalog.privacy_problems(doc), [])

    def test_case_ids_are_unique_across_catalogs(self):
        ids = [c["case_id"] for doc in catalog.definitions_catalogs() for c in doc["cases"]]
        self.assertEqual(len(ids), len(set(ids)))

    def test_definitions_carry_no_measurements(self):
        for doc in catalog.definitions_catalogs():
            for case in doc["cases"]:
                self.assertFalse([f for f in catalog.MEASURED_FIELDS if f in case], case["case_id"])

    def test_unknown_case_id_raises(self):
        with self.assertRaises(KeyError):
            catalog.case_by_id("does.not.exist")


class Parsing(unittest.TestCase):
    def test_unsupported_schema_version_is_rejected_with_the_version(self):
        doc = catalog_doc("definitions", definition())
        doc["schema_version"] = "tbccl-bench-baseline/2"
        errors = catalog.validate_catalog(doc)
        self.assertEqual(len(errors), 1)
        self.assertIn("tbccl-bench-baseline/2", errors[0])

    def test_missing_required_field_and_unknown_field_are_reported(self):
        bad = definition()
        del bad["metric_name"]
        bad["metrik"] = "typo"
        errors = catalog.validate_catalog(catalog_doc("definitions", bad))
        self.assertTrue(any("metric_name" in e for e in errors))
        self.assertTrue(any("metrik" in e for e in errors))

    def test_duplicate_case_ids_are_rejected(self):
        errors = catalog.validate_catalog(catalog_doc("definitions", definition(), definition()))
        self.assertTrue(any("duplicate" in e for e in errors))

    def test_empty_catalog_is_valid_and_loads(self):
        self.assertEqual(catalog.validate_catalog(catalog_doc("definitions")), [])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            path.write_text(json.dumps(catalog_doc("snapshot")))
            self.assertEqual(catalog.load_catalog(path)["cases"], [])

    def test_load_raises_with_all_errors(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "c.json"
            path.write_text(json.dumps(catalog_doc("definitions", definition(direction="sideways"))))
            with self.assertRaises(SchemaError) as ctx:
                catalog.load_catalog(path)
            self.assertTrue(ctx.exception.errors)

    def test_non_object_is_rejected(self):
        self.assertEqual(len(catalog.validate_catalog([])), 1)


class Definitions(unittest.TestCase):
    def test_measured_fields_are_rejected_in_definitions(self):
        errors = catalog.validate_catalog(catalog_doc("definitions", definition(value=1.0, observations=[1.0])))
        self.assertTrue(any("measured fields" in e for e in errors))

    def test_private_values_are_reported_by_category_without_the_text(self):
        for text, label in (("host at 198.51.100.40", "IPv4"), ("see /home/example/x", "private filesystem path"), ("aa:bb:cc:dd:ee:ff", "MAC"),
                            ("who@example.org", "e-mail"), ("fe80::1", "IPv6")):
            problems = catalog.privacy_problems({"notes": text})
            self.assertEqual(len(problems), 1, text)
            self.assertIn(label, problems[0])
            self.assertNotIn("example", problems[0])
            self.assertNotIn("198.51.100.40", problems[0])

    def test_loopback_and_versions_are_not_private(self):
        self.assertEqual(catalog.privacy_problems({"a": "127.0.0.1", "b": "0.0.0.0", "c": "mlx 0.32.4", "d": "exo-0.3.70;mlx-0.32"}), [])

    def test_private_value_in_a_definition_fails_validation(self):
        errors = catalog.validate_catalog(catalog_doc("definitions", definition(notes="192.0.2.15")))
        self.assertTrue(any(e.startswith("privacy:") for e in errors))


class Snapshots(unittest.TestCase):
    def test_valid_snapshot(self):
        self.assertEqual(catalog.validate_catalog(catalog_doc("snapshot", snapshot_case())), [])

    def test_snapshot_without_values_or_artifacts_is_rejected(self):
        bare = definition()
        errors = catalog.validate_catalog(catalog_doc("snapshot", bare))
        self.assertTrue(any("needs" in e for e in errors))

    def test_missing_metric_value_is_reported_for_valid_cases_only(self):
        case = snapshot_case()
        case["value"] = None
        self.assertTrue(catalog.validate_catalog(catalog_doc("snapshot", case)))
        invalid = copy.deepcopy(case)
        invalid["validity"] = {"status": "invalid"}
        invalid["observations"] = []
        self.assertEqual(catalog.validate_catalog(catalog_doc("snapshot", invalid)), [])

    def test_observation_count_must_equal_independent_units(self):
        case = snapshot_case(independent_lifecycle_count=5)
        errors = catalog.validate_catalog(catalog_doc("snapshot", case))
        self.assertTrue(any("independent" in e for e in errors))

    def test_value_must_be_the_median_of_the_observations(self):
        errors = catalog.validate_catalog(catalog_doc("snapshot", snapshot_case(value=99.0)))
        self.assertTrue(any("not the median" in e for e in errors))

    def test_non_finite_observations_are_rejected(self):
        errors = catalog.validate_catalog(catalog_doc("snapshot", snapshot_case(observations=[1.0, float("inf"), 3.0], value=3.0)))
        self.assertTrue(any("NaN or infinity" in e for e in errors))

    def test_pair_ids_must_match_observations(self):
        errors = catalog.validate_catalog(catalog_doc("snapshot", snapshot_case(pair_ids=["a", "b"])))
        self.assertTrue(any("pair_ids" in e for e in errors))

    def test_absolute_or_escaping_artifact_paths_are_rejected(self):
        for path in ("/abs/path.jsonl", "../outside.jsonl", "~/x"):
            case = snapshot_case(source_artifacts=[{"path": path, "sha256": None}])
            self.assertTrue(any("relative" in e for e in catalog.validate_catalog(catalog_doc("snapshot", case))), path)


class Provenance(unittest.TestCase):
    def test_present_matching_artifacts_pass(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "runA").mkdir()
            f = Path(d) / "runA" / "results.jsonl"
            f.write_text("{}\n")
            case = snapshot_case(source_artifacts=[{"path": "runA/results.jsonl", "sha256": catalog.sha256_file(f)}])
            self.assertEqual(catalog.verify_provenance(case, d), [])

    def test_missing_and_modified_artifacts_are_reported(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "runA").mkdir()
            f = Path(d) / "runA" / "results.jsonl"
            f.write_text("{}\n")
            digest = catalog.sha256_file(f)
            f.write_text("{} changed\n")
            problems = catalog.verify_provenance(snapshot_case(source_artifacts=[{"path": "runA/results.jsonl", "sha256": digest}, {"path": "runB/results.jsonl", "sha256": None}]), d)
            self.assertEqual(len(problems), 2)
            self.assertTrue(any("no longer matches" in p for p in problems))
            self.assertTrue(any("missing" in p for p in problems))

    def test_case_without_artifacts_cannot_be_traced(self):
        problems = catalog.verify_provenance(snapshot_case(source_artifacts=[]), ".")
        self.assertEqual(len(problems), 1)
        self.assertIn("traced", problems[0])


if __name__ == "__main__":
    unittest.main()
