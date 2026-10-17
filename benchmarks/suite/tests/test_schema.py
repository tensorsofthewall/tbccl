import copy
import json
import unittest

from benchmarks.suite import schema
from benchmarks.suite.importers import import_path
from benchmarks.suite.tests.common import FIXTURES


def good():
    return copy.deepcopy(import_path(FIXTURES / "collective_all_reduce_w2.csv").records[0])


class Validation(unittest.TestCase):
    def test_import_output_validates(self):
        self.assertEqual(schema.validate_record(good()), [])

    def test_required_and_unexpected_fields(self):
        r = good()
        del r["verification"]
        self.assertTrue(any("verification" in e for e in schema.validate_record(r)))
        r = good()
        r["surprise"] = 1
        self.assertTrue(any("unexpected" in e for e in schema.validate_record(r)))

    def test_enums_types_and_minimums(self):
        r = good()
        r["status"]["state"] = "great"
        self.assertTrue(schema.validate_record(r))
        r = good()
        r["configuration"]["rank_count"] = 0
        self.assertTrue(schema.validate_record(r))
        r = good()
        r["measurements"]["timing"]["statistics"]["median"] = "fast"
        self.assertTrue(schema.validate_record(r))
        r = good()
        r["measurements"]["throughput"][0]["value"] = -1
        self.assertTrue(schema.validate_record(r))

    def test_failed_records_never_carry_measurements(self):
        r = good()
        r["status"]["state"] = "failure"
        self.assertTrue(any("must not carry measurements" in e for e in schema.validate_record(r)))
        r["measurements"] = None
        self.assertEqual(schema.validate_record(r), [])

    def test_success_needs_measurements_and_passing_verification(self):
        r = good()
        r["measurements"] = None
        self.assertTrue(any("must carry measurements" in e for e in schema.validate_record(r)))
        r = good()
        r["verification"]["status"] = "failed"
        self.assertTrue(any("verification failed" in e for e in schema.validate_record(r)))

    def test_statistics_need_provenance_and_a_primary(self):
        r = good()
        del r["measurements"]["timing"]["statistics_provenance"]["median"]
        self.assertTrue(any("no provenance" in e for e in schema.validate_record(r)))
        r = good()
        r["measurements"]["timing"]["primary_statistic"] = "p99"
        self.assertTrue(any("primary statistic" in e for e in schema.validate_record(r)))
        r = good()
        r["measurements"]["timing"]["samples"] = [1, 2]
        self.assertTrue(any("sample_count" in e for e in schema.validate_record(r)))

    def test_version_check(self):
        with self.assertRaises(schema.SchemaError):
            schema.check_schema_version({"schema_version": "tbccl-bench-result/9"})

    def test_builtin_validator_agrees_with_jsonschema_when_installed(self):
        try:
            import jsonschema
        except ImportError:
            self.skipTest("jsonschema not installed")
        full = schema.load_schema()
        jsonschema.Draft202012Validator.check_schema(full)
        validator = jsonschema.Draft202012Validator(full)
        r = good()
        self.assertEqual(list(validator.iter_errors(r)), [])
        for mutate in (lambda x: x.update(schema_version="x"), lambda x: x["status"].update(state="bogus"), lambda x: x["configuration"].update(rank_count=0),
                       lambda x: x.pop("artifacts"), lambda x: x["measurements"]["throughput"][0].update(unit="parsecs")):
            bad = good()
            mutate(bad)
            self.assertTrue(list(validator.iter_errors(bad)))
            self.assertTrue(schema.validate_against(bad, full))


if __name__ == "__main__":
    unittest.main()
