"""python3 -m benchmarks.suite.baselines {validate,check-public,compare,provenance}"""
from __future__ import annotations

import argparse
import json
import sys

from . import catalog, comparison


def _cmd_validate(args):
    status = 0
    for path in args.catalog:
        try:
            doc = catalog.load_catalog(path)
            print(f"{path}: ok ({doc['kind']}, {len(doc['cases'])} cases)")
        except Exception as error:  # SchemaError and JSON errors alike
            print(f"{path}: INVALID\n  " + "\n  ".join(getattr(error, "errors", [str(error)])))
            status = 1
    return status


def _cmd_check_public(args):
    problems = []
    for path in args.catalog:
        doc = json.loads(open(path).read())
        if doc.get("kind") != "definitions":
            problems.append(f"{path}: only definitions may be published (kind is {doc.get('kind')!r})")
        problems += [f"{path}: {p}" for p in catalog.privacy_problems(doc)]
        problems += [f"{path}: {e}" for e in catalog.validate_catalog(doc)]
    print("\n".join(problems) if problems else "public catalogs are clean")
    return 1 if problems else 0


def _cmd_compare(args):
    base = catalog.index_cases(catalog.load_catalog(args.baseline))
    cand = catalog.index_cases(catalog.load_catalog(args.candidate))
    results = []
    for case_id, case in sorted(cand.items()):
        if args.case and case_id != args.case:
            continue
        ref = base.get(case_id)
        if ref is None:
            results.append({"case_id": case_id, "candidate_case_id": case_id, "classification": "incompatible", "reasons": ["no baseline case with this id"], "labels": [],
                            "software_version_differences": {}})
            continue
        results.append(comparison.compare(ref, case, purpose=args.purpose))
    print(comparison.to_markdown(results))
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(results, handle, indent=1, default=str)
    return 0


def _cmd_provenance(args):
    doc = catalog.load_catalog(args.catalog)
    problems = [p for case in doc["cases"] for p in catalog.verify_provenance(case, args.results_root)]
    print("\n".join(problems) if problems else "all source artifacts present and unchanged")
    return 1 if problems else 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="benchmarks.suite.baselines", description="Baseline catalogs and offline regression comparison.")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate", help="validate catalogs against the schema and invariants")
    p.add_argument("catalog", nargs="+")
    p.set_defaults(func=_cmd_validate)
    p = sub.add_parser("check-public", help="check that catalogs are publishable definitions without private values")
    p.add_argument("catalog", nargs="+")
    p.set_defaults(func=_cmd_check_public)
    p = sub.add_parser("compare", help="classify the cases of a candidate snapshot against a baseline snapshot")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--purpose", default="regression", choices=sorted(comparison.comparability.PURPOSES))
    p.add_argument("--case")
    p.add_argument("--json")
    p.set_defaults(func=_cmd_compare)
    p = sub.add_parser("provenance", help="verify that a snapshot's source artifacts exist and match their hashes")
    p.add_argument("catalog")
    p.add_argument("--results-root", required=True)
    p.set_defaults(func=_cmd_provenance)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
