"""Reading result files that an interrupted run may have left truncated.

A killed process can leave the last line of a JSONL file or a manifest half written. The final line of a JSONL file is dropped when it does not parse (and only then);
a malformed line anywhere else is real corruption and still raises. A run whose manifest or records cannot be read is returned as an `unreadable` run so reports list
it as excluded instead of failing for the whole campaign."""
from __future__ import annotations

import json
from pathlib import Path

from .. import schema


def read_jsonl(path) -> list:
    lines = [line for line in Path(path).read_text().splitlines() if line.strip()]
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            if index != len(lines) - 1:
                raise
    return rows


def load_run(directory, tier):
    """(tier, run_dir, manifest, records); an unreadable manifest or records file gives status 'unreadable' with the reason and no records."""
    d = Path(directory)
    try:
        manifest = json.loads((d / "manifest.json").read_text())
        records = schema.read_records(d / "results.jsonl")
    except (ValueError, OSError) as error:
        return tier, d, {"status": "unreadable", "reason": f"manifest.json or results.jsonl cannot be read ({type(error).__name__}); the run was probably interrupted",
                         "counts": {}}, []
    return tier, d, manifest, records
