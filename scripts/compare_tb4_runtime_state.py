#!/usr/bin/env python3
"""Structured diff of two capture_tb4_runtime_state.py JSON snapshots.

The sustained-session reproduction work. Read-only, does not mutate either input. Every leaf field
found in either snapshot is classified into exactly one of:

  changed     -- present in both, with a different value
  unchanged   -- present in both, with the same value
  unavailable -- missing (None or absent) in at least one snapshot, so no
                 real comparison is possible

A field must never be silently treated as "changed" just because it is
missing from one side (e.g. a historical the NHI ring-cadence work snapshot that predates
a field this tool now captures) -- that would fabricate a difference that
was never actually observed.

List-valued fields (e.g. pci_devices) are flattened by positional index.
capture_tb4_runtime_state.py sorts every list it produces by a stable key
(bdf, cpu number, etc.), so this is safe here, but a reordered list from
an untrusted source would misalign under this scheme -- not a concern for
this project's own snapshot format.
"""
import argparse
import json
from pathlib import Path


def _flatten(obj, prefix=""):
    """Yield (dotted.path, value) for every leaf (non-dict, non-list) value in obj."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(value, path)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            path = f"{prefix}.{index}" if prefix else str(index)
            yield from _flatten(value, path)
    else:
        yield prefix, obj


def diff_snapshots(before, after, ignore_prefixes=()):
    """Compare two snapshot dicts. Returns changed/unchanged/unavailable maps.

    ignore_prefixes: dotted-path prefixes to skip entirely (e.g. timestamps,
    raw command blobs not meant for field-by-field comparison).
    """
    before_flat = dict(_flatten(before))
    after_flat = dict(_flatten(after))
    all_paths = sorted(set(before_flat) | set(after_flat))

    changed = {}
    unchanged = {}
    unavailable = {}

    for path in all_paths:
        if any(path == p or path.startswith(p + ".") for p in ignore_prefixes):
            continue
        has_before = path in before_flat
        has_after = path in after_flat
        if not (has_before and has_after):
            unavailable[path] = {
                "before": before_flat.get(path, "<missing>"),
                "after": after_flat.get(path, "<missing>"),
            }
            continue
        b, a = before_flat[path], after_flat[path]
        if b is None or a is None:
            unavailable[path] = {"before": b, "after": a}
        elif b == a:
            unchanged[path] = b
        else:
            changed[path] = {"before": b, "after": a}

    return {
        "changed": changed,
        "unchanged_count": len(unchanged),
        "unavailable": unavailable,
        "total_fields_compared": len(all_paths),
    }


DEFAULT_IGNORE_PREFIXES = (
    "timestamp_utc",
    "commands.kernel",
    "kernel_events",
    "lspci_verbose",
    "ethtool_offloads",
    "ethtool_coalesce",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--include-noisy", action="store_true",
                         help="also diff free-text/timestamp fields normally ignored "
                              "(DEFAULT_IGNORE_PREFIXES)")
    parser.add_argument("--output")
    args = parser.parse_args()

    before = json.loads(Path(args.before).read_text())
    after = json.loads(Path(args.after).read_text())
    ignore = () if args.include_noisy else DEFAULT_IGNORE_PREFIXES
    result = diff_snapshots(before, after, ignore_prefixes=ignore)

    text = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text)
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
