#!/usr/bin/env python3
"""Phase 38: summarize tbccl_hetero_allreduce_bench JSON output pairs.

Each AllReduce run produces two JSON files (one per rank, rank 0's is
authoritative for timing since it's the side this script is pointed at by
convention -- both ranks' verify_ok flags are reported). Usage:

    analyze_phase38_allreduce.py file1_rank0.json file1_rank1.json [...]

Prints Markdown tables: Correctness, Sync vs Async, Root comparison.
"""
import json
import sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1

    records = [load(p) for p in argv[1:]]

    print("### Correctness\n")
    print("| label | rank | root | backend | sync_verify_ok | async_verify_ok |")
    print("|---|---:|---:|---|---|---|")
    for r in records:
        print(f"| {r.get('label','')} | {r.get('rank')} | {r.get('root')} | "
              f"{r.get('backend')} | {r.get('sync_verify_ok', 'n/a')} | "
              f"{r.get('async_verify_ok', 'n/a')} |")

    print("\n### Sync vs Async (rank 0 only, median us)\n")
    print("| label | bytes | sync_median_us | async_median_us | ratio (sync/async) |")
    print("|---|---:|---:|---:|---:|")
    for r in records:
        if r.get("rank") != 0:
            continue
        sync_us = r.get("sync_median_us")
        async_us = r.get("async_median_us")
        if sync_us and async_us:
            ratio = sync_us / async_us
            print(f"| {r.get('label','')} | {r.get('bytes')} | {sync_us:.1f} | "
                  f"{async_us:.1f} | {ratio:.3f}x |")

    print("\n### Root comparison (rank 0 only, async median us)\n")
    by_bytes = {}
    for r in records:
        if r.get("rank") != 0:
            continue
        key = r.get("bytes")
        by_bytes.setdefault(key, {})[r.get("root")] = r.get("async_median_us")
    print("| bytes | root 0 (Linux) us | root 1 (Mac) us |")
    print("|---:|---:|---:|")
    for b, roots in sorted(by_bytes.items()):
        print(f"| {b} | {roots.get(0, 'n/a')} | {roots.get(1, 'n/a')} |")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
