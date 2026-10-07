#!/usr/bin/env python3
"""Summarize tbccl_async_transfer_bench / tbccl_hetero_allreduce_bench
/ tbccl_bucketed_allreduce_bench JSON output for the Metal-direct A/B comparison.

Usage:
    analyze_metal_direct.py --p2p file1.json file2.json [...]
    analyze_metal_direct.py --allreduce file1.json file2.json [...]
    analyze_metal_direct.py --timing-log file.timing.log [...]

Prints Markdown tables pairing "*_adapter*" / "*_direct*" labeled runs.
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path


def load(path):
    text = Path(path).read_text()
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text)
    return obj


def pair_by_path(records):
    """Group records whose label differs only by _adapter/_direct suffix."""
    pairs = {}
    for name, r in records.items():
        label = r.get("label", name)
        for suffix in ("_adapter", "_direct"):
            if label.endswith(suffix):
                base = label[: -len(suffix)]
                pairs.setdefault(base, {})[suffix[1:]] = r
                break
    return pairs


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2p", nargs="*", default=[])
    parser.add_argument("--allreduce", nargs="*", default=[])
    parser.add_argument("--timing-log", nargs="*", default=[])
    args = parser.parse_args(argv)

    if args.p2p:
        records = {Path(p).stem: load(p) for p in args.p2p if load(p).get("role") == "sender"}
        pairs = pair_by_path(records)
        print("### P2P (sender-side authoritative timing)\n")
        print("| config | adapter_us | direct_us | direct vs adapter |")
        print("|---|---:|---:|---:|")
        for base, p in sorted(pairs.items()):
            a, d = p.get("adapter"), p.get("direct")
            if not a or not d:
                continue
            au, du = a["completion_median_us"], d["completion_median_us"]
            print(f"| {base} | {au:.1f} | {du:.1f} | {(au-du)/au*100:+.1f}% |")

    if args.allreduce:
        records = {Path(p).stem: load(p) for p in args.allreduce}
        pairs = pair_by_path(records)
        print("\n### AllReduce\n")
        print("| config | adapter_us | direct_us | direct vs adapter |")
        print("|---|---:|---:|---:|")
        for base, p in sorted(pairs.items()):
            a, d = p.get("adapter"), p.get("direct")
            if not a or not d:
                continue
            au, du = a["async_median_us"], d["async_median_us"]
            print(f"| {base} | {au:.1f} | {du:.1f} | {(au-du)/au*100:+.1f}% |")

    if args.timing_log:
        print("\n### Reduction decomposition (TBCCL_ALLREDUCE_TIMING)\n")
        print("| file | role | recv_us | reduce_us | send_us | total_us |")
        print("|---|---|---:|---:|---:|---:|")
        for path in args.timing_log:
            rows = []
            with open(path) as f:
                for line in f:
                    if "tbccl_allreduce_timing" not in line:
                        continue
                    rows.append(dict(re.findall(r"(\w+)=([\d.]+|\S+)", line.split("]", 1)[1])))
            if not rows:
                continue
            role = rows[0].get("role", "?")

            def med(key):
                vals = [float(r[key]) for r in rows if key in r]
                return statistics.median(vals) if vals else None

            print(f"| {Path(path).name} | {role} | {med('recv_leg_us')} | {med('reduce_us')} | "
                  f"{med('send_leg_us')} | {med('total_us')} |")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
