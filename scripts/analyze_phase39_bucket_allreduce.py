#!/usr/bin/env python3
"""Phase 39: summarize tbccl_bucketed_allreduce_bench JSON output.

Usage:
    analyze_phase39_bucket_allreduce.py file1.json file2.json [...]

Prints Markdown tables: Primary overlap (serial/overlap pairs matched by
label suffix), Reduction decomposition (if TBCCL_ALLREDUCE_TIMING logs are
passed via --timing-log), and a simple textual Gantt-like bucket timeline
(if --timeline CSV is passed).
"""
import argparse
import json
import re
import sys
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("jsons", nargs="*")
    parser.add_argument("--timing-log", action="append", default=[])
    parser.add_argument("--timeline", action="append", default=[])
    args = parser.parse_args(argv)

    records = {Path(p).stem: load(p) for p in args.jsons}

    print("### Primary overlap\n")
    print("| label | schedule | median_us | p95_us | submit_median_us | verify_ok |")
    print("|---|---|---:|---:|---:|---|")
    for name, r in sorted(records.items()):
        schedule = r.get("schedule", "")
        median = r.get(f"{schedule.replace('-', '_')}_median_us", r.get("serial_median_us") or r.get("overlap_median_us"))
        p95 = r.get(f"{schedule.replace('-', '_')}_p95_us")
        print(f"| {name} | {schedule} | {median} | {p95} | "
              f"{r.get('submit_median_us', 'n/a')} | {r.get('verify_ok')} |")

    # Pair serial/overlap by common prefix (everything before the schedule word)
    print("\n### Speedup (serial -> overlap)\n")
    print("| config | serial_us | overlap_us | speedup |")
    print("|---|---:|---:|---:|")
    serials = {k: v for k, v in records.items() if v.get("schedule") == "serial"}
    overlaps = {k: v for k, v in records.items() if v.get("schedule") == "overlap"}
    for sk, sv in sorted(serials.items()):
        config = sk.replace("sweep_serial_", "").replace("_rank0", "")
        ok = next((k for k in overlaps if config in k), None)
        if not ok:
            continue
        ov = overlaps[ok]
        s_us = sv.get("serial_median_us")
        o_us = ov.get("overlap_median_us")
        if s_us and o_us:
            print(f"| {config} | {s_us:.1f} | {o_us:.1f} | {s_us / o_us:.3f}x |")

    if args.timing_log:
        print("\n### Reduction decomposition (from TBCCL_ALLREDUCE_TIMING logs)\n")
        print("| file | role | recv_leg_us (median) | reduce_us (median) | send_leg_us (median) | total_us (median) |")
        print("|---|---|---:|---:|---:|---:|")
        import statistics
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

    if args.timeline:
        print("\n### Timeline (textual Gantt)\n")
        for path in args.timeline:
            print(f"\n`{path}`:\n")
            with open(path) as f:
                lines = f.read().strip().splitlines()
            header = lines[0].split(",")
            print("```")
            for line in lines[1:]:
                fields = dict(zip(header, line.split(",")))
                b = fields["bucket"]
                cs, ce = float(fields["compute_start_us"]), float(fields["compute_end_us"])
                sub, comp = float(fields["submit_us"]), float(fields["complete_us"])
                print(f"bucket{b}: compute=[{cs:8.1f} -> {ce:8.1f}]  "
                      f"submit={sub:8.1f}  collective_complete={comp:10.1f}")
            print("```")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
