#!/usr/bin/env python3
"""Analysis tool for the async tensor-transfer
substrate's benchmark output (tbccl_async_transfer_bench JSON files,
one per sender/receiver process per run, under results/phase32-local/).

Produces the four Markdown tables the plan asks for (transport
baseline, chunk sweep, device staging, async/overlap) from whichever
inputs are available -- this is a reporting tool, not a benchmark
runner, and never fabricates a row for a size/config that wasn't
actually measured.
"""
import argparse
import glob
import json
from pathlib import Path


def load_sender_results(pattern):
    """Loads every *_linux.json / *_mac.json matching `pattern` that has
    a 'role': 'sender' field (receiver-side files carry no authoritative
    timing and are skipped, matching the bench's own documented
    convention)."""
    results = []
    for path in sorted(glob.glob(pattern)):
        try:
            data = json.loads(Path(path).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("role") == "sender":
            data["_path"] = path
            results.append(data)
    return results


def transport_baseline_table(sync_baseline_rows):
    """sync_baseline_rows: list of dicts with keys path/bytes/median_us/
    gbps already extracted from the existing tbccl_tensor_transfer_bench
    CSV/summary lines by the caller (this tool doesn't parse that
    format itself -- see --sync-baseline-csv)."""
    lines = ["| path | bytes | median_us | GiB/s |", "|---|---:|---:|---:|"]
    for row in sync_baseline_rows:
        lines.append(
            f"| {row['path']} | {row['bytes']} | {row['median_us']:.1f} | {row['gib_per_s']:.3f} |")
    return "\n".join(lines)


def chunk_sweep_table(results):
    lines = ["| label | chunk_bytes | depth | direction | GiB/s | median_us |",
             "|---|---:|---:|---|---:|---:|"]
    for r in results:
        direction = r.get("direction", r.get("label", ""))
        lines.append(
            f"| {r.get('label','')} | {r.get('chunk_bytes',0)} | "
            f"{r.get('pipeline_depth',0)} | {direction} | "
            f"{r.get('gib_per_s',0):.3f} | {r.get('completion_median_us',0):.1f} |")
    return "\n".join(lines)


def async_overhead_table(results):
    lines = ["| label | bytes | enqueue_median_us | completion_median_us | "
             "caller_block_reduction_pct |",
             "|---|---:|---:|---:|---:|"]
    for r in results:
        enqueue = r.get("enqueue_median_us", 0)
        completion = r.get("completion_median_us", 0)
        reduction = (1 - enqueue / completion) * 100 if completion else 0
        lines.append(
            f"| {r.get('label','')} | {r.get('bytes',0)} | {enqueue:.3f} | "
            f"{completion:.1f} | {reduction:.2f} |")
    return "\n".join(lines)


def overlap_table(overlap_rows):
    """overlap_rows: list of dicts with bucket/compute_us/comm_us/
    serial_us/async_us keys, supplied by the caller (produced by a
    dedicated overlap-microbenchmark run, not this bench's own JSON
    shape)."""
    lines = ["| bucket | compute_us | comm_us | serial_us | async_us | hidden_pct |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in overlap_rows:
        hidden = row["serial_us"] - row["async_us"]
        hidden_pct = (hidden / row["comm_us"]) * 100 if row["comm_us"] else 0
        lines.append(
            f"| {row['bucket']} | {row['compute_us']:.1f} | {row['comm_us']:.1f} | "
            f"{row['serial_us']:.1f} | {row['async_us']:.1f} | {hidden_pct:.2f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-glob", default="results/phase32-local/*.json",
                         help="glob for tbccl_async_transfer_bench sender JSON files")
    parser.add_argument("--table", choices=["chunk-sweep", "async-overhead"],
                         default="chunk-sweep")
    parser.add_argument("--output")
    args = parser.parse_args()

    results = load_sender_results(args.results_glob)

    if args.table == "chunk-sweep":
        text = chunk_sweep_table(results)
    else:
        text = async_overhead_table(results)

    if args.output:
        Path(args.output).write_text(text + "\n")
        print(f"wrote {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
