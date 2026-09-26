#!/usr/bin/env python3
"""Repeatedly run the tb_pingpong client and aggregate latency results.

This script never touches the benchmark's wire protocol or CSV schema.
It shells out to an existing tb_pingpong binary once per trial, reads
the CSV it writes, and accumulates per-trial rows into a raw dataset
for one (direction, config) pair. After writing that raw dataset, it
rescans every "*-raw.csv" file already present in --output-dir and
regenerates the combined summary.csv and README.md, so partial runs
(one direction/config at a time, possibly from different machines)
converge to a complete report as each piece lands.

Percentile method: linear interpolation between order statistics,
matching NumPy's default `percentile` and Excel's PERCENTILE.INC. For
a sorted sample of size N, the p-th percentile's fractional rank
(0-based) is `rank = (p / 100) * (N - 1)`; when that rank falls
between two elements, the result is a linear interpolation between
them. This is documented here because the raw p95 numbers depend on
which convention was used.
"""

import argparse
import csv
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

RAW_HEADER = [
    "direction",
    "config",
    "busy_poll_us",
    "trial",
    "size_bytes",
    "iterations",
    "rtt_us",
    "gbps",
    "messages_per_second",
]

SUMMARY_HEADER = [
    "direction",
    "config",
    "busy_poll_us",
    "size_bytes",
    "trials",
    "min_rtt_us",
    "mean_rtt_us",
    "median_rtt_us",
    "p95_rtt_us",
    "max_rtt_us",
    "stddev_rtt_us",
]

SIZE_MULTIPLIERS = {"B": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}

DIRECTION_LABELS = {
    "mac-to-linux": "Mac→Linux",
    "linux-to-mac": "Linux→Mac",
}


# -----------------------------------------------------------------------------
# Size parsing (mirrors tb_pingpong's own parse_size/parse_sizes)
# -----------------------------------------------------------------------------


def parse_size(text):
    text = text.strip()

    if not text:
        raise ValueError("empty message size")

    suffix = text[-1].upper()

    if suffix in SIZE_MULTIPLIERS:
        digits = text[:-1]
        multiplier = SIZE_MULTIPLIERS[suffix]
    else:
        digits = text
        multiplier = 1

    if not digits or not digits.isdigit():
        raise ValueError(f"invalid message size: {text}")

    return int(digits) * multiplier


def parse_sizes(sizes_arg):
    return [parse_size(part) for part in sizes_arg.split(",") if part != ""]


def format_size(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} B"

    if num_bytes < 1024 ** 2:
        value = num_bytes / 1024
        unit = "KiB"
    else:
        value = num_bytes / 1024 ** 2
        unit = "MiB"

    if value == int(value):
        return f"{int(value)} {unit}"

    return f"{value:.1f} {unit}"


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------


def percentile(values, p):
    if not values:
        raise ValueError("percentile of empty sample")

    sorted_values = sorted(values)
    n = len(sorted_values)

    if n == 1:
        return sorted_values[0]

    rank = (p / 100.0) * (n - 1)
    lower_index = math.floor(rank)
    upper_index = math.ceil(rank)

    if lower_index == upper_index:
        return sorted_values[lower_index]

    fraction = rank - lower_index

    return (
        sorted_values[lower_index]
        + (sorted_values[upper_index] - sorted_values[lower_index]) * fraction
    )


def summarize(values):
    count = len(values)

    return {
        "trials": count,
        "min_rtt_us": min(values),
        "mean_rtt_us": statistics.mean(values),
        "median_rtt_us": statistics.median(values),
        "p95_rtt_us": percentile(values, 95),
        "max_rtt_us": max(values),
        "stddev_rtt_us": statistics.stdev(values) if count > 1 else 0.0,
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run tb_pingpong client repeatedly across message sizes and "
            "trials, recording raw per-trial latency measurements and an "
            "aggregated summary/README for --output-dir."
        )
    )

    parser.add_argument("--binary", required=True, help="Path to the tb_pingpong binary")
    parser.add_argument("--host", required=True, help="tb_pingpong server host")
    parser.add_argument("--port", type=int, default=18515, help="tb_pingpong server port")
    parser.add_argument(
        "--sizes",
        required=True,
        help="Comma-separated message sizes, e.g. 64,256,1K,4K,16K,64K,256K,1M",
    )
    parser.add_argument("--trials", type=int, default=10, help="Number of trials to run")

    parser.add_argument(
        "--busy-poll",
        type=int,
        default=None,
        help=(
            "Convenience: sets both --client-busy-poll and "
            "--effective-busy-poll to this value. Mutually exclusive with "
            "those two flags."
        ),
    )
    parser.add_argument(
        "--client-busy-poll",
        type=int,
        default=None,
        help=(
            "Value passed to tb_pingpong's own --busy-poll flag on this "
            "client's socket (use 0 on macOS, which does not support "
            "SO_BUSY_POLL)."
        ),
    )
    parser.add_argument(
        "--effective-busy-poll",
        type=int,
        default=None,
        help=(
            "The busy-poll duration actually in effect for this run, "
            "recorded in the output rows even if it was configured "
            "elsewhere (e.g. on a remote server) rather than on this "
            "client. Defaults to --client-busy-poll."
        ),
    )

    parser.add_argument("--direction", required=True, help="Label, e.g. mac-to-linux")
    parser.add_argument("--config", required=True, help="Label, e.g. normal or busy100")
    parser.add_argument("--output-dir", required=True, help="Directory for raw/summary output")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing raw dataset for this direction/config",
    )

    return parser


def resolve_busy_poll(args, parser):
    if args.busy_poll is not None:
        if args.client_busy_poll is not None or args.effective_busy_poll is not None:
            parser.error(
                "--busy-poll cannot be combined with "
                "--client-busy-poll/--effective-busy-poll"
            )

        return args.busy_poll, args.busy_poll

    client_busy_poll = args.client_busy_poll if args.client_busy_poll is not None else 0
    effective_busy_poll = (
        args.effective_busy_poll
        if args.effective_busy_poll is not None
        else client_busy_poll
    )

    return client_busy_poll, effective_busy_poll


# -----------------------------------------------------------------------------
# Trial execution
# -----------------------------------------------------------------------------


def run_trial(binary, host, port, sizes_arg, client_busy_poll, output_csv):
    command = [
        binary,
        "client",
        "--host",
        host,
        "--port",
        str(port),
        "--mode",
        "pingpong",
        "--sizes",
        sizes_arg,
        "--busy-poll",
        str(client_busy_poll),
        "--output",
        str(output_csv),
    ]

    try:
        result = subprocess.run(command, capture_output=True, text=True)
    except OSError as error:
        raise RuntimeError(f"failed to launch {binary}: {error}") from error

    if result.returncode != 0:
        raise RuntimeError(
            "trial failed (exit "
            f"{result.returncode}): {' '.join(command)}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )


def read_trial_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def collect_raw_rows(args, client_busy_poll, effective_busy_poll, expected_sizes):
    rows = []

    for trial in range(1, args.trials + 1):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_csv = Path(tmp_dir) / "trial.csv"

            print(
                f"[{args.direction}/{args.config}] trial {trial}/{args.trials}...",
                file=sys.stderr,
            )

            run_trial(
                args.binary,
                args.host,
                args.port,
                args.sizes,
                client_busy_poll,
                tmp_csv,
            )

            trial_rows = [
                row for row in read_trial_csv(tmp_csv) if row["mode"] == "pingpong"
            ]

            seen_sizes = [int(row["size_bytes"]) for row in trial_rows]

            if sorted(seen_sizes) != sorted(expected_sizes):
                raise RuntimeError(
                    f"trial {trial}: expected sizes {expected_sizes} exactly "
                    f"once each, got {seen_sizes}"
                )

            for row in trial_rows:
                rows.append(
                    {
                        "direction": args.direction,
                        "config": args.config,
                        "busy_poll_us": effective_busy_poll,
                        "trial": trial,
                        "size_bytes": int(row["size_bytes"]),
                        "iterations": int(row["iterations"]),
                        "rtt_us": row["rtt_us"],
                        "gbps": row["gbps"],
                        "messages_per_second": row["messages_per_second"],
                    }
                )

    return rows


def write_raw_csv(path, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_HEADER)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


# -----------------------------------------------------------------------------
# Aggregation across every raw dataset in --output-dir
# -----------------------------------------------------------------------------


def load_all_raw_rows(output_dir):
    rows = []

    for raw_path in sorted(output_dir.glob("*-raw.csv")):
        with open(raw_path, newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(row)

    return rows


def build_summary(rows):
    groups = {}
    config_busy_poll = {}

    for row in rows:
        direction = row["direction"]
        config = row["config"]
        busy_poll_us = int(row["busy_poll_us"])
        size_bytes = int(row["size_bytes"])
        rtt_us = float(row["rtt_us"])

        config_key = (direction, config)
        config_busy_poll.setdefault(config_key, set()).add(busy_poll_us)

        key = (direction, config, size_bytes)
        groups.setdefault(key, []).append(rtt_us)

    for config_key, busy_polls in config_busy_poll.items():
        if len(busy_polls) > 1:
            direction, config = config_key
            raise RuntimeError(
                f"direction={direction} config={config} has mixed "
                f"busy_poll_us values {sorted(busy_polls)} across raw "
                "datasets; normal and busy100 results must never be mixed "
                "under the same config label"
            )

    summary_rows = []

    for (direction, config, size_bytes), rtt_values in sorted(groups.items()):
        busy_poll_us = next(iter(config_busy_poll[(direction, config)]))
        stats = summarize(rtt_values)

        summary_rows.append(
            {
                "direction": direction,
                "config": config,
                "busy_poll_us": busy_poll_us,
                "size_bytes": size_bytes,
                "trials": stats["trials"],
                "min_rtt_us": stats["min_rtt_us"],
                "mean_rtt_us": stats["mean_rtt_us"],
                "median_rtt_us": stats["median_rtt_us"],
                "p95_rtt_us": stats["p95_rtt_us"],
                "max_rtt_us": stats["max_rtt_us"],
                "stddev_rtt_us": stats["stddev_rtt_us"],
            }
        )

    return summary_rows


def write_summary_csv(path, summary_rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_HEADER)
        writer.writeheader()

        for row in summary_rows:
            writer.writerow(row)


# -----------------------------------------------------------------------------
# Human-readable README
# -----------------------------------------------------------------------------


def render_markdown(summary_rows):
    by_key = {
        (row["direction"], row["config"], row["size_bytes"]): row
        for row in summary_rows
    }

    combos = [
        ("mac-to-linux", "normal", "Mac→Linux normal"),
        ("mac-to-linux", "busy100", "Mac→Linux busy100"),
        ("linux-to-mac", "normal", "Linux→Mac normal"),
        ("linux-to-mac", "busy100", "Linux→Mac busy100"),
    ]

    sizes = sorted({row["size_bytes"] for row in summary_rows})

    lines = []
    lines.append("# TCP latency vs. message size")
    lines.append("")
    lines.append(
        "Generated by `scripts/latency_sweep.py`. Every row below is a "
        "median across the trials actually collected for that cell; see "
        "the accompanying `summary.csv` for min/mean/p95/max/stddev, and "
        "the `*-raw.csv` files for every individual measurement."
    )
    lines.append("")

    if not sizes:
        lines.append("_No data collected yet._")
        lines.append("")
    else:
        header = ["Size"] + [label for _, _, label in combos]
        lines.append("| " + " | ".join(header) + " |")
        lines.append(
            "|"
            + "|".join(
                ["------"] + [":----------------:" for _ in combos]
            )
            + "|"
        )

        for size_bytes in sizes:
            cells = [format_size(size_bytes)]

            for direction, config, _ in combos:
                row = by_key.get((direction, config, size_bytes))

                if row is None:
                    cells.append("n/a")
                else:
                    cells.append(f"{row['median_rtt_us']:.2f}")

            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    lines.append("## Busy-poll latency reduction")
    lines.append("")
    lines.append(
        "`100 * (normal_median - busy100_median) / normal_median`, per "
        "direction and size. Positive means busy polling reduced "
        "latency; negative means it made this cell worse. No crossover "
        "size is asserted here beyond what these numbers show."
    )
    lines.append("")

    if not sizes:
        lines.append("_No data collected yet._")
        lines.append("")
    else:
        lines.append("| Size | Mac→Linux reduction | Linux→Mac reduction |")
        lines.append("|------|------------------------:|------------------------:|")

        for size_bytes in sizes:
            cells = [format_size(size_bytes)]

            for direction in ("mac-to-linux", "linux-to-mac"):
                normal = by_key.get((direction, "normal", size_bytes))
                busy = by_key.get((direction, "busy100", size_bytes))

                if normal is None or busy is None or normal["median_rtt_us"] == 0:
                    cells.append("n/a")
                else:
                    reduction = (
                        100.0
                        * (normal["median_rtt_us"] - busy["median_rtt_us"])
                        / normal["median_rtt_us"]
                    )
                    cells.append(f"{reduction:.1f}%")

            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")

    observed_trials = sorted({row["trials"] for row in summary_rows})
    trials_note = (
        f"{observed_trials[0]}"
        if len(observed_trials) == 1
        else f"varies by cell ({observed_trials})"
    )

    lines.append("## Reproducibility")
    lines.append("")
    lines.append("- Transport: TCP over ThunderboltIP")
    lines.append("- MTU: 9000")
    lines.append("- TCP_NODELAY: on")
    lines.append("- GRO: on")
    lines.append(f"- Trials per cell (observed): {trials_note}")
    lines.append("- Busy-poll configurations: 0 µs, 100 µs")
    lines.append("- Linux affinity: CPU 0 (P-core) where applicable, via `taskset -c 0`")
    lines.append("")
    lines.append("Hardware:")
    lines.append("")
    lines.append("- Mac: base M4 Mac mini, 16 GB, Thunderbolt 4")
    lines.append("- Linux: i7-12700HX / RTX 3070 Ti laptop")
    lines.append("")
    lines.append(
        "These numbers are specific to this pair of machines and this "
        "Thunderbolt link. They are not presented as representative of "
        "Thunderbolt networking in general."
    )
    lines.append("")

    return "\n".join(lines)


def aggregate_and_report(output_dir):
    rows = load_all_raw_rows(output_dir)
    summary_rows = build_summary(rows)

    write_summary_csv(output_dir / "summary.csv", summary_rows)

    with open(output_dir / "README.md", "w") as handle:
        handle.write(render_markdown(summary_rows))

    print(
        f"wrote {output_dir / 'summary.csv'} and {output_dir / 'README.md'} "
        f"({len(summary_rows)} summary rows from {len(rows)} raw rows)",
        file=sys.stderr,
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    try:
        client_busy_poll, effective_busy_poll = resolve_busy_poll(args, parser)

        expected_sizes = parse_sizes(args.sizes)

        if len(expected_sizes) != len(set(expected_sizes)):
            parser.error("--sizes must not contain duplicate sizes")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        raw_path = output_dir / f"{args.direction}-{args.config}-raw.csv"

        if raw_path.exists() and not args.overwrite:
            raise RuntimeError(
                f"refusing to overwrite existing raw dataset: {raw_path} "
                "(pass --overwrite to replace it)"
            )

        raw_rows = collect_raw_rows(
            args, client_busy_poll, effective_busy_poll, expected_sizes
        )

        write_raw_csv(raw_path, raw_rows)
        print(f"wrote {raw_path} ({len(raw_rows)} rows)", file=sys.stderr)

        aggregate_and_report(output_dir)
    except RuntimeError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
