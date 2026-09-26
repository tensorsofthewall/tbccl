#!/usr/bin/env python3
"""Plot median RTT vs. message size from latency_sweep.py's summary.csv.

This is a standalone, optional convenience script. It requires
matplotlib; latency_sweep.py itself never depends on it, so measurement
and aggregation always work even where matplotlib isn't installed.
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

SERIES = [
    ("mac-to-linux", "normal", "Mac→Linux normal", "tab:blue", "-"),
    ("mac-to-linux", "busy100", "Mac→Linux busy100", "tab:blue", "--"),
    ("linux-to-mac", "normal", "Linux→Mac normal", "tab:orange", "-"),
    ("linux-to-mac", "busy100", "Linux→Mac busy100", "tab:orange", "--"),
]


def load_summary(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(
        description="Plot median RTT vs. message size from summary.csv"
    )
    parser.add_argument(
        "--summary",
        default="results/latency-v1/summary.csv",
        help="Path to summary.csv produced by latency_sweep.py",
    )
    parser.add_argument(
        "--output",
        default="results/latency-v1/latency-vs-message-size.png",
        help="Path to write the PNG plot to",
    )
    args = parser.parse_args()

    if plt is None:
        print(
            "error: matplotlib is required for plot_latency.py "
            "(pip install matplotlib)",
            file=sys.stderr,
        )
        sys.exit(1)

    summary_path = Path(args.summary)

    if not summary_path.exists():
        print(f"error: summary file not found: {summary_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_summary(summary_path)

    fig, ax = plt.subplots(figsize=(8, 5.5))

    plotted_any = False

    for direction, config, label, color, linestyle in SERIES:
        points = sorted(
            (int(row["size_bytes"]), float(row["median_rtt_us"]))
            for row in rows
            if row["direction"] == direction and row["config"] == config
        )

        if not points:
            continue

        sizes = [p[0] for p in points]
        rtts = [p[1] for p in points]

        ax.plot(
            sizes,
            rtts,
            label=label,
            color=color,
            linestyle=linestyle,
            marker="o",
        )
        plotted_any = True

    if not plotted_any:
        print(f"error: no matching series found in {summary_path}", file=sys.stderr)
        sys.exit(1)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Message size (bytes, log2 scale)")
    ax.set_ylabel("Median RTT (µs)")
    ax.set_title("TCP ping-pong latency vs. message size")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5)
    ax.legend()

    fig.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)

    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
