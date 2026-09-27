#!/usr/bin/env python3
"""Launch a local multi-rank tbccl_collective_bench run as separate
processes, one per rank, all talking over loopback.

Captures rank 0's stdout (the CSV) separately from every rank's
stderr (diagnostics), and terminates every rank if any one of them
fails. With --algorithm both, runs "reference" then "ring" as two
separate process groups under identical settings and concatenates
their CSV output (one shared header).

Example:

    python scripts/run_collective_bench.py \\
        --binary ./build/tbccl_collective_bench \\
        --ranks 4 \\
        --collective all-gather \\
        --algorithm ring \\
        --sizes 64,1024,65536,1048576
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Launch a local multi-rank tbccl_collective_bench run."
    )
    parser.add_argument(
        "--binary", required=True, help="Path to the tbccl_collective_bench binary"
    )
    parser.add_argument(
        "--ranks", type=int, required=True, help="Number of ranks to launch"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Loopback host every rank binds/connects on",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=29500,
        help="Rank 0's port; rank i listens on base-port + i",
    )
    parser.add_argument(
        "--collective", default="all-gather", help="Collective to benchmark"
    )
    parser.add_argument(
        "--algorithm",
        default="reference",
        choices=["reference", "ring", "both"],
        help="Algorithm to benchmark, or 'both' to run reference then ring",
    )
    parser.add_argument(
        "--sizes",
        default=None,
        help="Comma-separated sizes (contribution bytes/rank for "
        "all-gather, output segment bytes/rank for reduce-scatter)",
    )
    parser.add_argument(
        "--datatype",
        default=None,
        choices=["int32", "int64", "float32", "float64"],
        help="Reduction datatype (reduce-scatter only)",
    )
    parser.add_argument(
        "--op",
        default=None,
        choices=["sum", "product", "min", "max"],
        help="Reduction operator (reduce-scatter only)",
    )
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--busy-poll", type=int, default=None)
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Overall wall-clock budget in seconds per algorithm run",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write the combined CSV here instead of only stdout",
    )
    return parser


def wait_for_all(processes, timeout):
    """Polls until every process exits 0, or one exits nonzero, or the
    timeout elapses. Returns True only if every process exited 0."""

    deadline = time.monotonic() + timeout
    finished = [False] * len(processes)

    while True:
        all_finished = True

        for i, process in enumerate(processes):
            if finished[i]:
                continue

            returncode = process.poll()

            if returncode is None:
                all_finished = False
                continue

            finished[i] = True

            if returncode != 0:
                return False

        if all_finished:
            return True

        if time.monotonic() >= deadline:
            return False

        time.sleep(0.05)


def signal_group(process, sig):
    # Each rank runs in its own session (start_new_session=True), so
    # signaling its process group also reaches any children it may
    # have spawned, not just the direct process.
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except ProcessLookupError:
        pass


def kill_all(processes):
    for process in processes:
        if process.poll() is None:
            signal_group(process, signal.SIGTERM)

    deadline = time.monotonic() + 2.0

    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())

        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            signal_group(process, signal.SIGKILL)
            process.wait()


def run_one_algorithm(binary, ranks, host, base_port, collective, algorithm,
                       sizes, datatype, op, iterations, warmup, busy_poll,
                       timeout):
    """Launches `ranks` processes for one algorithm run. Returns
    (success, rank0_stdout, [stderr per rank])."""

    peers = ",".join(f"{host}:{base_port + i}" for i in range(ranks))

    command_base = [
        binary,
        "--peers", peers,
        "--collective", collective,
        "--algorithm", algorithm,
    ]

    if sizes is not None:
        command_base += ["--sizes", sizes]
    if datatype is not None:
        command_base += ["--datatype", datatype]
    if op is not None:
        command_base += ["--op", op]
    if iterations is not None:
        command_base += ["--iterations", str(iterations)]
    if warmup is not None:
        command_base += ["--warmup", str(warmup)]
    if busy_poll is not None:
        command_base += ["--busy-poll", str(busy_poll)]

    processes = [
        subprocess.Popen(
            command_base + ["--rank", str(rank)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        for rank in range(ranks)
    ]

    success = wait_for_all(processes, timeout)

    if not success:
        print(
            f"error: algorithm={algorithm}: at least one rank failed, "
            f"hung, or the {timeout}s timeout elapsed; terminating "
            "remaining ranks",
            file=sys.stderr,
        )
        kill_all(processes)

    stdouts = []
    stderrs = []

    for process in processes:
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            signal_group(process, signal.SIGKILL)
            stdout, stderr = process.communicate()

        stdouts.append(stdout or "")
        stderrs.append(stderr or "")

    for rank, stderr in enumerate(stderrs):
        for line in stderr.splitlines():
            print(f"[algorithm={algorithm} rank={rank}] {line}", file=sys.stderr)

    if not success:
        return False, "", stderrs

    return True, stdouts[0], stderrs


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.ranks < 1:
        parser.error("--ranks must be >= 1")

    algorithms = ["reference", "ring"] if args.algorithm == "both" else [args.algorithm]

    combined_lines = []
    header = None

    for algorithm in algorithms:
        success, rank0_stdout, _ = run_one_algorithm(
            args.binary, args.ranks, args.host, args.base_port,
            args.collective, algorithm, args.sizes, args.datatype, args.op,
            args.iterations, args.warmup, args.busy_poll, args.timeout,
        )

        if not success:
            sys.exit(1)

        lines = [line for line in rank0_stdout.splitlines() if line]

        if not lines:
            print(
                f"error: algorithm={algorithm}: rank 0 produced no CSV output",
                file=sys.stderr,
            )
            sys.exit(1)

        if header is None:
            header = lines[0]
            combined_lines.append(header)
        elif lines[0] != header:
            print(
                "error: CSV header mismatch between algorithm runs",
                file=sys.stderr,
            )
            sys.exit(1)

        combined_lines.extend(lines[1:])

    csv_text = "\n".join(combined_lines) + "\n"

    sys.stdout.write(csv_text)

    if args.output:
        with open(args.output, "w") as f:
            f.write(csv_text)
        print(f"wrote {args.output}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
