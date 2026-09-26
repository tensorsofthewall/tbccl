#!/usr/bin/env python3
"""Launch a local multi-rank TBCCL World smoke test as separate processes.

Each rank runs as its own OS process (tbccl_world_smoke), all talking
to each other over one loopback host, so multi-rank bootstrap and
communication can be exercised without multiple physical machines.

Example:

    python scripts/run_local_world.py \\
        --binary ./build/tbccl_world_smoke \\
        --ranks 3
"""

import argparse
import os
import signal
import subprocess
import sys
import time


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Launch a local multi-rank TBCCL World smoke test."
    )
    parser.add_argument(
        "--binary", required=True, help="Path to the tbccl_world_smoke binary"
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
        default=18515,
        help="Rank 0's port; rank i listens on base-port + i",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Overall wall-clock budget in seconds for every rank to finish",
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


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.ranks < 1:
        parser.error("--ranks must be >= 1")

    peers = ",".join(
        f"{args.host}:{args.base_port + i}" for i in range(args.ranks)
    )

    processes = [
        subprocess.Popen(
            [args.binary, "--rank", str(rank), "--peers", peers],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        for rank in range(args.ranks)
    ]

    success = wait_for_all(processes, args.timeout)

    if not success:
        print(
            "error: at least one rank failed, hung, or the overall "
            f"{args.timeout}s timeout elapsed; terminating remaining ranks",
            file=sys.stderr,
        )
        kill_all(processes)

    outputs = []

    for process in processes:
        try:
            stdout, _ = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            signal_group(process, signal.SIGKILL)
            stdout, _ = process.communicate()

        outputs.append(stdout or "")

    results = []

    for rank, (process, output) in enumerate(zip(processes, outputs)):
        print(f"--- rank {rank} output ---")
        sys.stdout.write(output)

        if output and not output.endswith("\n"):
            print()

        passed = process.returncode == 0 and f"rank {rank}: PASS" in output
        results.append(passed)

        print(f"rank {rank}: {'PASS' if passed else 'FAIL'}")

    print()

    overall = success and all(results)

    print(f"{args.ranks}-rank TBCCL world: {'PASS' if overall else 'FAIL'}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
