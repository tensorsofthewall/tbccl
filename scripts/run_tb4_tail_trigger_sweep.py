#!/usr/bin/env python3
"""Bounded trigger-characterization sweep for the intermittent
~1.1-1.4ms Mac->Linux tail-latency mode (the tail-latency root-cause work).

Runs many short Mac->Linux host/64KiB/poll-0 bursts, varying the idle
duration between bursts (rotated order, not grouped) and recording session
metadata (requested/observed idle, process/session age, commit + binary
identity) alongside every raw per-iteration completion sample. Each burst
is a fresh process on both ends (this IS the "fresh process" condition; a
separate --long-lived-iterations run provides the long-lived-process
control, analyzed post-hoc by iteration-index range rather than needing
special runner support).

A small, focused wrapper rather than an extension of
run_tb4_busy_poll_sweep.py (the explicit alternative) --
that script's SSH-worker-heartbeat protocol is built for a different
shape of experiment (mid-sweep health guards across a large parameter
matrix); this one needs simple bounded local+SSH subprocess bursts with
inter-burst idle timing, which is most simply and safely done directly
here rather than retrofitted into that protocol.

No hardware state is modified. No builds, syncs or privileged writes.
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'TBCCL_DIAGNOSTIC '
BENCH = ROOT / 'build-release' / 'tbccl_tensor_transfer_bench'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5']
MAC_HOST = 'tbccl-mac'
MAC_ROOT = '~/projects/tbccl'


def git_commit():
    return subprocess.run(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
        capture_output=True, text=True, timeout=5).stdout.strip()


def binary_sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def parse_diagnostic(stderr_text):
    """Extracts the transfer_trace TBCCL_DIAGNOSTIC line, if present."""
    for line in stderr_text.splitlines():
        if line.startswith(PREFIX):
            return json.loads(line[len(PREFIX):])
    return None


def run_burst(session_index, idle_ms, warmup, measured, payload_bytes,
               base_port, run_id_prefix):
    """Launches one fresh Mac-source/Linux-sink burst (rank1 local, rank0
    over SSH), both with --trace, and returns the parsed result. Each
    process is fresh -- this itself is the 'fresh process' condition."""
    run_id = f'{run_id_prefix}_s{session_index}'
    linux_stderr_path = ROOT / 'results' / 'tail-trigger-local' / f'{run_id}_linux.trace'
    mac_stderr_path = ROOT / 'results' / 'tail-trigger-local' / f'{run_id}_mac.trace'
    peers = f'192.168.3.1:{base_port},192.168.3.2:{base_port + 1}'

    linux_argv = [
        str(BENCH), '--rank', '1', '--mode', 'end-to-end', '--peers', peers,
        '--local-backend', 'host', '--source-rank', '0', '--sizes', str(payload_bytes),
        '--warmup', str(warmup), '--iterations', str(measured), '--trace',
        '--run-id', run_id, '--physical-machine', 'linux',
    ]
    mac_argv = (
        f'cd {MAC_ROOT} && ./build-release/tbccl_tensor_transfer_bench --rank 0 '
        f'--mode end-to-end --peers {peers} --local-backend host --source-rank 0 '
        f'--sizes {payload_bytes} --warmup {warmup} --iterations {measured} --trace '
        f'--run-id {run_id} --physical-machine mac'
    )

    linux_stdout_path = ROOT / 'results' / 'tail-trigger-local' / f'{run_id}_linux.csv'
    mac_stdout_path = ROOT / 'results' / 'tail-trigger-local' / f'{run_id}_mac.csv'

    wall_launch_start = time.monotonic()
    with open(linux_stdout_path, 'w') as lout, open(linux_stderr_path, 'w') as lerr:
        linux_proc = subprocess.Popen(linux_argv, stdout=lout, stderr=lerr)
    with open(mac_stdout_path, 'w') as mout, open(mac_stderr_path, 'w') as merr:
        mac_proc = subprocess.Popen(SSH + [MAC_HOST, mac_argv], stdout=mout, stderr=merr)

    linux_rc = linux_proc.wait(timeout=60)
    mac_rc = mac_proc.wait(timeout=60)
    wall_launch_end = time.monotonic()

    linux_trace = parse_diagnostic(linux_stderr_path.read_text())
    samples_us = None
    if linux_trace is not None:
        samples_us = [
            (e['recv_end_ns'] - e['recv_begin_ns']) / 1000.0
            for e in linux_trace['entries']
            if e.get('recv_begin_ns') is not None and e.get('recv_end_ns') is not None
        ]

    return {
        'session_index': session_index,
        'run_id': run_id,
        'requested_idle_ms': idle_ms,
        'linux_returncode': linux_rc,
        'mac_returncode': mac_rc,
        'wall_burst_duration_us': (wall_launch_end - wall_launch_start) * 1e6,
        'samples_us': samples_us,
        'success': (linux_rc == 0 and mac_rc == 0 and samples_us is not None),
    }


def classify_thresholds(samples_us, thresholds=(500, 800, 1000)):
    counts = {t: 0 for t in thresholds}
    for value in samples_us or []:
        for t in thresholds:
            if value >= t:
                counts[t] += 1
    return counts


def rotate_conditions(idle_list_ms, repeats):
    """Round-robin across conditions, not grouped by condition (
    item 25's explicit requirement)."""
    order = []
    for _ in range(repeats):
        order.extend(idle_list_ms)
    return order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--idle-ms', default='0,10,100,500,1000,5000',
                        help='comma-separated idle durations between bursts')
    parser.add_argument('--bursts-per-condition', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=0)
    parser.add_argument('--measured', type=int, default=20)
    parser.add_argument('--payload-bytes', type=int, default=65536)
    parser.add_argument('--base-port', type=int, default=34000)
    parser.add_argument('--run-id-prefix', default='p24_idle')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    idle_list_ms = [int(x) for x in args.idle_ms.split(',') if x.strip()]
    order = rotate_conditions(idle_list_ms, args.bursts_per_condition)

    (ROOT / 'results' / 'tail-trigger-local').mkdir(parents=True, exist_ok=True)

    metadata = {
        'commit': git_commit(),
        'binary_sha256_linux': binary_sha256(BENCH),
        'idle_conditions_ms': idle_list_ms,
        'bursts_per_condition': args.bursts_per_condition,
        'condition_order': order,
        'warmup': args.warmup,
        'measured': args.measured,
        'payload_bytes': args.payload_bytes,
    }

    results = []
    previous_end_monotonic = None
    for session_index, idle_ms in enumerate(order):
        if previous_end_monotonic is not None:
            requested_sleep_s = idle_ms / 1000.0
            sleep_start = time.monotonic()
            if requested_sleep_s > 0:
                time.sleep(requested_sleep_s)
            observed_idle_us = (time.monotonic() - previous_end_monotonic) * 1e6
        else:
            observed_idle_us = None

        base_port = args.base_port + (session_index * 4) % 2000
        burst_start = time.monotonic()
        result = run_burst(
            session_index, idle_ms, args.warmup, args.measured,
            args.payload_bytes, base_port, args.run_id_prefix)
        result['observed_idle_us'] = observed_idle_us
        result['threshold_counts'] = classify_thresholds(result['samples_us'])
        results.append(result)
        previous_end_monotonic = time.monotonic()

        status = 'ok' if result['success'] else 'FAILED'
        n = len(result['samples_us'] or [])
        print(
            f"[{session_index+1}/{len(order)}] idle={idle_ms}ms "
            f"observed_idle_us={observed_idle_us} n={n} {status}",
            file=sys.stderr)

    output = {'metadata': metadata, 'results': results}
    args.output.write_text(json.dumps(output, indent=2))
    print(f'wrote {args.output}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
