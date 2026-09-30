#!/usr/bin/env python3
"""Phase 30 Part X: bounded, non-mutating sustained-session runner.

Launches one Mac(source)->Linux(sink) end-to-end host/64KiB session (the
Phase 24-27 trigger shape: a single continuous process/TCP connection,
warmup + measured iterations), optionally under the Linux-side ftrace
diagnostic stack (--include-receive-events --include-thunderbolt-events,
and --include-nhi-functions for Phase 29's Tier-1 function tracer), with
a read-only runtime-state snapshot taken immediately before and after.

This script does not change any system configuration -- CPU governor,
IRQ affinity, offloads, power management, and every other item in the
Phase 30 plan's "explicitly out of scope" list are left untouched. It
only starts/stops the benchmark process pair and the (already-existing,
Phase 22/26/27/29-built) tracefs capture, both of which are the same
kind of bounded, restoring action those phases already used.

Requires: passwordless sudo for capture_tb4_scheduler_trace.py's exact
path (already granted, see SESSION_HANDOFF.md), and SSH access to the
Mac (tbccl-mac, already configured).
"""
import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_tb4_trace as analyze  # noqa: E402
import capture_tb4_runtime_state as runtime_state  # noqa: E402

BENCH = ROOT / "build-release" / "tbccl_tensor_transfer_bench"
CAPTURE_SCRIPT = ROOT / "scripts" / "capture_tb4_scheduler_trace.py"
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
MAC_HOST = "tbccl-mac"
MAC_ROOT = "~/projects/tbccl"


def git_commit():
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=5).stdout.strip()


def irq_cpu_mapping(irq_numbers=(177, 178)):
    mapping = {}
    for irq in irq_numbers:
        path = Path(f"/proc/irq/{irq}/effective_affinity_list")
        try:
            mapping[str(irq)] = path.read_text().strip()
        except OSError:
            mapping[str(irq)] = None
    return mapping


def run_session(run_id, warmup, measured, payload_bytes, base_port,
                 out_dir, include_nhi_functions=False, timeout_s=120):
    out_dir.mkdir(parents=True, exist_ok=True)
    peers = f"192.168.3.1:{base_port},192.168.3.2:{base_port + 1}"

    linux_trace_path = out_dir / f"{run_id}_linux.trace"
    linux_csv_path = out_dir / f"{run_id}_linux.csv"
    sched_json_path = out_dir / f"{run_id}_sched.json"
    mac_trace_path = out_dir / f"{run_id}_mac.trace"
    mac_csv_path = out_dir / f"{run_id}_mac.csv"

    bench_argv = [
        str(BENCH), "--rank", "1", "--mode", "end-to-end", "--peers", peers,
        "--local-backend", "host", "--source-rank", "0", "--sizes", str(payload_bytes),
        "--warmup", str(warmup), "--iterations", str(measured), "--trace",
        "--run-id", run_id, "--physical-machine", "linux",
    ]
    capture_argv = [
        "sudo", "-n", sys.executable, str(CAPTURE_SCRIPT),
        "--output", str(sched_json_path),
        "--include-receive-events", "--include-thunderbolt-events",
        "--timeout-seconds", str(timeout_s + 10),
        "--run-as-user", "svb",
    ]
    if include_nhi_functions:
        capture_argv.append("--include-nhi-functions")
    capture_argv += ["--command", *bench_argv]

    mac_argv = (
        f"cd {MAC_ROOT} && ./build-release/tbccl_tensor_transfer_bench --rank 0 "
        f"--mode end-to-end --peers {peers} --local-backend host --source-rank 0 "
        f"--sizes {payload_bytes} --warmup {warmup} --iterations {measured} --trace "
        f"--run-id {run_id} --physical-machine mac"
    )

    pre_state = runtime_state.snapshot()

    launch_start = time.monotonic()
    with open(linux_csv_path, "w") as lout, open(linux_trace_path, "w") as lerr:
        linux_proc = subprocess.Popen(capture_argv, stdout=lout, stderr=lerr)
    with open(mac_csv_path, "w") as mout, open(mac_trace_path, "w") as merr:
        mac_proc = subprocess.Popen(SSH + [MAC_HOST, mac_argv], stdout=mout, stderr=merr)

    linux_rc = linux_proc.wait(timeout=timeout_s)
    mac_rc = mac_proc.wait(timeout=timeout_s)
    launch_end = time.monotonic()

    post_state = runtime_state.snapshot()

    # capture_tb4_scheduler_trace.py writes the benchmark's own stderr
    # (the TBCCL_DIAGNOSTIC line) interleaved with its own status lines
    # to stdout, and the trace JSON to --output; the benchmark's --trace
    # stderr is inherited, so the diagnostic line lands in linux_trace_path
    # only when the benchmark itself was invoked with --run-as-user and
    # its stderr passed through. capture_tb4_scheduler_trace.py inherits
    # the child's stderr by default, so linux_trace_path (mapped to this
    # process's stderr) receives it directly.
    app_trace = analyze.parse_diagnostic_file(linux_trace_path)
    sched_result = analyze.load(sched_json_path) if sched_json_path.exists() else None

    samples_us = []
    if app_trace is not None:
        for entry in app_trace.get("entries", []):
            if entry.get("recv_begin_ns") is not None and entry.get("recv_end_ns") is not None:
                samples_us.append((entry["recv_end_ns"] - entry["recv_begin_ns"]) / 1000.0)

    cadence_report = None
    if app_trace is not None and sched_result is not None:
        cadence_report = analyze.nhi_cadence_report(app_trace, sched_result, slow_threshold_us=1000)

    stats = {
        "n": len(samples_us),
        "median_us": statistics.median(samples_us) if samples_us else None,
        "p95_us": analyze.percentile(samples_us, .95) if samples_us else None,
        "p99_us": analyze.percentile(samples_us, .99) if samples_us else None,
        "max_us": max(samples_us) if samples_us else None,
        "count_ge_500us": sum(1 for v in samples_us if v >= 500),
        "count_ge_800us": sum(1 for v in samples_us if v >= 800),
        "count_ge_1000us": sum(1 for v in samples_us if v >= 1000),
    }

    return {
        "run_id": run_id,
        "git_commit": git_commit(),
        "boot_id": pre_state.get("boot_id"),
        "boot_id_stable": pre_state.get("boot_id") == post_state.get("boot_id"),
        "tracing_mode": "nhi-functions" if include_nhi_functions else "receive+thunderbolt",
        "irq_cpu_mapping": irq_cpu_mapping(),
        "warmup": warmup,
        "measured": measured,
        "payload_bytes": payload_bytes,
        "linux_returncode": linux_rc,
        "mac_returncode": mac_rc,
        "wall_duration_us": (launch_end - launch_start) * 1e6,
        "success": (linux_rc == 0 and mac_rc == 0 and app_trace is not None),
        "completion_stats": stats,
        "per_vector_baseline": (cadence_report or {}).get("per_vector_baseline"),
        "slow_events": (cadence_report or {}).get("slow_events"),
        "pre_state_path": None,
        "post_state_path": None,
        "timestamp_start_utc": pre_state.get("timestamp_utc"),
        "timestamp_end_utc": post_state.get("timestamp_utc"),
        "_pre_state": pre_state,
        "_post_state": post_state,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--payload-bytes", type=int, default=65536)
    parser.add_argument("--base-port", type=int, default=35000)
    parser.add_argument("--include-nhi-functions", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "phase30-local")
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_session(
        args.run_id, args.warmup, args.iterations, args.payload_bytes,
        args.base_port, args.out_dir, include_nhi_functions=args.include_nhi_functions,
        timeout_s=args.timeout_seconds)

    pre_path = args.out_dir / f"{args.run_id}_state_pre.json"
    post_path = args.out_dir / f"{args.run_id}_state_post.json"
    pre_path.write_text(json.dumps(result.pop("_pre_state"), indent=2))
    post_path.write_text(json.dumps(result.pop("_post_state"), indent=2))
    result["pre_state_path"] = str(pre_path)
    result["post_state_path"] = str(post_path)

    args.output.write_text(json.dumps(result, indent=2, default=str))
    stats = result["completion_stats"]
    print(f"[{args.run_id}] n={stats['n']} median={stats['median_us']} "
          f"p95={stats['p95_us']} ge1000us={stats['count_ge_1000us']} "
          f"success={result['success']}", file=sys.stderr)
    print(f"wrote {args.output}", file=sys.stderr)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
