#!/usr/bin/env python3
"""Bounded, permission-gated Linux ftrace capture for one benchmark run.

Must be run under sudo (tracefs at /sys/kernel/tracing is root-only on this
system: perf_event_paranoid=2, no bpftool/bpftrace). Enables only the
requested tracefs events, optionally filtered to a specific PID via
set_ftrace_pid, runs the given benchmark command for its natural duration,
captures the trace buffer, then restores every setting it touched -- even on
error or interruption (SIGINT/SIGTERM), so no global tracing state is left
enabled afterward.

Part of the tail-latency investigation. Not invoked automatically -- requires explicit user
authorization per run.
"""
import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

TRACEFS = Path("/sys/kernel/tracing")

DEFAULT_EVENTS = [
    "sched/sched_switch",
    "sched/sched_wakeup",
    "sched/sched_wakeup_new",
]


class TracefsUnavailable(RuntimeError):
    pass


def require_root():
    import os
    if os.geteuid() != 0:
        raise SystemExit(
            "capture_tb4_scheduler_trace.py must run under sudo "
            "(tracefs is root-only on this kernel)."
        )


def read_text(path):
    try:
        return path.read_text()
    except OSError as error:
        raise TracefsUnavailable(f"cannot read {path}: {error}") from error


def write_text(path, value):
    try:
        path.write_text(value)
    except OSError as error:
        raise TracefsUnavailable(f"cannot write {path}: {error}") from error


def event_enable_path(event):
    group, name = event.split("/", 1)
    return TRACEFS / "events" / group / name / "enable"


def event_filter_path(event):
    group, name = event.split("/", 1)
    return TRACEFS / "events" / group / name / "filter"


# set_ftrace_pid only scopes the function-tracer plugin -- it does NOT
# filter tracepoint events enabled via events/*/*/enable, which remain
# system-wide by default. Each sched tracepoint has its own field names
# for "which task", so the filter expression differs per event (both
# confirmed against this kernel's actual sched_switch/sched_wakeup output:
# sched_switch carries prev_pid/next_pid: the task can appear on either
# side of a switch; sched_wakeup(_new) carries pid: the task being woken).
EVENT_PID_FILTERS = {
    "sched/sched_switch": "prev_pid == {pid} || next_pid == {pid}",
    "sched/sched_wakeup": "pid == {pid}",
    "sched/sched_wakeup_new": "pid == {pid}",
}


def discover_supported_events(requested):
    supported, unsupported = [], []
    for event in requested:
        path = event_enable_path(event)
        (supported if path.exists() else unsupported).append(event)
    return supported, unsupported


class TraceSession:
    """Captures and restores exactly the tracefs state this script touches."""

    def __init__(self, events):
        self.events = events
        self.original_tracing_on = None
        self.original_set_ftrace_pid = None
        self.original_event_enable = {}
        self.original_event_filter = {}
        self.original_trace_clock = None
        self.touched = False

    def __enter__(self):
        self.original_tracing_on = read_text(TRACEFS / "tracing_on").strip()
        clock_path = TRACEFS / "trace_clock"
        if clock_path.exists():
            # Format is like "[local] global counter x86-tsc mono mono_raw
            # boot" -- the active one is bracketed.
            raw = read_text(clock_path)
            active = next((w.strip("[]") for w in raw.split() if w.startswith("[")), None)
            self.original_trace_clock = active
            if active != "mono":
                # "mono" == CLOCK_MONOTONIC, same clock/units/epoch as the
                # application trace -- switching to it (instead of the
                # default "local", a per-CPU clock not directly comparable
                # across CPUs or to CLOCK_MONOTONIC) avoids needing
                # statistical clock calibration entirely.
                write_text(clock_path, "mono")
        set_ftrace_pid_path = TRACEFS / "set_ftrace_pid"
        if set_ftrace_pid_path.exists():
            self.original_set_ftrace_pid = read_text(set_ftrace_pid_path).strip()
        for event in self.events:
            path = event_enable_path(event)
            self.original_event_enable[event] = read_text(path).strip()
            filter_path = event_filter_path(event)
            if filter_path.exists():
                self.original_event_filter[event] = read_text(filter_path).strip()

        self.touched = True
        write_text(TRACEFS / "trace", "")  # clear buffer
        for event in self.events:
            write_text(event_enable_path(event), "1")
        write_text(TRACEFS / "tracing_on", "1")
        return self

    def set_pid_filter(self, pid):
        """Restrict subsequent events to `pid`. Called once the traced
        process's PID is known (i.e. right after launching it) -- there is
        an unavoidable, small race between fork and this call, during which
        events are unfiltered; acceptable since the interval of interest is
        the steady-state send/recv loop, not process startup.

        Writes both set_ftrace_pid (for the function tracer, if later
        enabled) and a real per-event filter expression to each event's own
        filter file -- set_ftrace_pid alone does NOT scope tracepoint
        events enabled via events/*/*/enable, only the function tracer."""
        set_ftrace_pid_path = TRACEFS / "set_ftrace_pid"
        if set_ftrace_pid_path.exists():
            write_text(set_ftrace_pid_path, str(pid))
        for event in self.events:
            expression = EVENT_PID_FILTERS.get(event)
            if expression is None:
                continue
            filter_path = event_filter_path(event)
            if filter_path.exists():
                write_text(filter_path, expression.format(pid=pid))

    def __exit__(self, exc_type, exc, tb):
        if not self.touched:
            return False
        # Best-effort restore of every field we changed, in reverse order,
        # continuing past individual failures so a stuck event does not
        # prevent restoring tracing_on.
        try:
            write_text(TRACEFS / "tracing_on", self.original_tracing_on or "0")
        except TracefsUnavailable as error:
            print(f"WARNING: failed to restore tracing_on: {error}", file=sys.stderr)
        clock_path = TRACEFS / "trace_clock"
        if self.original_trace_clock and self.original_trace_clock != "mono" and clock_path.exists():
            try:
                write_text(clock_path, self.original_trace_clock)
            except TracefsUnavailable as error:
                print(f"WARNING: failed to restore trace_clock: {error}", file=sys.stderr)
        set_ftrace_pid_path = TRACEFS / "set_ftrace_pid"
        if self.original_set_ftrace_pid is not None and set_ftrace_pid_path.exists():
            try:
                write_text(set_ftrace_pid_path, self.original_set_ftrace_pid)
            except TracefsUnavailable as error:
                print(f"WARNING: failed to restore set_ftrace_pid: {error}", file=sys.stderr)
        for event, original in self.original_event_enable.items():
            try:
                write_text(event_enable_path(event), original)
            except TracefsUnavailable as error:
                print(f"WARNING: failed to restore {event}: {error}", file=sys.stderr)
        for event, original in self.original_event_filter.items():
            try:
                write_text(event_filter_path(event), original or "0")
            except TracefsUnavailable as error:
                print(f"WARNING: failed to restore {event} filter: {error}", file=sys.stderr)
        return False

    def read_trace(self):
        return read_text(TRACEFS / "trace")


def drop_privileges(uid, gid):
    """Returns a preexec_fn that drops root to (uid, gid) in the child
    *after* fork but *before* exec -- a single fork+exec, so the traced
    PID (from Popen.pid) is the benchmark process itself, not an
    intermediate supervisor. Using 'sudo -u user --' instead would add an
    extra fork that set_ftrace_pid does not follow (it does not propagate
    to descendants unless the event-fork/function-fork tracefs options are
    set), which is exactly what caused the initial version of this script
    to trace nearly the whole system instead of just the benchmark."""
    import os

    def _drop():
        os.setgid(gid)
        os.setuid(uid)

    return _drop


def run_benchmark(command_argv, timeout_seconds, session, run_as_user):
    # The whole script runs under sudo (root), but the benchmark itself must
    # not -- it opens sockets/files that should stay owned by the real user,
    # and running it as root would also change scheduling-relevant
    # properties we're trying to observe unmodified.
    preexec_fn = None
    if run_as_user:
        import pwd
        entry = pwd.getpwnam(run_as_user)
        preexec_fn = drop_privileges(entry.pw_uid, entry.pw_gid)
    process = subprocess.Popen(command_argv, preexec_fn=preexec_fn)
    session.set_pid_filter(process.pid)
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        raise
    return process.pid, process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--events", default=",".join(DEFAULT_EVENTS),
        help="comma-separated group/name tracefs events")
    parser.add_argument(
        "--run-as-user", default=None,
        help="launch --command as this user via 'sudo -u' instead of as "
             "root (this script itself must run under sudo for tracefs "
             "access, but the benchmark should not run as root)")
    parser.add_argument(
        "--command", nargs=argparse.REMAINDER, required=True,
        help="benchmark command to run while tracing (everything after "
             "this flag is passed through as argv)")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output", required=True, help="path for the JSON result")
    args = parser.parse_args()

    require_root()

    if not TRACEFS.exists():
        raise SystemExit(f"tracefs not mounted at {TRACEFS}")

    requested = [event.strip() for event in args.events.split(",") if event.strip()]
    supported, unsupported = discover_supported_events(requested)

    if not args.command:
        raise SystemExit("--command requires at least one argument (the benchmark argv)")

    trace_clock_path = TRACEFS / "trace_clock"
    trace_clock = read_text(trace_clock_path) if trace_clock_path.exists() else None

    result = {
        "requested_events": requested,
        "supported_events": supported,
        "unsupported_events": unsupported,
        "trace_clock": trace_clock,
        "command": args.command,
        "pid": None,
        "returncode": None,
        "trace_text": None,
        "error": None,
    }

    def handle_signal(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    signal.signal(signal.SIGTERM, handle_signal)

    if not supported:
        result["error"] = "no requested tracefs events are supported on this kernel"
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        return

    try:
        with TraceSession(supported) as session:
            pid, returncode = run_benchmark(
                args.command, args.timeout_seconds, session, args.run_as_user)
            result["pid"] = pid
            result["returncode"] = returncode
            result["trace_text"] = session.read_trace()
    except (TracefsUnavailable, subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        result["error"] = str(error)

    Path(args.output).write_text(json.dumps(result, indent=2))
    print(f"wrote {args.output} ({len(result.get('trace_text') or '')} bytes of trace)")
    if result["error"]:
        print(f"ERROR: {result['error']}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
