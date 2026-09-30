#!/usr/bin/env python3
"""Bounded, permission-gated Linux ftrace capture for one benchmark run.

Must be run under sudo (tracefs at /sys/kernel/tracing is root-only on this
system: perf_event_paranoid=2, no bpftool/bpftrace). Enables only the
requested tracefs events, optionally filtered to a specific PID via
set_ftrace_pid, runs the given benchmark command for its natural duration,
captures the trace buffer, then restores every setting it touched -- even on
error or interruption (SIGINT/SIGTERM), so no global tracing state is left
enabled afterward.

Phase 22 Part H. Not invoked automatically -- requires explicit user
authorization per run (see Part H item 42).

Phase 26 Part G extended this script with a receive-path event set
(RECEIVE_EVENTS) covering IRQ -> softirq -> NAPI -> GRO/skb-receive ->
socket-wakeup, discovered via --list-available-events against this
kernel's actual /sys/kernel/tracing/available_events (Part F item 19 --
no event name here was invented; see docs/phase26_report.md item 12 for
the full availability table). Unlike the scheduler events, none of these
support per-PID filtering (irq/softirq/napi/net/sock tracepoints have no
task-identifying field to filter on) -- they are inherently system-wide,
which is safe here only because capture windows are short and the only
traffic on thunderbolt0 during a benchmark run is the benchmark itself
(Part G item 26).
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

# Confirmed present and firing for thunderbolt0 traffic via a live smoke
# test (Phase 26 Part F item 19-20): irq_handler_entry fires twice per
# receive event (irq=177 then irq=178, both name=thunderbolt -- this
# driver's two MSI-X vectors), followed by softirq_entry vec=3
# [action=NET_RX], napi_poll for dev=thunderbolt0, and
# napi_gro_receive_entry dev=thunderbolt0 (GRO is active, so
# netif_receive_skb_entry does not fire for this device -- see the
# availability table). sk_data_ready fires once the socket layer is
# notified. Together these give a complete irq->softirq->napi->skb->
# socket-wakeup decomposition of the receive path.
RECEIVE_EVENTS = [
    "irq/irq_handler_entry",
    "irq/softirq_entry",
    "napi/napi_poll",
    "net/napi_gro_receive_entry",
    "net/netif_receive_skb_entry",
    "sock/sk_data_ready",
]

# Events with no task-identifying field to filter on -- system-wide even
# when a PID filter is requested for the rest of the session (documented,
# not silently pretended to be scoped; Part G item 26).
SYSTEM_WIDE_EVENTS = set(RECEIVE_EVENTS)


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


def looks_like_real_filter_expression(text):
    """A real ftrace filter expression contains a field comparison/logical
    operator, or is the bare literal "0"/"1". Anything else -- empty, or
    the "none\\nparse_error: ..." banner a filter file echoes back after a
    prior invalid write -- is not a filter to preserve/restore."""
    text = (text or "").strip()
    if not text:
        return False
    if text in ("0", "1"):
        return True
    return any(op in text for op in ("==", "!=", "&&", "||", ">=", "<=", " > ", " < "))


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
                # statistical clock calibration entirely (Part F item 32).
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
        unset_readbacks = (None, "-1", "", "no pid")
        if set_ftrace_pid_path.exists():
            if self.original_set_ftrace_pid in unset_readbacks:
                # The read-back value when no PID filter is set is "no pid"
                # on this kernel ("-1" on some others) -- writing either
                # back verbatim is rejected with EINVAL (confirmed via a
                # Phase 26 smoke test). The actual clear operation this
                # kernel accepts is writing a single space (also confirmed
                # empirically), which is what set_pid_filter() replaces
                # here rather than skipping the restore entirely -- a
                # stale pid left in set_ftrace_pid is functionally inert
                # (current_tracer stays "nop"), but restoring it properly
                # avoids leaking any session state into the next one.
                try:
                    write_text(set_ftrace_pid_path, " ")
                except TracefsUnavailable as error:
                    print(f"WARNING: failed to clear set_ftrace_pid: {error}", file=sys.stderr)
            else:
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
            if not looks_like_real_filter_expression(original):
                # Either genuinely empty, or (Phase 26 discovery: a real
                # kernel state this project's own earlier smoke-testing
                # produced) the file echoes a stale "none\nparse_error:
                # Field not found..." banner left over from a PREVIOUS
                # failed write elsewhere on the system -- that banner is
                # not a filter to restore, it is the kernel's read-back for
                # "no filter set, and here is why the last attempt to set
                # one failed." Treating it as literal text to write back
                # only reproduces the same parse error. Either way there is
                # no real filter to restore.
                continue
            try:
                write_text(event_filter_path(event), original)
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


def list_available_events(output_path):
    """Read-only discovery of every tracepoint this kernel exposes (Phase 26
    Part F item 19-20) -- does not enable, filter, or capture anything. Run
    under the same sudo grant as a normal capture (this script's own path is
    already covered by the existing NOPASSWD sudoers entry, so no new grant
    is needed for read-only discovery)."""
    events_path = TRACEFS / "available_events"
    all_events = [line.strip() for line in read_text(events_path).splitlines() if line.strip()]
    by_group = {}
    for event in all_events:
        group = event.split(":", 1)[0]
        by_group.setdefault(group, []).append(event)
    result = {"total_events": len(all_events), "groups": by_group}
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(f"wrote {output_path} ({len(all_events)} events across {len(by_group)} groups)")


def dump_state(events, output_path):
    """Read-only: print tracing_on/trace_clock/set_ftrace_pid and each
    event's current enable/filter content, without touching anything.
    Debugging aid for verifying TraceSession restoration against the real
    kernel (Part G item 24)."""
    state = {
        "tracing_on": read_text(TRACEFS / "tracing_on").strip(),
        "trace_clock": read_text(TRACEFS / "trace_clock").strip(),
        "set_ftrace_pid": read_text(TRACEFS / "set_ftrace_pid").strip()
        if (TRACEFS / "set_ftrace_pid").exists() else None,
        "current_tracer": read_text(TRACEFS / "current_tracer").strip()
        if (TRACEFS / "current_tracer").exists() else None,
        "events": {},
    }
    for event in events:
        enable_path = event_enable_path(event)
        filter_path = event_filter_path(event)
        state["events"][event] = {
            "enable": read_text(enable_path).strip() if enable_path.exists() else None,
            "filter": read_text(filter_path).strip() if filter_path.exists() else None,
        }
    Path(output_path).write_text(json.dumps(state, indent=2))
    print(json.dumps(state, indent=2))


def clear_set_ftrace_pid(output_path):
    """Best-effort: try each documented/observed way of clearing
    set_ftrace_pid back to its unset state, report which (if any)
    succeeded. Read-write but scoped to this single file. current_tracer
    is never touched by this project (stays "nop"), so a stale pid here is
    functionally inert -- this is cleanup, not a correctness requirement."""
    path = TRACEFS / "set_ftrace_pid"
    before = read_text(path).strip() if path.exists() else None
    attempts = []
    for candidate in (" ", "", "-1"):
        try:
            write_text(path, candidate)
            after = read_text(path).strip()
            attempts.append({"wrote": candidate, "error": None, "readback": after})
            if after in ("no pid", "-1", ""):
                break
        except TracefsUnavailable as error:
            attempts.append({"wrote": candidate, "error": str(error), "readback": None})
    result = {"before": before, "attempts": attempts,
              "after": read_text(path).strip() if path.exists() else None}
    Path(output_path).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def reset_state(events, output_path):
    """Emergency cleanup, read-write but conservative: force tracing_on to
    0, disable each of --events, and clear any real (non-empty) filter
    text by writing "" (the ftrace-documented way to clear a filter --
    unlike the bare "0" this script's restore path used before Phase 26,
    which is rejected by irq/napi/net/sock tracepoints). Does not touch
    trace_clock or set_ftrace_pid, since those are read back as valid
    values by TraceSession and don't get corrupted by a failed write the
    way filter files do."""
    report = {"tracing_on_before": None, "events": {}}
    tracing_on_path = TRACEFS / "tracing_on"
    report["tracing_on_before"] = read_text(tracing_on_path).strip()
    write_text(tracing_on_path, "0")
    for event in events:
        enable_path = event_enable_path(event)
        filter_path = event_filter_path(event)
        before = {
            "enable": read_text(enable_path).strip() if enable_path.exists() else None,
            "filter": read_text(filter_path).strip() if filter_path.exists() else None,
        }
        if enable_path.exists():
            write_text(enable_path, "0")
        if filter_path.exists():
            write_text(filter_path, "")
        report["events"][event] = {"before": before}
    Path(output_path).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


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
        "--command", nargs=argparse.REMAINDER, default=None,
        help="benchmark command to run while tracing (everything after "
             "this flag is passed through as argv); not required with "
             "--list-available-events")
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--output", required=True, help="path for the JSON result")
    parser.add_argument(
        "--list-available-events", action="store_true",
        help="read-only: dump every tracepoint under available_events, "
             "grouped by subsystem, and exit -- no capture is performed")
    parser.add_argument(
        "--include-receive-events", action="store_true",
        help="append RECEIVE_EVENTS (irq/softirq/napi/skb/socket-wakeup) "
             "to --events -- these are system-wide (no PID filter applies "
             "to them) even when the scheduler events are scoped to one pid")
    parser.add_argument(
        "--dump-state", action="store_true",
        help="read-only: print current tracing_on/trace_clock/"
             "set_ftrace_pid and each --events entry's enable/filter "
             "content, then exit -- no capture, nothing modified")
    parser.add_argument(
        "--reset-state", action="store_true",
        help="emergency cleanup: force tracing_on=0, disable each --events "
             "entry, and clear any real filter text -- for recovering from "
             "an interrupted/killed capture that TraceSession could not "
             "restore after (e.g. SIGKILL, which bypasses __exit__)")
    parser.add_argument(
        "--clear-set-ftrace-pid", action="store_true",
        help="best-effort cleanup of a stale set_ftrace_pid left over from "
             "an earlier session (functionally inert since current_tracer "
             "stays 'nop', but cleared for hygiene)")
    args = parser.parse_args()

    require_root()

    if not TRACEFS.exists():
        raise SystemExit(f"tracefs not mounted at {TRACEFS}")

    if args.list_available_events:
        list_available_events(args.output)
        return

    requested_events = [event.strip() for event in args.events.split(",") if event.strip()]
    if args.include_receive_events:
        for event in RECEIVE_EVENTS:
            if event not in requested_events:
                requested_events.append(event)

    if args.dump_state:
        dump_state(requested_events, args.output)
        return

    if args.reset_state:
        reset_state(requested_events, args.output)
        return

    if args.clear_set_ftrace_pid:
        clear_set_ftrace_pid(args.output)
        return

    if not args.command:
        raise SystemExit("--command requires at least one argument (the benchmark argv)")

    requested = requested_events
    supported, unsupported = discover_supported_events(requested)
    system_wide = sorted(SYSTEM_WIDE_EVENTS & set(supported))

    trace_clock_path = TRACEFS / "trace_clock"
    trace_clock = read_text(trace_clock_path) if trace_clock_path.exists() else None

    result = {
        "requested_events": requested,
        "supported_events": supported,
        "unsupported_events": unsupported,
        "system_wide_events": system_wide,
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
