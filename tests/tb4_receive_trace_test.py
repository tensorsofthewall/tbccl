#!/usr/bin/env python3
"""Tests for capture_tb4_scheduler_trace.py's TraceSession restoration
logic, extended in receive-path investigation with receive-path events (irq/softirq/
napi/net/sock). Runs entirely against a fake tracefs tree under a temp
directory -- no root privileges, no real /sys/kernel/tracing access --
since the real path requires sudo and this module's own TRACEFS constant
is monkeypatched per test (the receive-path investigation work item 87/89)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import capture_tb4_scheduler_trace as cap


def make_fake_tracefs(root, events, tracing_on="0", trace_clock_active="local",
                       set_ftrace_pid="no pid", current_tracer="nop",
                       set_ftrace_filter="#### all functions enabled ####"):
    root = Path(root)
    (root / "tracing_on").write_text(tracing_on)
    (root / "trace").write_text("")
    (root / "set_ftrace_pid").write_text(set_ftrace_pid)
    (root / "current_tracer").write_text(current_tracer)
    (root / "set_ftrace_filter").write_text(set_ftrace_filter)
    clocks = "local global counter x86-tsc mono mono_raw boot"
    tagged = " ".join(f"[{w}]" if w == trace_clock_active else w for w in clocks.split())
    (root / "trace_clock").write_text(tagged)
    for event in events:
        group, name = event.split("/", 1)
        event_dir = root / "events" / group / name
        event_dir.mkdir(parents=True, exist_ok=True)
        (event_dir / "enable").write_text("0")
        (event_dir / "filter").write_text("")
    return root


class TraceSessionRestorationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._original_tracefs = cap.TRACEFS

    def tearDown(self):
        cap.TRACEFS = self._original_tracefs
        self._tmp.cleanup()

    def test_restores_tracing_on_and_clock_and_enable(self):
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events, tracing_on="0", trace_clock_active="local")
        cap.TRACEFS = self.root
        with cap.TraceSession(events) as session:
            self.assertEqual((self.root / "tracing_on").read_text(), "1")
            self.assertEqual((self.root / "trace_clock").read_text(), "mono")
            self.assertEqual((self.root / "events/sched/sched_switch/enable").read_text(), "1")
        self.assertEqual((self.root / "tracing_on").read_text(), "0")
        # The fake tracefs just stores whatever is written verbatim (unlike
        # the real kernel, which echoes back a bracketed list) -- writing
        # the clock name back is what TraceSession does to restore it.
        self.assertEqual((self.root / "trace_clock").read_text(), "local")
        self.assertEqual((self.root / "events/sched/sched_switch/enable").read_text(), "0")

    def test_leaves_already_mono_clock_untouched(self):
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events, trace_clock_active="mono")
        cap.TRACEFS = self.root
        with cap.TraceSession(events):
            pass
        # No write attempted/needed -- still mono, no restore-to-non-mono step.
        self.assertEqual(
            next(w.strip("[]") for w in (self.root / "trace_clock").read_text().split()
                 if w.startswith("[")),
            "mono")

    def test_pid_filter_applied_only_to_events_with_a_filter_expression(self):
        events = ["sched/sched_switch", "sched/sched_wakeup", "irq/irq_handler_entry"]
        make_fake_tracefs(self.root, events)
        cap.TRACEFS = self.root
        with cap.TraceSession(events) as session:
            session.set_pid_filter(4242)
            self.assertEqual(
                (self.root / "events/sched/sched_switch/filter").read_text(),
                "prev_pid == 4242 || next_pid == 4242")
            self.assertEqual(
                (self.root / "events/sched/sched_wakeup/filter").read_text(),
                "pid == 4242")
            # irq/irq_handler_entry has no PID-identifying field -- no
            # filter expression is defined for it, so it stays untouched
            # (system-wide), not silently given a bogus filter.
            self.assertEqual((self.root / "events/irq/irq_handler_entry/filter").read_text(), "")

    def test_restores_nonempty_filter_that_was_set_before_the_session(self):
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events)
        (self.root / "events/sched/sched_switch/filter").write_text("prev_pid == 1")
        cap.TRACEFS = self.root
        with cap.TraceSession(events) as session:
            session.set_pid_filter(999)
            self.assertEqual(
                (self.root / "events/sched/sched_switch/filter").read_text(),
                "prev_pid == 999 || next_pid == 999")
        self.assertEqual(
            (self.root / "events/sched/sched_switch/filter").read_text(),
            "prev_pid == 1")

    def test_skips_restoring_filter_that_was_originally_empty(self):
        # Regression test: writing "0" back to an originally-empty filter
        # file for irq/napi/net/sock events raised EINVAL on the real
        # kernel (confirmed via a receive-path investigation live smoke
        # test against thunderbolt0 traffic) -- these event groups' field
        # sets don't accept a bare "0" operand the way
        # sched_switch/sched_wakeup's pid fields do. The fix: never write
        # back an originally-empty filter at all, since empty is already
        # the default state.
        events = ["napi/napi_poll"]
        make_fake_tracefs(self.root, events)
        napi_filter = self.root / "events/napi/napi_poll/filter"
        cap.TRACEFS = self.root
        with cap.TraceSession(events) as session:
            pass  # no set_pid_filter call -- filter file stays empty throughout
        # Must not have raised, and the file is untouched (still empty).
        self.assertEqual(napi_filter.read_text(), "")

    def test_clears_set_ftrace_pid_with_a_space_when_originally_unset(self):
        # Regression test: writing "-1" (or the real readback "no pid")
        # back to set_ftrace_pid verbatim raised EINVAL on the real kernel.
        # The clear operation this kernel actually accepts is writing a
        # single space (confirmed empirically in receive-path
        # investigation) -- __exit__ must use that, not the literal
        # original readback text, and must not skip clearing altogether
        # (that would leak the session's pid into whatever runs next).
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events, set_ftrace_pid="-1")
        set_ftrace_pid_path = self.root / "set_ftrace_pid"

        original_write = Path.write_text

        def guarded_write(self_path, value):
            if self_path == set_ftrace_pid_path and value in ("-1", "no pid"):
                raise OSError(22, "Invalid argument: not accepted on restore")
            return original_write(self_path, value)

        cap.TRACEFS = self.root
        Path.write_text = guarded_write
        try:
            with cap.TraceSession(events) as session:
                session.set_pid_filter(555)
                self.assertEqual(set_ftrace_pid_path.read_text(), "555")
            self.assertEqual(set_ftrace_pid_path.read_text(), " ")
        finally:
            Path.write_text = original_write

    def test_clears_set_ftrace_pid_when_readback_is_no_pid(self):
        # This kernel's actual readback for "unset" is the string "no pid"
        # (confirmed via a real /sys/kernel/tracing read in receive-path
        # investigation), not "-1" -- both must be treated as the unset
        # default and cleared with a space rather than restored verbatim.
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events, set_ftrace_pid="no pid")
        set_ftrace_pid_path = self.root / "set_ftrace_pid"
        cap.TRACEFS = self.root
        with cap.TraceSession(events) as session:
            session.set_pid_filter(555)
            self.assertEqual(set_ftrace_pid_path.read_text(), "555")
        self.assertEqual(set_ftrace_pid_path.read_text(), " ")

    def test_restores_real_set_ftrace_pid_when_originally_set(self):
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events, set_ftrace_pid="123")
        cap.TRACEFS = self.root
        with cap.TraceSession(events) as session:
            session.set_pid_filter(999)
            self.assertEqual((self.root / "set_ftrace_pid").read_text(), "999")
        self.assertEqual((self.root / "set_ftrace_pid").read_text(), "123")

    def test_restoration_continues_past_a_missing_event_enable_file(self):
        # If one event's enable file disappears mid-session (simulated by
        # deleting it before __exit__ runs), restoration of the other
        # events and of tracing_on must still complete rather than
        # aborting on the first failure.
        events = ["sched/sched_switch", "sched/sched_wakeup"]
        make_fake_tracefs(self.root, events)
        cap.TRACEFS = self.root
        session = cap.TraceSession(events)
        session.__enter__()
        (self.root / "events/sched/sched_switch/enable").unlink()
        session.__exit__(None, None, None)
        self.assertEqual((self.root / "tracing_on").read_text(), "0")
        self.assertEqual((self.root / "events/sched/sched_wakeup/enable").read_text(), "0")


class FilterExpressionClassificationTests(unittest.TestCase):
    def test_empty_is_not_a_real_filter(self):
        self.assertFalse(cap.looks_like_real_filter_expression(""))
        self.assertFalse(cap.looks_like_real_filter_expression(None))

    def test_stale_parse_error_banner_is_not_a_real_filter(self):
        # Regression test: a real /sys/kernel/tracing filter file, once an
        # invalid expression has ever been written to it, permanently
        # echoes this banner on read even after writing "" -- confirmed via
        # a live the receive-path investigation work capture. It must never
        # be treated as a filter expression to restore (attempting to write
        # it back verbatim always fails with EINVAL, since it isn't valid
        # filter syntax).
        banner = ("none\n    ^\nparse_error: Field not found\n     ^\n"
                  "parse_error: Field not found")
        self.assertFalse(cap.looks_like_real_filter_expression(banner))

    def test_real_pid_filter_expression_is_recognized(self):
        self.assertTrue(cap.looks_like_real_filter_expression("prev_pid == 4242"))
        self.assertTrue(cap.looks_like_real_filter_expression(
            "prev_pid == 4242 || next_pid == 4242"))

    def test_bare_zero_or_one_is_recognized(self):
        self.assertTrue(cap.looks_like_real_filter_expression("0"))
        self.assertTrue(cap.looks_like_real_filter_expression("1"))


class ReceiveEventDiscoveryTests(unittest.TestCase):
    def test_receive_events_are_all_marked_system_wide(self):
        # RECEIVE_EVENTS has no PID-filterable member -- every one of them
        # must appear in SYSTEM_WIDE_EVENTS, or the capture tool would
        # silently under-report which events are unscoped (the
        # receive-path investigation work item 26: "Do not pretend
        # system-wide events belong uniquely to TBCCL without additional
        # correlation").
        for event in cap.RECEIVE_EVENTS:
            self.assertIn(event, cap.SYSTEM_WIDE_EVENTS)

    def test_receive_events_have_no_pid_filter_expression(self):
        for event in cap.RECEIVE_EVENTS:
            self.assertNotIn(event, cap.EVENT_PID_FILTERS)

    def test_list_available_events_groups_by_subsystem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "available_events").write_text(
                "irq:irq_handler_entry\nnapi:napi_poll\nirq:softirq_entry\n")
            cap.TRACEFS = root
            output = root / "out.json"
            cap.list_available_events(str(output))
            data = json.loads(output.read_text())
            self.assertEqual(data["total_events"], 3)
            self.assertEqual(sorted(data["groups"]["irq"]),
                              ["irq:irq_handler_entry", "irq:softirq_entry"])
            self.assertEqual(data["groups"]["napi"], ["napi:napi_poll"])


class FunctionTracingTests(unittest.TestCase):
    """Tier-1 narrow function tracing (--include-nhi-
    functions) -- current_tracer/set_ftrace_filter save/restore, and the
    real bug this phase found: set_ftrace_pid must NOT be written when a
    function_filter is active, since ring_msix/ring_work/tb_ring_poll run
    in interrupt/workqueue context, never attributed to the benchmark's
    own PID -- writing it would scope the function tracer away from every
    context those functions actually run in (confirmed via a live smoke
    test that captured zero function-trace lines before this fix)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._original_tracefs = cap.TRACEFS

    def tearDown(self):
        cap.TRACEFS = self._original_tracefs
        self._tmp.cleanup()

    def test_function_filter_sets_current_tracer_and_filter(self):
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events)
        cap.TRACEFS = self.root
        with cap.TraceSession(events, function_filter=["ring_msix", "ring_work"]):
            self.assertEqual((self.root / "current_tracer").read_text(), "function")
            self.assertEqual((self.root / "set_ftrace_filter").read_text(),
                              "ring_msix\nring_work\n")

    def test_function_filter_restored_to_nop(self):
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events, current_tracer="nop")
        cap.TRACEFS = self.root
        with cap.TraceSession(events, function_filter=["ring_msix"]):
            pass
        self.assertEqual((self.root / "current_tracer").read_text(), "nop")

    def test_unset_filter_placeholder_restored_as_empty(self):
        # Regression test: writing back the literal "#### all functions
        # enabled ####" placeholder (this kernel's readback for "no
        # filter set") raises EINVAL -- must be normalized to "" instead.
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events,
                           set_ftrace_filter="#### all functions enabled ####")
        filter_path = self.root / "set_ftrace_filter"
        original_write = Path.write_text

        def guarded_write(self_path, value):
            if self_path == filter_path and "all functions enabled" in value:
                raise OSError(22, "Invalid argument: placeholder not accepted on restore")
            return original_write(self_path, value)

        cap.TRACEFS = self.root
        Path.write_text = guarded_write
        try:
            with cap.TraceSession(events, function_filter=["ring_msix"]):
                pass
            self.assertEqual(filter_path.read_text(), "")
        finally:
            Path.write_text = original_write

    def test_real_function_filter_restored_verbatim(self):
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events, set_ftrace_filter="some_other_func\n")
        cap.TRACEFS = self.root
        with cap.TraceSession(events, function_filter=["ring_msix"]):
            pass
        self.assertEqual((self.root / "set_ftrace_filter").read_text(), "some_other_func")

    def test_set_pid_filter_does_not_write_set_ftrace_pid_when_function_filter_active(self):
        # The real the NHI DMA-ring work bug: NHI functions run in
        # interrupt/workqueue context, not the benchmark's own PID --
        # set_ftrace_pid must stay untouched (system-wide function
        # tracing) or every NHI function trace line is silently filtered
        # away.
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events, set_ftrace_pid="no pid")
        cap.TRACEFS = self.root
        with cap.TraceSession(events, function_filter=["ring_msix"]) as session:
            session.set_pid_filter(4242)
            self.assertEqual((self.root / "set_ftrace_pid").read_text(), "no pid")
            # Tracepoint PID filters (unrelated to the function tracer)
            # must still be applied normally.
            self.assertEqual(
                (self.root / "events/sched/sched_switch/filter").read_text(),
                "prev_pid == 4242 || next_pid == 4242")

    def test_set_pid_filter_still_writes_set_ftrace_pid_without_function_filter(self):
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events, set_ftrace_pid="no pid")
        cap.TRACEFS = self.root
        with cap.TraceSession(events) as session:  # function_filter=None (default)
            session.set_pid_filter(4242)
            self.assertEqual((self.root / "set_ftrace_pid").read_text(), "4242")

    def test_no_function_tracing_when_filter_is_none(self):
        # Default behavior (every prior phase) must be completely
        # untouched: current_tracer/set_ftrace_filter are never read or
        # written when function_filter is None.
        events = ["sched/sched_switch"]
        make_fake_tracefs(self.root, events, current_tracer="nop")
        cap.TRACEFS = self.root
        with cap.TraceSession(events):
            self.assertEqual((self.root / "current_tracer").read_text(), "nop")


class CheckFilterFunctionsTests(unittest.TestCase):
    def test_reports_presence_and_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "available_filter_functions").write_text(
                "ring_msix\nring_work\ntb_ring_poll\nunrelated_func [thunderbolt]\n")
            cap.TRACEFS = root
            output = root / "out.json"
            cap.check_filter_functions(
                ["ring_msix", "__ring_interrupt", "tb_ring_poll"], str(output))
            data = json.loads(output.read_text())
            self.assertTrue(data["ring_msix"])
            self.assertTrue(data["tb_ring_poll"])
            self.assertFalse(data["__ring_interrupt"])


if __name__ == "__main__":
    unittest.main()
