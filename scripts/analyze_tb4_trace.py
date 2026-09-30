#!/usr/bin/env python3
"""Analyze bounded TBCCL transfer traces and correlate Linux kernel events.

Application and journal timestamps are compared only on Linux, within one boot.
Durations are calculated only from timestamps recorded by one process. Absolute
timestamps from different hosts are never subtracted.
"""
import argparse
import datetime
import json
import math
from pathlib import Path
import re
import statistics

import tb4_health_snapshot as health

PREFIX = 'TBCCL_DIAGNOSTIC '
BDF = re.compile(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]')


def load(path):
    return json.loads(Path(path).read_text())


def diagnostics(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(PREFIX):
            result.append(json.loads(line[len(PREFIX):]))
    return result


def percentile(values, fraction):
    """Empirical nearest-rank order statistic; no population-p99 claim."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def duration(entry, begin, end):
    a, b = entry.get(begin), entry.get(end)
    return None if a is None or b is None else (b - a) / 1000.0


def trace_interval(trace):
    if trace.get('clock') != 'CLOCK_MONOTONIC' or trace.get('clock_units') != 'nanoseconds':
        raise ValueError('trace does not use CLOCK_MONOTONIC nanoseconds')
    entries = trace.get('entries', [])
    if not entries:
        raise ValueError('empty trace')
    starts = [e['iteration_begin_ns'] for e in entries if e.get('iteration_begin_ns') is not None]
    ends = [e['iteration_end_ns'] for e in entries if e.get('iteration_end_ns') is not None]
    if not starts or not ends:
        raise ValueError('trace has no complete local interval')
    return min(starts), max(ends)


def event_provenance(event):
    message = event.get('message', '')
    devices = BDF.findall(message)
    received = re.search(
        r'pcieport\s+(' + BDF.pattern + r').*received from\s+(' + BDF.pattern + r')',
        message, re.I)
    severity = ('fatal' if re.search(r'\bfatal\b', message, re.I)
                else 'uncorrectable' if re.search(r'uncorrect', message, re.I)
                else 'correctable' if re.search(r'correct', message, re.I)
                else 'unknown')
    return {'cursor': event.get('cursor'), 'monotonic_us': event.get('monotonic_us'),
            'source': event.get('source', 'unknown'),
            'reporting_pci_device': (received.group(1) if received else
                                     devices[0] if len(devices) > 1 else None),
            'affected_pci_device': (received.group(2) if received else
                                    devices[1] if len(devices) > 1 else
                                    devices[0] if devices else None),
            'severity': severity, 'description': message or None}


def correlate_kernel_events(trace, before, after):
    if not before.get('boot_id') or before.get('boot_id') != after.get('boot_id'):
        raise ValueError('kernel event correlation rejected across reboot boundary')
    trace_boot = trace.get('boot_id')
    if trace_boot in (None, '', 'unavailable') or trace_boot != after.get('boot_id'):
        raise ValueError('application trace and kernel snapshot boot IDs differ')
    start_ns, end_ns = trace_interval(trace)
    prior = {e.get('cursor') for e in before.get('kernel_events', []) if e.get('cursor')}
    output = []
    for event in after.get('kernel_events', []):
        if event.get('cursor') and event.get('cursor') in prior:
            continue
        row = event_provenance(event)
        try:
            timestamp_ns = int(event.get('monotonic_us')) * 1000
        except (TypeError, ValueError):
            row['interval_relation'] = 'unknown-missing-monotonic-timestamp'
        else:
            row['interval_relation'] = ('inside' if start_ns <= timestamp_ns <= end_ns
                                        else 'before' if timestamp_ns < start_ns else 'after')
        output.append(row)
    return {'same_boot': True, 'clock_relationship':
            'Linux CLOCK_MONOTONIC ns compared with journal __MONOTONIC_TIMESTAMP us',
            'application_interval_ns': [start_ns, end_ns], 'events': output}


def clusters(indices):
    groups = []
    for index in sorted(indices):
        if not groups or index != groups[-1][-1] + 1:
            groups.append([index])
        else:
            groups[-1].append(index)
    return groups


def analyze_pair(source_trace, receiver_trace, samples=None):
    source = source_trace.get('entries', [])
    receiver = receiver_trace.get('entries', [])
    if [e.get('iteration_index') for e in source] != [e.get('iteration_index') for e in receiver]:
        raise ValueError('source and receiver iteration identities differ')
    if not source or source[0].get('local_rank') != source[0].get('source_rank'):
        raise ValueError('first trace is not the source trace')
    if receiver and receiver[0].get('local_rank') == receiver[0].get('source_rank'):
        raise ValueError('second trace is not the receiver trace')
    if samples is not None and len(samples) != len(source):
        raise ValueError('raw completion sample count differs from trace')
    rows = []
    for position, (sent, received) in enumerate(zip(source, receiver)):
        timer_begin = ('source_ready_ns' if sent.get('timing_scope') == 'ready'
                       else 'iteration_begin_ns')
        completion = (samples[position] if samples is not None
                      else duration(sent, timer_begin, 'ack_received_ns'))
        rows.append({
            'iteration_index': sent['iteration_index'],
            'completion_us': completion,
            'source_preparation_us': duration(sent, 'iteration_begin_ns', 'source_ready_ns'),
            'source_staging_us': duration(sent, 'staging_begin_ns', 'staging_end_ns'),
            'sender_send_us': duration(sent, 'send_begin_ns', 'send_end_ns'),
            'ack_wait_us': duration(sent, 'ack_wait_begin_ns', 'ack_received_ns'),
            'receiver_recv_us': duration(received, 'recv_begin_ns', 'recv_end_ns'),
            'destination_staging_us': duration(
                received, 'destination_staging_begin_ns', 'destination_staging_end_ns'),
            'destination_sync_us': duration(
                received, 'destination_staging_end_ns', 'destination_sync_end_ns'),
            'observed_source_gap_us': (None if sent.get('observed_source_gap_ns') is None
                                       else sent['observed_source_gap_ns'] / 1000.0),
        })
    values = [r['completion_us'] for r in rows]
    slow500 = [r['iteration_index'] for r in rows if r['completion_us'] > 500]
    slow1000 = [r['iteration_index'] for r in rows if r['completion_us'] > 1000]
    med = statistics.median(values)
    return {'clock_note': 'all durations are process-local; receiver and sender absolute clocks are not subtracted',
            'rows': rows,
            'summary': {'samples': len(values), 'median_us': med,
                        'p95_empirical_us': percentile(values, .95),
                        'p99_empirical_us': percentile(values, .99), 'max_us': max(values),
                        'over_500_us': len(slow500), 'over_1000_us': len(slow1000),
                        'over_median_plus_500_us': sum(v > med + 500 for v in values),
                        'slow_clusters_over_500_us': clusters(slow500),
                        'slow_clusters_over_1000_us': clusters(slow1000)}}


def analyze_run(directory):
    directory = Path(directory)
    metadata = load(directory / 'metadata.json')
    config = metadata['configuration']
    traces = [load(directory / f'rank{rank}.trace.json') for rank in (0, 1)]
    source_rank = config['source_rank']
    source_events = diagnostics(directory / f'rank{source_rank}.stderr')
    sample_events = [e for e in source_events if e.get('kind') == 'samples' and
                     e.get('metric') == 'completion_confirmed']
    samples = sample_events[0]['values_us'] if len(sample_events) == 1 else None
    result = {'configuration': config,
              'tail_analysis': analyze_pair(traces[source_rank], traces[1-source_rank], samples)}
    health_after_path = directory / 'health-after.json'
    health_delta_path = directory / 'health-delta.json'
    if health_after_path.exists() and health_delta_path.exists():
        after_all, delta_all = load(health_after_path), load(health_delta_path)
        after = after_all.get('linux', {})
        delta = delta_all.get('linux', {})
        # Reconstruct the prior cursor set for the already-computed same-boot delta.
        new_cursors = {e.get('cursor') for e in delta.get('new_kernel_events', [])}
        before = dict(after, kernel_events=[e for e in after.get('kernel_events', [])
                                            if e.get('cursor') not in new_cursors])
        linux_rank = metadata.get('physical_ranks', []).index('linux')
        result['kernel_correlation'] = correlate_kernel_events(
            traces[linux_rank], before, after)
    return result


def historical_incidents(root):
    """Build snapshot-association records from preserved Phase 20 artifacts."""
    rows = []
    seen = set()
    root = Path(root)
    for journal_path in sorted(root.glob('*/kernel-live.jsonl')):
        candidates = []
        for metadata_path in journal_path.parent.glob('run-*/metadata.json'):
            try:
                metadata = load(metadata_path)
                started = datetime.datetime.fromisoformat(metadata['start_utc']).timestamp() * 1_000_000
                candidates.append((started, metadata_path, metadata))
            except (OSError, ValueError, KeyError):
                continue
        candidates.sort()
        recent = []
        for line in journal_path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            recent.append(event)
            recent = recent[-8:]
            if not health.TB_TIMEOUT.search(str(event.get('MESSAGE', ''))):
                continue
            realtime = int(event.get('__REALTIME_TIMESTAMP', 0))
            matches = [candidate for candidate in candidates if candidate[0] <= realtime]
            if not matches:
                continue
            _, metadata_path, metadata = matches[-1]
            after_path = metadata_path.with_name('health-after.json')
            after = load(after_path).get('linux', {}) if after_path.exists() else {}
            relevant = {d.get('bdf') for d in after.get('pci_devices', [])}
            message = str(event.get('MESSAGE', ''))
            bdfs = set(BDF.findall(message))
            normalized = {'cursor': event.get('__CURSOR'),
                          'monotonic_us': event.get('__MONOTONIC_TIMESTAMP'),
                          'realtime_us': event.get('__REALTIME_TIMESTAMP'),
                          'message': message,
                          'source': ('thunderbolt' if bdfs & relevant else
                                     'other-pci' if bdfs else 'unattributed')}
            identity = normalized.get('cursor') or (normalized.get('monotonic_us'), message)
            if identity in seen:
                continue
            seen.add(identity)
            config = metadata.get('configuration', {})
            interface = after.get('interface', {})
            provenance = event_provenance(normalized)
            context = []
            for item in recent:
                context_message = str(item.get('MESSAGE', ''))
                context.append({'monotonic_us': item.get('__MONOTONIC_TIMESTAMP'),
                                'message': context_message})
                received = re.search(
                    r'pcieport\s+(' + BDF.pattern + r').*received from\s+(' + BDF.pattern + r')',
                    context_message, re.I)
                if received:
                    provenance['reporting_pci_device'] = received.group(1)
                    provenance['affected_pci_device'] = received.group(2)
                    provenance['severity'] = 'correctable'
            rows.append({'incident': len(rows) + 1,
                         'group_run_id': str(metadata_path.parent.relative_to(root)),
                         'direction': config.get('source'), 'rank0': config.get('rank0'),
                         'backend': config.get('backend_pair'),
                         'payload_bytes': config.get('payload_bytes'),
                         'busy_poll_us': config.get('busy_poll_us'),
                         'snapshot_interval': metadata.get('start_utc'),
                         'kernel_event': provenance, 'kernel_context': context,
                         'kernel_realtime_us': normalized['realtime_us'],
                         'application_success': metadata.get('success', metadata.get('outcome', {}).get('success')),
                         'interface_present_after': interface.get('present'),
                         'carrier_after': interface.get('carrier'),
                         'classification': 'historical snapshot-interval association; causality not established'})
    if rows:
        rows.sort(key=lambda row: int(row.get('kernel_realtime_us') or 0))
        for index, row in enumerate(rows, 1):
            row['incident'] = index
        return rows
    # Older artifact bundles may contain snapshots but no live journal.
    for delta_path in sorted(root.glob('**/run-*/health-delta.json')):
        try:
            delta = load(delta_path).get('linux', {})
            metadata = load(delta_path.with_name('metadata.json'))
            after = load(delta_path.with_name('health-after.json')).get('linux', {})
        except (OSError, ValueError, KeyError):
            continue
        events = [e for e in delta.get('new_kernel_events', [])
                  if e.get('source') == 'thunderbolt' and health.TB_TIMEOUT.search(e.get('message', ''))]
        for event in events:
            identity = event.get('cursor') or (event.get('monotonic_us'), event.get('message'))
            if identity in seen:
                continue
            seen.add(identity)
            config = metadata.get('configuration', {})
            interface = after.get('interface', {})
            row = {'incident': len(rows) + 1, 'group_run_id': str(delta_path.parent.relative_to(root)),
                   'direction': config.get('source'), 'rank0': config.get('rank0'),
                   'backend': config.get('backend_pair'), 'payload_bytes': config.get('payload_bytes'),
                   'busy_poll_us': config.get('busy_poll_us'),
                   'snapshot_interval': metadata.get('start_utc'),
                   'kernel_event': event_provenance(event),
                   'application_success': metadata.get('success', metadata.get('outcome', {}).get('success')),
                   'interface_present_after': interface.get('present'),
                   'carrier_after': interface.get('carrier'),
                   'classification': 'historical snapshot-interval association; causality not established'}
            rows.append(row)
    return rows


def parse_diagnostic_file(path):
    """Reads one rank's --trace stderr output (TBCCL_DIAGNOSTIC line(s))."""
    events = diagnostics(path)
    batches = [e for e in events if e.get('kind') == 'transfer_trace']
    if not batches:
        raise ValueError(f'{path}: no transfer_trace diagnostic found')
    return batches[-1]


# Phase 22 Part H/M: correlate a Linux receiver's application trace with a
# bounded scheduler trace (from capture_tb4_scheduler_trace.py) and,
# optionally, a packet capture, to classify slow iterations per item 70.
# The scheduler trace's tracefs events use trace_clock=mono, which the
# capture script sets to be identical (same clock, units, epoch) to the
# application trace's CLOCK_MONOTONIC -- no calibration needed between
# those two. A packet capture's timestamps are in CLOCK_REALTIME, so
# correlating against it requires the realtime<->monotonic offset the
# caller derives from the application trace's own bracketing clock
# samples (clock_sample_monotonic_ns / clock_sample_realtime_ns).

def sched_cycles_for_pid(sched_result, pid):
    """Returns [(sleep_ts, wakeup_ts, run_ts)] sleep/wakeup/resume cycles
    for one PID, in trace_clock=mono seconds (== CLOCK_MONOTONIC). Each
    cycle is: the PID goes to sleep (sched_switch, prev_state S/D/Z) ->
    later woken (sched_wakeup) -> actually put on a CPU (sched_switch,
    prev_state R, i.e. was runnable, not the wakeup itself)."""
    if sched_result.get('error'):
        return []
    pid_str = str(pid)
    events = []  # (ts, 'sleep'|'wakeup'|'run')
    for line in (sched_result.get('trace_text') or '').splitlines():
        if not line or line.startswith('#'):
            continue
        ts_match = re.search(r'(\d+\.\d+):\s*sched_', line)
        if not ts_match:
            continue
        ts = float(ts_match.group(1))
        if 'sched_wakeup' in line and f'pid={pid_str} ' in line + ' ':
            events.append((ts, 'wakeup'))
        elif 'sched_switch' in line:
            if f'next_pid={pid_str} ' in line + ' ':
                events.append((ts, 'run'))
            if f'prev_pid={pid_str} ' in line + ' ' and 'prev_state=R' not in line:
                events.append((ts, 'sleep'))
    events.sort()
    cycles = []
    pending_sleep = None
    pending_wakeup = None
    for ts, kind in events:
        if kind == 'sleep':
            pending_sleep = ts
        elif kind == 'wakeup' and pending_sleep is not None:
            pending_wakeup = ts
        elif kind == 'run' and pending_sleep is not None and pending_wakeup is not None:
            cycles.append((pending_sleep, pending_wakeup, ts))
            pending_sleep = pending_wakeup = None
    return cycles


# Phase 23 Part E/F/G: classify packets in a capture (application-level
# completion ACK vs. plain TCP ACK vs. tensor payload) and compute, from
# ONE machine's own local capture clock only, the interval between a prior
# completion ACK and the next tensor payload -- never comparing timestamps
# from two different machines' captures directly (Part F item 28/29).
#
# Classification is by observed payload length only (confirmed against
# real captures in this phase): the benchmark's completion ACK is always a
# single application byte (TCP payload length 1); a bare TCP ACK carries no
# payload (length 0); a tensor payload segment is large (length >= 1000,
# comfortably above the connection handshake's small control messages of
# 24/40 bytes and far below a real 64 KiB tensor's segments of ~2900/62636
# bytes after GRO coalescing).
TCPDUMP_LINE = re.compile(
    r'^(?P<ts>\d+\.\d+)\s+IP\s+(?P<src>\S+)\s+>\s+(?P<dst>\S+):\s+'
    r'Flags\s+\[(?P<flags>[^\]]*)\].*?length\s+(?P<length>\d+)')


def parse_tcpdump_text(text):
    """Pure function over tcpdump -tt output text -- kept separate from
    run_tcpdump so it can be unit tested without invoking tcpdump."""
    packets = []
    for line in text.splitlines():
        match = TCPDUMP_LINE.match(line)
        if not match:
            continue
        packets.append({
            'ts': float(match.group('ts')),
            'src': match.group('src'),
            'dst': match.group('dst'),
            'flags': match.group('flags'),
            'length': int(match.group('length')),
        })
    return packets


def run_tcpdump(pcap_path):
    import subprocess
    proc = subprocess.run(
        ['tcpdump', '-tt', '-r', str(pcap_path)],
        capture_output=True, text=True, timeout=30)
    return parse_tcpdump_text(proc.stdout)


def classify_packet(packet, payload_min_length=1000):
    if packet['length'] == 0:
        return 'tcp_ack_only'
    if packet['length'] == 1:
        return 'application_ack'
    if packet['length'] >= payload_min_length:
        return 'tensor_payload'
    return 'other'  # e.g. connection-handshake control messages


def ack_to_payload_intervals(
        packets, payload_min_length=1000, max_gap_us=5000, expected_count=None):
    """For each application-completion-ack packet, the local-capture-clock
    gap until the next tensor-payload packet on the SAME capture. Both
    timestamps come from one machine's own capture -- comparing the
    resulting distributions between two machines' captures (rather than
    subtracting their timestamps) is Part F's dual-end method.

    The final measured iteration's completion ack has no legitimate next
    payload (the run ends after it) -- without a bound, it would pair with
    whatever unrelated packet happens to appear later in the capture
    (observed in this phase: a ~2ms "interval" that was actually the last
    iteration's ack spuriously matched against post-run traffic). max_gap_us
    (default 5000, well above any interval this investigation is chasing)
    excludes such pairings; every dropped ack is still reported, in
    unmatched_ack_timestamps, rather than silently discarded, since a
    genuinely large real delay must never be hidden by this bound.

    expected_count, when given (the caller's known measured-iteration
    count), truncates the returned intervals to the first expected_count
    temporally-ordered pairs. This excludes the benchmark's own untimed
    post-loop verification round (tensor_transfer_bench.cpp resends one
    full, explicitly untimed tensor for byte verification after the
    measured loop) -- discovered in this phase when that round's payload
    was found spuriously paired with the last measured iteration's ack,
    producing a ~2ms "interval" that was not a real measurement. Anything
    beyond expected_count is reported in extra_beyond_expected, never
    silently dropped."""
    classified = sorted(
        (p['ts'], classify_packet(p, payload_min_length)) for p in packets)
    intervals = []
    unmatched = []
    pending_ack_ts = None
    for ts, kind in classified:
        if kind == 'application_ack':
            if pending_ack_ts is not None:
                unmatched.append(pending_ack_ts)
            pending_ack_ts = ts
        elif kind == 'tensor_payload' and pending_ack_ts is not None:
            gap_us = (ts - pending_ack_ts) * 1e6
            if gap_us <= max_gap_us:
                intervals.append({
                    'ack_ts': pending_ack_ts, 'payload_ts': ts, 'gap_us': gap_us})
                pending_ack_ts = None
            # else: leave pending_ack_ts set -- a later, closer payload
            # (if any) is preferred over this distant one.
    if pending_ack_ts is not None:
        unmatched.append(pending_ack_ts)
    extra = []
    if expected_count is not None and len(intervals) > expected_count:
        extra = intervals[expected_count:]
        intervals = intervals[:expected_count]
    return intervals, unmatched, extra


def packet_silence_gaps(pcap_path, min_gap_us=200):
    """Shells out to tcpdump -tt to list inter-packet gaps on the capture
    above min_gap_us. Timestamps are CLOCK_REALTIME (unix epoch seconds,
    tcpdump's native format) -- never compared directly to CLOCK_MONOTONIC
    without the caller applying an offset. tcpdump is required (no extra
    Python packet-parsing dependency)."""
    import shutil
    import subprocess
    if not shutil.which('tcpdump'):
        return None
    proc = subprocess.run(
        ['tcpdump', '-tt', '-r', str(pcap_path)],
        capture_output=True, text=True, timeout=30)
    timestamps = []
    for line in proc.stdout.splitlines():
        try:
            timestamps.append(float(line.split()[0]))
        except (IndexError, ValueError):
            continue
    timestamps.sort()
    gaps = []
    for a, b in zip(timestamps, timestamps[1:]):
        gap_us = (b - a) * 1e6
        if gap_us >= min_gap_us:
            gaps.append((a, b, gap_us))
    return gaps


def classify_slow_iterations(
        app_trace, sched_result=None, pcap_path=None, slow_threshold_us=500, gaps=None):
    """Part M items 68-70: per-iteration diagnostics + classification for
    every receiver-side iteration slower than slow_threshold_us.

    Classifications are only made from intervals actually observed in this
    run's evidence -- absence of a signal is reported as
    'insufficient evidence', never inferred.

    `gaps` (a list of (start_realtime_s, end_realtime_s, gap_us) tuples)
    can be passed directly instead of `pcap_path`, to avoid recomputing
    them for repeated calls or invoking tcpdump in tests."""
    entries = app_trace.get('entries', [])
    pid = app_trace.get('process_id')
    cycles = sched_cycles_for_pid(sched_result, pid) if sched_result and pid else []

    realtime_offset_ns = None
    if app_trace.get('clock_sample_monotonic_ns') is not None and \
            app_trace.get('clock_sample_realtime_ns') is not None:
        realtime_offset_ns = (app_trace['clock_sample_realtime_ns'] -
                              app_trace['clock_sample_monotonic_ns'])

    if gaps is None and pcap_path is not None:
        gaps = packet_silence_gaps(pcap_path, min_gap_us=200)

    rows = []
    for entry in entries:
        recv_us = duration(entry, 'recv_begin_ns', 'recv_end_ns')
        if recv_us is None or recv_us < slow_threshold_us:
            continue
        recv_begin_s = entry['recv_begin_ns'] / 1e9
        recv_end_s = entry['recv_end_ns'] / 1e9

        # Find the sleep/wakeup/run cycle enclosing this iteration's recv:
        # the thread records recv_begin while still running (from a PRIOR
        # cycle's wakeup), then blocks -- that block is this cycle's sleep
        # event -- and later resumes (this cycle's run event) and records
        # recv_end while running again. So: sleep must occur at/after
        # recv_begin, and run must occur at/before recv_end.
        enclosing = next(
            (c for c in cycles if c[0] >= recv_begin_s and c[2] <= recv_end_s + 0.0005),
            None)

        row = {
            'iteration_index': entry.get('iteration_index'),
            'recv_us': recv_us,
            'trace_quality': 'application-only',
            'classification': 'insufficient evidence',
            'wakeup_to_run_us': None,
            'sleep_to_wakeup_us': None,
            'wire_silence_overlap_us': None,
        }

        if enclosing is not None:
            sleep_ts, wakeup_ts, run_ts = enclosing
            row['wakeup_to_run_us'] = (run_ts - wakeup_ts) * 1e6
            row['sleep_to_wakeup_us'] = (wakeup_ts - sleep_ts) * 1e6
            row['trace_quality'] = 'application + scheduler'

            if row['wakeup_to_run_us'] > 100:
                row['classification'] = 'scheduler wakeup delay observed'
            elif gaps is not None and realtime_offset_ns is not None:
                row['trace_quality'] = 'application + scheduler + packet'
                sleep_realtime = sleep_ts + realtime_offset_ns / 1e9
                wakeup_realtime = wakeup_ts + realtime_offset_ns / 1e9
                overlap = next(
                    (g for g in gaps if g[0] <= wakeup_realtime and g[1] >= sleep_realtime),
                    None)
                if overlap is not None:
                    row['wire_silence_overlap_us'] = overlap[2]
                    row['classification'] = 'late local packet observation'
                else:
                    row['classification'] = 'mixed/ambiguous'
            else:
                row['classification'] = ('scheduler wakeup delay observed'
                                         if row['wakeup_to_run_us'] > 100
                                         else 'mixed/ambiguous')
        rows.append(row)
    return rows


# Phase 24: aggregate a run_tb4_tail_trigger_sweep.py output into a
# condition-level slow-event summary, and a position-in-burst ("first-N")
# breakdown. Kept here rather than in the sweep script itself so both the
# sweep runner and any future ad-hoc trigger data can be analyzed with the
# same code (Part N item 52's "avoid duplicating packet/trace parsing").

def summarize_by_condition(sweep_output, thresholds=(500, 800, 1000)):
    """sweep_output: the parsed JSON from run_tb4_tail_trigger_sweep.py
    (a dict with 'results', each having 'requested_idle_ms' and
    'samples_us'). Returns one row per distinct requested_idle_ms with
    session/iteration counts, threshold counts, and basic percentiles.
    Sessions with samples_us=None (a failed burst) are excluded from the
    iteration counts but still counted as attempted sessions."""
    by_condition = {}
    for result in sweep_output.get('results', []):
        condition = result['requested_idle_ms']
        by_condition.setdefault(condition, {'sessions': 0, 'samples': []})
        by_condition[condition]['sessions'] += 1
        if result.get('samples_us'):
            by_condition[condition]['samples'].extend(result['samples_us'])

    rows = []
    for condition in sorted(by_condition):
        samples = by_condition[condition]['samples']
        row = {
            'requested_idle_ms': condition,
            'sessions': by_condition[condition]['sessions'],
            'iterations': len(samples),
        }
        for threshold in thresholds:
            row[f'over_{threshold}_us'] = sum(1 for v in samples if v >= threshold)
        if samples:
            row['median_us'] = statistics.median(samples)
            row['p95_us'] = percentile(samples, .95)
            row['max_us'] = max(samples)
        else:
            row['median_us'] = row['p95_us'] = row['max_us'] = None
        rows.append(row)
    return rows


def summarize_by_position(sweep_output):
    """Position-in-burst ("first-N") breakdown: for each 0-indexed
    position within a burst, the median/mean/sample count across every
    session that reached that position -- reveals whether early
    iterations of a fresh burst are systematically slower than later ones,
    independent of which idle condition preceded the burst."""
    by_position = {}
    for result in sweep_output.get('results', []):
        for position, value in enumerate(result.get('samples_us') or []):
            by_position.setdefault(position, []).append(value)

    rows = []
    for position in sorted(by_position):
        values = by_position[position]
        rows.append({
            'position': position,
            'samples': len(values),
            'median_us': statistics.median(values),
            'mean_us': statistics.mean(values),
        })
    return rows


# Phase 25 Part G/M: decompose the Mac sender boundary preceding a slow
# Linux iteration, and classify which sub-interval expands. Discovered in
# this phase's own data that two genuinely distinct mechanisms exist
# among slow events: (1) the completion-ack packet is visible on Mac's
# own bridge0 capture promptly, but Mac's TCP-level ack / application
# ack_received lags far behind it (implicates Mac-local processing); (2)
# nothing at all is observed on Mac's bridge0 until near the very end of
# the wait (implicates something upstream of Mac's local capture point --
# Linux's own emission or network/TB4 transit). Distinguishing these does
# not require cross-host timestamps: both are computed purely from Mac's
# own local capture and Mac's own application trace, in Mac's own clock.

def mac_realtime_offset(mac_app_trace):
    return mac_app_trace['clock_sample_realtime_ns'] - mac_app_trace['clock_sample_monotonic_ns']


def first_application_ack_packet(mac_packets, window_start_s, window_end_s):
    """First length==1 (application completion ACK) packet observed on
    Mac's own capture within [window_start_s, window_end_s] realtime."""
    candidates = [
        p for p in mac_packets
        if classify_packet(p, payload_min_length=1000) == 'application_ack'
        and window_start_s <= p['ts'] <= window_end_s
    ]
    return min(candidates, key=lambda p: p['ts']) if candidates else None


def classify_mac_sender_boundary(
        linux_iteration_index, mac_entries_by_index, mac_packets, offset_ns,
        gap_threshold_us=200):
    """mac_entries_by_index: {iteration_index: entry} from the Mac's own
    application trace. Returns None if the previous/current Mac entries
    are unavailable (e.g. the very first measured iteration has no N-1).
    """
    previous = mac_entries_by_index.get(linux_iteration_index - 1)
    current = mac_entries_by_index.get(linux_iteration_index)
    if previous is None or current is None:
        return {
            'iteration_index': linux_iteration_index,
            'classification': 'insufficient evidence',
            'trace_quality': 'missing adjacent Mac trace entries',
        }

    ack_wait_us = (previous['ack_received_ns'] - previous['ack_wait_begin_ns']) / 1000.0
    app_ack_to_next_iter_us = (current['iteration_begin_ns'] - previous['iteration_end_ns']) / 1000.0
    iter_to_ready_us = (current['source_ready_ns'] - current['iteration_begin_ns']) / 1000.0
    ready_to_send_us = (current['send_begin_ns'] - current['source_ready_ns']) / 1000.0
    send_us = (current['send_end_ns'] - current['send_begin_ns']) / 1000.0

    row = {
        'iteration_index': linux_iteration_index,
        'mac_ack_wait_us': ack_wait_us,
        'mac_app_ack_to_next_iter_us': app_ack_to_next_iter_us,
        'mac_iter_to_ready_us': iter_to_ready_us,
        'mac_ready_to_send_us': ready_to_send_us,
        'mac_send_us': send_us,
        'mac_ack_packet_to_app_us': None,
        'mac_send_to_packet_us': None,
        'classification': 'insufficient evidence',
        'trace_quality': 'application-only',
    }

    if mac_packets is None:
        return row

    row['trace_quality'] = 'application + packet'

    # send_end -> this same iteration's own payload becoming visible on
    # Mac's own capture (Part I item 40 / Part M classification G).
    # send() is synchronous, so the payload is often already (mostly or
    # fully) visible on the wire microseconds BEFORE send_end is
    # timestamped, not strictly after -- confirmed empirically in this
    # phase's own data (a payload's trailing segment observed ~1us
    # before its iteration's app-level send_end). Searching only for
    # packets with ts >= send_end therefore skips the current iteration's
    # own (already-sent) payload and incorrectly finds the NEXT
    # iteration's payload instead, which measures a completely different,
    # much larger interval. Search a window that starts at send_begin
    # (payload can start streaming while send() is still in progress)
    # and take the payload packet closest to send_end, whichever side.
    send_begin_s = current['send_begin_ns'] / 1e9 + offset_ns / 1e9
    send_end_s = current['send_end_ns'] / 1e9 + offset_ns / 1e9
    payload_candidates = [
        p for p in mac_packets
        if classify_packet(p, payload_min_length=1000) == 'tensor_payload'
        and send_begin_s - 0.001 <= p['ts'] <= send_end_s + 0.003
    ]
    if payload_candidates:
        payload_packet = min(payload_candidates, key=lambda p: abs(p['ts'] - send_end_s))
        # Only report as a positive (post-send) delay; a packet observed
        # at/before send_end means emission was already prompt.
        row['mac_send_to_packet_us'] = max(0.0, (payload_packet['ts'] - send_end_s) * 1e6)

    window_start_s = previous['ack_wait_begin_ns'] / 1e9 + offset_ns / 1e9
    window_end_s = previous['ack_received_ns'] / 1e9 + offset_ns / 1e9
    ack_packet = first_application_ack_packet(mac_packets, window_start_s, window_end_s)

    if ack_packet is not None:
        # mac_ack_packet_to_app_us = time from the packet's arrival on
        # Mac's own capture to Mac's app-level ack_received. LARGE means
        # the packet was visible early but Mac was slow to process it
        # (Mac-local delay). SMALL means the packet only became visible
        # shortly before ack_received -- most of the wait elapsed BEFORE
        # the packet was even observable (delay upstream of Mac's
        # capture point: Linux's own emission or network/TB4 transit).
        mac_ack_packet_to_app_us = (window_end_s - ack_packet['ts']) * 1e6
        row['mac_ack_packet_to_app_us'] = mac_ack_packet_to_app_us

        if ack_wait_us < gap_threshold_us:
            # The wait itself wasn't elevated -- look at the other
            # sender-boundary sub-intervals instead.
            if ready_to_send_us >= gap_threshold_us and iter_to_ready_us < gap_threshold_us:
                row['classification'] = 'delayed send invocation'
            elif iter_to_ready_us >= gap_threshold_us:
                row['classification'] = 'delayed source preparation'
            elif send_us >= gap_threshold_us:
                row['classification'] = 'blocking send syscall'
            elif app_ack_to_next_iter_us >= gap_threshold_us:
                row['classification'] = 'delayed iteration start'
            elif row['mac_send_to_packet_us'] is not None and row['mac_send_to_packet_us'] >= gap_threshold_us:
                row['classification'] = 'delayed local packet emission after send'
            elif row['mac_send_to_packet_us'] is not None and row['mac_send_to_packet_us'] < gap_threshold_us:
                # Every Mac-local interval (ack_wait through post-send
                # emission) was normal -- by elimination, the delay lies
                # after Mac's own local packet observation (Part M
                # classification H): network/TB4 transit, or Linux-side
                # processing before Linux's own capture point.
                row['classification'] = 'delay after Mac local packet observation'
            else:
                row['classification'] = 'mixed/ambiguous'
        elif mac_ack_packet_to_app_us >= ack_wait_us * 0.5:
            row['classification'] = 'delayed Mac application ACK reception'
        else:
            row['classification'] = 'late completion ACK arrival at Mac'
    else:
        # No application-ack packet observed anywhere in the wait window
        # at all -- the delay is upstream of Mac's own local capture
        # point (Linux's own emission timing, or network/TB4 transit).
        row['classification'] = 'late completion ACK arrival at Mac'

    return row


def sender_boundary_report(
        linux_app_trace, mac_app_trace, mac_packets, slow_threshold_us=1000):
    """Part T item 81/82: full per-slow-event table plus a same-run normal
    baseline (excluding candidate tails, >=800us, from the baseline itself
    -- Part T item 82)."""
    linux_entries_by_index = {e['iteration_index']: e for e in linux_app_trace['entries']}
    mac_entries_by_index = {e['iteration_index']: e for e in mac_app_trace['entries']}
    offset_ns = mac_realtime_offset(mac_app_trace)

    slow_rows = []
    normal_ack_wait, normal_send_to_packet = [], []
    for index, entry in linux_entries_by_index.items():
        if entry.get('recv_begin_ns') is None or entry.get('recv_end_ns') is None:
            continue
        recv_us = (entry['recv_end_ns'] - entry['recv_begin_ns']) / 1000.0
        if recv_us >= slow_threshold_us:
            row = classify_mac_sender_boundary(
                index, mac_entries_by_index, mac_packets, offset_ns)
            row['linux_recv_us'] = recv_us
            slow_rows.append(row)
        elif recv_us < 800:
            # Same-run normal baseline, excluding candidate tails
            # (Part T item 82).
            row = classify_mac_sender_boundary(
                index, mac_entries_by_index, mac_packets, offset_ns)
            if row.get('mac_ack_wait_us') is not None:
                normal_ack_wait.append(row['mac_ack_wait_us'])
            if row.get('mac_send_to_packet_us') is not None:
                normal_send_to_packet.append(row['mac_send_to_packet_us'])

    def stats(values):
        if not values:
            return {'median_us': None, 'p95_us': None, 'max_us': None}
        return {'median_us': statistics.median(values), 'p95_us': percentile(values, .95),
                'max_us': max(values)}

    from collections import Counter
    classification_counts = dict(Counter(row['classification'] for row in slow_rows))

    return {
        'slow_events': slow_rows,
        'classification_counts': classification_counts,
        'same_run_normal_baseline': {
            'mac_ack_wait_us': stats(normal_ack_wait),
            'mac_send_to_packet_us': stats(normal_send_to_packet),
        },
    }


# Phase 26 Part F/H/P: Linux receive-path decomposition (irq -> softirq ->
# napi/skb-receive -> socket-wakeup), from capture_tb4_scheduler_trace.py's
# --include-receive-events trace_text. trace_clock=mono means these
# timestamps are already in the same seconds-since-boot CLOCK_MONOTONIC
# domain as the application trace's nanosecond timestamps (divided by 1e9)
# -- no calibration needed, same as the existing sched_cycles_for_pid path.
#
# This kernel exposes irq_handler_entry, softirq_entry (vec=3 == NET_RX),
# napi_poll, napi_gro_receive_entry, netif_receive_skb_entry, and
# sk_data_ready (confirmed via a live smoke test against real thunderbolt0
# traffic -- Part F item 19-20; see docs/phase26_report.md for the full
# available_events table). It does NOT expose a separate NAPI poll *entry*
# tracepoint (only one combined post-poll napi_poll event), so "NAPI
# processing" and "capture/skb-receive visibility" cannot be independently
# timed on this kernel -- napi_gro_receive_entry (GRO is active per `ethtool
# -k thunderbolt0`, so netif_receive_skb_entry does not fire for this
# device) is used as the single proxy for both, and this collapsing is
# documented rather than silently implied (Part P item 62: never infer a
# stage that was not traced).
FTRACE_RECEIVE_LINE = re.compile(
    r'(?P<ts>\d+\.\d+):\s*(?P<event>irq_handler_entry|softirq_entry|napi_poll|'
    r'napi_gro_receive_entry|netif_receive_skb_entry|sk_data_ready):\s*(?P<rest>.*)')
IRQ_NUMBER_LINE = re.compile(r'irq=(\d+)')


def parse_receive_path_trace(trace_text, device='thunderbolt0'):
    """Flat, time-sorted list of receive-path kernel events relevant to
    `device`, extracted from raw ftrace trace text. These tracepoints are
    system-wide (no per-PID filter applies to them, Part G item 26) --
    filtering by name=thunderbolt / dev=<device> narrows to the interface
    of interest for irq_handler_entry/napi_gro_receive_entry/napi_poll.
    sk_data_ready carries no per-device field at all; it is included
    unfiltered here and attributed to a specific iteration only by the
    caller's time-window correlation (receive_path_events_in_window), never
    claimed to belong uniquely to TBCCL without that correlation."""
    events = []
    for line in (trace_text or '').splitlines():
        if not line or line.startswith('#'):
            continue
        match = FTRACE_RECEIVE_LINE.search(line)
        if not match:
            continue
        ts = float(match.group('ts'))
        event = match.group('event')
        rest = match.group('rest')
        if event == 'irq_handler_entry' and 'name=thunderbolt' not in rest:
            continue
        if event == 'softirq_entry' and 'action=NET_RX' not in rest:
            continue
        if event == 'napi_poll' and f'device {device}' not in rest:
            continue
        if event in ('napi_gro_receive_entry', 'netif_receive_skb_entry') \
                and f'dev={device}' not in rest:
            continue
        if event == 'sk_data_ready' and 'family=2' not in rest:
            continue
        entry = {'ts': ts, 'event': event, 'raw': line.strip()}
        if event == 'irq_handler_entry':
            irq_match = IRQ_NUMBER_LINE.search(rest)
            entry['irq'] = int(irq_match.group(1)) if irq_match else None
        events.append(entry)
    events.sort(key=lambda e: e['ts'])
    return events


def receive_path_events_in_window(events, window_start_s, window_end_s, margin_s=0.0005):
    """LAST occurrence of each event kind within [window_start_s - margin_s,
    window_end_s + margin_s] -- i.e. the receive-path cycle closest to
    when the window actually ended (closest to the recv() call's actual
    return). Deliberately not the first occurrence: this device's
    irq_handler_entry/softirq/napi/wakeup events fire in a continuous
    background cadence roughly every 60-90us regardless of real TCP
    traffic (Phase 26 finding -- this is the thunderbolt-net driver's own
    RX ring polling substrate), so "first in window" would pick up an
    unrelated earlier cycle rather than the one that actually gated this
    iteration's data delivery. The margin allows a little slack since the
    window boundary (from the application trace) and these kernel-event
    timestamps come from different measurement points not expected to
    align to the microsecond. In practice this fallback path is rarely
    reached: every slow event in this phase's live capture was resolved
    by the irq-gap check in classify_receive_path_event before reaching
    here (see receive_path_report)."""
    lo = window_start_s - margin_s
    hi = window_end_s + margin_s
    by_kind = {}
    for event in events:
        if lo <= event['ts'] <= hi:
            by_kind[event['event']] = event['ts']
    return by_kind


def irq_cadence_gaps(events, event_kind='irq_handler_entry'):
    """Consecutive-timestamp gaps (seconds, sorted) for one event kind --
    the basis for irq_silence_overlap(). Discovered in Phase 26 live
    capture: this driver's irq_handler_entry (and the softirq/napi/skb
    events that follow each one within a few microseconds) fires in a
    tight, continuous ~60-90us cadence throughout the run, REGARDLESS of
    whether a given cycle carries real TBCCL TCP payload -- i.e. this is
    the thunderbolt-net driver's own RX ring polling substrate, not a
    per-TCP-segment signal. A slow iteration's window reliably overlaps a
    single anomalous ~1ms gap in this cadence (9/9 events, this phase's
    live run) -- a far stronger and more direct signal than checking
    "first event after window_start" (which always finds a normal-cadence
    event just before window_start regardless of what happens later
    inside the window, since the cadence is continuous)."""
    ts = sorted(e['ts'] for e in events if e['event'] == event_kind)
    return [(ts[i], ts[i + 1], (ts[i + 1] - ts[i]) * 1e6) for i in range(len(ts) - 1)]


def irq_cadence_baseline(gaps, anomaly_threshold_us=500):
    """Same-run normal irq-to-irq cadence stats (Part Q), excluding gaps
    already >= anomaly_threshold_us so a real silence event doesn't
    inflate its own baseline."""
    normal = [g[2] for g in gaps if g[2] < anomaly_threshold_us]
    if not normal:
        return {'median_us': None, 'p95_us': None, 'max_us': None, 'n': 0}
    return {'median_us': statistics.median(normal), 'p95_us': percentile(normal, .95),
            'max_us': max(normal), 'n': len(normal)}


def irq_silence_overlap(window_start_s, window_end_s, gaps, anomaly_threshold_us=500):
    """The largest cadence gap (if any) that overlaps [window_start_s,
    window_end_s], among gaps >= anomaly_threshold_us. Returns None if no
    anomalous gap overlaps the window (i.e. the irq cadence continued
    normally throughout)."""
    overlapping = [g for g in gaps
                   if g[2] >= anomaly_threshold_us and g[0] <= window_end_s and g[1] >= window_start_s]
    if not overlapping:
        return None
    return max(overlapping, key=lambda g: g[2])


def classify_receive_path_event(
        window_start_s, window_end_s, receive_events, irq_gaps=None,
        baseline=None, prompt_threshold_us=150, irq_anomaly_threshold_us=500):
    """Part R items 66-72's decision tree for one slow iteration.

    Primary signal (Part R item 66, Part H classification A): does an
    anomalous gap in the continuous irq_handler_entry cadence overlap this
    iteration's window? If so, the delay is upstream of all traced
    receiver network processing -- softirq/NAPI/socket-wakeup are never
    even reached during the gap, so checking their timing is moot. Only
    when no such gap overlaps the window does the finer irq->softirq->
    skb->wakeup decomposition (Part R items 67-71) apply, using the FIRST
    receive-path event cycle actually inside the window (not "nearest to
    window_start", since the cadence is continuous and unrelated cycles
    would otherwise be picked up -- see irq_cadence_gaps' docstring).

    window_start_s/window_end_s must be in the same clock domain as
    receive_events and irq_gaps (this receiving host's own mono/
    CLOCK_MONOTONIC timeline)."""
    detail = {'irq_gap_us': None, 'irq_us': None, 'softirq_us': None,
              'skb_us': None, 'wakeup_us': None}

    if irq_gaps is not None:
        overlap = irq_silence_overlap(window_start_s, window_end_s, irq_gaps,
                                       irq_anomaly_threshold_us)
        if overlap is not None:
            detail['irq_gap_us'] = overlap[2]
            return 'delayed before receiver kernel-observable ingress', detail

    irq_ts = receive_events.get('irq_handler_entry')
    softirq_ts = receive_events.get('softirq_entry')
    skb_ts = (receive_events.get('napi_gro_receive_entry')
              or receive_events.get('netif_receive_skb_entry'))
    wakeup_ts = receive_events.get('sk_data_ready')

    def prompt(value_us, key):
        if baseline and baseline.get(key, {}).get('p95_us') is not None:
            return value_us <= max(baseline[key]['p95_us'] * 2, prompt_threshold_us)
        return value_us <= prompt_threshold_us

    if irq_ts is None:
        return 'delayed before receiver kernel-observable ingress', detail
    detail['irq_us'] = (irq_ts - window_start_s) * 1e6

    if softirq_ts is None:
        return 'insufficient evidence', detail
    detail['softirq_us'] = (softirq_ts - irq_ts) * 1e6
    if not prompt(detail['softirq_us'], 'softirq_us'):
        return 'delayed softirq scheduling', detail

    if skb_ts is None:
        return 'insufficient evidence', detail
    detail['skb_us'] = (skb_ts - softirq_ts) * 1e6
    if not prompt(detail['skb_us'], 'skb_us'):
        return 'prolonged NAPI processing', detail

    if wakeup_ts is None:
        return 'insufficient evidence', detail
    detail['wakeup_us'] = (wakeup_ts - skb_ts) * 1e6
    if not prompt(detail['wakeup_us'], 'wakeup_us'):
        return 'delayed socket delivery', detail

    return 'receiver path normal / delay upstream', detail


def receive_path_baseline(events, normal_windows, device='thunderbolt0'):
    """Part Q: same-run normal irq/softirq/skb/wakeup sub-interval
    distributions, computed from normal_windows -- a list of
    (window_start_s, window_end_s) pairs for NON-slow iterations on this
    same receiving host in this same run. Never uses another phase's
    historical numbers as the reference (item 65)."""
    irq_us, softirq_us, skb_us, wakeup_us = [], [], [], []
    for start, end in normal_windows:
        found = receive_path_events_in_window(events, start, end)
        irq_ts = found.get('irq_handler_entry')
        softirq_ts = found.get('softirq_entry')
        skb_ts = found.get('napi_gro_receive_entry') or found.get('netif_receive_skb_entry')
        wakeup_ts = found.get('sk_data_ready')
        if irq_ts is not None:
            irq_us.append((irq_ts - start) * 1e6)
        if irq_ts is not None and softirq_ts is not None:
            softirq_us.append((softirq_ts - irq_ts) * 1e6)
        if softirq_ts is not None and skb_ts is not None:
            skb_us.append((skb_ts - softirq_ts) * 1e6)
        if skb_ts is not None and wakeup_ts is not None:
            wakeup_us.append((wakeup_ts - skb_ts) * 1e6)

    def stats(values):
        if not values:
            return {'median_us': None, 'p95_us': None, 'max_us': None, 'n': 0}
        return {'median_us': statistics.median(values), 'p95_us': percentile(values, .95),
                'max_us': max(values), 'n': len(values)}

    return {'irq_us': stats(irq_us), 'softirq_us': stats(softirq_us),
            'skb_us': stats(skb_us), 'wakeup_us': stats(wakeup_us)}


def receive_path_report(app_trace, sched_result, slow_threshold_us=1000, normal_threshold_us=800):
    """Part P: per-slow-iteration receive-path classification for one
    receiving host, from its own application trace (recv_begin_ns/
    recv_end_ns bracket the blocking recv() call -- exactly the window a
    slow iteration's delay must fall within) and a
    capture_tb4_scheduler_trace.py --include-receive-events result.
    Same-run baseline (Part Q) is computed from this run's own
    recv_us < normal_threshold_us iterations, never from another phase's
    historical numbers (item 65)."""
    events = parse_receive_path_trace((sched_result or {}).get('trace_text') or '')
    slow_windows, normal_windows = [], []
    for entry in app_trace['entries']:
        if entry.get('recv_begin_ns') is None or entry.get('recv_end_ns') is None:
            continue
        recv_us = (entry['recv_end_ns'] - entry['recv_begin_ns']) / 1000.0
        window = (entry['iteration_index'], entry['recv_begin_ns'] / 1e9,
                  entry['recv_end_ns'] / 1e9, recv_us)
        if recv_us >= slow_threshold_us:
            slow_windows.append(window)
        elif recv_us < normal_threshold_us:
            normal_windows.append(window)

    baseline = receive_path_baseline(events, [(s, e) for _, s, e, _ in normal_windows])
    irq_gaps = irq_cadence_gaps(events)
    irq_baseline = irq_cadence_baseline(irq_gaps)

    slow_events = []
    for index, start_s, end_s, recv_us in slow_windows:
        found = receive_path_events_in_window(events, start_s, end_s)
        classification, detail = classify_receive_path_event(
            start_s, end_s, found, irq_gaps, baseline)
        slow_events.append({
            'iteration': index, 'recv_us': recv_us, 'classification': classification,
            **detail,
        })

    from collections import Counter
    return {
        'slow_events': slow_events,
        'classification_counts': dict(Counter(e['classification'] for e in slow_events)),
        'same_run_baseline': baseline,
        'irq_cadence_baseline': irq_baseline,
        'normal_iteration_count': len(normal_windows),
    }


# Phase 27 Part E/F/G/H: separate the aggregate irq_handler_entry cadence
# (Phase 26 treated all `name=thunderbolt` IRQs as one stream) by MSI-X
# vector (irq number), test whether each vector's own cadence is closer to
# the driver's configured interrupt-moderation constant, and correlate
# Thunderbolt control-plane (tb_tx/tb_rx/tb_event) and USB4NET data-plane
# (tbnet_tx_ip_frame/tbnet_rx_ip_frame/tbnet_tx_skb/tbnet_rx_skb) events
# against the silence windows. thunderbolt:tb_tx/tb_rx/tb_event are
# confirmed (via their own format files, Part G item 32-34) to reference
# TB_CFG_PKG_* symbols (READ/WRITE/ERROR/NOTIFY_ACK/EVENT/XDOMAIN_REQ/
# XDOMAIN_RESP/OVERRIDE/RESET/ICM_EVENT/ICM_CMD/ICM_RESP on this kernel) --
# Thunderbolt router configuration-space and ICM-firmware control traffic,
# never USB4NET IP payload framing. tbnet_tx_ip_frame/tbnet_rx_ip_frame
# carry id/size/index/count (a large IP packet's fragment sequence within
# one ThunderboltIP frame) -- genuine data-plane granularity.
THUNDERBOLT_CONTROL_LINE = re.compile(
    r'(?P<ts>\d+\.\d+):\s*(?P<event>tb_tx|tb_rx|tb_event):\s*(?P<rest>.*)')
TBNET_DATA_LINE = re.compile(
    r'(?P<ts>\d+\.\d+):\s*(?P<event>tbnet_tx_skb|tbnet_tx_ip_frame|'
    r'tbnet_rx_ip_frame|tbnet_rx_skb):\s*(?P<rest>.*)')
TBNET_FIELD = {
    'id': re.compile(r'\bid=(\d+)'), 'index': re.compile(r'\bindex=(\d+)'),
    'count': re.compile(r'\bcount=(\d+)'), 'size': re.compile(r'\bsize=(\d+)'),
    'len': re.compile(r'\blen=(\d+)'),
}


def parse_thunderbolt_control_trace(trace_text):
    """Flat, time-sorted tb_tx/tb_rx/tb_event control-plane events."""
    events = []
    for line in (trace_text or '').splitlines():
        match = THUNDERBOLT_CONTROL_LINE.search(line)
        if not match:
            continue
        events.append({'ts': float(match.group('ts')), 'event': match.group('event'),
                       'raw': line.strip()})
    events.sort(key=lambda e: e['ts'])
    return events


def parse_tbnet_data_trace(trace_text):
    """Flat, time-sorted tbnet_*_skb/tbnet_*_ip_frame data-plane events,
    with id/index/count/size/len parsed where present (never fabricated
    for events that don't carry a given field)."""
    events = []
    for line in (trace_text or '').splitlines():
        match = TBNET_DATA_LINE.search(line)
        if not match:
            continue
        entry = {'ts': float(match.group('ts')), 'event': match.group('event'),
                  'raw': line.strip()}
        rest = match.group('rest')
        for field, pattern in TBNET_FIELD.items():
            field_match = pattern.search(rest)
            if field_match:
                entry[field] = int(field_match.group(1))
        events.append(entry)
    events.sort(key=lambda e: e['ts'])
    return events


def per_vector_irq_gaps(events):
    """irq_cadence_gaps(), computed separately per MSI-X vector (irq
    number) instead of aggregated across all `name=thunderbolt` IRQs --
    Phase 27's primary extension of Phase 26's single-stream analysis
    (Part E item 21: 'do not aggregate all name=thunderbolt events into
    one sequence'). Returns {irq_number: [(start, end, gap_us), ...]}."""
    by_irq = {}
    for event in events:
        if event['event'] == 'irq_handler_entry' and event.get('irq') is not None:
            by_irq.setdefault(event['irq'], []).append(event['ts'])
    result = {}
    for irq, timestamps in by_irq.items():
        timestamps.sort()
        result[irq] = [(timestamps[i], timestamps[i + 1],
                        (timestamps[i + 1] - timestamps[i]) * 1e6)
                       for i in range(len(timestamps) - 1)]
    return result


def per_vector_cadence_baseline(per_vector_gaps, anomaly_threshold_us=500):
    """irq_cadence_baseline() applied independently to each vector's own
    gap list -- tests Part F's 128us-per-vector hypothesis without
    assuming it (item 28: 'do not fit the result to 128us by assumption')."""
    return {irq: irq_cadence_baseline(gaps, anomaly_threshold_us)
            for irq, gaps in per_vector_gaps.items()}


def simultaneous_vector_silence(window_start_s, window_end_s, per_vector_gaps,
                                 anomaly_threshold_us=500):
    """For every vector, the largest anomalous gap (if any) overlapping
    the window -- lets the caller determine whether multiple vectors go
    silent together (Part F item 29-30) or only one does (Part S outcomes
    D/E). Returns {irq: gap_us_or_None}."""
    result = {}
    for irq, gaps in per_vector_gaps.items():
        overlap = irq_silence_overlap(window_start_s, window_end_s, gaps, anomaly_threshold_us)
        result[irq] = overlap[2] if overlap else None
    return result


def events_near_window(events, window_start_s, window_end_s, before_us=500, after_us=500):
    """Every event (from parse_thunderbolt_control_trace or
    parse_tbnet_data_trace) within [window_start_s - before_us,
    window_end_s + after_us], each tagged with its position relative to
    the window (Part K item 56): 'before' (strictly before window_start,
    within before_us), 'inside' (within the window itself), or 'after'
    (strictly after window_end, within after_us)."""
    lo = window_start_s - before_us / 1e6
    hi = window_end_s + after_us / 1e6
    tagged = []
    for event in events:
        if not (lo <= event['ts'] <= hi):
            continue
        if event['ts'] < window_start_s:
            position = 'before'
        elif event['ts'] > window_end_s:
            position = 'after'
        else:
            position = 'inside'
        tagged.append({**event, 'position': position})
    return tagged


def nhi_cadence_report(app_trace, sched_result, slow_threshold_us=1000,
                        normal_threshold_us=800, control_window_us=500):
    """Part Q: ties per-vector cadence, the 128us hypothesis, and
    control/data-plane correlation together for every slow iteration.
    Requires a sched_result captured with --include-receive-events
    --include-thunderbolt-events (Phase 27 Part G item 36)."""
    trace_text = (sched_result or {}).get('trace_text') or ''
    events = parse_receive_path_trace(trace_text)
    control_events = parse_thunderbolt_control_trace(trace_text)
    tbnet_events = parse_tbnet_data_trace(trace_text)

    per_vector_gaps = per_vector_irq_gaps(events)
    vector_baseline = per_vector_cadence_baseline(per_vector_gaps)

    slow_windows, normal_count = [], 0
    for entry in app_trace['entries']:
        if entry.get('recv_begin_ns') is None or entry.get('recv_end_ns') is None:
            continue
        recv_us = (entry['recv_end_ns'] - entry['recv_begin_ns']) / 1000.0
        if recv_us >= slow_threshold_us:
            slow_windows.append((entry['iteration_index'], entry['recv_begin_ns'] / 1e9,
                                 entry['recv_end_ns'] / 1e9, recv_us))
        elif recv_us < normal_threshold_us:
            normal_count += 1

    slow_events = []
    for index, start_s, end_s, recv_us in slow_windows:
        vector_gaps = simultaneous_vector_silence(start_s, end_s, per_vector_gaps)
        silenced = {irq: gap for irq, gap in vector_gaps.items() if gap is not None}
        both_vectors_silent = len(silenced) >= 2 if len(per_vector_gaps) >= 2 else None
        nearby_control = events_near_window(control_events, start_s, end_s, control_window_us,
                                             control_window_us)
        nearby_data = events_near_window(tbnet_events, start_s, end_s, control_window_us,
                                          control_window_us)
        ratios = {}
        for irq, gap_us in silenced.items():
            median = vector_baseline.get(irq, {}).get('median_us')
            ratios[irq] = {
                'gap_us': gap_us,
                'ratio_to_vector_median': (gap_us / median) if median else None,
                'ratio_to_128us_reference': gap_us / 128.0,
            }
        slow_events.append({
            'iteration': index, 'recv_us': recv_us,
            'vector_gaps_us': vector_gaps,
            'both_vectors_silent': both_vectors_silent,
            'ratios': ratios,
            'control_events_before': sum(1 for e in nearby_control if e['position'] == 'before'),
            'control_events_inside': sum(1 for e in nearby_control if e['position'] == 'inside'),
            'control_events_after': sum(1 for e in nearby_control if e['position'] == 'after'),
            'data_events_before': sum(1 for e in nearby_data if e['position'] == 'before'),
            'data_events_inside': sum(1 for e in nearby_data if e['position'] == 'inside'),
            'data_events_after': sum(1 for e in nearby_data if e['position'] == 'after'),
        })

    return {
        'per_vector_baseline': vector_baseline,
        'slow_events': slow_events,
        'total_control_events': len(control_events),
        'total_data_events': len(tbnet_events),
        'normal_iteration_count': normal_count,
    }


# Phase 29 Part F/G/K/L: parse Tier-1 narrow function-trace output
# (capture_tb4_scheduler_trace.py --include-nhi-functions) and use it to
# (a) resolve TX/RX vector role by call-site (ring_msix on a TX-ring IRQ
# leads to ring_work/process_one_work in a kworker thread; ring_msix on an
# RX-ring IRQ leads to tb_ring_poll/tb_ring_poll_complete inside
# tbnet_poll's NAPI softirq context -- confirmed from the exact v6.18.34
# source, Phase 28/29's provenance: tb_ring_alloc_tx() always passes
# start_poll=NULL, so TX servicing is unconditionally workqueue-based
# while RX is unconditionally NAPI-based) and (b) count tb_ring_poll
# "bursts" (consecutive calls with no large gap) as the safe, source-
# grounded way to run the plan's completion-accumulation test (Part L)
# without any dynamic probe: tbnet_poll() calls tb_ring_poll() once per
# already-hardware-completed descriptor in a tight loop, so N consecutive
# calls in one burst means N descriptors were already completed when that
# NAPI cycle started servicing the ring.
FTRACE_FUNCTION_LINE = re.compile(
    r'(?P<task>\S+)-\d+\s+\[(?P<cpu>\d+)\].*?(?P<ts>\d+\.\d+):\s*'
    r'(?P<func>\w+)\s*<-(?P<caller>\S+)')


def parse_nhi_function_trace(trace_text, functions=('ring_msix', 'ring_work',
                              'tb_ring_poll', 'tb_ring_poll_complete')):
    """Flat, time-sorted list of {ts, cpu, func, caller} for the given
    function names, extracted from raw function-tracer output."""
    wanted = set(functions)
    events = []
    for line in (trace_text or '').splitlines():
        if not line or line.startswith('#'):
            continue
        match = FTRACE_FUNCTION_LINE.search(line)
        if not match or match.group('func') not in wanted:
            continue
        events.append({
            'ts': float(match.group('ts')), 'cpu': int(match.group('cpu')),
            'func': match.group('func'), 'caller': match.group('caller'),
        })
    events.sort(key=lambda e: e['ts'])
    return events


def resolve_cpu_roles(function_events):
    """Maps each CPU to 'tx' or 'rx' by which follow-on function runs on
    it after ring_msix: ring_work (TX, always workqueue-based on this
    kernel -- Part E: tb_ring_alloc_tx() always passes start_poll=NULL) or
    tb_ring_poll (RX, always NAPI-based). Returns {cpu: 'tx'|'rx'}; a CPU
    that shows neither is simply absent, never guessed."""
    cpu_role = {}
    for event in function_events:
        if event['func'] == 'ring_work':
            cpu_role.setdefault(event['cpu'], 'tx')
        elif event['func'] == 'tb_ring_poll':
            cpu_role.setdefault(event['cpu'], 'rx')
    return cpu_role


def vector_roles_from_affinity(cpu_role_by_cpu, cpu_by_irq):
    """Combines resolve_cpu_roles()'s {cpu: 'tx'|'rx'} with a caller-
    supplied {irq_number: cpu} affinity mapping (from /proc/interrupts,
    Part E item 19 -- read-only, never changed) to produce the final
    {irq_number: 'tx'|'rx'|'unknown'}."""
    return {irq: cpu_role_by_cpu.get(cpu, 'unknown') for irq, cpu in cpu_by_irq.items()}


def tb_ring_poll_bursts(function_events, max_gap_us=50):
    """Groups consecutive tb_ring_poll calls into bursts (a new burst
    starts when the gap since the previous tb_ring_poll exceeds
    max_gap_us). Each burst's size is the number of already-hardware-
    completed descriptors tbnet_poll() drained in that NAPI cycle (Part L
    item 58-59) -- the safe, tracepoint-only completion-accumulation
    signal, since no independent/periodic hardware-completion sampling
    exists in this driver (Part E's architectural finding: completion
    detection is 100% interrupt-gated, no polling thread). Returns a list
    of {start_ts, end_ts, count}."""
    polls = sorted(e['ts'] for e in function_events if e['func'] == 'tb_ring_poll')
    bursts = []
    current = None
    for ts in polls:
        if current is None:
            current = {'start_ts': ts, 'end_ts': ts, 'count': 1}
        elif (ts - current['end_ts']) * 1e6 <= max_gap_us:
            current['end_ts'] = ts
            current['count'] += 1
        else:
            bursts.append(current)
            current = {'start_ts': ts, 'end_ts': ts, 'count': 1}
    if current is not None:
        bursts.append(current)
    return bursts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-dir', type=Path)
    group.add_argument('--historical-root', type=Path)
    group.add_argument('--correlate', action='store_true',
                       help='classify slow iterations from --application-trace, '
                            'optionally with --sched-trace-json / --packet-pcap')
    group.add_argument('--boundary', action='store_true',
                       help='Phase 23 Part F: dual-end completion-ack-to-'
                            'next-payload local interval distributions from '
                            '--linux-pcap and/or --mac-pcap (each analyzed '
                            'using only its own capture clock)')
    group.add_argument('--trigger-summary', type=Path,
                       help='Phase 24 Part N: condition-level and '
                            'position-in-burst summary of a '
                            'run_tb4_tail_trigger_sweep.py output JSON file')
    group.add_argument('--sender-boundary', action='store_true',
                       help='Phase 25 Part G/M: Mac sender-boundary '
                            'decomposition and classification of slow '
                            'Linux iterations, from --application-trace '
                            '(Linux), --mac-application-trace, --mac-pcap')
    group.add_argument('--receive-path', action='store_true',
                       help='Phase 26 Part P: Linux receive-path (irq -> '
                            'softirq -> napi/skb -> socket-wakeup) '
                            'classification of slow iterations, from '
                            '--application-trace and --sched-trace-json '
                            '(captured with --include-receive-events)')
    group.add_argument('--nhi-cadence', action='store_true',
                       help='Phase 27 Part Q: per-MSI-X-vector IRQ '
                            'cadence, 128us-hypothesis ratios, and '
                            'Thunderbolt control/data-plane correlation, '
                            'from --application-trace and '
                            '--sched-trace-json (captured with '
                            '--include-receive-events '
                            '--include-thunderbolt-events)')
    parser.add_argument('--application-trace', type=Path,
                        help='a rank stderr file containing a TBCCL_DIAGNOSTIC '
                             'transfer_trace line (--correlate/--sender-boundary mode)')
    parser.add_argument('--mac-application-trace', type=Path,
                        help='--sender-boundary mode: the Mac source rank\'s '
                             'stderr file')
    parser.add_argument('--sched-trace-json', type=Path,
                        help='output of capture_tb4_scheduler_trace.py (--correlate mode)')
    parser.add_argument('--packet-pcap', type=Path,
                        help='tcpdump/tshark-readable capture (--correlate mode)')
    parser.add_argument('--linux-pcap', type=Path, help='--boundary mode')
    parser.add_argument('--mac-pcap', type=Path, help='--boundary mode')
    parser.add_argument('--payload-min-length', type=int, default=1000,
                        help='--boundary mode: minimum TCP payload length '
                             'classified as tensor data rather than control '
                             'traffic (default 1000; observed real tensor '
                             'segments are ~2900/62636 bytes, observed '
                             'control/handshake messages are 24/40 bytes)')
    parser.add_argument('--expected-count', type=int, default=None,
                        help='--boundary mode: total warmup+measured '
                             'iterations, to exclude the untimed post-loop '
                             'verification round from the tail')
    parser.add_argument('--slow-threshold-us', type=float, default=500)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.correlate:
        if not args.application_trace:
            parser.error('--correlate requires --application-trace')
        app_trace = parse_diagnostic_file(args.application_trace)
        sched_result = load(args.sched_trace_json) if args.sched_trace_json else None
        result = {'slow_iterations': classify_slow_iterations(
            app_trace, sched_result, args.packet_pcap, args.slow_threshold_us)}
    elif args.boundary:
        if not args.linux_pcap and not args.mac_pcap:
            parser.error('--boundary requires --linux-pcap and/or --mac-pcap')
        result = {}
        for label, pcap_path in (('linux', args.linux_pcap), ('mac', args.mac_pcap)):
            if pcap_path is None:
                continue
            intervals, unmatched, extra = ack_to_payload_intervals(
                run_tcpdump(pcap_path), args.payload_min_length,
                expected_count=args.expected_count)
            gaps_us = [i['gap_us'] for i in intervals]
            result[f'{label}_ack_to_payload'] = {
                'samples': len(gaps_us),
                'median_us': statistics.median(gaps_us) if gaps_us else None,
                'p95_us': percentile(gaps_us, .95) if gaps_us else None,
                'max_us': max(gaps_us) if gaps_us else None,
                'unmatched_ack_count': len(unmatched),
                'excluded_beyond_expected_count': len(extra),
                'intervals': intervals,
            }
    elif args.trigger_summary:
        sweep_output = load(args.trigger_summary)
        result = {
            'by_condition': summarize_by_condition(sweep_output),
            'by_position': summarize_by_position(sweep_output),
        }
    elif args.sender_boundary:
        if not args.application_trace or not args.mac_application_trace:
            parser.error('--sender-boundary requires --application-trace '
                          '(Linux) and --mac-application-trace')
        linux_trace = parse_diagnostic_file(args.application_trace)
        mac_trace = parse_diagnostic_file(args.mac_application_trace)
        mac_packets = run_tcpdump(args.mac_pcap) if args.mac_pcap else None
        result = sender_boundary_report(
            linux_trace, mac_trace, mac_packets, args.slow_threshold_us)
    elif args.receive_path:
        if not args.application_trace or not args.sched_trace_json:
            parser.error('--receive-path requires --application-trace and '
                          '--sched-trace-json')
        app_trace = parse_diagnostic_file(args.application_trace)
        sched_result = load(args.sched_trace_json)
        result = receive_path_report(app_trace, sched_result, args.slow_threshold_us)
    elif args.nhi_cadence:
        if not args.application_trace or not args.sched_trace_json:
            parser.error('--nhi-cadence requires --application-trace and '
                          '--sched-trace-json')
        app_trace = parse_diagnostic_file(args.application_trace)
        sched_result = load(args.sched_trace_json)
        result = nhi_cadence_report(app_trace, sched_result, args.slow_threshold_us)
    else:
        result = analyze_run(args.run_dir) if args.run_dir else {
            'incidents': historical_incidents(args.historical_root)}
    text = json.dumps(result, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
