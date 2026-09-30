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
    """Build snapshot-association records from preserved the TB4 latency work artifacts."""
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


# Correlate a Linux receiver's application trace with a bounded scheduler
# trace (from capture_tb4_scheduler_trace.py) and, optionally, a packet
# capture, to classify slow iterations per item 70. The scheduler trace's
# tracefs events use trace_clock=mono, which the capture script sets to be
# identical (same clock, units, epoch) to the application trace's
# CLOCK_MONOTONIC -- no calibration needed between those two. A packet
# capture's timestamps are in CLOCK_REALTIME, so correlating against it
# requires the realtime<->monotonic offset the caller derives from the
# application trace's own bracketing clock samples
# (clock_sample_monotonic_ns / clock_sample_realtime_ns).

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


# Classify packets in a capture (application-level
# completion ACK vs. plain TCP ACK vs. tensor payload) and compute, from
# ONE machine's own local capture clock only, the interval between a prior
# completion ACK and the next tensor payload -- never comparing timestamps
# from two different machines' captures directly.
#
# Classification is by observed payload length only (confirmed against
# real captures): the benchmark's completion ACK is always a
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
    subtracting their timestamps) is the dual-end method.

    The final measured iteration's completion ack has no legitimate next
    payload (the run ends after it) -- without a bound, it would pair with
    whatever unrelated packet happens to appear later in the capture
    (observed in a real capture: a ~2ms "interval" that was actually the last
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
    measured loop) -- discovered when that round's payload
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
    """Per-iteration diagnostics + classification for
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


# Aggregate a run_tb4_tail_trigger_sweep.py output into a condition-level
# slow-event summary, and a position-in-burst ("first-N") breakdown. Kept
# here rather than in the sweep script itself so both the sweep runner and
# any future ad-hoc trigger data can be analyzed with the same code (the
# "avoid duplicating packet/trace parsing").

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


# Decompose the Mac sender boundary preceding a slow Linux iteration, and
# classify which sub-interval expands. Observed in captured
# data that two genuinely distinct mechanisms exist among slow events:
# (1) the completion-ack packet is visible on Mac's own bridge0 capture
# promptly, but Mac's TCP-level ack / application ack_received lags far
# behind it (implicates Mac-local processing); (2) nothing at all is
# observed on Mac's bridge0 until near the very end of the wait
# (implicates something upstream of Mac's local capture point -- Linux's
# own emission or network/TB4 transit). Distinguishing these does not
# require cross-host timestamps: both are computed purely from Mac's own
# local capture and Mac's own application trace, in Mac's own clock.

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
    # Mac's own capture.
    # send() is synchronous, so the payload is often already (mostly or
    # fully) visible on the wire microseconds BEFORE send_end is
    # timestamped, not strictly after -- confirmed empirically in
    # captured data (a payload's trailing segment observed ~1us
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
                # after Mac's own local packet observation (
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
    """Full per-slow-event table plus a same-run normal
    baseline (excluding candidate tails, >=800us, from the baseline itself
)."""
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
            # Same-run normal baseline, excluding candidate
            # tails.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run-dir', type=Path)
    group.add_argument('--historical-root', type=Path)
    group.add_argument('--correlate', action='store_true',
                       help='classify slow iterations from --application-trace, '
                            'optionally with --sched-trace-json / --packet-pcap')
    group.add_argument('--boundary', action='store_true',
                       help='dual-end completion-ack-to-'
                            'next-payload local interval distributions from '
                            '--linux-pcap and/or --mac-pcap (each analyzed '
                            'using only its own capture clock)')
    group.add_argument('--trigger-summary', type=Path,
                       help='condition-level and '
                            'position-in-burst summary of a '
                            'run_tb4_tail_trigger_sweep.py output JSON file')
    group.add_argument('--sender-boundary', action='store_true',
                       help='Mac sender-boundary '
                            'decomposition and classification of slow '
                            'Linux iterations, from --application-trace '
                            '(Linux), --mac-application-trace, --mac-pcap')
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
