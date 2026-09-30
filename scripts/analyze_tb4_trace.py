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
    parser.add_argument('--application-trace', type=Path,
                        help='a rank stderr file containing a TBCCL_DIAGNOSTIC '
                             'transfer_trace line (--correlate mode)')
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
