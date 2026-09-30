#!/usr/bin/env python3
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_tb4_trace as analysis


def entry(index,rank,source,start):
    common=dict(run_id='run',iteration_index=index,local_rank=rank,source_rank=source,
                physical_machine='linux' if rank==1 else 'mac',direction='host->host',
                payload_bytes=65536,busy_poll_us=200,timing_scope='ready',source_gap_us=0,
                iteration_begin_ns=start,iteration_end_ns=start+2_000_000,
                source_ready_ns=None,staging_begin_ns=None,staging_end_ns=None,
                send_begin_ns=None,send_end_ns=None,ack_wait_begin_ns=None,ack_received_ns=None,
                recv_begin_ns=None,recv_end_ns=None,destination_staging_begin_ns=None,
                destination_staging_end_ns=None,destination_sync_end_ns=None,
                ack_send_begin_ns=None,ack_send_end_ns=None,observed_source_gap_ns=None)
    if rank==source:
        common.update(source_ready_ns=start+100_000,staging_begin_ns=start+100_000,
                      staging_end_ns=start+120_000,send_begin_ns=start+120_000,
                      send_end_ns=start+300_000,ack_wait_begin_ns=start+300_000,
                      ack_received_ns=start+2_000_000,observed_source_gap_ns=500_000)
    else:
        common.update(recv_begin_ns=start,recv_end_ns=start+1_500_000,
                      destination_staging_begin_ns=start+1_500_000,
                      destination_staging_end_ns=start+1_600_000,
                      destination_sync_end_ns=start+1_600_000,
                      ack_send_begin_ns=start+1_600_000,ack_send_end_ns=start+1_700_000,
                      iteration_end_ns=start+1_700_000)
    return common


def trace(rank,source=0,boot='boot'):
    return dict(kind='transfer_trace',clock='CLOCK_MONOTONIC',clock_units='nanoseconds',
                boot_id=boot,entries=[entry(0,rank,source,1_000_000),
                                      entry(1,rank,source,4_000_000),
                                      entry(2,rank,source,7_000_000)])


class AnalysisTests(unittest.TestCase):
    def test_tail_rows_are_local_and_clustered(self):
        result=analysis.analyze_pair(trace(0),trace(1),[400,1200,1300])
        self.assertEqual(result['summary']['over_1000_us'],2)
        self.assertEqual(result['summary']['slow_clusters_over_1000_us'],[[1,2]])
        self.assertEqual(result['rows'][0]['sender_send_us'],180)
        self.assertEqual(result['rows'][0]['receiver_recv_us'],1500)

    def test_iteration_identity_and_missing_stage(self):
        receiver=trace(1);receiver['entries'][1]['iteration_index']=9
        with self.assertRaisesRegex(ValueError,'identities'): analysis.analyze_pair(trace(0),receiver)
        source=trace(0);source['entries'][0]['staging_begin_ns']=None
        self.assertIsNone(analysis.analyze_pair(source,trace(1))['rows'][0]['source_staging_us'])

    def test_same_boot_inside_outside_and_provenance(self):
        before=dict(boot_id='boot',kernel_events=[dict(cursor='old')])
        after=copy.deepcopy(before)
        after['kernel_events'] += [
            dict(cursor='inside',monotonic_us='2000',source='thunderbolt',
                 message='pcieport 0000:00:1d.4: AER: Corrected error received: 0000:08:00.0'),
            dict(cursor='outside',monotonic_us='20000',source='other-pci',
                 message='pcieport 0000:00:1b.4: AER event'),
            dict(cursor='missing',source='thunderbolt',message='[12] Timeout')]
        result=analysis.correlate_kernel_events(trace(1),before,after)
        self.assertEqual([e['interval_relation'] for e in result['events']],
                         ['inside','after','unknown-missing-monotonic-timestamp'])
        self.assertEqual(result['events'][0]['reporting_pci_device'],'0000:00:1d.4')
        self.assertEqual(result['events'][0]['affected_pci_device'],'0000:08:00.0')
        self.assertEqual(result['events'][1]['source'],'other-pci')

    def test_reboot_and_boot_mismatch_rejected(self):
        before=dict(boot_id='old',kernel_events=[])
        after=dict(boot_id='new',kernel_events=[])
        with self.assertRaisesRegex(ValueError,'reboot'): analysis.correlate_kernel_events(trace(1),before,after)
        before['boot_id']=after['boot_id']='new'
        with self.assertRaisesRegex(ValueError,'boot IDs differ'):
            analysis.correlate_kernel_events(trace(1,boot='old'),before,after)


def sched_line(ts,kind,pid,comm='tbccl_tensor_tr',other_pid=999,state='S'):
    """Builds one synthetic ftrace text line matching this kernel's actual
    sched_switch/sched_wakeup format (verified against real captured
    output in the tail-latency investigation's manual analysis)."""
    if kind=='wakeup':
        return f'          <idle>-0       [000] dN.2. {ts:.6f}: sched_wakeup: comm={comm} pid={pid} prio=120 target_cpu=000'
    if kind=='sleep':
        return (f'{comm}-{pid}  [000] d..2. {ts:.6f}: sched_switch: prev_comm={comm} '
                f'prev_pid={pid} prev_prio=120 prev_state={state} ==> next_comm=swapper/0 next_pid=0 next_prio=120')
    if kind=='run':
        return (f'          <idle>-0       [000] d..2. {ts:.6f}: sched_switch: prev_comm=swapper/0 '
                f'prev_pid=0 prev_prio=120 prev_state=R ==> next_comm={comm} next_pid={pid} next_prio=120')
    raise ValueError(kind)


def sched_result(cycles,pid,error=None):
    """cycles: list of (sleep_ts, wakeup_ts, run_ts) in trace_clock=mono
    seconds (matches capture_tb4_scheduler_trace.py setting trace_clock to
    'mono', i.e. directly comparable to CLOCK_MONOTONIC)."""
    lines=['# tracer: nop','#']
    for sleep_ts,wakeup_ts,run_ts in cycles:
        lines.append(sched_line(sleep_ts,'sleep',pid))
        lines.append(sched_line(wakeup_ts,'wakeup',pid))
        lines.append(sched_line(run_ts,'run',pid))
    return dict(trace_text='\n'.join(lines),pid=pid,error=error,returncode=0)


def app_trace_for_classify(recv_pairs_ns,process_id=12345,
                            mono_sample_ns=0,realtime_sample_ns=1_700_000_000_000_000_000):
    entries=[]
    for index,(begin_ns,end_ns) in enumerate(recv_pairs_ns):
        entries.append(dict(iteration_index=index,recv_begin_ns=begin_ns,recv_end_ns=end_ns))
    return dict(kind='transfer_trace',clock='CLOCK_MONOTONIC',clock_units='nanoseconds',
                process_id=process_id,thread_id=process_id,
                clock_sample_monotonic_ns=mono_sample_ns,
                clock_sample_realtime_ns=realtime_sample_ns,entries=entries)


class ClassifyTests(unittest.TestCase):
    def test_sched_cycles_parses_wakeup_run_sleep(self):
        # A single sleep(1.000)->wakeup(2.300)->run(2.301) cycle for pid 42.
        result=sched_result([(1.000,2.300,2.301)],pid=42)
        cycles=analysis.sched_cycles_for_pid(result,42)
        self.assertEqual(cycles,[(1.000,2.300,2.301)])

    def test_sched_cycles_ignores_other_pids(self):
        result=sched_result([(1.000,2.300,2.301)],pid=42)
        self.assertEqual(analysis.sched_cycles_for_pid(result,7),[])

    def test_below_threshold_excluded(self):
        trace=app_trace_for_classify([(1_000_000_000,1_000_100_000)])  # 100us, below 500us default
        rows=analysis.classify_slow_iterations(trace,slow_threshold_us=500)
        self.assertEqual(rows,[])

    def test_late_packet_observation_classified_with_wire_overlap(self):
        # recv spans [1.000, 2.352]s mono (1352us, slow). The enclosing
        # cycle: sleeps at 1.050 (after recv_begin), woken+resumed at
        # 2.350/2.351 (both before recv_end) -- 1us wakeup-to-run (fast),
        # ~1300us sleep-to-wakeup (the actual cost). A pcap gap overlapping
        # [1.050, 2.350] in realtime should yield 'late local packet
        # observation'.
        begin_ns,end_ns=1_000_000_000,2_352_000_000
        trace=app_trace_for_classify([(begin_ns,end_ns)],
                                     mono_sample_ns=0,realtime_sample_ns=100_000_000_000)
        sched=sched_result([(1.050,2.350,2.350001)],pid=12345)
        realtime_offset_s=100.0
        gaps=[(1.050+realtime_offset_s-0.01,2.350+realtime_offset_s+0.01,1200.0)]
        rows=analysis.classify_slow_iterations(trace,sched,gaps=gaps)
        self.assertEqual(len(rows),1)
        row=rows[0]
        self.assertEqual(row['classification'],'late local packet observation')
        self.assertEqual(row['trace_quality'],'application + scheduler + packet')
        self.assertAlmostEqual(row['wakeup_to_run_us'],1.0,delta=1.0)
        self.assertIsNotNone(row['wire_silence_overlap_us'])

    def test_scheduler_wakeup_delay_classified_without_packet_check(self):
        # Same recv window, but wakeup-to-run is 500us (>100us threshold)
        # -- a genuine scheduler descheduling signature, classified before
        # any packet evidence is even consulted.
        begin_ns,end_ns=1_000_000_000,2_352_000_000
        trace=app_trace_for_classify([(begin_ns,end_ns)])
        sched=sched_result([(1.050,2.350,2.350500)],pid=12345)
        rows=analysis.classify_slow_iterations(trace,sched,gaps=[])
        self.assertEqual(rows[0]['classification'],'scheduler wakeup delay observed')

    def test_mixed_ambiguous_when_no_wire_gap_overlaps(self):
        begin_ns,end_ns=1_000_000_000,2_352_000_000
        trace=app_trace_for_classify([(begin_ns,end_ns)],
                                     mono_sample_ns=0,realtime_sample_ns=100_000_000_000)
        sched=sched_result([(1.050,2.350,2.350001)],pid=12345)
        # A gap far outside the [sleep,wakeup] realtime window -- no overlap.
        gaps=[(500.0,500.5,500.0)]
        rows=analysis.classify_slow_iterations(trace,sched,gaps=gaps)
        self.assertEqual(rows[0]['classification'],'mixed/ambiguous')

    def test_insufficient_evidence_without_scheduler_trace(self):
        begin_ns,end_ns=1_000_000_000,2_352_000_000
        trace=app_trace_for_classify([(begin_ns,end_ns)])
        rows=analysis.classify_slow_iterations(trace,sched_result=None)
        self.assertEqual(rows[0]['classification'],'insufficient evidence')
        self.assertEqual(rows[0]['trace_quality'],'application-only')

    def test_scheduler_trace_error_treated_as_unavailable(self):
        begin_ns,end_ns=1_000_000_000,2_352_000_000
        trace=app_trace_for_classify([(begin_ns,end_ns)])
        failed=dict(trace_text=None,pid=None,error='no requested tracefs events are supported')
        rows=analysis.classify_slow_iterations(trace,failed)
        self.assertEqual(rows[0]['classification'],'insufficient evidence')


def tcpdump_line(ts,src,dst,length,flags='P.'):
    """Builds one synthetic tcpdump -tt text line matching the
    real captured format closely enough for parse_tcpdump_text."""
    return (f'{ts:.6f} IP {src} > {dst}: Flags [{flags}], seq 1:2, ack 1, '
            f'win 100, options [nop,nop,TS val 1 ecr 1], length {length}')


class BoundaryTests(unittest.TestCase):
    def test_parse_tcpdump_text_extracts_fields(self):
        text=tcpdump_line(100.5,'mac.33400','linux.44914',65536)
        packets=analysis.parse_tcpdump_text(text)
        self.assertEqual(len(packets),1)
        self.assertEqual(packets[0]['ts'],100.5)
        self.assertEqual(packets[0]['length'],65536)

    def test_parse_tcpdump_text_ignores_unparseable_lines(self):
        text='not a packet line\n'+tcpdump_line(1.0,'a','b',0)
        self.assertEqual(len(analysis.parse_tcpdump_text(text)),1)

    def test_classify_packet_by_length(self):
        self.assertEqual(analysis.classify_packet(dict(length=0)),'tcp_ack_only')
        self.assertEqual(analysis.classify_packet(dict(length=1)),'application_ack')
        self.assertEqual(analysis.classify_packet(dict(length=2900)),'tensor_payload')
        self.assertEqual(analysis.classify_packet(dict(length=24)),'other')

    def _packets(self,*specs):
        return [dict(ts=ts,src='s',dst='d',flags='.',length=length) for ts,length in specs]

    def test_ack_to_payload_pairs_in_order(self):
        packets=self._packets((1.000,1),(1.000050,62636),(1.000300,1),(1.000400,62636))
        intervals,unmatched,extra=analysis.ack_to_payload_intervals(packets)
        self.assertEqual(len(intervals),2)
        self.assertAlmostEqual(intervals[0]['gap_us'],50.0,delta=0.1)
        self.assertAlmostEqual(intervals[1]['gap_us'],100.0,delta=0.1)
        self.assertEqual(unmatched,[])
        self.assertEqual(extra,[])

    def test_trailing_ack_without_payload_is_unmatched(self):
        # Matches the real macOS sender-silence shape: N
        # acks/payloads alternating, then one final trailing ack (e.g.
        # the destination's verify_ok) with no payload after it at all.
        packets=self._packets((1.000,1),(1.000050,62636),(1.000300,1))
        intervals,unmatched,extra=analysis.ack_to_payload_intervals(packets)
        self.assertEqual(len(intervals),1)
        self.assertEqual(unmatched,[1.000300])
        self.assertEqual(extra,[])

    def test_verification_round_excluded_via_expected_count(self):
        # Reproduces the real bug found earlier: the benchmark's
        # own untimed post-loop verification round (one more full tensor
        # resend after the measured loop) gets structurally paired with
        # the last measured iteration's ack, since ack/payload alternate
        # correctly in temporal order regardless of which round they
        # belong to. With 2 real iterations, only 1 meaningful transition
        # exists (iteration 0's ack -> iteration 1's payload); the 2nd
        # successful pairing found (iteration 1's ack -> the verification
        # round's payload) must be excluded via expected_count, not
        # silently treated as a real measurement.
        packets=self._packets(
            (1.000,1),(1.000050,62636),      # iteration 0 ack -> iter 1 payload
            (1.000300,1),(1.003300,62636))   # iteration 1 ack -> verification payload (3ms later)
        intervals,unmatched,extra=analysis.ack_to_payload_intervals(
            packets,expected_count=1)
        self.assertEqual(len(intervals),1)
        self.assertAlmostEqual(intervals[0]['gap_us'],50.0,delta=0.1)
        self.assertEqual(len(extra),1)
        self.assertAlmostEqual(extra[0]['gap_us'],3000.0,delta=1.0)

    def test_max_gap_excludes_implausible_pairing_without_expected_count(self):
        # Even without expected_count, an implausibly distant "payload"
        # (here, deliberately far beyond max_gap_us) is never paired --
        # it is left unmatched instead of inflating the distribution.
        packets=self._packets((1.000,1),(1.010,62636))  # 10ms gap
        intervals,unmatched,extra=analysis.ack_to_payload_intervals(
            packets,max_gap_us=5000)
        self.assertEqual(intervals,[])
        self.assertEqual(unmatched,[1.000])

    def test_dual_end_intervals_from_independent_captures_never_subtract_clocks(self):
        # Sanity check on the analysis contract: mac and linux boundary
        # results are computed from entirely separate packet lists, each
        # using only its own local timestamps.
        mac_packets=self._packets((10.0,1),(10.0001,62636))
        linux_packets=self._packets((500.0,1),(500.0003,62636))
        mac_intervals,_,_=analysis.ack_to_payload_intervals(mac_packets)
        linux_intervals,_,_=analysis.ack_to_payload_intervals(linux_packets)
        self.assertAlmostEqual(mac_intervals[0]['gap_us'],100.0,delta=0.1)
        self.assertAlmostEqual(linux_intervals[0]['gap_us'],300.0,delta=0.1)


if __name__=='__main__': unittest.main()
