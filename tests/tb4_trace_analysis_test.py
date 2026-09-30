#!/usr/bin/env python3
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze_tb4_trace as analysis
import run_tb4_tail_trigger_sweep as sweep


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
    output in Phase 22's manual investigation)."""
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
    """Builds one synthetic tcpdump -tt text line matching this phase's
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
        # Matches the real Phase 23 shape: N acks/payloads alternating,
        # then one final trailing ack (e.g. the destination's verify_ok)
        # with no payload after it at all.
        packets=self._packets((1.000,1),(1.000050,62636),(1.000300,1))
        intervals,unmatched,extra=analysis.ack_to_payload_intervals(packets)
        self.assertEqual(len(intervals),1)
        self.assertEqual(unmatched,[1.000300])
        self.assertEqual(extra,[])

    def test_verification_round_excluded_via_expected_count(self):
        # Reproduces the real bug found in this phase: the benchmark's
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


def sweep_result(idle_ms,samples_us,success=True):
    return dict(requested_idle_ms=idle_ms,samples_us=samples_us,success=success)


class TriggerSummaryTests(unittest.TestCase):
    def test_summarize_by_condition_counts_thresholds(self):
        sweep=dict(results=[
            sweep_result(0,[100,600,900]),
            sweep_result(0,[200,1100]),
            sweep_result(500,[150,160,170]),
        ])
        rows=analysis.summarize_by_condition(sweep,thresholds=(500,800,1000))
        by_idle={r['requested_idle_ms']:r for r in rows}
        self.assertEqual(by_idle[0]['sessions'],2)
        self.assertEqual(by_idle[0]['iterations'],5)
        self.assertEqual(by_idle[0]['over_500_us'],3)
        self.assertEqual(by_idle[0]['over_800_us'],2)
        self.assertEqual(by_idle[0]['over_1000_us'],1)
        self.assertEqual(by_idle[500]['over_500_us'],0)
        self.assertEqual(by_idle[500]['median_us'],160)

    def test_summarize_by_condition_excludes_failed_sessions_from_iterations(self):
        # A failed burst (samples_us=None) is still counted as an attempted
        # session, but contributes zero iterations/events.
        sweep=dict(results=[
            sweep_result(0,[100,200]),
            sweep_result(0,None,success=False),
        ])
        rows=analysis.summarize_by_condition(sweep)
        row=rows[0]
        self.assertEqual(row['sessions'],2)
        self.assertEqual(row['iterations'],2)

    def test_summarize_by_condition_handles_all_sessions_empty(self):
        sweep=dict(results=[sweep_result(0,None,success=False)])
        rows=analysis.summarize_by_condition(sweep)
        self.assertEqual(rows[0]['iterations'],0)
        self.assertIsNone(rows[0]['median_us'])

    def test_summarize_by_position_first_n_breakdown(self):
        sweep=dict(results=[
            sweep_result(0,[500,300,300]),
            sweep_result(10,[560,320,310]),
        ])
        rows=analysis.summarize_by_position(sweep)
        by_pos={r['position']:r for r in rows}
        self.assertEqual(by_pos[0]['samples'],2)
        self.assertEqual(by_pos[0]['median_us'],530)
        self.assertEqual(by_pos[1]['median_us'],310)
        self.assertEqual(by_pos[2]['median_us'],305)

    def test_summarize_by_position_handles_uneven_burst_lengths(self):
        # A shorter burst simply contributes fewer positions; later
        # positions aggregate only over the sessions that reached them.
        sweep=dict(results=[
            sweep_result(0,[100,200,300]),
            sweep_result(0,[110,210]),
        ])
        rows=analysis.summarize_by_position(sweep)
        by_pos={r['position']:r for r in rows}
        self.assertEqual(by_pos[2]['samples'],1)
        self.assertEqual(by_pos[0]['samples'],2)


class TriggerSweepRunnerTests(unittest.TestCase):
    def test_classify_thresholds_counts_at_and_above_boundary(self):
        counts=sweep.classify_thresholds([100,499,500,799,800,999,1000,1500],
                                          thresholds=(500,800,1000))
        self.assertEqual(counts[500],6)  # 500,799,800,999,1000,1500 -- inclusive
        self.assertEqual(counts[800],4)  # 800,999,1000,1500
        self.assertEqual(counts[1000],2) # 1000,1500

    def test_classify_thresholds_empty_samples(self):
        counts=sweep.classify_thresholds(None)
        self.assertEqual(counts,{500:0,800:0,1000:0})

    def test_rotate_conditions_interleaves_not_grouped(self):
        order=sweep.rotate_conditions([0,10,100],repeats=3)
        self.assertEqual(len(order),9)
        # Not grouped: the first occurrence of each condition should all
        # appear before the second occurrence of any condition (Part F
        # item 25's explicit "rotate order" requirement).
        first_occurrence_positions=[order.index(c) for c in (0,10,100)]
        second_occurrence_positions=[order.index(c,3) for c in (0,10,100)]
        self.assertTrue(max(first_occurrence_positions)<min(second_occurrence_positions))


def mac_entry(index,ack_wait_begin,ack_received,iteration_begin,source_ready,
              send_begin,send_end):
    return dict(iteration_index=index,ack_wait_begin_ns=ack_wait_begin,
                ack_received_ns=ack_received,iteration_end_ns=ack_received,
                iteration_begin_ns=iteration_begin,source_ready_ns=source_ready,
                send_begin_ns=send_begin,send_end_ns=send_end)


def mac_packet(ts,application_ack=False,tensor_payload=False):
    length = 1 if application_ack else (2900 if tensor_payload else 0)
    return dict(ts=ts,src='a',dst='b',flags='.',length=length)


# All synthetic timestamps use whole seconds for iteration_index*10 + ns
# offsets, with offset_ns=0 (mono realtime == mac's own monotonic here),
# to keep the arithmetic easy to read.
BASE_S = 1000.0


class SenderBoundaryTests(unittest.TestCase):
    def entries_for(self,ack_wait_begin_s,ack_received_s,iter_begin_s,
                     source_ready_s,send_begin_s,send_end_s):
        # index 9 is "previous" (N-1), index 10 is "current" (N) -- the
        # slow Linux iteration under test is always index 10.
        previous=mac_entry(9,int(ack_wait_begin_s*1e9),int(ack_received_s*1e9),
                            0,0,0,0)
        current=mac_entry(10,0,0,int(iter_begin_s*1e9),int(source_ready_s*1e9),
                           int(send_begin_s*1e9),int(send_end_s*1e9))
        return {9:previous,10:current}

    def test_delayed_mac_application_ack_reception(self):
        # ack_wait elevated (1.2ms); the ack packet WAS visible on Mac's
        # capture near the start of the wait (1.0ms before ack_received)
        # -- Mac itself was slow to process an already-arrived packet.
        entries=self.entries_for(BASE_S,BASE_S+0.0012,BASE_S+0.0012,
                                  BASE_S+0.00121,BASE_S+0.00121,BASE_S+0.00122)
        packets=[mac_packet(BASE_S+0.0001,application_ack=True),
                 mac_packet(BASE_S+0.00122,tensor_payload=True)]
        row=analysis.classify_mac_sender_boundary(10,entries,packets,offset_ns=0)
        self.assertEqual(row['classification'],'delayed Mac application ACK reception')

    def test_late_completion_ack_arrival_no_packet_in_window(self):
        # ack_wait elevated; no application-ack packet observed anywhere
        # in the wait window at all.
        entries=self.entries_for(BASE_S,BASE_S+0.0012,BASE_S+0.0012,
                                  BASE_S+0.00121,BASE_S+0.00121,BASE_S+0.00122)
        packets=[mac_packet(BASE_S+0.00122,tensor_payload=True)]
        row=analysis.classify_mac_sender_boundary(10,entries,packets,offset_ns=0)
        self.assertEqual(row['classification'],'late completion ACK arrival at Mac')

    def test_late_completion_ack_arrival_packet_near_window_end(self):
        # ack_wait elevated; the ack packet only becomes visible right
        # before ack_received -- most of the wait elapsed BEFORE it was
        # even observable.
        entries=self.entries_for(BASE_S,BASE_S+0.0012,BASE_S+0.0012,
                                  BASE_S+0.00121,BASE_S+0.00121,BASE_S+0.00122)
        packets=[mac_packet(BASE_S+0.00119,application_ack=True),
                 mac_packet(BASE_S+0.00122,tensor_payload=True)]
        row=analysis.classify_mac_sender_boundary(10,entries,packets,offset_ns=0)
        self.assertEqual(row['classification'],'late completion ACK arrival at Mac')

    def test_delayed_source_preparation(self):
        # ack_wait normal; iteration_begin -> source_ready elevated.
        entries=self.entries_for(BASE_S,BASE_S+0.00015,BASE_S+0.00015,
                                  BASE_S+0.0015,BASE_S+0.0015,BASE_S+0.00151)
        packets=[mac_packet(BASE_S+0.0001,application_ack=True),
                 mac_packet(BASE_S+0.00151,tensor_payload=True)]
        row=analysis.classify_mac_sender_boundary(10,entries,packets,offset_ns=0)
        self.assertEqual(row['classification'],'delayed source preparation')

    def test_delayed_send_invocation(self):
        # ack_wait/iter_to_ready normal; source_ready -> send_begin elevated.
        entries=self.entries_for(BASE_S,BASE_S+0.00015,BASE_S+0.00015,
                                  BASE_S+0.00016,BASE_S+0.0016,BASE_S+0.00161)
        packets=[mac_packet(BASE_S+0.0001,application_ack=True),
                 mac_packet(BASE_S+0.00161,tensor_payload=True)]
        row=analysis.classify_mac_sender_boundary(10,entries,packets,offset_ns=0)
        self.assertEqual(row['classification'],'delayed send invocation')

    def test_blocking_send_syscall(self):
        # everything normal until send_begin -> send_end.
        entries=self.entries_for(BASE_S,BASE_S+0.00015,BASE_S+0.00015,
                                  BASE_S+0.00016,BASE_S+0.00017,BASE_S+0.0017)
        packets=[mac_packet(BASE_S+0.0001,application_ack=True),
                 mac_packet(BASE_S+0.0017,tensor_payload=True)]
        row=analysis.classify_mac_sender_boundary(10,entries,packets,offset_ns=0)
        self.assertEqual(row['classification'],'blocking send syscall')

    def test_delay_after_mac_local_packet_observation(self):
        # Every Mac-local interval normal, including a prompt post-send
        # emission -- by elimination (this function is only ever called
        # for a Linux iteration already known to be slow), the delay is
        # upstream of Mac's own local capture point.
        entries=self.entries_for(BASE_S,BASE_S+0.00015,BASE_S+0.00015,
                                  BASE_S+0.00016,BASE_S+0.00017,BASE_S+0.00019)
        packets=[mac_packet(BASE_S+0.0001,application_ack=True),
                 mac_packet(BASE_S+0.00019,tensor_payload=True)]  # at send_end
        row=analysis.classify_mac_sender_boundary(10,entries,packets,offset_ns=0)
        self.assertEqual(row['classification'],'delay after Mac local packet observation')

    def test_send_to_packet_matches_own_payload_not_next_iteration(self):
        # Regression test for the bug found in this phase: the payload
        # segment for the CURRENT iteration can appear on the capture
        # microseconds BEFORE send_end is timestamped (send() is
        # synchronous). Searching only for packets strictly after
        # send_end would skip past it and incorrectly match against the
        # NEXT iteration's payload instead, hugely inflating the measured
        # interval.
        send_end_s=BASE_S+0.00019
        entries=self.entries_for(BASE_S,BASE_S+0.00015,BASE_S+0.00015,
                                  BASE_S+0.00016,BASE_S+0.00017,send_end_s)
        packets=[mac_packet(BASE_S+0.0001,application_ack=True),
                 mac_packet(send_end_s-0.000001,tensor_payload=True),  # this iteration's own payload, just before send_end
                 mac_packet(send_end_s+0.0012,tensor_payload=True)]    # NEXT iteration's payload, much later
        row=analysis.classify_mac_sender_boundary(10,entries,packets,offset_ns=0)
        self.assertEqual(row['mac_send_to_packet_us'],0.0)
        self.assertEqual(row['classification'],'delay after Mac local packet observation')

    def test_insufficient_evidence_missing_adjacent_entries(self):
        row=analysis.classify_mac_sender_boundary(10,{},[],offset_ns=0)
        self.assertEqual(row['classification'],'insufficient evidence')

    def test_insufficient_evidence_without_packets(self):
        entries=self.entries_for(BASE_S,BASE_S+0.0012,BASE_S+0.0012,
                                  BASE_S+0.00121,BASE_S+0.00121,BASE_S+0.00122)
        row=analysis.classify_mac_sender_boundary(10,entries,None,offset_ns=0)
        self.assertEqual(row['classification'],'insufficient evidence')
        self.assertEqual(row['trace_quality'],'application-only')

    def test_sender_boundary_report_counts_and_excludes_tails_from_baseline(self):
        # One slow (>=1000us) iteration and one normal (<800us) iteration;
        # the slow one's own ack_wait must not pollute the normal baseline.
        linux_trace=dict(entries=[
            dict(iteration_index=9,recv_begin_ns=0,recv_end_ns=200_000),   # 200us, normal
            dict(iteration_index=10,recv_begin_ns=0,recv_end_ns=1_200_000),# 1200us, slow
        ])
        mac_trace=dict(
            clock_sample_monotonic_ns=0,clock_sample_realtime_ns=int(BASE_S*1e9),
            entries=[
                mac_entry(8,int((BASE_S-0.001)*1e9),int((BASE_S-0.0008)*1e9),
                          0,0,0,0),
                mac_entry(9,int((BASE_S-0.0008)*1e9),int((BASE_S-0.00065)*1e9),
                          int((BASE_S-0.00065)*1e9),int((BASE_S-0.00064)*1e9),
                          int((BASE_S-0.00063)*1e9),int((BASE_S-0.00061)*1e9)),
                mac_entry(10,int((BASE_S-0.00061)*1e9),int(BASE_S*1e9),
                          int(BASE_S*1e9),int((BASE_S+0.00001)*1e9),
                          int((BASE_S+0.00002)*1e9),int((BASE_S+0.00003)*1e9)),
            ])
        packets=[mac_packet(BASE_S-0.0007,application_ack=True),
                 mac_packet(BASE_S-0.00061,tensor_payload=True),
                 mac_packet(BASE_S-0.0001,application_ack=True),
                 mac_packet(BASE_S+0.00003,tensor_payload=True)]
        report=analysis.sender_boundary_report(linux_trace,mac_trace,packets)
        self.assertEqual(len(report['slow_events']),1)
        self.assertEqual(report['slow_events'][0]['iteration_index'],10)
        self.assertIn(sum(report['classification_counts'].values()),(1,))


def ftrace_receive_line(ts, event, rest):
    return f"          <idle>-0       [013] ..s1. {ts:.6f}: {event}: {rest}"


class ReceivePathParsingTests(unittest.TestCase):
    def test_parses_and_filters_thunderbolt_irq(self):
        text = "\n".join([
            ftrace_receive_line(1.0001, 'irq_handler_entry', 'irq=177 name=thunderbolt'),
            ftrace_receive_line(1.0002, 'irq_handler_entry', 'irq=42 name=i915'),
        ])
        events = analysis.parse_receive_path_trace(text)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['event'], 'irq_handler_entry')

    def test_filters_softirq_to_net_rx_only(self):
        text = "\n".join([
            ftrace_receive_line(1.0, 'softirq_entry', 'vec=3 [action=NET_RX]'),
            ftrace_receive_line(1.0001, 'softirq_entry', 'vec=9 [action=RCU]'),
        ])
        events = analysis.parse_receive_path_trace(text)
        self.assertEqual(len(events), 1)

    def test_filters_napi_poll_and_skb_by_device(self):
        text = "\n".join([
            ftrace_receive_line(1.0, 'napi_poll', 'napi poll on napi struct X for device thunderbolt0 work 1 budget 64'),
            ftrace_receive_line(1.0001, 'napi_poll', 'napi poll on napi struct Y for device enp6s0 work 1 budget 64'),
            ftrace_receive_line(1.0002, 'napi_gro_receive_entry', 'dev=thunderbolt0 len=84'),
            ftrace_receive_line(1.0003, 'napi_gro_receive_entry', 'dev=wlan0 len=46'),
        ])
        events = analysis.parse_receive_path_trace(text, device='thunderbolt0')
        self.assertEqual(len(events), 2)
        self.assertTrue(all('thunderbolt0' in e['raw'] for e in events))

    def test_filters_sk_data_ready_to_inet(self):
        text = "\n".join([
            ftrace_receive_line(1.0, 'sk_data_ready', 'family=2 protocol=1 func=sock_def_readable'),
            ftrace_receive_line(1.0001, 'sk_data_ready', 'family=1 protocol=0 func=unix_stream'),
        ])
        events = analysis.parse_receive_path_trace(text)
        self.assertEqual(len(events), 1)

    def test_events_returned_time_sorted(self):
        text = "\n".join([
            ftrace_receive_line(2.0, 'irq_handler_entry', 'irq=177 name=thunderbolt'),
            ftrace_receive_line(1.0, 'irq_handler_entry', 'irq=178 name=thunderbolt'),
        ])
        events = analysis.parse_receive_path_trace(text)
        self.assertEqual([e['ts'] for e in events], [1.0, 2.0])


class ReceivePathWindowTests(unittest.TestCase):
    def test_last_occurrence_per_kind_within_window(self):
        events = [
            {'ts': 1.0001, 'event': 'irq_handler_entry', 'raw': ''},
            {'ts': 1.0005, 'event': 'irq_handler_entry', 'raw': ''},  # later irq -- wins
            {'ts': 1.0002, 'event': 'softirq_entry', 'raw': ''},
            {'ts': 5.0, 'event': 'sk_data_ready', 'raw': ''},  # outside window
        ]
        found = analysis.receive_path_events_in_window(events, 1.0, 1.001)
        self.assertEqual(found['irq_handler_entry'], 1.0005)
        self.assertEqual(found['softirq_entry'], 1.0002)
        self.assertNotIn('sk_data_ready', found)


class IrqCadenceTests(unittest.TestCase):
    def test_gaps_computed_between_consecutive_irq_timestamps(self):
        events = [
            {'ts': 1.0000, 'event': 'irq_handler_entry', 'raw': ''},
            {'ts': 1.0001, 'event': 'irq_handler_entry', 'raw': ''},
            {'ts': 1.0100, 'event': 'irq_handler_entry', 'raw': ''},  # 9.9ms-scale big gap
            {'ts': 1.0102, 'event': 'softirq_entry', 'raw': ''},  # wrong kind -- ignored
        ]
        gaps = analysis.irq_cadence_gaps(events)
        self.assertEqual(len(gaps), 2)
        self.assertAlmostEqual(gaps[0][2], 100, delta=1)
        self.assertAlmostEqual(gaps[1][2], 9900, delta=1)

    def test_baseline_excludes_anomalous_gaps(self):
        gaps = [(0, 0, 60.0), (0, 0, 70.0), (0, 0, 80.0), (0, 0, 1200.0)]
        baseline = analysis.irq_cadence_baseline(gaps, anomaly_threshold_us=500)
        self.assertEqual(baseline['n'], 3)
        self.assertEqual(baseline['max_us'], 80.0)

    def test_silence_overlap_finds_gap_spanning_window(self):
        gaps = [(10.0, 10.00006, 60.0), (10.0500, 10.0512, 1200.0), (10.09, 10.091, 70.0)]
        overlap = analysis.irq_silence_overlap(10.0505, 10.0511, gaps)
        self.assertIsNotNone(overlap)
        self.assertAlmostEqual(overlap[2], 1200.0)

    def test_silence_overlap_none_when_window_fully_inside_normal_cadence(self):
        gaps = [(10.0, 10.00006, 60.0), (10.00006, 10.00013, 70.0)]
        overlap = analysis.irq_silence_overlap(10.00002, 10.00004, gaps)
        self.assertIsNone(overlap)

    def test_silence_overlap_ignores_gaps_below_threshold(self):
        gaps = [(10.0, 10.0003, 300.0)]  # below default 500us anomaly threshold
        overlap = analysis.irq_silence_overlap(10.0, 10.0003, gaps)
        self.assertIsNone(overlap)


class ReceivePathClassificationTests(unittest.TestCase):
    def test_no_irq_at_all_is_delayed_before_ingress(self):
        classification, detail = analysis.classify_receive_path_event(1.0, 1.0012, {})
        self.assertEqual(classification, 'delayed before receiver kernel-observable ingress')
        self.assertIsNone(detail['irq_us'])

    def test_irq_gap_overlap_short_circuits_the_finer_decomposition(self):
        # Phase 26's actual live-capture finding: even though a
        # (irrelevant, earlier-cycle) irq/softirq/skb/wakeup quadruple is
        # present in `found` (as would happen with the continuous
        # background cadence), an anomalous gap overlapping the window
        # must take priority and short-circuit the finer per-stage
        # decomposition, which would otherwise misleadingly report
        # "receiver path normal" using an unrelated earlier cycle.
        found = {'irq_handler_entry': 0.9999, 'softirq_entry': 0.99991,
                 'napi_gro_receive_entry': 0.99992, 'sk_data_ready': 0.99993}
        irq_gaps = [(0.9999, 2.001, 1002100.0)]  # huge gap spanning the window
        classification, detail = analysis.classify_receive_path_event(
            1.0, 1.0012, found, irq_gaps=irq_gaps)
        self.assertEqual(classification, 'delayed before receiver kernel-observable ingress')
        self.assertAlmostEqual(detail['irq_gap_us'], 1002100.0)

    def test_no_gap_overlap_falls_through_to_fine_decomposition(self):
        found = {'irq_handler_entry': 1.00001, 'softirq_entry': 1.00002,
                  'napi_gro_receive_entry': 1.00003, 'sk_data_ready': 1.00004}
        irq_gaps = [(0.5, 0.5001, 100.0)]  # nowhere near the window
        classification, _ = analysis.classify_receive_path_event(
            1.0, 1.0012, found, irq_gaps=irq_gaps)
        self.assertEqual(classification, 'receiver path normal / delay upstream')

    def test_prompt_irq_but_missing_softirq_is_insufficient_evidence(self):
        found = {'irq_handler_entry': 1.00001}
        classification, _ = analysis.classify_receive_path_event(1.0, 1.0012, found)
        self.assertEqual(classification, 'insufficient evidence')

    def test_delayed_softirq_scheduling(self):
        found = {'irq_handler_entry': 1.00001, 'softirq_entry': 1.0015}
        classification, detail = analysis.classify_receive_path_event(1.0, 1.0016, found)
        self.assertEqual(classification, 'delayed softirq scheduling')
        self.assertGreater(detail['softirq_us'], 150)

    def test_prolonged_napi_processing(self):
        found = {'irq_handler_entry': 1.00001, 'softirq_entry': 1.00002,
                  'napi_gro_receive_entry': 1.0015}
        classification, detail = analysis.classify_receive_path_event(1.0, 1.0016, found)
        self.assertEqual(classification, 'prolonged NAPI processing')
        self.assertGreater(detail['skb_us'], 150)

    def test_delayed_socket_delivery(self):
        found = {'irq_handler_entry': 1.00001, 'softirq_entry': 1.00002,
                  'napi_gro_receive_entry': 1.00003, 'sk_data_ready': 1.0015}
        classification, detail = analysis.classify_receive_path_event(1.0, 1.0016, found)
        self.assertEqual(classification, 'delayed socket delivery')
        self.assertGreater(detail['wakeup_us'], 150)

    def test_all_stages_prompt_is_receiver_path_normal(self):
        found = {'irq_handler_entry': 1.00001, 'softirq_entry': 1.00002,
                  'napi_gro_receive_entry': 1.00003, 'sk_data_ready': 1.00004}
        classification, _ = analysis.classify_receive_path_event(1.0, 1.0012, found)
        self.assertEqual(classification, 'receiver path normal / delay upstream')

    def test_baseline_widens_prompt_threshold(self):
        # A 300us softirq gap would normally be classified as delayed
        # (>150us fixed threshold), but if same-run normal iterations
        # regularly see ~200us softirq gaps (p95), 2x that (400us) covers
        # it -- baseline must be consulted, not just the fixed default.
        found = {'irq_handler_entry': 1.00001, 'softirq_entry': 1.0003}
        baseline = {'softirq_us': {'p95_us': 200.0}}
        classification, _ = analysis.classify_receive_path_event(
            1.0, 1.0016, found, baseline=baseline)
        self.assertEqual(classification, 'insufficient evidence')  # no napi event given
        # Without baseline, the same gap (~290us) would be flagged delayed:
        classification_no_baseline, _ = analysis.classify_receive_path_event(1.0, 1.0016, found)
        self.assertEqual(classification_no_baseline, 'delayed softirq scheduling')


class ReceivePathBaselineTests(unittest.TestCase):
    def test_computes_stage_stats_from_normal_windows(self):
        events = [
            {'ts': 10.0001, 'event': 'irq_handler_entry', 'raw': ''},
            {'ts': 10.0002, 'event': 'softirq_entry', 'raw': ''},
            {'ts': 10.0003, 'event': 'napi_gro_receive_entry', 'raw': ''},
            {'ts': 10.0004, 'event': 'sk_data_ready', 'raw': ''},
        ]
        baseline = analysis.receive_path_baseline(events, [(10.0, 10.001)])
        self.assertEqual(baseline['irq_us']['n'], 1)
        self.assertAlmostEqual(baseline['irq_us']['median_us'], 100, delta=1)
        self.assertAlmostEqual(baseline['softirq_us']['median_us'], 100, delta=1)

    def test_empty_normal_windows_yields_null_stats(self):
        baseline = analysis.receive_path_baseline([], [])
        self.assertIsNone(baseline['irq_us']['median_us'])
        self.assertEqual(baseline['irq_us']['n'], 0)


class ReceivePathReportTests(unittest.TestCase):
    def test_end_to_end_classifies_slow_iteration_and_builds_baseline(self):
        # iteration 5: normal (150us recv), a normal irq/softirq/skb/wakeup
        # cycle within its window. iteration 6: slow (1200us recv) with a
        # continuous background irq cadence (~100us apart, no gap >=500us
        # anywhere) but a delayed socket wakeup within its own cycle --
        # must fall through the irq-gap check (no anomalous gap exists) to
        # the finer per-stage decomposition, and must not pollute the
        # baseline (which should reflect only iteration 5's cycle).
        app_trace = dict(entries=[
            dict(iteration_index=5, recv_begin_ns=10_000_000_000,
                 recv_end_ns=10_000_150_000),
            dict(iteration_index=6, recv_begin_ns=10_001_000_000,
                 recv_end_ns=10_002_200_000),
        ])
        trace_lines = [
            ftrace_receive_line(10.0000001, 'irq_handler_entry', 'irq=177 name=thunderbolt'),
            ftrace_receive_line(10.0000002, 'softirq_entry', 'vec=3 [action=NET_RX]'),
            ftrace_receive_line(10.0000003, 'napi_gro_receive_entry', 'dev=thunderbolt0 len=84'),
            ftrace_receive_line(10.0000004, 'sk_data_ready', 'family=2 protocol=1 func=x'),
        ]
        # A continuous ~100us-spaced background irq cadence spanning both
        # windows, mimicking the real driver's RX ring polling substrate
        # (Phase 26 finding) -- no gap here reaches the 500us anomaly
        # threshold, so classify_receive_path_event must fall through to
        # the fine decomposition rather than reporting "upstream".
        ts = 10.0005
        while ts < 10.0010:
            trace_lines.append(
                ftrace_receive_line(ts, 'irq_handler_entry', 'irq=177 name=thunderbolt'))
            ts += 0.0001
        # The slow iteration's OWN cycle: prompt irq/softirq/skb, delayed
        # socket wakeup (1.5ms after skb -- must be flagged).
        trace_lines += [
            ftrace_receive_line(10.0010001, 'irq_handler_entry', 'irq=177 name=thunderbolt'),
            ftrace_receive_line(10.0010002, 'softirq_entry', 'vec=3 [action=NET_RX]'),
            ftrace_receive_line(10.0010003, 'napi_gro_receive_entry', 'dev=thunderbolt0 len=84'),
            ftrace_receive_line(10.0025000, 'sk_data_ready', 'family=2 protocol=1 func=x'),
        ]
        sched_result = {'trace_text': "\n".join(trace_lines)}
        report = analysis.receive_path_report(app_trace, sched_result, slow_threshold_us=1000)
        self.assertEqual(len(report['slow_events']), 1)
        self.assertEqual(report['slow_events'][0]['iteration'], 6)
        self.assertEqual(report['slow_events'][0]['classification'], 'delayed socket delivery')
        self.assertEqual(report['normal_iteration_count'], 1)
        self.assertEqual(report['same_run_baseline']['irq_us']['n'], 1)

    def test_missing_sched_trace_text_yields_insufficient_evidence_not_a_crash(self):
        app_trace = dict(entries=[
            dict(iteration_index=1, recv_begin_ns=0, recv_end_ns=1_200_000),
        ])
        report = analysis.receive_path_report(app_trace, {'trace_text': None, 'error': 'x'})
        self.assertEqual(len(report['slow_events']), 1)
        self.assertEqual(report['slow_events'][0]['classification'],
                          'delayed before receiver kernel-observable ingress')

    def test_irq_silence_gap_overlapping_slow_window_is_flagged_upstream(self):
        # Phase 26's live-capture central finding, reproduced as a fixture:
        # a slow iteration whose window overlaps a real ~1ms gap in an
        # otherwise-continuous irq cadence must be classified as delayed
        # upstream of all traced receiver network processing -- even
        # though softirq/napi/skb events DO eventually appear later in the
        # window (just not promptly relative to where the cadence broke).
        app_trace = dict(entries=[
            dict(iteration_index=1, recv_begin_ns=10_000_000_000,
                 recv_end_ns=10_001_100_000),
        ])
        trace_lines = [
            ftrace_receive_line(9.9995, 'irq_handler_entry', 'irq=177 name=thunderbolt'),
            ftrace_receive_line(9.9996, 'irq_handler_entry', 'irq=177 name=thunderbolt'),
            # ~1.05ms silence in the irq cadence spanning the recv window:
            ftrace_receive_line(10.00105, 'irq_handler_entry', 'irq=177 name=thunderbolt'),
            ftrace_receive_line(10.00106, 'softirq_entry', 'vec=3 [action=NET_RX]'),
            ftrace_receive_line(10.00107, 'napi_gro_receive_entry', 'dev=thunderbolt0 len=84'),
            ftrace_receive_line(10.00109, 'sk_data_ready', 'family=2 protocol=1 func=x'),
        ]
        sched_result = {'trace_text': "\n".join(trace_lines)}
        report = analysis.receive_path_report(app_trace, sched_result, slow_threshold_us=1000)
        self.assertEqual(len(report['slow_events']), 1)
        event = report['slow_events'][0]
        self.assertEqual(event['classification'], 'delayed before receiver kernel-observable ingress')
        self.assertAlmostEqual(event['irq_gap_us'], 1450, delta=5)


if __name__=='__main__': unittest.main()
