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


if __name__=='__main__': unittest.main()
