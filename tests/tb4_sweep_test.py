#!/usr/bin/env python3
import copy
import csv
import json
import io
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_tb4_busy_poll_sweep as sweep


class SweepTests(unittest.TestCase):
    def args(self, *extra):
        return sweep.parser().parse_args(['--loopback', '--output', '/tmp/not-created', *extra])

    def test_parsing_and_rejection(self):
        for value in ['-1', '1000001', '0,0', 'abc', '']:
            with self.assertRaises(Exception):
                sweep.integer_list(value, 0, 1000000)
        for extra in [('--timeout', 'nan'), ('--iterations', '0'), ('--warmup', '-1'),
                      ('--order', 'abba'), ('--busy-poll', '100'), ('--source', 'mac', '--source-rank', '0')]:
            with self.assertRaises(ValueError):
                sweep.validate(self.args(*extra))

    def test_order_and_physical_mapping(self):
        args = self.args('--sizes', '65536', '--busy-poll', '0,100', '--order', 'abba',
                         '--repetitions', '2', '--rank0', 'both', '--source-rank', '1')
        runs = list(sweep.experiments(args))
        self.assertEqual([r['busy_poll_us'] for r in runs[:8]], [0,100,100,0]*2)
        self.assertEqual(runs[0]['source'], 'linux')
        self.assertEqual(runs[8]['source'], 'mac')
        self.assertEqual([r['order'] for r in runs], list(range(len(runs))))
        args.order = 'rotate'
        args.busy_poll = [0,50,100,150,200]
        args.repetitions = 3
        runs = list(sweep.experiments(args))
        self.assertNotEqual([r['busy_poll_us'] for r in runs[:5]], [r['busy_poll_us'] for r in runs[5:10]])
        self.assertEqual(sorted(r['busy_poll_us'] for r in runs[5:10]), args.busy_poll)

    def test_aggregation_pairs_repetitions(self):
        rows = []
        for rep, baseline, candidate in [(0,100,80), (1,200,210), (2,300,240)]:
            for poll, value in [(0, baseline),(100,candidate)]:
                r = dict(backend_pair='host', timing_scope='ready', rank0='mac', source='linux',
                         payload_bytes=64, repetition=rep, busy_poll_us=poll)
                for metric in ['min_us','median_us','p95_us','p99_us','max_us','effective_GBps',
                               'linux_process_cpu_pct','mac_process_cpu_pct','linux_user_seconds',
                               'linux_system_seconds','mac_user_seconds','mac_system_seconds']:
                    r[metric] = value
                rows.append(r)
        result = sweep.aggregate(rows)[1]
        self.assertEqual(result['improvement_pct'], 20)
        self.assertEqual(result['improvement_min_pct'], -5)
        self.assertEqual(result['positive_repetitions'], 2)
        self.assertEqual(result['median_us'], 210)

    def fixture(self, directory):
        config = next(sweep.experiments(self.args('--sizes','64','--iterations','3')))
        row = dict(mode='end-to-end', timing_scope='ready', payload_bytes=64, iterations=3,
                   source_rank=config['source_rank'], source_backend='host', destination_backend='host')
        for i, key in enumerate(['min_us','median_us','p95_us','p99_us','max_us']): row[key] = i+1
        for key in ['effective_GBps','source_staging_us','sender_network_us','receiver_network_us',
                    'destination_staging_us','completion_confirmed_us']: row[key] = 1
        for rank in [0,1]:
            with (directory/f'rank{rank}.stdout').open('w') as f:
                if rank == config['source_rank']:
                    w=csv.DictWriter(f, fieldnames=row); w.writeheader(); w.writerow(row)
            cpu=dict(kind='cpu',rank=rank,payload_bytes=64,wall_seconds=1,user_seconds=.2,system_seconds=.1,process_cpu_pct=30)
            poll=dict(kind='busy_poll',requested_us=0,status='supported',sockets=[{'effective_us':0}])
            events=[cpu,poll]
            if rank==config['source_rank']:
                events.append(dict(kind='samples',rank=rank,payload_bytes=64,metric='completion_confirmed',values_us=[1,2,3]))
            (directory/f'rank{rank}.stderr').write_text(''.join(sweep.PREFIX+json.dumps(e)+'\n' for e in events))
        return config

    def test_csv_and_unsupported_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            d=Path(tmp); config=self.fixture(d)
            self.assertEqual(sweep.parse_result(d,config,True)['linux_process_cpu_pct'],30)
            p=d/'rank0.stderr'
            original=p.read_text()
            p.write_text(original.replace('"supported"','"unsupported"').replace('[{"effective_us": 0}]','[]'))
            result=sweep.parse_result(d,config,False)
            self.assertIsNone(result['mac_effective_us'])
            with self.assertRaises(ValueError): sweep.parse_result(d,config,True)
            p.write_text(original.replace('"effective_us": 0','"effective_us": 50'))
            with self.assertRaises(ValueError): sweep.parse_result(d,config,True)
            self.fixture(d)
            (d/f"rank{config['source_rank']}.stdout").write_text('bad,csv\n1,2\n')
            with self.assertRaises((ValueError,KeyError)): sweep.parse_result(d,config,True)

    def worker(self, code, timeout=5):
        return [sys.executable,'-u','-c',sweep.WORKER,json.dumps({
            'argv':[sys.executable,'-c',code], 'cwd':str(sweep.ROOT), 'timeout':timeout})]

    def test_timeout_cleanup_and_unrelated_listener(self):
        with tempfile.TemporaryDirectory() as tmp, socket.socket() as unrelated:
            unrelated.bind(('127.0.0.1',0)); unrelated.listen()
            pidfile=Path(tmp)/'child.pid'
            code=f'import os,time; open({str(pidfile)!r},"w").write(str(os.getpid())); time.sleep(60)'
            result=sweep.run_processes([self.worker(code)],Path(tmp),.5)
            self.assertFalse(result['success']); self.assertIn('timeout',result['error'])
            pid=int(pidfile.read_text())
            with self.assertRaises(ProcessLookupError): os.kill(pid,0)
            self.assertGreater(unrelated.getsockname()[1],0)
            self.assertIn('TBCCL_WORKER ',(Path(tmp)/'rank0.stderr').read_text())

    def test_failed_peer_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            started=time.monotonic()
            result=sweep.run_processes([self.worker('import sys;sys.exit(7)'),self.worker('import time;time.sleep(60)')],Path(tmp),10)
            self.assertFalse(result['success']); self.assertIn('peer failed',result['error'])
            self.assertLess(time.monotonic()-started,6)
            self.assertEqual(result['exit_codes'][0],7)

    def test_endpoint_eof_watchdog(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile=Path(tmp)/'pid'
            code=f'import os,time; open({str(pidfile)!r},"w").write(str(os.getpid())); time.sleep(60)'
            p=subprocess.Popen(self.worker(code),stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                deadline=time.monotonic()+5
                while not pidfile.exists() and time.monotonic()<deadline: time.sleep(.01)
                self.assertTrue(pidfile.exists())
                p.stdin.close(); self.assertEqual(p.wait(timeout=4),124)
                with self.assertRaises(ProcessLookupError): os.kill(int(pidfile.read_text()),0)
            finally:
                sweep.stop_processes([p])

    def test_spawn_failure_preserves_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=sweep.run_processes([['/no/such/benchmark']],Path(tmp),1)
            self.assertFalse(result['success'])

    def test_live_guard_stops_on_fatal_and_carrier_loss(self):
        guard=sweep.LinkGuard.__new__(sweep.LinkGuard)
        guard.baseline=dict(system='Linux',boot_id='abc',interface={'present':True},
                            kernel_log_available=True,pci_devices=[],kernel_events=[])
        guard.relevant={'0000:42:00.0'}
        guard.process=Mock();guard.process.poll.return_value=None
        guard.pending='';guard.events=[]
        guard.reader=io.StringIO(json.dumps({'__CURSOR':'new','MESSAGE':'pcieport 0000:42:00.0 AER: Uncorrectable (Fatal)'})+'\n')
        with patch.object(Path,'exists',return_value=True), patch.object(sweep.health,'read',return_value='1'):
            with self.assertRaisesRegex(RuntimeError,'fatal/uncorrectable'): guard.check()
        with patch.object(Path,'exists',return_value=True), patch.object(sweep.health,'read',return_value='0'):
            with self.assertRaisesRegex(RuntimeError,'carrier lost'): guard.check()

    def test_health_failure_excludes_completed_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args('--output',str(Path(tmp)/'result'), '--sizes','64',
                           '--busy-poll','0','--repetitions','1','--source','linux',
                           '--mac-root','/tmp/repo','--mac-binary','/tmp/repo/benchmark')
            args.loopback=False
            before={p:dict(system='Linux',boot_id='abc',interface={'present':True},
                           kernel_log_available=True,pci_devices=[],kernel_events=[]) for p in ['linux','mac']}
            after=copy.deepcopy(before);after['linux']['interface']['present']=False
            def completed(commands,directory,timeout,guard):
                for rank in [0,1]: (directory/f'rank{rank}.stderr').write_text('')
                return dict(success=True,error=None,exit_codes=[0,0])
            with patch.object(sweep,'capture_health',side_effect=[before,after,after]), \
                 patch.object(sweep,'healthy'), patch.object(sweep,'LinkGuard'), \
                 patch.object(sweep,'free_port',return_value=12345), \
                 patch.object(sweep,'run_processes',side_effect=completed), \
                 patch.object(sweep,'parse_result',return_value={}):
                self.assertEqual(sweep.run(args),1)
            self.assertFalse((args.output/'runs.csv').exists())
            self.assertFalse(json.loads((args.output/'run-0000/metadata.json').read_text())['success'])
            self.assertEqual(json.loads((args.output/'status.json').read_text())['completed'],0)

    def test_dry_run_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            args=self.args('--dry-run','--output',str(Path(tmp)/'result'))
            self.assertEqual(sweep.run(args),0)
            metadata=json.loads((args.output/'plan.json').read_text())
            self.assertEqual(metadata['environment'],'loopback')
            self.assertEqual(metadata['experiments'][0]['busy_poll_us'],0)
            with self.assertRaises(FileExistsError): sweep.run(args)


if __name__=='__main__': unittest.main()
