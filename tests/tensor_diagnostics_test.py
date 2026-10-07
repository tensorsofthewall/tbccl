#!/usr/bin/env python3
"""Loopback compatibility and real telemetry tests; usable with a sanitized host build."""
import csv
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_tb4_busy_poll_sweep as sweep
BINARY = sys.argv.pop(1)


def ephemeral_floor():
    try:
        return int(Path('/proc/sys/net/ipv4/ip_local_port_range').read_text().split()[0])
    except (OSError, ValueError):
        return 49152


def ports():
    # The bench processes bind their own endpoints, so an OS-assigned socket cannot be handed to them. The only other users of
    # the OS ephemeral range are concurrent dynamic-port tests (bind port 0, connect source ports), which can take a port
    # between a bind-0/close probe here and the bench's bind. So pick below that range: a fixed-port test cannot run beside
    # this one (all of them share the tbccl_loopback_ports lock).
    floor = ephemeral_floor()
    start = 20000 + (os.getpid() * 7) % max(1, floor - 22000)
    found = []
    port = start
    while len(found) < 2:
        if port + 1000 >= floor:
            port = 20000
        with socket.socket() as a, socket.socket() as b:
            for sock in (a, b):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                a.bind(('127.0.0.1', port))
                b.bind(('127.0.0.1', port + 1000))
                found.append(port)
            except OSError:
                pass
        port += 2
    return found[0], found[1]


class DiagnosticsTests(unittest.TestCase):
    def pair(self,mode,scope='ready',source=0,diagnostics=False,warmup=2,
             trace=False,source_gap=0):
        a,b=ports()
        base=[BINARY,'--mode',mode,'--sizes','0,64,65536','--warmup',str(warmup),'--iterations','5',
              '--peers',f'127.0.0.1:{a},127.0.0.1:{b}','--source-rank',str(source),'--timing-scope',scope]
        if diagnostics: base+=['--diagnostics']
        base += ['--source-gap-us',str(source_gap)]
        if trace: base += ['--trace','--run-id','fixture-run','--physical-machine','fixture-host']
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            outcome=sweep.run_processes([base+['--rank',str(r)] for r in [0,1]],directory,15)
            self.assertTrue(outcome['success'],str(outcome)+'\n'+''.join((directory/f'rank{r}.stderr').read_text() for r in [0,1]))
            rows=list(csv.DictReader(io.StringIO((directory/f'rank{source}.stdout').read_text())))
            self.assertEqual([int(r['payload_bytes']) for r in rows],[0,64,65536])
            if diagnostics:
                for rank in [0,1]:
                    events=sweep.telemetry((directory/f'rank{rank}.stderr').read_text())
                    self.assertEqual(len([e for e in events if e['kind']=='busy_poll']),1)
                    cpus=[e for e in events if e['kind']=='cpu']
                    self.assertEqual(len(cpus),3)
                    for cpu in cpus:
                        self.assertGreater(cpu['wall_seconds'],0)
                        expected=100*(cpu['user_seconds']+cpu['system_seconds'])/cpu['wall_seconds']
                        self.assertAlmostEqual(cpu['process_cpu_pct'],expected,places=6)
                    samples=[e for e in events if e['kind']=='samples']
                    self.assertEqual(len(samples),3 if rank==source else 0)
                    if samples:
                        for event,row in zip(samples,rows):
                            values=sorted(event['values_us'])
                            self.assertEqual(len(values),5)
                            self.assertAlmostEqual(float(row['median_us']),values[2],delta=max(.001,values[2]*1e-5))
                    traces=[e for e in events if e['kind']=='transfer_trace']
                    self.assertEqual(len(traces),3 if trace else 0)
                    for event in traces:
                        self.assertEqual(event['clock'],'CLOCK_MONOTONIC')
                        self.assertEqual(event['clock_units'],'nanoseconds')
                        self.assertEqual(event['capacity'],5)
                        self.assertEqual([e['iteration_index'] for e in event['entries']],list(range(5)))
                        self.assertTrue(all(e['run_id']=='fixture-run' for e in event['entries']))
                        for entry in event['entries']:
                            self.assertLessEqual(entry['iteration_begin_ns'],entry['iteration_end_ns'])
                            if rank==source:
                                self.assertIsNone(entry['recv_begin_ns'])
                                self.assertLessEqual(entry['source_ready_ns'],entry['ack_received_ns'])
                            else:
                                self.assertIsNone(entry['source_ready_ns'])
                                self.assertLessEqual(entry['recv_begin_ns'],entry['ack_send_end_ns'])
            return list(rows[0])

    def test_legacy_modes_and_header(self):
        header=self.pair('end-to-end')
        for mode in ['network-only','latency-floor','ack-calibration']:
            self.assertEqual(header,self.pair(mode))
        p=subprocess.run([BINARY,'--mode','staging-only','--sizes','64','--warmup','1','--iterations','2'],capture_output=True,text=True,timeout=10)
        self.assertEqual(p.returncode,0,p.stderr)
        self.assertEqual(header,next(csv.reader(io.StringIO(p.stdout))))
        self.assertEqual(header,self.pair('end-to-end',diagnostics=True))

    def test_diagnostics_scopes_and_sources(self):
        for scope in ['ready','produce']:
            for source in [0,1]: self.pair('end-to-end',scope,source,True,0)
        self.pair('ack-calibration',diagnostics=True)

    def test_trace_and_source_gap(self):
        self.pair('end-to-end',diagnostics=True,trace=True,source_gap=2000)
        for value in ['-1','10000001']:
            p=subprocess.run([BINARY,'--mode','staging-only','--source-gap-us',value],
                             capture_output=True,text=True,timeout=5)
            self.assertNotEqual(p.returncode,0)
        p=subprocess.run([BINARY,'--mode','staging-only','--trace'],
                         capture_output=True,text=True,timeout=5)
        self.assertNotEqual(p.returncode,0)

    def test_invalid_diagnostic_mode(self):
        p=subprocess.run([BINARY,'--mode','staging-only','--diagnostics'],capture_output=True,text=True,timeout=5)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('requires end-to-end',p.stderr)


if __name__=='__main__': unittest.main()
