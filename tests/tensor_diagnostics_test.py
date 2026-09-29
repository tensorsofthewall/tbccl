#!/usr/bin/env python3
"""Loopback compatibility and real telemetry tests; usable with a sanitized host build."""
import csv
import io
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_tb4_busy_poll_sweep as sweep
BINARY = sys.argv.pop(1)


def ports():
    with socket.socket() as a, socket.socket() as b:
        a.bind(('127.0.0.1',0));b.bind(('127.0.0.1',0))
        return a.getsockname()[1],b.getsockname()[1]


class DiagnosticsTests(unittest.TestCase):
    def pair(self,mode,scope='ready',source=0,diagnostics=False,warmup=2):
        a,b=ports()
        base=[BINARY,'--mode',mode,'--sizes','0,64,65536','--warmup',str(warmup),'--iterations','5',
              '--peers',f'127.0.0.1:{a},127.0.0.1:{b}','--source-rank',str(source),'--timing-scope',scope]
        if diagnostics: base+=['--diagnostics']
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

    def test_invalid_diagnostic_mode(self):
        p=subprocess.run([BINARY,'--mode','staging-only','--diagnostics'],capture_output=True,text=True,timeout=5)
        self.assertNotEqual(p.returncode,0)
        self.assertIn('requires end-to-end',p.stderr)


if __name__=='__main__': unittest.main()
