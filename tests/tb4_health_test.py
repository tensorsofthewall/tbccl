#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import tb4_health_snapshot as health


class HealthTests(unittest.TestCase):
    def fixture_command(self, argv):
        return {'command':argv,'returncode':None,'stdout':'','stderr':'unavailable command'}

    def test_missing_interface_pci_and_commands_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); boot=root/'proc/sys/kernel/random/boot_id'
            boot.parent.mkdir(parents=True); boot.write_text('fixture-boot')
            initial=sorted(str(p) for p in root.rglob('*'))
            with patch.object(health,'command',self.fixture_command):
                result=health.linux_snapshot(sysroot=root/'sys',procroot=root/'proc')
            self.assertFalse(result['interface']['present'])
            self.assertEqual(result['pci_devices'],[])
            self.assertEqual(result['boot_id'],'fixture-boot')
            self.assertIn('unavailable',result['commands']['pci_tree']['stderr'])
            self.assertEqual(initial,sorted(str(p) for p in root.rglob('*')))
            with patch('shutil.which',return_value=None):
                self.assertIsNone(health.command(['missing-command'])['returncode'])

    def test_topology_discovers_changed_bdfs_and_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            switch=root/'devices/pci0000:00/0000:00:02.0/0000:42:00.0'
            nhi=switch/'0000:43:00.0/0000:44:00.0'
            domain=nhi/'domain0';domain.mkdir(parents=True)
            sibling=switch/'0000:43:01.0';(sibling/'power').mkdir(parents=True)
            (sibling/'power/control').write_text('auto')
            (sibling/'power/runtime_status').write_text('suspended')
            (root/'bus/thunderbolt/devices').mkdir(parents=True)
            (root/'bus/thunderbolt/devices/domain0').symlink_to(domain)
            (root/'bus/pci/devices').mkdir(parents=True)
            for p in [switch,nhi.parent,nhi,sibling,switch.parent]:
                (root/'bus/pci/devices'/p.name).symlink_to(p)
            evidence,devices=health.pci_devices(root)
            self.assertTrue(evidence)
            self.assertIn(sibling.name,[d['bdf'] for d in devices])
            d=next(d for d in devices if d['bdf']==sibling.name)
            self.assertEqual(d['power']['control'],'auto')
            self.assertEqual(d['power']['runtime_status'],'suspended')
            self.assertNotIn('0000:09:01.0',[d['bdf'] for d in devices])

    def base(self):
        return dict(system='Linux',boot_id='abc',interface={'present':True,'counters':{'rx_errors':'1'}},
                    kernel_log_available=True,kernel_events=[],pci_devices=[])

    def test_deltas_and_reboot(self):
        before=self.base();after=copy.deepcopy(before)
        after['interface']['counters']['rx_errors']='3'
        after['kernel_events']=[dict(cursor='1',source='other-pci',message='0000:00:1b.4 AER: Corrected error'),
                                dict(cursor='2',source='thunderbolt',message='0000:42:00.0 AER: Uncorrectable (Fatal)')]
        result=health.compare(before,after)
        self.assertEqual(result['counter_deltas']['rx_errors'],2)
        self.assertEqual(len(result['stop_reasons']),1)
        after['boot_id']='new'
        result=health.compare(before,after)
        self.assertFalse(result['same_boot']);self.assertEqual(result['counter_deltas'],{})
        self.assertEqual(result['new_kernel_events'],[])

    def test_stop_conditions_and_nvme_separation(self):
        before=self.base();after=copy.deepcopy(before)
        after['kernel_events']=[dict(cursor=str(i),source='other-pci',message='AER: Corrected error') for i in range(12)]
        self.assertFalse(health.compare(before,after)['stop_reasons'])
        for e in after['kernel_events']: e['source']='thunderbolt'
        self.assertTrue(health.compare(before,after)['stop_reasons'])
        after['interface']['present']=False
        self.assertIn('TB4 interface missing',health.compare(before,after)['stop_reasons'])

    def test_mac_link_counters_ignore_address_duplicates(self):
        text='Name Mtu Network Address Ipkts Ierrs Ibytes Opkts Oerrs Obytes Coll Drop\n'
        text+='bridge0 9000 <Link#17> 00:11:22:33:44:55 100 1 1234 200 2 5678 0 3\n'
        text+='bridge0 9000 192.168.3 192.168.3.1 100 - 1234 200 - 5678 - -\n'
        self.assertEqual(health.mac_counters(text),dict(rx_bytes='1234',tx_bytes='5678',rx_errors='1',tx_errors='2',drops='3'))
        self.assertEqual(health.mac_counters(''),{})

    def test_existing_events_not_new(self):
        before=self.base()
        before['kernel_events']=[dict(cursor='x',source='thunderbolt',message='AER: Fatal')]
        self.assertFalse(health.compare(before,copy.deepcopy(before))['stop_reasons'])


if __name__=='__main__': unittest.main()
