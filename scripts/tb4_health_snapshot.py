#!/usr/bin/env python3
"""Read-only Linux/macOS TB4 snapshot and same-boot comparison. No sudo."""
import argparse
import datetime
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess

PCI = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")
ERROR = re.compile(r"AER|PCIe.*error|thunderbolt|controller.*recover", re.I)
TB_TIMEOUT = re.compile(r"\[12\]\s+Timeout|correctable.*timeout|timeout.*correctable", re.I)


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def command(argv):
    if not shutil.which(argv[0]):
        return {"command": argv, "returncode": None, "stdout": "", "stderr": "unavailable command"}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"command": argv, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": argv, "returncode": None, "stdout": "", "stderr": str(error)}


def parse_nstat(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            result[fields[0]] = int(fields[1])
    return result


def aer_counts(text):
    """Parse named sysfs AER counters without conflating error classes."""
    result = {}
    for line in (text or '').splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[1].isdigit():
            result[fields[0]] = int(fields[1])
    return result


def pci_devices(sysroot):
    """Discover the NHI via Thunderbolt domain symlinks or its bound driver.

    Include ancestors and the NHI's switch subtree, including downstream ports
    that have no connected devices. Never select historical BDFs by address.
    """
    sysroot = Path(sysroot)
    anchors = list((sysroot / 'bus/thunderbolt/devices').glob('domain*'))
    anchors += [p for p in (sysroot / 'bus/pci/drivers/thunderbolt').glob('*') if PCI.match(p.name)]
    selected = set()
    evidence = []
    devices = list((sysroot / 'bus/pci/devices').glob('*'))
    for anchor in anchors:
        resolved = anchor.resolve()
        chain = [p for p in [resolved, *resolved.parents] if PCI.match(p.name)]
        if not chain:
            continue
        evidence.append({"anchor": str(anchor), "resolved": str(resolved), "chain": [p.name for p in reversed(chain)]})
        selected.update(p.name for p in chain)
        # NHI -> upstream-facing bridge -> switch upstream bridge (Maple Ridge).
        # Include all descendants, retaining class/vendor/device identity for review.
        subtree = chain[min(2, len(chain) - 1)]
        for device in devices:
            if subtree == device.resolve() or subtree in device.resolve().parents:
                selected.add(device.name)
    result = []
    for bdf in sorted(selected):
        p = sysroot / 'bus/pci/devices' / bdf
        result.append({"bdf": bdf, "path": str(p.resolve()), "present": p.exists(),
                       "vendor": read(p / 'vendor'), "device": read(p / 'device'),
                       "class": read(p / 'class'),
                       "aer": {f: read(p / f) for f in
                               ['aer_dev_correctable', 'aer_dev_nonfatal', 'aer_dev_fatal']},
                       "driver": (p / 'driver').resolve().name if (p / 'driver').exists() else None,
                       "power": {f: read(p / 'power' / f) for f in
                                 ['control', 'runtime_status', 'runtime_active_time',
                                  'runtime_suspended_time', 'runtime_usage', 'autosuspend_delay_ms']}})
    return evidence, result


def linux_snapshot(interface='thunderbolt0', sysroot='/sys', procroot='/proc'):
    root = Path(sysroot)
    interface_path = root / 'class/net' / interface
    evidence, devices = pci_devices(root)
    snapshot = {"boot_id": read(Path(procroot) / 'sys/kernel/random/boot_id'),
                "topology_evidence": evidence, "pci_devices": devices,
                "interface": {"name": interface, "present": interface_path.exists(),
                              "operstate": read(interface_path / 'operstate'),
                              "carrier": read(interface_path / 'carrier'),
                              "mtu": read(interface_path / 'mtu'),
                              "counters": {f: read(interface_path / 'statistics' / f) for f in
                                           ['rx_bytes', 'tx_bytes', 'rx_errors', 'tx_errors', 'rx_dropped', 'tx_dropped']}},
                "thunderbolt_devices": []}
    for p in sorted((root / 'bus/thunderbolt/devices').glob('*')):
        snapshot['thunderbolt_devices'].append({"name": p.name, "path": str(p.resolve()),
            **{f: read(p / f) for f in ['device_name', 'vendor_name', 'unique_id', 'authorized', 'security']}})
    snapshot['commands'] = {name: command(argv) for name, argv in {
        'pci_tree': ['lspci', '-D', '-tv'], 'pci_identity': ['lspci', '-D', '-nn'],
        'links': ['ip', '-br', 'link'], 'addresses': ['ip', '-br', 'addr'],
        'interface': ['ip', '-s', 'link', 'show', interface],
        'addresses_json': ['ip', '-j', 'addr', 'show', 'dev', interface],
        'network_stats': ['nstat', '-az'],
        'kernel': ['journalctl', '-k', '-b', '--no-pager', '-o', 'json'],
    }.items()}
    snapshot['network_stats'] = parse_nstat(
        snapshot['commands']['network_stats']['stdout'])
    kernel = snapshot['commands']['kernel']
    snapshot['kernel_log_available'] = kernel['returncode'] == 0 and bool(kernel['stdout'].strip())
    events = []
    relevant = {d['bdf'] for d in devices}
    for line in kernel['stdout'].splitlines():
        try:
            item = json.loads(line)
        except ValueError:
            continue
        message = item.get('MESSAGE', '')
        if not isinstance(message, str):
            continue
        bdfs = set(re.findall(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', message))
        # Keep controller-specific continuation records such as '[12] Timeout',
        # which do not contain the words AER or error.
        if not ERROR.search(message) and not bdfs & relevant:
            continue
        source = 'thunderbolt' if bdfs & relevant or 'thunderbolt' in message.lower() else ('other-pci' if bdfs else 'unattributed')
        events.append({"cursor": item.get('__CURSOR'), "monotonic_us": item.get('__MONOTONIC_TIMESTAMP'),
                       "message": message, "source": source})
    snapshot['kernel_events'] = events
    # Retain relevant messages in structured form; avoid copying the entire boot journal.
    kernel.pop('stdout')
    return snapshot


def mac_counters(text):
    lines = text.splitlines()
    if not lines:
        return {}
    columns = lines[0].split()
    mapping = {'Ibytes': 'rx_bytes', 'Obytes': 'tx_bytes', 'Ierrs': 'rx_errors',
               'Oerrs': 'tx_errors', 'Drop': 'drops'}
    for line in lines[1:]:
        values = line.split()
        if len(values) != len(columns) or '<Link#' not in line:
            continue
        row = dict(zip(columns, values))
        return {name: row[field] for field, name in mapping.items()
                if row.get(field, '').isdigit()}
    return {}


def snapshot():
    result = {"timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "system": platform.system(), "hostname": platform.node(), "uname": list(platform.uname())}
    if platform.system() == 'Linux':
        result.update(linux_snapshot())
    elif platform.system() == 'Darwin':
        result['commands'] = {name: command(argv) for name, argv in {
            'enumeration': ['system_profiler', 'SPThunderboltDataType', '-json'],
            'interface': ['ifconfig', 'bridge0'],
            'counters': ['netstat', '-ibdn', '-I', 'bridge0'],
            'boot': ['sysctl', '-n', 'kern.boottime'],
        }.items()}
        result['boot_id'] = result['commands']['boot']['stdout'].strip()
        result['interface'] = {"name": 'bridge0', "present": result['commands']['interface']['returncode'] == 0,
                               "counters": mac_counters(result['commands']['counters']['stdout'])}
    else:
        result['error'] = 'unsupported platform'
    return result


def compare(before, after, stop_on_any_tb_timeout=False):
    result = {"same_boot": bool(before.get('boot_id')) and before.get('boot_id') == after.get('boot_id'),
              "stop_reasons": [], "new_kernel_events": [], "counter_deltas": {}, "power_changes": [],
              "aer_counter_deltas": {}, "new_tb_timeout": False}
    if not after.get('interface', {}).get('present'):
        result['stop_reasons'].append('TB4 interface missing')
    if not result['same_boot']:
        result['stop_reasons'].append('boot changed or unknown; establish a new baseline')
        return result
    old = {e['cursor'] for e in before.get('kernel_events', []) if e.get('cursor')}
    new = [e for e in after.get('kernel_events', []) if e.get('cursor') not in old]
    result['new_kernel_events'] = new
    result['new_tb_timeout'] = any(
        e.get('source') == 'thunderbolt' and TB_TIMEOUT.search(e.get('message', ''))
        for e in new)
    if after.get('system') == 'Linux' and not after.get('kernel_log_available'):
        result['stop_reasons'].append('kernel log unavailable')
    tb_errors = [e for e in new if e['source'] == 'thunderbolt' and re.search(r'error|failed|failure', e['message'], re.I)]
    for e in new:
        if e['source'] != 'other-pci' and re.search(r'\bfatal\b|\buncorrect(?:able|ed)\b', e['message'], re.I):
            # "non-fatal" is still an uncorrectable condition and deliberately stops.
            result['stop_reasons'].append('new fatal/uncorrectable PCIe event')
    if len(tb_errors) >= 10:
        result['stop_reasons'].append('rapid Thunderbolt error growth (>=10 events per group)')
    if sum(bool(re.search(r'recover.*fail|fail.*recover', e['message'], re.I)) for e in new if e['source'] == 'thunderbolt') >= 2:
        result['stop_reasons'].append('repeated controller recovery failures')
    for name, value in after.get('interface', {}).get('counters', {}).items():
        previous = before.get('interface', {}).get('counters', {}).get(name)
        if value is not None and previous is not None:
            delta = int(value) - int(previous)
            result['counter_deltas'][name] = delta
            if delta < 0:
                result['stop_reasons'].append('interface counters reset; establish a new baseline')
    for name, value in after.get('network_stats', {}).items():
        previous = before.get('network_stats', {}).get(name)
        if previous is not None:
            result.setdefault('network_stat_deltas', {})[name] = value - previous
    prior = {d['bdf']: d for d in before.get('pci_devices', [])}
    current = {d['bdf']: d for d in after.get('pci_devices', [])}
    if set(prior) != set(current):
        result['stop_reasons'].append('Thunderbolt PCI topology changed')
    for bdf, d in current.items():
        p = prior.get(bdf)
        if p and (p.get('vendor'), p.get('device')) != (d.get('vendor'), d.get('device')):
            result['stop_reasons'].append('PCI identity changed')
        if p:
            for field, total_name in [('aer_dev_correctable', 'TOTAL_ERR_COR'),
                                      ('aer_dev_nonfatal', 'TOTAL_ERR_NONFATAL'),
                                      ('aer_dev_fatal', 'TOTAL_ERR_FATAL')]:
                def total(device):
                    text = device.get('aer', {}).get(field) or ''
                    match = re.search(r'^' + total_name + r' (\d+)$', text, re.M)
                    return int(match[1]) if match else None
                old_total, new_total = total(p), total(d)
                if old_total is not None and new_total is not None:
                    delta = new_total - old_total
                    result['aer_counter_deltas'][bdf + '/' + field] = delta
                    if delta < 0:
                        result['stop_reasons'].append('AER counter reset; establish a new baseline')
                    if delta > 0 and field != 'aer_dev_correctable':
                        result['stop_reasons'].append('fatal/uncorrectable Thunderbolt AER counter increased')
                    if delta >= 10 and field == 'aer_dev_correctable':
                        result['stop_reasons'].append('rapid Thunderbolt correctable AER counter growth')
                if field == 'aer_dev_correctable':
                    old_timeout = aer_counts(p.get('aer', {}).get(field)).get('Timeout')
                    new_timeout = aer_counts(d.get('aer', {}).get(field)).get('Timeout')
                    if (old_timeout is not None and new_timeout is not None and
                            new_timeout > old_timeout):
                        result['new_tb_timeout'] = True
        if p and p['power'] != d['power']:
            result['power_changes'].append({'bdf': bdf, 'before': p['power'], 'after': d['power']})
    if stop_on_any_tb_timeout and result['new_tb_timeout']:
        result['stop_reasons'].append('new Thunderbolt bridge AER Timeout')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare', nargs=2, metavar=('BEFORE', 'AFTER'))
    args = parser.parse_args()
    if args.compare:
        result = compare(*(json.loads(Path(p).read_text()) for p in args.compare))
    else:
        result = snapshot()
    print(json.dumps(result, indent=2))
    return int(bool(result.get('stop_reasons')))


if __name__ == '__main__':
    raise SystemExit(main())
