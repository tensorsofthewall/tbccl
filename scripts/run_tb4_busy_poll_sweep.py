#!/usr/bin/env python3
"""Controlled two-host TB4 sweep. No builds, syncs, privileged writes or recovery.

Use --loopback for local fixture/CPU testing, never for TB4 conclusions.
Real hardware runs require healthy pre/post snapshots and stop on failures.
"""
import argparse
import csv
import io
import itertools
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time

import tb4_health_snapshot as health

PREFIX = 'TBCCL_DIAGNOSTIC '
ROOT = Path(__file__).resolve().parents[1]
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
       '-o', 'ServerAliveInterval=2', '-o', 'ServerAliveCountMax=3']

# This supervisor runs on the endpoint, including over SSH. The child has its
# own process group. EOF, missing heartbeat, timeout or supervisor termination
# terminates only that group. Killing the local SSH client alone is insufficient.
WORKER = r'''
import hashlib,json,os,platform,select,signal,subprocess,sys,time
from pathlib import Path
cfg=json.loads(sys.argv[1])
p=None
cancelled=False
def cancel(*args):
    global cancelled
    cancelled=True
signal.signal(signal.SIGTERM,cancel)
signal.signal(signal.SIGINT,cancel)
try:
    binary=Path(cfg['argv'][0])
    revision=subprocess.run(['git','-C',cfg['cwd'],'rev-parse','HEAD'],capture_output=True,text=True,timeout=5)
    diff=subprocess.run(['git','-C',cfg['cwd'],'diff','HEAD','--'],capture_output=True,timeout=5)
    print('TBCCL_WORKER '+json.dumps({'hostname':platform.node(),'system':platform.system(),
        'uname':list(platform.uname()),'git_commit':revision.stdout.strip(),
        'tracked_diff_sha256':hashlib.sha256(diff.stdout).hexdigest(),
        'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),'command':cfg['argv']}),file=sys.stderr,flush=True)
    p=subprocess.Popen(cfg['argv'],cwd=cfg['cwd'],start_new_session=True)
    deadline=time.monotonic()+cfg['timeout']
    heartbeat=time.monotonic()
    while p.poll() is None:
        now=time.monotonic()
        if cancelled or now>=deadline or now-heartbeat>10:
            raise RuntimeError('endpoint cancelled, timed out, or lost controller heartbeat')
        readable,_,_=select.select([sys.stdin],[],[],0.05)
        if readable:
            data=os.read(sys.stdin.fileno(),4096)
            if not data:
                raise RuntimeError('controller disconnected')
            heartbeat=time.monotonic()
    code=p.returncode
except BaseException as error:
    print('TBCCL_WORKER_ERROR '+str(error),file=sys.stderr,flush=True)
    code=124
finally:
    if p is not None and p.poll() is None:
        try: os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError: pass
        try: p.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try: os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait()
sys.exit(code if code>=0 else 128-code)
'''


def integer_list(text, lower=0, upper=None):
    try:
        values = [int(x) for x in text.split(',')]
    except ValueError as error:
        raise argparse.ArgumentTypeError('expected comma-separated integers') from error
    if not values or len(set(values)) != len(values) or any(v < lower or (upper is not None and v > upper) for v in values):
        raise argparse.ArgumentTypeError('values must be unique and within the allowed range')
    return values


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--linux-binary', default=str(ROOT / 'build-release/tbccl_tensor_transfer_bench'))
    p.add_argument('--linux-root', default=str(ROOT))
    p.add_argument('--mac-ssh', default='tbccl-mac')
    p.add_argument('--mac-root')
    p.add_argument('--mac-binary')
    p.add_argument('--mac-python', default='/usr/bin/python3')
    p.add_argument('--linux-address', default='192.168.3.2')
    p.add_argument('--mac-address', default='192.168.3.1')
    p.add_argument('--loopback', action='store_true')
    p.add_argument('--backend-pair', choices=['host', 'cuda-metal'], default='host')
    p.add_argument('--rank0', choices=['linux', 'mac', 'both'], default='mac')
    p.add_argument('--source', choices=['linux', 'mac', 'both'], default='both')
    p.add_argument('--source-rank', type=int, choices=[0, 1], help='alternative to --source (default both)')
    p.add_argument('--sizes', type=lambda s: integer_list(s, 1), default=[64, 4096, 65536, 262144, 1048576, 4194304])
    p.add_argument('--busy-poll', type=lambda s: integer_list(s, 0, 1000000), default=[0, 50, 100, 150, 200])
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--large-iterations', type=int, default=50)
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--order', choices=['rotate', 'abba'], default='rotate')
    p.add_argument('--timing-scope', choices=['ready', 'produce'], default='ready')
    p.add_argument('--timeout', type=float, default=45)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true', help='write experiment plan only; no SSH or benchmarks')
    return p


def validate(args):
    if min(args.iterations, args.large_iterations, args.repetitions) <= 0 or args.warmup < 0 or not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError('positive iterations, repetitions and timeout; nonnegative warmup required')
    if args.order == 'abba' and (len(args.busy_poll) != 2 or args.busy_poll[0] != 0):
        raise ValueError('ABBA requires exactly two busy-poll values starting with 0')
    if 0 not in args.busy_poll:
        raise ValueError('include busy-poll 0 as the contemporaneous baseline')
    if args.source_rank is not None and args.source != 'both':
        raise ValueError('use --source or --source-rank, not both')
    if args.mac_ssh.startswith('-'):
        raise ValueError('invalid SSH destination')
    if not args.loopback and not args.dry_run and (not args.mac_root or not args.mac_binary):
        raise ValueError('real TB4 runs require absolute --mac-root and --mac-binary paths')
    for path in [args.linux_binary, args.linux_root, args.mac_root, args.mac_binary]:
        if path is not None and not Path(path).is_absolute():
            raise ValueError('binary/repository paths must be absolute')
    if args.loopback and args.backend_pair != 'host':
        raise ValueError('loopback fixture supports host backend pair only')
    for address in [args.linux_address, args.mac_address]:
        socket.inet_pton(socket.AF_INET, address)


def experiments(args):
    order_index = 0
    assignments = ['mac', 'linux'] if args.rank0 == 'both' else [args.rank0]
    for rank0, size in itertools.product(assignments, args.sizes):
        physical = [rank0, 'mac' if rank0 == 'linux' else 'linux']
        sources = ['linux', 'mac'] if args.source == 'both' else [args.source]
        if args.source_rank is not None:
            sources = [physical[args.source_rank]]
        for source in sources:
            for repetition in range(args.repetitions):
                values = args.busy_poll
                if args.order == 'abba':
                    ordered = [values[0], values[1], values[1], values[0]]
                else:
                    shift = repetition % len(values)
                    ordered = values[shift:] + values[:shift]
                for poll in ordered:
                    yield dict(order=order_index, repetition=repetition, rank0=rank0, source=source,
                               source_rank=physical.index(source), payload_bytes=size, busy_poll_us=poll,
                               iterations=args.large_iterations if size >= 4194304 else args.iterations,
                               warmup=args.warmup, backend_pair=args.backend_pair, timing_scope=args.timing_scope)
                    order_index += 1


def remote_python(args, code, arguments=()):
    return SSH + [args.mac_ssh, shlex.join([args.mac_python, '-u', '-c', code, *arguments])]


def endpoint_command(args, physical, code, arguments=()):
    if args.loopback or physical == 'linux':
        return [sys.executable, '-u', '-c', code, *arguments]
    return remote_python(args, code, arguments)


def read_endpoint(args, physical, code):
    p = subprocess.run(endpoint_command(args, physical, code), capture_output=True, text=True, timeout=35)
    if p.returncode:
        raise RuntimeError(f'{physical} diagnostic failed: {p.stderr}')
    return json.loads(p.stdout)


def free_port(args, physical):
    address = '127.0.0.1' if args.loopback else getattr(args, physical + '_address')
    # Bind-and-release reduces collisions; it cannot reserve the port until World
    # opens it. A racing bind is an explicit failed run, never a reason to kill.
    code = 'import socket,json; s=socket.socket(); s.bind((' + repr(address) + ',0)); print(json.dumps(s.getsockname()[1]))'
    return read_endpoint(args, physical, code)


def health_snapshot(args, physical):
    if physical == 'linux':
        return health.snapshot()
    # Stream code for a read-only query; install/sync nothing on the endpoint.
    code = Path(health.__file__).read_text().split("if __name__ == '__main__':")[0]
    return read_endpoint(args, physical, code + '\nprint(json.dumps(snapshot()))')


def healthy(args, snapshots):
    linux, mac = snapshots['linux'], snapshots['mac']
    if not linux.get('interface', {}).get('present') or linux['interface'].get('carrier') != '1':
        raise RuntimeError('Linux TB4 interface missing or carrier down')
    if not linux.get('topology_evidence') or not linux.get('kernel_log_available'):
        raise RuntimeError('cannot establish PCI topology/kernel-log safety baseline')
    if not mac.get('interface', {}).get('present'):
        raise RuntimeError('Mac bridge0 missing')
    if linux['interface'].get('mtu') != '9000':
        raise RuntimeError('Linux TB4 MTU is not 9000')
    addresses = linux['commands']['addresses_json']
    info = json.loads(addresses['stdout']) if addresses['returncode'] == 0 else []
    if not any(a.get('local') == args.linux_address for entry in info for a in entry.get('addr_info', [])):
        raise RuntimeError('Linux TB4 address missing')
    mac_interface = mac['commands']['interface']['stdout']
    if f'inet {args.mac_address} ' not in mac_interface or 'status: active' not in mac_interface or 'mtu 9000' not in mac_interface:
        raise RuntimeError('Mac TB4 address, active link, or MTU missing')
    for physical, destination in [('linux', args.mac_address), ('mac', args.linux_address)]:
        source = getattr(args, physical + '_address')
        argv = ['ping', '-n', '-c', '2', '-I' if physical == 'linux' else '-S', source, destination]
        result = read_endpoint(args, physical, 'import json,subprocess; p=subprocess.run(' + repr(argv) + ',capture_output=True,text=True,timeout=5); print(json.dumps({"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}))')
        snapshots[physical]['reachability'] = result
        if result['returncode']:
            raise RuntimeError(f'{physical} TB4 ping failed')


def stop_processes(processes):
    # Closing SSH stdin reaches the remote watchdog; wait for its child cleanup
    # before terminating the local client. A remote watchdog also has a deadline.
    for p in processes:
        if p.stdin and not p.stdin.closed:
            p.stdin.close()
    deadline = time.monotonic() + 4
    for p in processes:
        try:
            p.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()


def run_processes(commands, directory, timeout, guard=None):
    processes, files = [], []
    error = None
    try:
        for rank, command in enumerate(commands):
            stdout = (directory / f'rank{rank}.stdout').open('wb')
            stderr = (directory / f'rank{rank}.stderr').open('wb')
            files += [stdout, stderr]
            processes.append(subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout,
                                               stderr=stderr, start_new_session=True))
        deadline = time.monotonic() + timeout
        next_heartbeat = 0
        while True:
            now = time.monotonic()
            codes = [p.poll() for p in processes]
            if any(c is not None and c != 0 for c in codes):
                raise RuntimeError(f'peer failed: {codes}')
            if all(c == 0 for c in codes):
                break
            if now >= deadline:
                raise TimeoutError('per-run timeout')
            if guard:
                guard()
            if now >= next_heartbeat:
                for p in processes:
                    if p.poll() is None:
                        p.stdin.write(b'heartbeat\n')
                        p.stdin.flush()
                next_heartbeat = now + 1
            time.sleep(0.025)
    except (Exception, KeyboardInterrupt) as exception:
        error = str(exception) or type(exception).__name__
    finally:
        stop_processes(processes)
        for f in files:
            f.close()
    return {'success': error is None, 'error': error, 'exit_codes': [p.returncode for p in processes]}


def telemetry(text):
    return [json.loads(line[len(PREFIX):]) for line in text.splitlines() if line.startswith(PREFIX)]


def parse_result(directory, config, loopback=False):
    csv_text = (directory / f"rank{config['source_rank']}.stdout").read_text()
    rows = list(csv.DictReader(io.StringIO(csv_text)))
    if len(rows) != 1:
        raise ValueError('expected exactly one benchmark CSV row')
    row = rows[0]
    if row.get('mode') != 'end-to-end' or row.get('timing_scope') != config['timing_scope'] or int(row['payload_bytes']) != config['payload_bytes'] or int(row['iterations']) != config['iterations'] or int(row['source_rank']) != config['source_rank']:
        raise ValueError('CSV configuration mismatch')
    numeric = ['min_us', 'median_us', 'p95_us', 'p99_us', 'max_us', 'effective_GBps',
               'source_staging_us', 'sender_network_us', 'receiver_network_us', 'destination_staging_us',
               'completion_confirmed_us']
    result = dict(config)
    for key in numeric:
        result[key] = float(row[key])
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError(f'invalid CSV field: {key}')
    if not all(result[a] <= result[b] for a, b in zip(numeric[:4], numeric[1:5])):
        raise ValueError('invalid percentile order')
    physical = [config['rank0'], 'mac' if config['rank0'] == 'linux' else 'linux']
    expected_backends = {'linux': 'host', 'mac': 'host'} if config['backend_pair'] == 'host' else {'linux': 'cuda-pinned', 'mac': 'metal-shared'}
    if row['source_backend'] != expected_backends[config['source']] or row['destination_backend'] != expected_backends[physical[1-config['source_rank']]]:
        raise ValueError('CSV backend mismatch')
    for rank, machine in enumerate(physical):
        events = telemetry((directory / f'rank{rank}.stderr').read_text())
        cpus = [e for e in events if e['kind'] == 'cpu' and e['rank'] == rank and e['payload_bytes'] == config['payload_bytes']]
        polls = [e for e in events if e['kind'] == 'busy_poll']
        if len(cpus) != 1 or len(polls) != 1:
            raise ValueError('missing or duplicate CPU/socket telemetry')
        cpu, poll = cpus[0], polls[0]
        for key in ['wall_seconds', 'user_seconds', 'system_seconds', 'process_cpu_pct']:
            if not math.isfinite(cpu[key]) or cpu[key] < 0 or (key == 'wall_seconds' and cpu[key] == 0):
                raise ValueError('invalid CPU telemetry')
            result[machine + '_' + key] = cpu[key]
        if poll['requested_us'] != config['busy_poll_us']:
            raise ValueError('requested socket configuration mismatch')
        expected_status = 'supported' if machine == 'linux' or loopback else 'unsupported'
        if poll['status'] != expected_status:
            raise ValueError('unexpected busy-poll support; cannot compare as configured')
        result[machine + '_busy_poll_status'] = poll['status']
        observed = [s['effective_us'] for s in poll['sockets']]
        result[machine + '_effective_us'] = observed[0] if len(observed) == 1 else None
        if expected_status == 'supported' and (len(observed) != 1 or observed[0] != config['busy_poll_us']):
            raise ValueError('requested vs effective socket setting differs or could not be observed')
        if rank == config['source_rank']:
            samples = [e for e in events if e['kind'] == 'samples' and e['metric'] == 'completion_confirmed' and e['payload_bytes'] == config['payload_bytes']]
            if len(samples) != 1 or len(samples[0]['values_us']) != config['iterations'] or any(not math.isfinite(x) or x < 0 for x in samples[0]['values_us']):
                raise ValueError('missing or invalid raw completion samples')
    (directory / 'benchmark.csv').write_text(csv_text)
    return result


def aggregate(results):
    """Medians of independent run statistics, never pooled-sample percentiles.

    Baseline comparisons use the same repetition/size/direction/assignment/scope.
    ABBA duplicates are reduced inside each repetition before comparing.
    """
    keys = ['backend_pair', 'timing_scope', 'rank0', 'source', 'payload_bytes']
    metrics = ['min_us', 'median_us', 'p95_us', 'p99_us', 'max_us', 'effective_GBps',
               'linux_process_cpu_pct', 'mac_process_cpu_pct', 'linux_user_seconds',
               'linux_system_seconds', 'mac_user_seconds', 'mac_system_seconds']
    groups = {}
    for row in results:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key, {}).setdefault(row['busy_poll_us'], []).append(row)
    output = []
    for key, polls in groups.items():
        baseline = polls.get(0, [])
        for poll, rows in sorted(polls.items()):
            row = dict(zip(keys, key), busy_poll_us=poll, runs=len(rows), repetitions=len({r['repetition'] for r in rows}))
            row.update({m: statistics.median(r[m] for r in rows) for m in metrics})
            improvements = []
            for rep in sorted({r['repetition'] for r in rows}):
                a = [r['median_us'] for r in baseline if r['repetition'] == rep]
                b = [r['median_us'] for r in rows if r['repetition'] == rep]
                if a and b and statistics.median(a) > 0:
                    improvements.append(100 * (statistics.median(a) - statistics.median(b)) / statistics.median(a))
            row['improvement_pct'] = statistics.median(improvements) if improvements else None
            row['improvement_min_pct'] = min(improvements) if improvements else None
            row['improvement_max_pct'] = max(improvements) if improvements else None
            row['positive_repetitions'] = sum(x > 0 for x in improvements)
            row['paired_repetitions'] = len(improvements)
            row['run_median_min_us'] = min(r['median_us'] for r in rows)
            row['run_median_max_us'] = max(r['median_us'] for r in rows)
            output.append(row)
    return output


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + '\n')


class LinkGuard:
    """Read-only live monitoring; post-run snapshots remain the authoritative delta."""
    def __init__(self, directory, baseline):
        self.baseline = baseline
        self.writer = (directory / 'kernel-live.jsonl').open('wb')
        self.reader = (directory / 'kernel-live.jsonl').open('r')
        self.errors = (directory / 'kernel-live.stderr').open('wb')
        self.process = subprocess.Popen(['journalctl', '-k', '-b', '-f', '-n', '0', '-o', 'json'],
                                        stdout=self.writer, stderr=self.errors)
        self.pending = ''
        self.events = []
        self.relevant = {d['bdf'] for d in baseline['pci_devices']}

    def check(self):
        if not Path('/sys/class/net/thunderbolt0').exists():
            raise RuntimeError('STOP: thunderbolt0 disappeared')
        if health.read('/sys/class/net/thunderbolt0/carrier') != '1':
            raise RuntimeError('STOP: TB4 carrier lost')
        if self.process.poll() is not None:
            raise RuntimeError('STOP: live kernel monitor exited')
        self.pending += self.reader.read()
        lines = self.pending.split('\n')
        self.pending = lines.pop()
        for line in lines:
            try:
                entry = json.loads(line)
            except ValueError:
                continue
            message = entry.get('MESSAGE', '')
            if not isinstance(message, str) or not health.ERROR.search(message):
                continue
            bdfs = set(health.re.findall(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', message))
            source = 'thunderbolt' if bdfs & self.relevant or 'thunderbolt' in message.lower() else ('other-pci' if bdfs else 'unattributed')
            self.events.append({'cursor': entry.get('__CURSOR'), 'message': message, 'source': source})
        after = dict(self.baseline, kernel_events=self.events)
        reasons = health.compare(dict(self.baseline, kernel_events=[]), after)['stop_reasons']
        if reasons:
            raise RuntimeError('STOP: ' + '; '.join(reasons))

    def close(self):
        self.process.terminate()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        for f in [self.writer, self.reader, self.errors]:
            f.close()


def capture_health(args):
    # Preserve the surviving machine's snapshot even when its peer disappears.
    result = {}
    for physical in ['linux', 'mac']:
        try:
            result[physical] = health_snapshot(args, physical)
        except Exception as error:
            result[physical] = {'error': str(error), 'interface': {'present': False}}
    return result


def run(args):
    validate(args)
    args.output.mkdir(parents=True, exist_ok=False)
    planned = list(experiments(args))
    dump(args.output / 'plan.json', {'arguments': vars(args), 'experiments': planned,
                                   'environment': 'loopback' if args.loopback else 'real-tb4',
                                   'aggregation': 'median of per-run statistics; paired independent repetitions'})
    if args.dry_run:
        return 0
    results = []
    failure = None
    monitor = None
    try:
        pre = {}
        if not args.loopback:
            pre = capture_health(args)
            dump(args.output / 'health-before.json', pre)
            try:
                healthy(args, pre)
            finally:
                dump(args.output / 'health-before.json', pre)
            monitor = LinkGuard(args.output, pre['linux'])
        for config in planned:
            directory = args.output / f"run-{config['order']:04d}"
            directory.mkdir()
            meta = {'configuration': config, 'start_utc': health.datetime.datetime.now(health.datetime.timezone.utc).isoformat()}
            dump(directory / 'metadata.json', meta)
            print(f"run {config['order']+1}/{len(planned)}: {config}", flush=True)
            try:
                ports = {p: free_port(args, p) for p in ['linux', 'mac']}
                if args.loopback:
                    while ports['mac'] == ports['linux']:
                        ports['mac'] = free_port(args, 'mac')
                physical = [config['rank0'], 'mac' if config['rank0'] == 'linux' else 'linux']
                addresses = {p: '127.0.0.1' if args.loopback else getattr(args, p + '_address') for p in physical}
                peers = ','.join(f'{addresses[p]}:{ports[p]}' for p in physical)
                commands = []
                benchmark_commands = []
                for rank, machine in enumerate(physical):
                    local = args.loopback or machine == 'linux'
                    binary = args.linux_binary if local else args.mac_binary
                    cwd = args.linux_root if local else args.mac_root
                    backend = 'host' if config['backend_pair'] == 'host' else ('cuda-pinned' if machine == 'linux' else 'metal-shared')
                    argv = [binary, '--rank', str(rank), '--peers', peers, '--bind', addresses[machine],
                            '--mode', 'end-to-end', '--local-backend', backend,
                            '--source-rank', str(config['source_rank']), '--sizes', str(config['payload_bytes']),
                            '--busy-poll', str(config['busy_poll_us']), '--warmup', str(config['warmup']),
                            '--iterations', str(config['iterations']), '--timing-scope', config['timing_scope'], '--diagnostics']
                    benchmark_commands.append(argv)
                    commands.append(endpoint_command(args, machine, WORKER, [json.dumps({'argv': argv, 'cwd': cwd, 'timeout': args.timeout})]))
                meta.update(commands=benchmark_commands, launch_commands=commands, physical_ranks=physical)
                dump(directory / 'metadata.json', meta)
                outcome = run_processes(commands, directory, args.timeout, monitor.check if monitor else None)
                meta.update(outcome)
                meta['machines'] = []
                for rank in range(len(outcome['exit_codes'])):
                    for line in (directory / f'rank{rank}.stderr').read_text().splitlines():
                        if line.startswith('TBCCL_WORKER '):
                            meta['machines'].append(json.loads(line[len('TBCCL_WORKER '):]))
                if not outcome['success']:
                    raise RuntimeError(outcome['error'])
                result = parse_result(directory, config, args.loopback)
                results.append(result)
                write_csv(args.output / 'runs.csv', results)
                write_csv(args.output / 'summary.csv', aggregate(results))
            except BaseException as error:
                meta.update(success=False, error=str(error))
                raise
            finally:
                dump(directory / 'metadata.json', meta)
                if not args.loopback:
                    post = capture_health(args)
                    dump(directory / 'health-after.json', post)
                    deltas = {p: health.compare(pre[p], post[p]) for p in pre}
                    dump(directory / 'health-delta.json', deltas)
                    stop = [reason for delta in deltas.values() for reason in delta['stop_reasons']]
                    pre = post
                    if stop:
                        raise RuntimeError('STOP: ' + '; '.join(stop))
    except (Exception, KeyboardInterrupt) as error:
        failure = str(error) or type(error).__name__
        print(f'STOP: {failure}', file=sys.stderr)
    finally:
        if monitor:
            monitor.close()
        if not args.loopback:
            dump(args.output / 'health-final.json', capture_health(args))
        dump(args.output / 'status.json', {'success': failure is None, 'error': failure,
                                          'completed': len(results), 'planned': len(planned)})
    return int(failure is not None)


def main():
    p = parser()
    args = p.parse_args()
    try:
        return run(args)
    except (ValueError, OSError) as error:
        p.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
