#!/usr/bin/env python3
"""Node0 controller: SSH deployment, warmup gate, optional probe, result checks."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import subprocess as sp
import time
import uuid


def command(args, **kw):
    return sp.run(args, check=True, text=True, timeout=30, **kw)


def ssh(host, args, **kw):
    return command(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                    host, shlex.join([str(x) for x in args])], **kw)


def resolve_counter(ifname, counter):
    """Derive the default only when --counter was omitted."""
    return Path(f'/sys/class/net/{ifname}/statistics/tx_bytes') if counter is None else counter


def read_timings(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def check_trace(path, timing):
    """Streaming check: no byte conversion or resampling, align same-host RAW only."""
    anchor = None
    stop = None
    first = last = None
    valid = invalid = decreases = 0
    previous = None
    with path.open() as f:
        for line in f:
            if line.startswith('# anchor_start '):
                anchor = int(re.search(r'mono_raw_ns=(\d+)', line).group(1))
            elif line.startswith('# stop_reason='):
                stop = re.search(r'stop_reason=(\w+)', line).group(1)
            elif line.startswith('#') or line.startswith('ta_ns,'):
                continue
            else:
                ta, tb, value, ok = map(int, line.strip().split(','))
                if not ok:
                    invalid += 1
                    continue
                if previous is not None and value < previous:
                    decreases += 1
                previous = value
                if first is None:
                    first = (ta, tb, value)
                last = (ta, tb, value)
                valid += 1
    if anchor is None or first is None or not timing:
        raise RuntimeError('Probe has no valid coverage or missing metadata')
    # Require a completed sample before the first barrier, and a sample begun
    # after the last measured collective. No cross-node clock conversion.
    covered = (anchor + first[1] <= int(timing[0]['barrier_start_raw_ns']) and
               anchor + last[0] >= int(timing[-1]['end_raw_ns']))
    result = dict(anchor_raw_ns=anchor, valid=valid, invalid=invalid,
                  counter_decreases=decreases, stop_reason=stop,
                  total_delta_raw=last[2]-first[2], full_timing_coverage=covered)
    if not covered or invalid or decreases or stop == 'buffer_full':
        raise RuntimeError(f'Probe trace failed acceptance checks: {result}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['smoke', 'off', 'on'])
    p.add_argument('--hosts', nargs='+', default=['node0', 'node1'])
    p.add_argument('--master', default='10.10.1.1')
    p.add_argument('--port', type=int, default=29531)
    p.add_argument('--ifname', default='enp65s0f0np0')
    p.add_argument('--bench-cpu', type=int, default=10)
    p.add_argument('--probe-cpu', type=int, default=6)
    p.add_argument('--peer-mib', type=int, default=32)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--gap-ms', type=float, default=200)
    p.add_argument('--guard-ms', type=float, default=20)
    p.add_argument('--probe', type=Path, default=Path.home() / 'burst/counter_probe')
    p.add_argument('--counter', type=Path, default=None,
                   help='Default: /sys/class/net/IFNAME/statistics/tx_bytes; explicit path is preserved')
    p.add_argument('--results', type=Path, default=Path.home() / 'results')
    p.add_argument('--run-timeout', type=int, default=240)
    a = p.parse_args()
    a.counter = resolve_counter(a.ifname, a.counter)
    if not 2 <= len(a.hosts) <= 4 or len(set(a.hosts)) != len(a.hosts):
        p.error('Supply 2..4 distinct nodes, with this node first')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+', h) or h.startswith('-') for h in a.hosts):
        p.error('Use plain SSH host aliases/hostnames (no username or options)')
    if a.mode == 'smoke':
        a.peer_mib, a.iters = 1, 2
    if a.peer_mib <= 0 or a.iters <= 0 or a.run_timeout < 30:
        p.error('Positive sizes/counts and timeout >=30 are required')
    if not 0 < a.port < 65536 or min(a.bench_cpu, a.probe_cpu) < 0:
        p.error('Invalid port or CPU')
    root = Path(__file__).resolve().parent
    run_id = time.strftime('%Y%m%d_%H%M%S') + '_' + a.mode + '_' + uuid.uuid4().hex[:6]
    out = a.results.expanduser().resolve() / run_id
    out.mkdir(parents=True, exist_ok=False)
    remote = '/tmp/gloo-main-' + run_id
    processes, logs = [], []
    probe = None
    summary = dict(success=False, mode=a.mode, run_id=run_id)
    local_rank = out / 'rank0'
    local_rank.mkdir()
    probe_path = a.probe.expanduser().resolve()
    started = time.monotonic()

    def ensure_alive():
        for proc in processes:
            if proc.poll() is not None:
                raise RuntimeError('A rank exited before GO; inspect rank logs')
        if time.monotonic() - started > a.run_timeout:
            raise TimeoutError('Timed out waiting for rank readiness')

    def snapshot(name):
        with (out / name).open('w') as f:
            command(['ethtool', '-S', a.ifname], stdout=f, stderr=sp.STDOUT)

    try:
        # Confirm that rank0 is the node running this controller.
        local_machine = Path('/etc/machine-id').read_text().strip()
        remote_machine = ssh(a.hosts[0], ['cat', '/etc/machine-id'], capture_output=True).stdout.strip()
        if local_machine != remote_machine:
            raise RuntimeError('Run this controller on hosts[0], normally node0')
        with (out / 'preflight.txt').open('w') as f:
            for host in a.hosts:
                f.write(f'\nHOST {host}\n'); f.flush()
                ssh(host, ['uname', '-r'], stdout=f)
                ssh(host, ['ip', '-br', 'addr', 'show', 'dev', a.ifname], stdout=f)
                ssh(host, ['ethtool', '-i', a.ifname], stdout=f)
                ssh(host, ['python3', '-c', 'import torch; print(torch.__version__)'], stdout=f)
                ssh(host, ['ip', 'route', 'get', a.master], stdout=f)
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
        (out / 'config.json').write_text(json.dumps(config, indent=2))
        if a.mode == 'on':
            if not os.access(probe_path, os.X_OK):
                raise RuntimeError(f'Probe is not executable: {probe_path}')
            with a.counter.open() as f:
                int(f.read().strip())  # No sudo needed for a readable sysfs counter.
        for rank, host in enumerate(a.hosts):
            dest = local_rank if rank == 0 else Path(remote) / f'rank{rank}'
            if rank:
                ssh(host, ['mkdir', '-p', str(dest)])
                command(['scp', '-q', str(root / 'coll_bench.py'), f'{host}:{remote}/coll_bench.py'])
            worker = str(root / 'coll_bench.py') if rank == 0 else remote + '/coll_bench.py'
            args = ['python3', '-u', worker, '--rank', str(rank), '--world', str(len(a.hosts)),
                    '--master', a.master, '--port', str(a.port), '--ifname', a.ifname,
                    '--cpu', str(a.bench_cpu), '--peer-mib', str(a.peer_mib),
                    '--iters', str(a.iters), '--gap-ms', str(a.gap_ms),
                    '--guard-ms', str(a.guard_ms), '--out', str(dest)]
            # Remote timeout also bounds a worker left behind by a lost SSH connection.
            if rank:
                args = ['ssh', '-o', 'BatchMode=yes', host,
                        shlex.join(['timeout', '--signal=TERM', '--kill-after=10s',
                                    str(a.run_timeout + 30)] + args)]
            f = (out / f'rank{rank}.log').open('w'); logs.append(f)
            processes.append(sp.Popen(args, stdout=f, stderr=sp.STDOUT, start_new_session=True))
        for rank, host in enumerate(a.hosts):
            while True:
                ensure_alive()
                if rank == 0:
                    ready = (local_rank / 'READY').exists()
                else:
                    ready = sp.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5', host,
                        shlex.join(['test', '-f', f'{remote}/rank{rank}/READY'])],
                        timeout=10).returncode == 0
                if ready:
                    break
                time.sleep(0.2)
        print('All ranks warmed up and validated. Starting measurement window.', flush=True)
        snapshot('ethtool_before.txt')
        if a.mode == 'on':
            f = (out / 'probe.log').open('w'); logs.append(f)
            probe = sp.Popen([str(probe_path), str(a.counter), str(a.probe_cpu),
                              str(a.run_timeout), str(out / 'counter.csv')],
                             stdout=f, stderr=sp.STDOUT, start_new_session=True)
        # Same lead-in for off/on. Old probe has no READY flag; prove coverage later.
        time.sleep(3)
        if probe is not None and probe.poll() is not None:
            raise RuntimeError('Probe exited during preparation; inspect probe.log')
        # GO travels via SSH/control network; rank0 begins only after remote GO.
        for rank, host in enumerate(a.hosts[1:], 1):
            ssh(host, ['touch', f'{remote}/rank{rank}/GO'])
        (local_rank / 'GO').touch()
        while any(proc.poll() is None for proc in processes):
            if any(proc.poll() not in (None, 0) for proc in processes):
                raise RuntimeError('Collective worker failed; inspect rank logs')
            if probe is not None and probe.poll() is not None:
                raise RuntimeError('Probe ended before workers; possible buffer limit')
            if time.monotonic() - started > a.run_timeout:
                raise TimeoutError('Experiment timeout')
            time.sleep(0.1)
        if any(proc.returncode != 0 for proc in processes):
            raise RuntimeError('Worker failed')
        time.sleep(2)
        if probe is not None:
            if probe.poll() is not None:
                raise RuntimeError('Probe ended unexpectedly; inspect probe.log')
            probe.send_signal(signal.SIGINT)
            probe.wait(timeout=180)  # Wait for CSV flush BEFORE any next run.
            if probe.returncode != 0:
                raise RuntimeError('Probe save failed')
        snapshot('ethtool_after.txt')
        for rank, host in enumerate(a.hosts[1:], 1):
            command(['scp', '-q', '-r', f'{host}:{remote}/rank{rank}', str(out)])
        by_rank = []
        versions = set()
        for rank in range(len(a.hosts)):
            meta = json.loads((out / f'rank{rank}/metadata.json').read_text())
            timing = read_timings(out / f'rank{rank}/timing.csv')
            if not meta['success'] or len(timing) != a.iters:
                raise RuntimeError(f'Incomplete rank {rank}')
            versions.add(meta['torch_version'])
            values = [int(row['duration_ns']) / 1e6 for row in timing]
            by_rank.append(dict(rank=rank, median_ms=statistics.median(values),
                                min_ms=min(values), max_ms=max(values),
                                mean_ms=statistics.mean(values)))
        if len(versions) != 1:
            raise RuntimeError('PyTorch versions differ across ranks')
        summary.update(per_rank=by_rank,
                       measured_remote_payload_per_rank=a.iters * (len(a.hosts)-1) * a.peer_mib * 1024**2)
        if a.mode == 'on':
            summary['probe'] = check_trace(out / 'counter.csv', read_timings(local_rank / 'timing.csv'))
        summary['success'] = True
    except BaseException as exc:
        summary['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        # Only signal processes launched by this run, never broad pkill commands.
        if probe is not None and probe.poll() is None:
            probe.send_signal(signal.SIGINT)
            try:
                probe.wait(timeout=180)
            except sp.TimeoutExpired:
                os.killpg(probe.pid, signal.SIGKILL)
                probe.wait()
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except sp.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        for f in logs:
            f.close()
        (out / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(f'Results: {out}', flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
