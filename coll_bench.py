#!/usr/bin/env python3
"""One CPU rank per node. Run via run.sh; no measurement-time file logging."""
import argparse
import csv
import json
import os
from pathlib import Path
import platform
import time
from datetime import timedelta


def raw_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def clock_anchor():
    before = raw_ns()
    realtime = time.time_ns()
    after = raw_ns()
    return dict(raw_before_ns=before, realtime_ns=realtime, raw_after_ns=after)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rank', type=int, required=True)
    p.add_argument('--world', type=int, required=True)
    p.add_argument('--master', required=True)
    p.add_argument('--port', type=int, default=29531)
    p.add_argument('--ifname', required=True)
    p.add_argument('--cpu', type=int, default=10)
    p.add_argument('--peer-mib', type=int, default=32)
    p.add_argument('--iters', type=int, default=20)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--gap-ms', type=float, default=200)
    p.add_argument('--guard-ms', type=float, default=20)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if not (2 <= a.world <= 4 and 0 <= a.rank < a.world):
        p.error('world must be 2..4 with a valid rank')
    if min(a.peer_mib, a.iters, a.warmup, a.timeout) <= 0:
        p.error('sizes, iterations, warmup and timeout must be positive')
    if not (0 <= a.gap_ms <= 10000 and 0 <= a.guard_ms <= 10000):
        p.error('gap/guard must be finite and between 0 and 10000 ms')
    os.sched_setaffinity(0, {a.cpu})
    os.environ['GLOO_SOCKET_IFNAME'] = a.ifname
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    import torch
    import torch.distributed as dist
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    metadata = dict(vars(a), out=str(a.out), torch_version=torch.__version__,
                    hostname=platform.node(), kernel=platform.release(),
                    cpu_affinity=sorted(os.sched_getaffinity(0)), backend='gloo',
                    operation='all_to_all_single', dtype='float32',
                    anchor_start=clock_anchor(), success=False)
    peer_bytes = a.peer_mib * 1024 * 1024
    metadata.update(peer_bytes=peer_bytes, input_bytes=a.world * peer_bytes,
                    remote_payload_per_iter=(a.world - 1) * peer_bytes,
                    measured_remote_payload=a.iters * (a.world - 1) * peer_bytes,
                    warmup_remote_payload=a.warmup * (a.world - 1) * peer_bytes)
    initialized = False
    try:
        dist.init_process_group('gloo', init_method=f'tcp://{a.master}:{a.port}',
                                rank=a.rank, world_size=a.world,
                                timeout=timedelta(seconds=a.timeout))
        initialized = True
        elements = peer_bytes // 4
        send = torch.empty(a.world * elements, dtype=torch.float32)
        receive = torch.empty_like(send)
        for dst in range(a.world):
            send[dst * elements:(dst + 1) * elements].fill_(a.rank * a.world + dst)
        receive.fill_(-1)  # Touch receive pages before warmup.

        def exchange():
            # Tensor API, NOT the list-based dist.all_to_all(). No fallback.
            dist.all_to_all_single(receive, send, async_op=False)

        def verify():
            for src in range(a.world):
                block = receive[src * elements:(src + 1) * elements]
                if not torch.all(block == src * a.world + a.rank).item():
                    raise RuntimeError(f'Wrong data from rank {src}')

        for _ in range(a.warmup):
            exchange()
        verify()
        dist.barrier()
        # Control files are written BEFORE timing starts, on each node's disk.
        (a.out / 'READY').write_text('warmup and full receive check passed\n')
        end_wait = time.monotonic() + a.timeout
        while not (a.out / 'GO').exists():
            if time.monotonic() > end_wait:
                raise TimeoutError('Timed out waiting for GO')
            time.sleep(0.05)

        # These are per-rank local timestamps. Never subtract clocks across nodes.
        for iteration in range(a.iters):
            b0 = raw_ns()
            dist.barrier()
            b1 = raw_ns()
            time.sleep(a.guard_ms / 1000)  # Separate barrier traffic from the call.
            t0 = raw_ns()
            exchange()
            t1 = raw_ns()
            rows.append(dict(iteration=iteration, barrier_start_raw_ns=b0,
                             barrier_end_raw_ns=b1, start_raw_ns=t0,
                             end_raw_ns=t1, duration_ns=t1-t0))
            if iteration + 1 < a.iters:
                time.sleep(a.gap_ms / 1000)
        verify()  # Outside timed collective calls.
        metadata['success'] = True
    except BaseException as exc:
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        metadata['anchor_end'] = clock_anchor()
        with (a.out / 'timing.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['iteration', 'barrier_start_raw_ns',
                'barrier_end_raw_ns', 'start_raw_ns', 'end_raw_ns', 'duration_ns'])
            writer.writeheader()
            writer.writerows(rows)
        (a.out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        if initialized:
            dist.destroy_process_group()
    print(f'rank={a.rank} success iterations={len(rows)}', flush=True)


if __name__ == '__main__':
    main()
