#!/usr/bin/env python3
"""Host-clock offset estimation via application 4-timestamp TCP exchange.

All clock readings use CLOCK_MONOTONIC_RAW. Offsets are measurement estimates,
NOT a hard-synchronized clock. Designed for pre/post training checks over the
CONTROL NIC (so the experiment-NIC TX/RX samples exclude sync packets).

Remote server: python clock_sync_raw.py server --bind 128.110.x.x --port 29790
Node0 client:  python clock_sync_raw.py client --peer 128.110.x.x --output sync_pre_node1.csv
"""
import argparse
import csv
import os
import socket
import struct
import time

Q = struct.Struct('!Q')
QQQ = struct.Struct('!QQQ')


def raw_ns():
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


def recv_exact(sock, n):
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            if not out:
                return None
            raise ConnectionError('peer disconnected mid-message')
        out.extend(chunk)
    return bytes(out)


def server(args):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as ss:
        ss.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        ss.bind((args.bind, args.port))
        ss.listen(8)
        print(f'[raw-clock server] listening at {args.bind}:{args.port}', flush=True)
        while True:
            conn, address = ss.accept()
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.settimeout(10)
                print(f'[raw-clock server] connection from {address}', flush=True)
                try:
                    while True:
                        message = recv_exact(conn, Q.size)
                        if message is None:
                            break
                        t2 = raw_ns()
                        (seq,) = Q.unpack(message)
                        t3 = raw_ns()
                        conn.sendall(QQQ.pack(seq, t2, t3))
                except (BrokenPipeError, ConnectionError, TimeoutError, socket.timeout):
                    pass


def client(args):
    host = args.peer
    with socket.create_connection((host, args.port), timeout=8) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(8)
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow(['seq', 't1_node0_ns', 't2_peer_ns', 't3_peer_ns',
                             't4_node0_ns', 'reference_mid_ns', 'peer_mid_ns',
                             'theta_peer_minus_node0_ns', 'roundtrip_adjusted_ns',
                             'half_rtt_ns'])
            for i in range(args.samples):
                t1 = raw_ns()
                conn.sendall(Q.pack(i))
                message = recv_exact(conn, QQQ.size)
                t4 = raw_ns()
                if message is None:
                    raise ConnectionError('remote server closed connection')
                seq, t2, t3 = QQQ.unpack(message)
                if seq != i:
                    raise RuntimeError('sequence mismatch')
                # NTP-style estimate assumes approximately symmetric one-way delays.
                theta = ((t2-t1) + (t3-t4)) / 2.0
                rtt = (t4-t1) - (t3-t2)
                writer.writerow([i,t1,t2,t3,t4,(t1+t4)//2,(t2+t3)//2,
                                 round(theta),rtt,max(0,rtt)/2])
                if args.interval_ms:
                    time.sleep(args.interval_ms/1000)
    print(f'[raw-clock] saved {args.samples} offset samples to {args.output}')
    import statistics
    with open(args.output, newline='') as f:
        rows = list(csv.DictReader(f))
    rows.sort(key=lambda x: int(x['roundtrip_adjusted_ns']))
    k=max(5, min(len(rows), int(len(rows)*0.1)))
    best=rows[:k]
    print(f'[raw-clock] lowest-RTT 10% median theta={statistics.median(int(r["theta_peer_minus_node0_ns"]) for r in best)/1000:.3f} us, '
          f'RTT median={statistics.median(int(r["roundtrip_adjusted_ns"]) for r in best)/1000:.3f} us')
    print('NOTE: half-RTT is a rough delay bound under symmetric-path assumptions, NOT a guaranteed offset error bound.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    sub=ap.add_subparsers(dest='mode', required=True)
    ps=sub.add_parser('server')
    ps.add_argument('--bind', required=True, help='control NIC IPv4 address')
    ps.add_argument('--port', type=int, default=29790)
    pc=sub.add_parser('client')
    pc.add_argument('--peer', required=True, help='control NIC IPv4 address of server')
    pc.add_argument('--port', type=int, default=29790)
    pc.add_argument('--samples', type=int, default=300)
    pc.add_argument('--interval-ms', type=float, default=2)
    pc.add_argument('--output', required=True)
    a=ap.parse_args()
    if a.mode=='server': server(a)
    else: client(a)


if __name__=='__main__':
    main()
