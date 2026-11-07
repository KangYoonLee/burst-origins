#!/usr/bin/env python3
"""Check actual probe coverage, bad reads, probe deadline and trace integrity."""
import argparse
import csv
import re
from pathlib import Path
import numpy as np
from burst_analysis import application_events, semantic_intervals


def scan(path):
    anchor=None;first=last=prev_time=prev_val=None
    bad=reset=nonmono=0;delays=[];meta={}
    with open(path,buffering=1024*1024) as f:
        for line in f:
            if line.startswith('#'):
                m=re.search(r'anchor_start realtime_ns=\d+ mono_raw_ns=(\d+)',line)
                if m:anchor=int(m.group(1))
                for key in ['stop_reason','target_interval_us']:
                    m=re.search(r'\b'+key+r'=(\w+)',line)
                    if m:meta[key]=m.group(1)
                continue
            if line.startswith('ta_ns,'):continue
            if not line.strip():continue
            if anchor is None:raise RuntimeError(f'missing anchor: {path}')
            ta,tb,val,ok=line.strip().split(',')
            if ok!='1':bad+=1;prev_time=prev_val=None;continue
            ts=anchor+(int(ta)+int(tb))//2; val=int(val)
            if first is None:first=ts
            last=ts
            if prev_time is not None:
                diff=ts-prev_time
                if diff<=0:nonmono+=1
                else:delays.append(diff)
                if val<prev_val:reset+=1
            prev_time=ts;prev_val=val
    if first is None:raise RuntimeError(f'no valid samples: {path}')
    percentiles=np.percentile(delays,[50,95,99,100]) / 1000 if delays else [float('nan')]*4
    return {'first':first,'last':last,'bad':bad,'resets':reset,'nonmono':nonmono,
            'n_intervals':len(delays),'dt_us_p50_p95_p99_max':[round(float(x),3) for x in percentiles],**meta}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',required=True)
    ap.add_argument('--rank',type=int,required=True)
    a=ap.parse_args()
    folder=Path(a.directory)
    ev=application_events(folder/f'events_rank{a.rank}.csv')
    sem=semantic_intervals(folder/f'moe_semantic_trace_rank{a.rank}.csv')
    b=min(e['iter_start'][0] for e in ev.values());end=max(e['iter_end'][0] for e in ev.values())
    print(f'[events] rank={a.rank} measured_iters={len(ev)} sem_iternum={len(sem)} train_time={(end-b)/1e9:.3f}s')
    all_ok=len(ev)==len(sem)
    for direction in ['tx','rx']:
        info=scan(folder/f'{direction}.csv')
        cover=info['first']<b and info['last']>end
        no_failure=(info['bad']==0 and info['resets']==0 and info['nonmono']==0)
        stop_ok=info.get('stop_reason')!='buffer_full'
        all_ok &= cover and no_failure and stop_ok
        print(f'[{direction.upper()}] covers entire measured run={cover}; no_failures/resets={no_failure}; stop={info.get("stop_reason")}')
        print(f'           lead={(b-info["first"])/1e3:.1f} us, tail={(info["last"]-end)/1e3:.1f} us, '
              f'poll gaps p50,p95,p99,max={info["dt_us_p50_p95_p99_max"]} us')
    print('VALIDATION:', 'PASS (basic coverage checks)' if all_ok else 'FAIL (inspect errors; do not claim complete capture)')
    if not all_ok:raise SystemExit(2)


if __name__=='__main__':main()
