#!/usr/bin/env python3
"""Offline utilities for CLOCK_MONOTONIC_RAW MoE and NIC data.

Counter-rate estimation spreads each observed delta uniformly between two
read timestamps; this is explicitly an interpolation, NOT packet timing.
A bin is plotted as NaN if time coverage <95% or if its longest sample
interval is >2 bins; coverage masking avoids zero == missing ambiguity.
"""
import csv
import re
from collections import defaultdict
from pathlib import Path
import numpy as np


def application_events(path):
    result=defaultdict(lambda:defaultdict(list))
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            result[int(row['iter'])][row['event']].append(int(row['t_mono_raw_ns']))
    return result


def semantic_intervals(path):
    by_iter=defaultdict(list)
    starts=defaultdict(list)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            i=int(row['iteration']); event=row['event']; layer=int(row['layer_id']); cid=int(row['collective_id']); t=int(row['t_mono_raw_ns'])
            key=(i,event[:-6],layer,cid) if event.endswith('_start') else (i,event[:-4],layer,cid)
            if event.endswith('_start'):
                starts[key].append((t,row))
            elif event.endswith('_end') and starts[key]:
                begin,first=starts[key].pop(0)
                if t < begin:
                    raise ValueError(f'end before start: {key}')
                by_iter[i].append({'name':event[:-4],'layer':layer,'cid':cid,
                                   'start':begin,'end':t,'bytes':int(first['bytes']),
                                   'group_size':int(first['group_size'])})
    return by_iter


def app_range(events, name):
    a=events.get(f'{name}_start',[])
    b=events.get(f'{name}_end',[])
    return (min(a),max(b)) if a and b else None


def sample_clock_model(pre_file,post_file):
    """Estimate peer_minus_reference offset as an affine function of PEER time."""
    def get(path):
        with open(path, newline='') as f: rows=list(csv.DictReader(f))
        if len(rows)<10: raise ValueError(f'too few samples in {path}')
        rows.sort(key=lambda r:float(r['roundtrip_adjusted_ns']))
        # Low-RTT subset reduces scheduling tails, but cannot remove network asymmetry.
        keep=rows[:max(5,int(len(rows)*.10))]
        t=float(np.median([int(r['peer_mid_ns']) for r in keep]))
        theta=float(np.median([int(r['theta_peer_minus_node0_ns']) for r in keep]))
        low_rtt=float(np.median([float(r['roundtrip_adjusted_ns']) for r in keep]))
        spread=float(np.ptp([int(r['theta_peer_minus_node0_ns']) for r in keep]))
        return t, theta, low_rtt, spread
    pre=get(pre_file); post=get(post_file)
    if post[0]<=pre[0]: raise ValueError('post sync must be after pre sync')
    drift=(post[1]-pre[1])/(post[0]-pre[0])
    return dict(pre=pre,post=post,drift_ns_per_ns=drift,
                rough_rtt_half_us=max(pre[2],post[2])/2000)


def to_ref(ts,model):
    if model is None: return ts
    t0,theta0,*_=model['pre']
    off=theta0 + model['drift_ns_per_ns']*(np.asarray(ts,dtype=np.float64)-t0)
    return np.asarray(ts,dtype=np.float64)-off


def nic_bins_aligned(path,edges,model=None,max_gap_factor=2.0):
    edges=np.asarray(edges,dtype=np.float64)
    bw=edges[1]-edges[0]
    out=np.zeros(len(edges)-1,dtype=float)
    cov=np.zeros(len(edges)-1,dtype=float)
    anchor=None; prev=None
    stats={'failures':0,'resets':0,'long_gaps':0,'samples':0,'max_gap_us':0.0,'stop_reason':None}
    with open(path,buffering=1024*1024) as f:
        for line in f:
            if line.startswith('#'):
                mt=re.search(r'anchor_start realtime_ns=\d+ mono_raw_ns=(\d+)',line)
                if mt: anchor=int(mt.group(1))
                mr=re.search(r'stop_reason=(\w+)',line)
                if mr: stats['stop_reason']=mr.group(1)
                continue
            if line.startswith('ta_ns,'): continue
            if not line.strip():continue
            if anchor is None:raise ValueError(f'No raw clock anchor in {path}')
            ta,tb,val,ok=line.strip().split(',')
            stats['samples']+=1
            if ok!='1':
                stats['failures']+=1;prev=None;continue
            ts=anchor+(int(ta)+int(tb))//2
            t=float(to_ref(ts,model)); v=int(val)
            if prev is None:
                prev=(t,v);continue
            p,pv=prev
            prev=(t,v)
            dt=t-p;d=v-pv
            if d<0:
                stats['resets']+=1;prev=None;continue
            if dt<=0:
                stats['resets']+=1;prev=None;continue
            stats['max_gap_us']=max(stats['max_gap_us'],dt/1000)
            if dt>max_gap_factor*bw:
                stats['long_gaps']+=1
                continue
            if t<=edges[0] or p>=edges[-1]:continue
            a=max(p,edges[0]);b=min(t,edges[-1]);
            if b<=a:continue
            lo=max(0,int(np.searchsorted(edges,a,side='right')-1))
            hi=min(len(out)-1,int(np.searchsorted(edges,b,side='left')))
            for j in range(lo,hi+1):
                length=max(0.0,min(b,edges[j+1])-max(a,edges[j]))
                if length:
                    cov[j]+=length
                    out[j]+=d*length/dt
    coverage=np.clip(cov/bw,0,1)
    gbps=out*8/bw # byte/ns *8 = Gbit/s
    gbps[coverage<0.95]=np.nan
    return gbps,coverage,stats


def union_ns(intervals,left,right):
    seg=sorted((max(a,left),min(b,right)) for a,b in intervals if min(b,right)>max(a,left))
    if not seg:return 0
    merged=[]
    for a,b in seg:
        if not merged or merged[-1][1]<a:merged.append([a,b])
        else:merged[-1][1]=max(merged[-1][1],b)
    return sum(b-a for a,b in merged)
