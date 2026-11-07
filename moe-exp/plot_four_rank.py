#!/usr/bin/env python3
"""Four-rank raw-clock-corrected TX, RX, and collective entry/exit plots.

Input layout after scp: <root>/node0..node3 each contain events/semantics,
TX/RX raw counter CSV.  Node0 collected <sync>/node{1,2,3}_{pre,post}.csv.
Do not interpret synchronous sending as switch congestion without queue data.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from burst_analysis import (application_events, semantic_intervals,
                            sample_clock_model, to_ref, nic_bins_aligned)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True,help='parent directory with node0..node3')
    ap.add_argument('--sync',required=True,help='directory with nodeR_pre.csv and nodeR_post.csv')
    ap.add_argument('--iteration',type=int,default=0)
    ap.add_argument('--bin-us',type=int,default=500)
    ap.add_argument('--output-dir',default=None)
    args=ap.parse_args()
    root=Path(args.root);sync=Path(args.sync)
    out=Path(args.output_dir) if args.output_dir else root/'figures_4rank'
    out.mkdir(parents=True,exist_ok=True)
    models={0:None}
    for rank in [1,2,3]:
        models[rank]=sample_clock_model(sync/f'node{rank}_pre.csv',sync/f'node{rank}_post.csv')
    app={r:application_events(root/f'node{r}'/f'events_rank{r}.csv') for r in range(4)}
    sem={r:semantic_intervals(root/f'node{r}'/f'moe_semantic_trace_rank{r}.csv') for r in range(4)}
    orig=app[0][args.iteration]['iter_start'][0]
    last=app[0][args.iteration]['iter_end'][0]
    left=orig-2_000_000;right=last+2_000_000
    bin_ns=args.bin_us*1000
    edges=np.arange(left,right+2*bin_ns,bin_ns,dtype=np.int64)
    time=(edges[:-1]-orig)/1e6
    report=[]
    with open(out/'clock_offset_summary.csv','w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['rank','pre_theta_ns','post_theta_ns','drift_ppm','rough_max_half_rtt_us','note'])
        for r,m in models.items():
            if m is None:
                writer.writerow([r,0,0,0,0,'Reference clock'])
            else:
                writer.writerow([r,round(m['pre'][1]),round(m['post'][1]),
                                 round(m['drift_ns_per_ns']*1e6,5),
                                 round(m['rough_rtt_half_us'],3),
                                 'RTT/2 is not guaranteed synchronization accuracy'])
    for traffic in ['tx','rx']:
        fig,ax=plt.subplots(figsize=(14.5,5.5))
        for r in range(4):
            rate,coverage,stats=nic_bins_aligned(root/f'node{r}'/f'{traffic}.csv',edges,models[r])
            ax.step(time,rate,where='post',linewidth=1.1,label=f'Rank {r}')
            report.append((traffic,r,round(float(np.mean(coverage>=.95)),4),stats))
        ax.grid(alpha=.2)
        ax.legend(ncol=4,loc='upper right')
        ax.set_xlabel('Node0 CLOCK_MONOTONIC_RAW time relative to Node0 iteration start (ms)')
        ax.set_ylabel(f'Estimated {traffic.upper()} Gbps')
        ax.set_title(f'4-rank {traffic.upper()} · Iteration {args.iteration} · corrected clock offset (uncertainty applies)')
        fig.tight_layout()
        output=out/f'4rank_{traffic}_iter{args.iteration}.png'
        fig.savefig(output,dpi=160);plt.close(fig)
        print('[plot]',output)
    # Collective Entry/Exit in same reference domain
    choices={'dispatch_a2a','return_a2a','bwd_return_a2a','bwd_dispatch_a2a','grad_allreduce_nonexpert'}
    fig,ax=plt.subplots(figsize=(14,7.8))
    ticks=[]; labels=[]; i=0
    for r in range(4):
        entries=[]
        for s in sem[r][args.iteration]:
            if s['name'] not in choices:continue
            a=float(to_ref(s['start'],models[r])); b=float(to_ref(s['end'],models[r]))
            entries.append((a,b,s['name'],s['layer']))
        entries.sort()
        for a,b,name,layer in entries:
            if b<=a:continue
            ax.broken_barh([((a-orig)/1e6,(b-a)/1e6)],(i-.36,.72),alpha=.82)
            if (b-a)>=500000:
                ax.text(((a+b)/2-orig)/1e6,i,name,ha='center',va='center',fontsize=7)
            ticks.append(i);labels.append(f'R{r} L{layer} {name}');i+=1
        i+=1
    ax.set_yticks(ticks,labels,fontsize=8)
    ax.set_ylim(i-.5,-1)
    ax.grid(axis='x',alpha=.25)
    ax.set_xlabel('Node0 reference time relative to Node0 iteration start (ms)')
    ax.set_title(f'4-rank collective entry/exit intervals · Iteration {args.iteration} · offset-corrected')
    fig.tight_layout()
    path=out/f'4rank_collectives_iter{args.iteration}.png'
    fig.savefig(path,dpi=160);plt.close(fig)
    print('[plot]',path)
    for traffic,r,fraction,stats in report:
        print(f'[NIC] {traffic.upper()} rank={r} valid_bins={fraction*100:.1f}% {stats}')
    print('[clock] clock_offset_summary.csv includes a rough RTT/2 diagnostic; do not treat it as a verified clock-accuracy bound.')


if __name__=='__main__':main()
