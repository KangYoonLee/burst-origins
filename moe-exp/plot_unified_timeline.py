#!/usr/bin/env python3
"""Plot a single rank/iteration across Application, Model, MoE, Backward and NIC.

Runs offline on the saved events and semantics. NIC CSVs optional for older smoke.
Does not claim that unlabeled intervals mean idle CPU or zero network.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from burst_analysis import application_events, semantic_intervals, app_range, nic_bins_aligned, union_ns


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',required=True)
    ap.add_argument('--rank',type=int,default=0)
    ap.add_argument('--iteration',type=int,default=0)
    ap.add_argument('--bin-us',type=int,default=250)
    ap.add_argument('--output',default=None)
    args=ap.parse_args()
    base=Path(args.directory)
    app=application_events(base/f'events_rank{args.rank}.csv')
    sem=semantic_intervals(base/f'moe_semantic_trace_rank{args.rank}.csv')
    ev=app[args.iteration]
    intervals=sem[args.iteration]
    if not ev.get('iter_start') or not ev.get('iter_end'):
        raise ValueError('missing iteration boundaries')
    origin=ev['iter_start'][0];end=ev['iter_end'][0]
    left=origin-500_000;right=end+500_000
    edges=np.arange(left,right+args.bin_us*1000,args.bin_us*1000,dtype=np.int64)
    if edges[-1]<right:edges=np.append(edges,edges[-1]+args.bin_us*1000)
    f,axes=plt.subplots(6,1,figsize=(16,11.5),sharex=True,gridspec_kw={'height_ratios':[2.2,1,1.35,1.8,1.7,.75]})
    nic_ax,app_ax,model_ax,moe_ax,bwd_ax,cov_ax=axes
    x=(edges[:-1]-origin)/1e6
    coverage_labels=[]
    for label,filename in [('TX','tx.csv'),('RX','rx.csv')]:
        path=base/filename
        if path.exists():
            rate,coverage,stats=nic_bins_aligned(path,edges)
            nic_ax.step(x,rate,where='post',label=f'{label} estimated Gbps',linewidth=1.0)
            cov_ax.plot(x,coverage,label=f'{label} coverage',linewidth=1.2)
            coverage_labels.append(f'{label}: {stats}')
        else:
            coverage_labels.append(f'{label}: MISSING')
    if not (base/'tx.csv').exists() or not (base/'rx.csv').exists():
        nic_ax.text(.02,.60,'NIC CSV not available (application/semantic timeline only)',transform=nic_ax.transAxes,fontsize=11)
    nic_ax.set_ylabel('NIC rate\n(Gbps)')
    if nic_ax.get_legend_handles_labels()[0]: nic_ax.legend(loc='upper right',fontsize=8)
    nic_ax.grid(alpha=.18)

    def add_bar(ax,name,interval,y,h=.7,alpha=.85):
        if interval is None:return
        a,b=interval
        if b<=a:return
        start=(a-origin)/1e6; width=(b-a)/1e6
        ax.broken_barh([(start,width)],(y-h/2,h),alpha=alpha)
        if width>=0.7:
            ax.text(start+width/2,y,name,ha='center',va='center',fontsize=8)

    for y,name in enumerate(['forward','loss','backward','step']):
        add_bar(app_ax,name,app_range(ev,name),y)
    app_ax.set_yticks(range(4),['Forward','Loss','Backward','Step']);app_ax.set_ylim(-.6,3.6);app_ax.invert_yaxis()
    app_ax.set_title('Application phases (outer boundaries)',loc='left',fontsize=10)

    mods=[('embedding',0),('layer0_attention',1),('layer0_moe',2),('layer1_attention',3),('layer1_moe',4),('lm_head',5)]
    for name,y in mods:add_bar(model_ax,name,app_range(ev,name),y)
    model_ax.set_yticks(list(range(6)),['Embedding','Attention L0','App MoE L0','Attention L1','App MoE L1','LM Head'])
    model_ax.set_ylim(-.6,5.6);model_ax.invert_yaxis()
    model_ax.set_title('Model events (App MoE includes residual/norm)',loc='left',fontsize=10)

    fwd_types=['router','dispatch_prep','dispatch_a2a','expert','return_a2a','combine']
    for it in intervals:
        if it['name'] in fwd_types:
            y=fwd_types.index(it['name'])+7*it['layer']
            add_bar(moe_ax,it['name'],(it['start'],it['end']),y,h=.72)
    labels=[f'L0 {a}' for a in fwd_types]+['']+[f'L1 {a}' for a in fwd_types]
    vals=list(range(6))+[6]+list(range(7,13))
    moe_ax.set_yticks(vals,labels)
    moe_ax.set_ylim(-.6,12.6);moe_ax.invert_yaxis()
    moe_ax.set_title('DeepSpeed MoE Forward (only instrumented intervals)',loc='left',fontsize=10)

    bwd_types=['bwd_return_a2a','expert_bwd','bwd_dispatch_a2a','router_bwd','grad_allreduce_nonexpert','grad_allreduce_expert']
    y_labels=[]
    y=0
    for layer in [1,0]:
        for name in bwd_types[:4]:
            for it in intervals:
                if it['name']==name and it['layer']==layer:add_bar(bwd_ax,name,(it['start'],it['end']),y,h=.7)
            y_labels.append(f'L{layer} {name}');y+=1
    for name in bwd_types[4:]:
        for it in intervals:
            if it['name']==name:add_bar(bwd_ax,name,(it['start'],it['end']),y,h=.7)
        y_labels.append(name);y+=1
    bwd_ax.set_yticks(range(y),y_labels)
    bwd_ax.set_ylim(-.6,y-.4);bwd_ax.invert_yaxis()
    bwd_ax.set_title('DeepSpeed Backward (untracked segments are not necessarily idle)',loc='left',fontsize=10)

    if not (base/'tx.csv').exists() or not (base/'rx.csv').exists():
        cov_ax.text(.02,.5,'No simultaneous NIC capture in supplied files',transform=cov_ax.transAxes)
    cov_ax.set_ylim(-.05,1.05);cov_ax.set_ylabel('Counter\ncoverage')
    cov_ax.set_xlabel('Time relative to this rank iteration start (ms)')
    if cov_ax.get_legend_handles_labels()[0]: cov_ax.legend(loc='upper right',fontsize=8)
    cov_ax.grid(alpha=.18)
    for ax in axes:
        ax.axvline(0,color='gray',linewidth=.8,alpha=.6)
        ax.axvline((end-origin)/1e6,color='gray',linewidth=.8,alpha=.6)
        ax.grid(axis='x',alpha=.13)
    nic_ax.set_xlim((left-origin)/1e6,(right-origin)/1e6)
    f.suptitle(f'Rank {args.rank} · Iteration {args.iteration} · Unified Execution and NIC Timeline',fontsize=14)
    f.tight_layout(rect=[0,0,1,.976])
    dst=Path(args.output) if args.output else base/f'unified_rank{args.rank}_iter{args.iteration}.png'
    dst.parent.mkdir(parents=True,exist_ok=True)
    f.savefig(dst,dpi=150);plt.close(f)

    # Accounting must use non-overlapping direct children, NOT nested sum.
    accounting=[]
    model_start=sorted(ev['forward_start'])[1] if len(ev['forward_start'])>=2 else None
    model_end=sorted(ev['forward_end'])[0] if len(ev['forward_end'])>=2 else None
    if model_start is not None and model_end is not None:
        children=[app_range(ev,k) for k in ['embedding','layer0','layer1','lm_head']]
        children=[p for p in children if p]
        recorded=union_ns(children,model_start,model_end)
        accounting.append(('model_forward',model_start,model_end,recorded,model_end-model_start-recorded))
    for layer in [0,1]:
        parent=next((it for it in intervals if it['name']=='moe' and it['layer']==layer),None)
        if parent:
            children=[(it['start'],it['end']) for it in intervals if it['layer']==layer and it['name'] in fwd_types]
            recorded=union_ns(children,parent['start'],parent['end'])
            accounting.append((f'deepspeed_moe_l{layer}',parent['start'],parent['end'],recorded,parent['end']-parent['start']-recorded))
    bwd=app_range(ev,'backward')
    if bwd:
        children=[(it['start'],it['end']) for it in intervals if it['name'] in bwd_types]
        recorded=union_ns(children,*bwd)
        accounting.append(('application_backward',bwd[0],bwd[1],recorded,bwd[1]-bwd[0]-recorded))
        first_bwd=min((it['start'] for it in intervals if it['name'] in bwd_types),default=None)
        if first_bwd is not None:accounting.append(('backward_start_to_first_semantic_event',bwd[0],first_bwd,0,first_bwd-bwd[0]))
    acct=dst.with_suffix('.accounting.csv')
    with open(acct,'w',newline='') as fp:
        w=csv.writer(fp);w.writerow(['scope','start_ns','end_ns','total_ms','instrumented_union_ms','unattributed_ms'])
        for label,a,b,known,unknown in accounting:
            w.writerow([label,a,b,round((b-a)/1e6,6),round(known/1e6,6),round(unknown/1e6,6)])
    print(f'[unified] {dst}');print(f'[accounting] {acct}')
    for entry in coverage_labels:print('[coverage]',entry)


if __name__=='__main__':main()
