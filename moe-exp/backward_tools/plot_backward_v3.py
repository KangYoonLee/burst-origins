#!/usr/bin/env python3
"""Plot backward gradient checkpoints, DeepSpeed MoE and NIC traffic.

Expected directory structure:
  baseline02/burst_analysis.py
  baseline02/node0/events_rank0.csv
  baseline02/node0/moe_semantic_trace_rank0.csv
  baseline02/node0/backward_tensor_trace_rank0.csv
  baseline02/node0/tx.csv,rx.csv

Usage:
  python plot_backward_v3.py --root . --rank 0 --iteration 0 --bin-us 250

No cross-node timestamp alignment needed: the figure is single-node.
NOTE: gradient readiness hooks are checkpoints, NOT exact CPU kernel
execution start/end or disjoint pure-compute/communication/wait durations.
"""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLORS={
 'tx':'#18669f','rx':'#d97936',
 'bwd_return_a2a':'#d6a14f','bwd_dispatch_a2a':'#c16a4e',
 'expert_bwd':'#489878','router_bwd':'#846aae',
 'grad_allreduce_nonexpert':'#6960ba','grad_allreduce_expert':'#a49bcc',
 'engine_autograd':'#89a5c5','engine_epilogue':'#a989c5',
}
SHORT={
 'bwd_return_a2a':'Backward return A2A','bwd_dispatch_a2a':'Backward dispatch A2A',
 'expert_bwd':'Expert backward','router_bwd':'Router backward',
 'grad_allreduce_nonexpert':'Non-expert gradient AllReduce',
 'grad_allreduce_expert':'Expert gradient AllReduce'
}
EVENT_LABELS={
 'logits_grad':'LM head output gradient',
 'final_norm_output_grad':'Final Norm output gradient',
 'layers_output_grad':'Layer stack output gradient',
 'embedding_output_grad':'Embedding output gradient',
}
for i in (0,1):
    EVENT_LABELS.update({
       f'layer{i}_output_grad':f'Layer {i} output gradient',
       f'layer{i}_moe_output_grad':f'Layer {i} MoE output gradient',
       f'layer{i}_after_attention_grad':f'Layer {i} attention output gradient',
       f'layer{i}_input_grad':f'Layer {i} input gradient',
    })


def get_grad_rows(path,iteration):
    if not path.is_file():
        raise FileNotFoundError(f'Missing new backward trace: {path}. Run the patched training first.')
    with path.open(newline='') as f:
        return sorted([dict(name=r['event'],time=int(r['t_mono_raw_ns']))
                       for r in csv.DictReader(f) if int(r['iter'])==iteration],
                      key=lambda r:r['time'])


def write_events(path,orig,app,sem,grad):
    events=[('application','backward_start',app['backward_start'][0]),
            ('application','backward_end',app['backward_end'][0])]
    for e in sem:
        if e['name'].startswith('engine_'):
            events.extend([('DeepSpeed',e['name']+'_start',e['start']),
                           ('DeepSpeed',e['name']+'_end',e['end'])])
    events.extend([('gradient_checkpoint',e['name'],e['time']) for e in grad])
    events.sort(key=lambda x:x[2])
    with path.open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['source','event','t_mono_raw_ns','ms_relative_to_iteration_start','delta_from_previous_checkpoint_ms'])
        prev=None
        for source,name,t in events:
            w.writerow([source,name,t,f'{(t-orig)/1e6:.6f}',
                        '' if prev is None else f'{(t-prev)/1e6:.6f}'])
            prev=t


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',default='.')
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--iteration',type=int,default=0)
    p.add_argument('--bin-us',type=int,default=250)
    p.add_argument('--outdir',default=None)
    args=p.parse_args()
    root=Path(args.root).expanduser().resolve()
    sys.path.insert(0,str(root))
    from burst_analysis import application_events,semantic_intervals,nic_bins_aligned
    node=root/f'node{args.rank}'
    app=application_events(node/f'events_rank{args.rank}.csv')[args.iteration]
    sem=semantic_intervals(node/f'moe_semantic_trace_rank{args.rank}.csv')[args.iteration]
    grad=get_grad_rows(node/f'backward_tensor_trace_rank{args.rank}.csv',args.iteration)
    if not grad:
        raise RuntimeError('No gradient callbacks recorded for this iteration. Verify patched actual_moe_train.py was executed.')
    origin=app['iter_start'][0];bstart=app['backward_start'][0];bend=app['backward_end'][0]
    print('Backward', (bend-bstart)/1e6, 'ms; gradient boundary checkpoints',len(grad))
    eg={e['name']:e for e in sem if e['name'] in ('engine_autograd','engine_epilogue')}
    for name in ('engine_autograd','engine_epilogue'):
        if name not in eg:raise RuntimeError(f'Missing {name} from semantic trace. Check engine.py/trace.py patch installation.')
    bw=args.bin_us*1000
    edges=np.arange(int(bstart)-2*bw,int(bend)+3*bw,bw,dtype=np.int64)
    xs=(edges[:-1]-origin)/1e6
    rates={}
    for direction in ('tx','rx'):
        rates[direction],cov,st=nic_bins_aligned(node/f'{direction}.csv',edges,None)
        print(direction,'coverage_min',float(cov.min()),'long_gaps',st['long_gaps'])
    out=(Path(args.outdir) if args.outdir else root/'backward_figures_v3').resolve();out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans',
                         'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(16,12),layout='constrained')
    gs=fig.add_gridspec(4,1,height_ratios=[2.2,.9,2.1,3.8],hspace=.14)
    a=fig.add_subplot(gs[0]);b=fig.add_subplot(gs[1],sharex=a)
    c=fig.add_subplot(gs[2],sharex=a);d=fig.add_subplot(gs[3],sharex=a)
    a.step(xs,rates['tx'],where='post',color=COLORS['tx'],label='TX',lw=1.5)
    a.step(xs,rates['rx'],where='post',color=COLORS['rx'],label='RX',lw=1.5)
    a.set_ylabel('Estimated NIC rate\n(Gbps)');a.legend(frameon=False,ncol=2)
    a.set_title(f'Rank {args.rank} · Iteration {args.iteration} — Where does backward time go?',
                loc='left',fontweight='bold',fontsize=15)
    a.grid(alpha=.16)
    for j,name in enumerate(('engine_autograd','engine_epilogue')):
        e=eg[name]
        x=(e['start']-origin)/1e6;dur=(e['end']-e['start'])/1e6
        b.broken_barh([(x,dur)],(j-.28,.56),facecolors=COLORS[name])
        if dur>.8:b.text(x+dur/2,j,f'{dur:.2f}ms',ha='center',va='center',fontsize=9)
    b.set_yticks([0,1],['Autograd execution','DeepSpeed epilogue']);b.set_ylim(1.7,-.65)
    b.grid(axis='x',alpha=.15)
    tracked=[e for e in sem if e['name'] in SHORT]
    rows={('bwd',1):0,('bwd',0):1,('allreduce',-1):2}
    for e in tracked:
        if e['name'].startswith('grad_allreduce'): row=2
        else:row=0 if e['layer']==1 else 1
        x=(e['start']-origin)/1e6;dur=(e['end']-e['start'])/1e6
        c.broken_barh([(x,dur)],(row-.34,.68),facecolors=COLORS.get(e['name'],'#aaaaaa'))
    c.set_ylim(2.65,-.65);c.set_yticks([0,1,2],['DeepSpeed MoE L1','DeepSpeed MoE L0','Gradient AllReduce'])
    c.grid(axis='x',alpha=.15)
    c.legend(handles=[Patch(facecolor=COLORS[k],label=SHORT[k]) for k in SHORT],
             frameon=False,ncol=3,fontsize=8,loc='upper right')
    expected=['logits_grad','final_norm_output_grad','layers_output_grad',
              'layer1_output_grad','layer1_moe_output_grad','layer1_after_attention_grad','layer1_input_grad',
              'layer0_output_grad','layer0_moe_output_grad','layer0_after_attention_grad','layer0_input_grad',
              'embedding_output_grad']
    byname={e['name']:e['time'] for e in grad}
    ordered=[name for name in expected if name in byname]
    for j,name in enumerate(ordered):
        x=(byname[name]-origin)/1e6
        d.plot(x,j,marker='o',ms=7,color='#305b83',zorder=5)
        d.hlines(j,(bstart-origin)/1e6,x,color='#dfe5eb',lw=.7,zorder=1)
    d.set_yticks(np.arange(len(ordered)),[EVENT_LABELS.get(n,n) for n in ordered],fontsize=9)
    d.set_ylim(len(ordered)-.3,-.7);d.grid(axis='x',alpha=.15)
    d.set_ylabel('Gradient readiness\ncheckpoints')
    # Dots report when a gradient becomes available; not when an operator executes exclusively.
    d.set_xlabel(f'Time relative to Rank {args.rank} iteration start (ms)')
    for ax in (a,b,c):ax.tick_params(labelbottom=False)
    for ax in (a,b,c,d):ax.set_xlim((bstart-origin)/1e6-.4,(bend-origin)/1e6+.4)
    fig.suptitle('Gradient checkpoints ≠ precise operator execution intervals; collective bars include possible waiting.\n'
                 'Gloo/TCP, counter-derived NIC rates; not hardware packet timestamps.',
                 fontsize=10,color='#475569')
    stem=f'backward_rank{args.rank}_iter{args.iteration}'
    fig.savefig(out/f'{stem}.png',dpi=190,bbox_inches='tight')
    fig.savefig(out/f'{stem}.svg',bbox_inches='tight')
    plt.close(fig)
    write_events(out/f'{stem}_checkpoint_sequence.csv',origin,app,sem,grad)
    print('SAVED',out/f'{stem}.png')
    print('SAVED',out/f'{stem}_checkpoint_sequence.csv')

if __name__=='__main__':main()
