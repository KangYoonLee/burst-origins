#!/usr/bin/env python3
"""Add lightweight backward boundary tracing to current MoE training code.

Patches only:
- moe-exp/actual_moe_train.py (gradient-availability tensor hooks and CSV logging)
- imported deepspeed/moe/trace.py (four event IDs)
- imported deepspeed/runtime/engine.py (autograd vs epilogue brackets)

Does NOT change sharded_moe.py, collective algorithms, model parameters or optimizer.
Will refuse if source layout differs or already patched. Always backs up all three.

Usage on CloudLab (with 0.19.7 installed in ~/moe-env):
  ~/moe-env/bin/python patch_backward_trace.py --check
  ~/moe-env/bin/python patch_backward_trace.py --apply

For testing elsewhere:
  python patch_backward_trace.py --check --app ... --engine ... --trace ...
"""
import argparse
import datetime
import hashlib
import os
import pathlib
import shutil
import sys
import uuid

VERSION = 'BURST_BWD_TRACE_V1'


def replace_exact(src, old, new, label):
    count = src.count(old)
    if count != 1:
        raise ValueError(f'{label}: expected 1 match, found {count}')
    return src.replace(old, new, 1)


def patch_app(txt):
    helper = '''\n\n# BURST_BWD_TRACE_V1: lightweight gradient-availability checkpoints.\n# A tensor hook runs when its gradient becomes available; this is not\n# the exact operator start/end or pure CPU compute/waiting duration.\n_BWD_GRAD_ROWS = []\n_BWD_TRACE_ITER = -1\n_BWD_TRACE_RANK = -1\n\ndef watch_backward_gradient(tensor, name, enabled):\n    if not enabled or not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:\n        return\n    iteration = _BWD_TRACE_ITER\n    rank = _BWD_TRACE_RANK\n    def callback(grad):\n        _BWD_GRAD_ROWS.append((rank, iteration, name, now_raw_ns()))\n        return grad\n    tensor.register_hook(callback)\n\n'''
    txt=replace_exact(txt, '\n\n# ============================================================\n# One Transformer block:',helper+'\n# ============================================================\n# One Transformer block:', 'app: helper')
    txt=replace_exact(txt,
        '        self._mark(f"layer{self.layer_id}_start", record_events)\n',
        '        self._mark(f"layer{self.layer_id}_start", record_events)\n'
        '        watch_backward_gradient(x, f"layer{self.layer_id}_input_grad", record_events)\n',
        'app: layer input gradient')
    txt=replace_exact(txt,
        '        x = self.norm1(x + attn_out)\n',
        '        x = self.norm1(x + attn_out)\n'
        '        watch_backward_gradient(x, f"layer{self.layer_id}_after_attention_grad", record_events)\n',
        'app: post attention gradient')
    txt=replace_exact(txt,
        '        moe_out, l_aux, exp_counts = self.moe(x)\n        x = self.norm2(x + moe_out)\n',
        '        moe_out, l_aux, exp_counts = self.moe(x)\n'
        '        watch_backward_gradient(moe_out, f"layer{self.layer_id}_moe_output_grad", record_events)\n'
        '        x = self.norm2(x + moe_out)\n'
        '        watch_backward_gradient(x, f"layer{self.layer_id}_output_grad", record_events)\n',
        'app: MoE output and postnorm gradient')
    txt=replace_exact(txt,
        '        self._mark("embedding_end", record_events)\n',
        '        watch_backward_gradient(x, "embedding_output_grad", record_events)\n'
        '        self._mark("embedding_end", record_events)\n',
        'app: embedding gradient')
    txt=replace_exact(txt,
        '        self._mark("lm_head_start", record_events)\n\n        x = self.final_norm(x)\n        logits = self.lm_head(x)\n',
        '        self._mark("lm_head_start", record_events)\n'
        '        watch_backward_gradient(x, "layers_output_grad", record_events)\n\n'
        '        x = self.final_norm(x)\n'
        '        watch_backward_gradient(x, "final_norm_output_grad", record_events)\n'
        '        logits = self.lm_head(x)\n'
        '        watch_backward_gradient(logits, "logits_grad", record_events)\n',
        'app: logits/final norm gradient')
    txt=replace_exact(txt,
        'def main():\n    parser = argparse.ArgumentParser()\n',
        'def main():\n    global _BWD_TRACE_ITER, _BWD_TRACE_RANK\n    parser = argparse.ArgumentParser()\n',
        'app: global in main')
    txt=replace_exact(txt,
        '        is_measured = global_iter >= args.warmup\n        measured_iter = global_iter - args.warmup\n',
        '        is_measured = global_iter >= args.warmup\n        measured_iter = global_iter - args.warmup\n'
        '        _BWD_TRACE_ITER = measured_iter if is_measured else -1\n'
        '        _BWD_TRACE_RANK = rank\n',
        'app: iteration context')
    txt=replace_exact(txt,
        '    save_csv(str(semantic_trace_path))\n',
        '    save_csv(str(semantic_trace_path))\n\n'
        '    # BURST_BWD_TRACE_V1: flush in-memory gradient checkpoints after run.\n'
        '    grad_trace_path = Path(args.outdir) / f"backward_tensor_trace_rank{rank}.csv"\n'
        '    with grad_trace_path.open("w", newline="") as grad_f:\n'
        '        grad_writer = csv.writer(grad_f)\n'
        '        grad_writer.writerow(["rank", "iter", "event", "t_mono_raw_ns"])\n'
        '        grad_writer.writerows(_BWD_GRAD_ROWS)\n'
        '    print(f"[rank {rank}] backward tensor trace: {grad_trace_path} ({len(_BWD_GRAD_ROWS)} checkpoints)", flush=True)\n',
        'app: gradient trace output')
    return txt


def patch_trace(txt):
    return replace_exact(txt,
        '    35: "grad_allreduce_unknown_end",\n',
        '    35: "grad_allreduce_unknown_end",\n'
        '    # BURST_BWD_TRACE_V1: coarse DeepSpeed backward envelope.\n'
        '    40: "engine_autograd_start",\n'
        '    41: "engine_autograd_end",\n'
        '    42: "engine_epilogue_start",\n'
        '    43: "engine_epilogue_end",\n',
        'trace: event map')


def patch_engine(txt):
    begin=txt.find('    def backward(self, loss, retain_graph=False, scale_wrt_gas=True):')
    if begin<0:raise ValueError('engine: could not find DeepSpeedEngine.backward()')
    end=txt.find('    def is_gradient_accumulation_boundary(self):',begin)
    if end<0:raise ValueError('engine: could not find end of backward()')
    section=txt[begin:end]
    section=replace_exact(section,
        '            with compiled_autograd(self._is_compiled_autograd_enabled, self._compile_kwargs):\n'
        '                if self.zero_optimization() or not self.amp_enabled():\n',
        '            with compiled_autograd(self._is_compiled_autograd_enabled, self._compile_kwargs):\n'
        '                mark(40, -1)  # BURST_BWD_TRACE_V1: autograd begins\n'
        '                if self.zero_optimization() or not self.amp_enabled():\n',
        'engine: autograd start')
    section=replace_exact(section,
        '                self._backward_epilogue()\n\n            return gas_scaled_loss\n',
        '                mark(41, -1)  # BURST_BWD_TRACE_V1: autograd ends\n'
        '                mark(42, -1)  # epilogue starts (includes gradient synchronization)\n'
        '                self._backward_epilogue()\n'
        '                mark(43, -1)  # epilogue ends\n\n'
        '            return gas_scaled_loss\n',
        'engine: epilogue boundaries')
    return txt[:begin]+section+txt[end:]


def write_atomic(path,content):
    temp=path.with_name(path.name+'.burst_tmp_'+uuid.uuid4().hex)
    try:
        temp.write_text(content,encoding='utf-8')
        os.chmod(temp,path.stat().st_mode & 0o777)
        os.replace(temp,path)
    finally:
        if temp.exists():temp.unlink()


def main():
    a=argparse.ArgumentParser(description=__doc__)
    action=a.add_mutually_exclusive_group(required=True)
    action.add_argument('--check',action='store_true')
    action.add_argument('--apply',action='store_true')
    a.add_argument('--app',default='~/moe-burst/moe-exp/actual_moe_train.py')
    a.add_argument('--engine',default=None)
    a.add_argument('--trace',default=None)
    a.add_argument('--backup-dir',default='~/moe-burst/backward-trace-backups')
    args=a.parse_args()
    app=pathlib.Path(args.app).expanduser()
    if args.engine and args.trace:
        engine=pathlib.Path(args.engine).expanduser()
        trace=pathlib.Path(args.trace).expanduser()
    elif args.engine or args.trace:
        raise SystemExit('Pass both --engine and --trace or neither')
    else:
        try:
            import deepspeed
        except ImportError as e:
            raise SystemExit('DeepSpeed import failed. Use ~/moe-env/bin/python or supply --engine and --trace') from e
        ds=pathlib.Path(deepspeed.__file__).parent
        engine=ds/'runtime/engine.py'
        trace=ds/'moe/trace.py'
    files={'app':app,'engine':engine,'trace':trace}
    original={k:p.read_text(encoding='utf-8') for k,p in files.items()}
    for k,txt in original.items():
        if VERSION in txt:
            raise SystemExit(f'Already patched ({k}): {files[k]} — refusing to apply twice.')
    patched={'app':patch_app(original['app']),
             'engine':patch_engine(original['engine']),
             'trace':patch_trace(original['trace'])}
    for k,t in patched.items():
        compile(t,str(files[k]),'exec')
        print(f'[VALID] {k}: {files[k]} (SHA256 before {hashlib.sha256(original[k].encode()).hexdigest()[:12]})')
    if args.check:
        print('PASS: expected source patterns found; no files changed.')
        return
    backup_base=pathlib.Path(args.backup_dir).expanduser()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup=backup_base/stamp
    backup.mkdir(parents=True,exist_ok=False)
    for k,p in files.items():shutil.copy2(p,backup/f'{k}.py')
    updated=[]
    try:
        for k,p in files.items():
            write_atomic(p,patched[k]); updated.append(k)
        print('APPLIED successfully to all 3 sources')
        print('BACKUP',backup)
        print('Remember: do not run install_instrumentation.sh --apply afterward; it would overwrite engine.py/trace.py changes.')
    except Exception:
        for k in updated:shutil.copy2(backup/f'{k}.py',files[k])
        raise

if __name__=='__main__':main()
