#!/usr/bin/env python3
"""
actual_moe_train.py

4-node CPU DeepSpeed MoE training workload for traffic measurement.

Topology
--------
Physical:
  4 CloudLab nodes, 1 rank/node, Gloo/TCP over enp65s0f0np0.

Expert Parallel:
  world_size = 4
  ep_size    = 4
  experts    = 4
  -> one local Expert per rank.

Model:
  token embedding + positional embedding
  -> Transformer-MoE block x MOE_LAYERS (default 2; D5K2 uses 5)
       self-attention
       DeepSpeed Top-K MoE (default 1; D5K2 uses 2)
         Router
         Dispatch AllToAll
         local Expert FFN
         Return/Combine AllToAll
  -> LM head
  -> next-token CrossEntropy + MoE auxiliary loss
  -> backward
  -> optimizer step

Measurement
-----------
Application events use CLOCK_MONOTONIC_RAW, the same local clock domain
used by counter_probe.c. The app records iteration/layer/MoE boundaries.
The instrumented DeepSpeed MoE source additionally records Router,
Dispatch/Return AllToAll, Expert, Combine, and selected backward events
into an in-memory semantic trace that is flushed after the run.
"""

import argparse
import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import deepspeed

from deepspeed.moe.layer import MoE
from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
from deepspeed.moe.trace import (
    init_trace,
    enable_trace,
    set_iteration,
    clear_trace,
    save_csv,
)


# ============================================================
# Workload configuration
# ============================================================
HIDDEN = 128
FFN_HIDDEN = 4 * HIDDEN

NUM_HEADS = 4
NUM_LAYERS = int(os.environ.get("MOE_LAYERS", "2"))

VOCAB_SIZE = 1024
BATCH = 2
SEQ_LEN = 256

NUM_EXPERTS = 4
EP_SIZE = 4
TOP_K = int(os.environ.get("MOE_TOP_K", "1"))

AUX_LOSS_WEIGHT = 0.01
LR = 1e-3


def now_raw_ns():
    """Same local clock domain as counter_probe.c."""
    return time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)


# ============================================================
# One Transformer block:
# Attention -> MoE FFN
# ============================================================
class TransformerMoEBlock(nn.Module):
    def __init__(self, layer_id: int, rank: int):
        super().__init__()
        self.layer_id = layer_id
        self.last_events = []

        # ----------------------------------------------------
        # Replicated Attention:
        # same initialization on all four ranks.
        # ----------------------------------------------------
        torch.manual_seed(10_000 + layer_id)

        self.attn = nn.MultiheadAttention(
            embed_dim=HIDDEN,
            num_heads=NUM_HEADS,
            dropout=0.0,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(HIDDEN)
        self.norm2 = nn.LayerNorm(HIDDEN)

        # ----------------------------------------------------
        # Expert FFN:
        # one different local Expert per rank.
        #
        # Expert = Linear -> GELU -> Linear
        # 128 -> 512 -> 128
        # ----------------------------------------------------
        torch.manual_seed(20_000 + 100 * layer_id + rank)

        local_expert = nn.Sequential(
            nn.Linear(HIDDEN, FFN_HIDDEN),
            nn.GELU(),
            nn.Linear(FFN_HIDDEN, HIDDEN),
        )

        # ----------------------------------------------------
        # Replicated Router:
        # reset to the same seed before constructing MoE,
        # so the gate/router starts identically on all ranks.
        # ----------------------------------------------------
        torch.manual_seed(30_000 + layer_id)

        self.moe = MoE(
            hidden_size=HIDDEN,
            expert=local_expert,
            num_experts=NUM_EXPERTS,
            ep_size=EP_SIZE,
            k=TOP_K,
            capacity_factor=1.0,
            eval_capacity_factor=1.0,
            min_capacity=4,
            drop_tokens=True,
            use_rts=True,
        )

    def set_expert_parallelism(self):
        self.moe.set_deepspeed_parallelism()

    def _mark(self, event: str, enabled: bool):
        if enabled:
            self.last_events.append((event, now_raw_ns()))

    def forward(self, x, causal_mask, record_events=False):
        self.last_events = []

        self._mark(f"layer{self.layer_id}_start", record_events)

        # ---------------- Attention ----------------
        self._mark(f"layer{self.layer_id}_attention_start", record_events)

        attn_out, _ = self.attn(
            x, x, x,
            attn_mask=causal_mask,
            need_weights=False,
        )
        x = self.norm1(x + attn_out)

        self._mark(f"layer{self.layer_id}_attention_end", record_events)

        # ---------------- MoE ----------------
        # Inside this DeepSpeed call:
        # Router -> Dispatch AllToAll -> Expert FFN
        #        -> Return/Combine AllToAll.
        self._mark(f"layer{self.layer_id}_moe_start", record_events)

        moe_out, l_aux, exp_counts = self.moe(x)
        x = self.norm2(x + moe_out)

        self._mark(f"layer{self.layer_id}_moe_end", record_events)
        self._mark(f"layer{self.layer_id}_end", record_events)

        return x, l_aux, exp_counts


# ============================================================
# Tiny trainable Transformer-MoE language model
# ============================================================
class TinyTransformerMoE(nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
        self.last_events = []

        # Replicated token/position embeddings.
        torch.manual_seed(9_000)
        self.token_embedding = nn.Embedding(VOCAB_SIZE, HIDDEN)
        self.position_embedding = nn.Embedding(SEQ_LEN, HIDDEN)

        self.layers = nn.ModuleList(
            [
                TransformerMoEBlock(layer_id=i, rank=rank)
                for i in range(NUM_LAYERS)
            ]
        )

        # Replicated output head.
        torch.manual_seed(9_001)
        self.final_norm = nn.LayerNorm(HIDDEN)
        self.lm_head = nn.Linear(HIDDEN, VOCAB_SIZE, bias=False)

    def set_expert_parallelism(self):
        for layer in self.layers:
            layer.set_expert_parallelism()

    def _mark(self, event: str, enabled: bool):
        if enabled:
            self.last_events.append((event, now_raw_ns()))

    def forward(self, token_ids, causal_mask, record_events=False):
        self.last_events = []

        self._mark("forward_start", record_events)

        # ---------------- Embedding ----------------
        self._mark("embedding_start", record_events)

        positions = torch.arange(
            token_ids.size(1),
            device=token_ids.device,
        ).unsqueeze(0)

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(positions)
        )

        self._mark("embedding_end", record_events)

        # ---------------- Transformer layers ----------------
        aux_losses = []
        expert_counts = []

        for layer in self.layers:
            x, l_aux, exp_counts = layer(
                x,
                causal_mask=causal_mask,
                record_events=record_events,
            )

            if record_events:
                self.last_events.extend(layer.last_events)

            aux_losses.append(l_aux)
            expert_counts.append(exp_counts.detach().cpu())

        # ---------------- LM head ----------------
        self._mark("lm_head_start", record_events)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        self._mark("lm_head_end", record_events)
        self._mark("forward_end", record_events)

        aux_loss = torch.stack(aux_losses).mean()

        return logits, aux_loss, expert_counts


# ============================================================
# DeepSpeed MoE optimizer parameter groups
# ============================================================
def create_moe_param_groups(model):
    params = {
        "params": [p for p in model.parameters() if p.requires_grad],
        "name": "parameters",
    }

    return split_params_into_different_moe_groups_for_optimizer(params)


# ============================================================
# Router synchronization sanity check
# ============================================================
def get_router_checksum(model, layer_id):
    w = model.layers[layer_id].moe.deepspeed_moe.gate.wg.weight.detach()
    return w.double().sum().reshape(1)


def check_router_sync(model, world_size, rank, label, fail_on_mismatch):
    """
    Each layer's Router is a replicated parameter, so it should be the
    same on all ranks.  Different input batches can still produce
    different routing decisions.
    """
    all_ok = True

    for layer_id in range(NUM_LAYERS):
        local = get_router_checksum(model, layer_id)

        gathered = [
            torch.zeros_like(local)
            for _ in range(world_size)
        ]

        dist.all_gather(gathered, local)

        values = [x.item() for x in gathered]
        spread = max(values) - min(values)

        if rank == 0:
            print(
                f"[router-check:{label}] "
                f"layer={layer_id} "
                f"checksums={values} "
                f"spread={spread:.3e}",
                flush=True,
            )

        if spread > 1e-10:
            all_ok = False

    if fail_on_mismatch and not all_ok:
        raise RuntimeError(
            f"Router parameters differ across ranks during '{label}' check."
        )

    return all_ok


# ============================================================
# Logging
# ============================================================
def write_logs(
    outdir,
    rank,
    event_rows,
    iter_rows,
    anchor_realtime_ns,
    anchor_mono_raw_ns,
):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Metadata / clock anchor
    meta_path = outdir / f"meta_rank{rank}.txt"

    with meta_path.open("w") as f:
        f.write(f"rank={rank}\n")
        f.write("clock=CLOCK_MONOTONIC_RAW\n")
        f.write(f"anchor_realtime_ns={anchor_realtime_ns}\n")
        f.write(f"anchor_mono_raw_ns={anchor_mono_raw_ns}\n")
        f.write(f"hidden={HIDDEN}\n")
        f.write(f"ffn_hidden={FFN_HIDDEN}\n")
        f.write(f"heads={NUM_HEADS}\n")
        f.write(f"layers={NUM_LAYERS}\n")
        f.write(f"vocab_size={VOCAB_SIZE}\n")
        f.write(f"batch={BATCH}\n")
        f.write(f"seq_len={SEQ_LEN}\n")
        f.write(f"num_experts={NUM_EXPERTS}\n")
        f.write(f"ep_size={EP_SIZE}\n")
        f.write(f"top_k={TOP_K}\n")
        f.write(f"capacity_factor=1.0\n")
        f.write(f"drop_tokens=True\n")
        f.write(f"use_rts=True\n")

    # Fine-grained application events
    events_path = outdir / f"events_rank{rank}.csv"

    with events_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["rank", "iter", "event", "t_mono_raw_ns"]
        )
        writer.writerows(event_rows)

    # Per-iteration summary
    iter_path = outdir / f"iters_rank{rank}.csv"

    fields = [
        "rank",
        "iter",
        "lm_loss",
        "aux_loss",
        "total_loss",
        "iter_ms",
        "forward_ms",
        "loss_ms",
        "backward_ms",
        "step_ms",
    ]

    for layer_id in range(NUM_LAYERS):
        fields.append(f"layer{layer_id}_exp_counts")

    with iter_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(iter_rows)


# ============================================================
# Main training loop
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations that are not logged.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Measured training iterations.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="actual_moe_run",
        help="Output directory for event/iteration logs.",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # 1. Distributed initialization
    # --------------------------------------------------------
    deepspeed.init_distributed(
        dist_backend="gloo",
        auto_mpi_discovery=False,
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 4:
        raise RuntimeError(
            f"Expected world_size=4, got {world_size}"
        )

    if rank == 0:
        print()
        print("=== Actual-ish DeepSpeed Transformer-MoE Training ===")
        print(f"world_size     = {world_size}")
        print(f"layers         = {NUM_LAYERS}")
        print(f"experts/layer  = {NUM_EXPERTS}")
        print(f"ep_size        = {EP_SIZE}")
        print(f"top_k          = {TOP_K}")
        print(f"batch/rank     = {BATCH}")
        print(f"seq_len        = {SEQ_LEN}")
        print(f"tokens/rank    = {BATCH * SEQ_LEN}")
        print(f"hidden         = {HIDDEN}")
        print(f"interface      = {os.environ.get('GLOO_SOCKET_IFNAME')}")
        print(f"warmup         = {args.warmup}")
        print(f"measured       = {args.iters}")
        print()

    # --------------------------------------------------------
    # 2. Build model
    # --------------------------------------------------------
    model = TinyTransformerMoE(rank=rank)

    # Do NOT create the Expert Parallel process group manually here.
    # deepspeed.initialize() detects MoE layers and creates/attaches the
    # EP/EDP process groups itself. Creating them twice causes:
    #   AssertionError: Attempting to override an existing ep_group
    model.train()

    # Verify that replicated Routers start identically.
    check_router_sync(
        model,
        world_size,
        rank,
        label="initial",
        fail_on_mismatch=True,
    )

    # --------------------------------------------------------
    # 3. Optimizer + DeepSpeed Engine
    # --------------------------------------------------------
    param_groups = create_moe_param_groups(model)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=LR,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    ds_config = {
        "train_batch_size": BATCH * world_size,
        "train_micro_batch_size_per_gpu": BATCH,
        "gradient_accumulation_steps": 1,
        "steps_per_print": 1000000,
        "zero_optimization": {
            "stage": 0,
        },
        "fp16": {
            "enabled": False,
        },
        "wall_clock_breakdown": False,
    }

    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=param_groups,
        optimizer=optimizer,
        config=ds_config,
        dist_init_required=False,
    )

    # --------------------------------------------------------
    # 3a. DeepSpeed MoE semantic trace context
    #
    # Each DeepSpeed MOELayer needs to know which Transformer
    # layer it belongs to. The trace itself is process-local.
    # Warmup is excluded; measured iterations are labeled 0..N-1.
    # --------------------------------------------------------
    for layer_id, layer in enumerate(engine.module.layers):
        layer.moe.deepspeed_moe.trace_layer_id = layer_id

    init_trace(rank)
    clear_trace()
    enable_trace(False)

    # Causal self-attention:
    # True means "this position cannot attend to that future position".
    causal_mask = torch.triu(
        torch.ones(
            SEQ_LEN,
            SEQ_LEN,
            dtype=torch.bool,
        ),
        diagonal=1,
    )

    # --------------------------------------------------------
    # 4. Reproducible but changing local batch every iteration
    # --------------------------------------------------------
    data_generator = torch.Generator(device="cpu")
    data_generator.manual_seed(50_000 + rank)

    # Application clock anchor.
    anchor_realtime_ns = time.clock_gettime_ns(
        time.CLOCK_REALTIME
    )
    anchor_mono_raw_ns = now_raw_ns()

    event_rows = []
    iter_rows = []

    # One barrier only before the training run.
    # We intentionally do NOT add a barrier every iteration.
    dist.barrier()

    total_iterations = args.warmup + args.iters

    # --------------------------------------------------------
    # 5. Training
    # --------------------------------------------------------
    for global_iter in range(total_iterations):
        is_measured = global_iter >= args.warmup
        measured_iter = global_iter - args.warmup

        # New local batch every iteration.
        token_ids = torch.randint(
            low=0,
            high=VOCAB_SIZE,
            size=(BATCH, SEQ_LEN),
            generator=data_generator,
            dtype=torch.long,
        )

        # Next-token target:
        # position t predicts token at t+1.
        targets = token_ids[:, 1:].contiguous()

        # ----------------------------------------------------
        # DeepSpeed semantic tracing
        #
        # Keep tracing enabled across BOTH forward and backward
        # for measured iterations. Disable it again before the
        # optimizer step so this v1 trace stays scoped to the
        # MoE forward/backward execution path.
        # ----------------------------------------------------
        if is_measured:
            set_iteration(measured_iter)
            enable_trace(True)
        else:
            enable_trace(False)

        # ==================== ITERATION START ====================
        iter_start = now_raw_ns()

        # ---------------- Forward ----------------
        forward_start = now_raw_ns()

        logits, aux_loss, expert_counts = engine(
            token_ids,
            causal_mask,
            record_events=is_measured,
        )

        forward_end = now_raw_ns()

        # ---------------- Loss ----------------
        loss_start = now_raw_ns()

        next_token_logits = logits[:, :-1, :].contiguous()

        lm_loss = F.cross_entropy(
            next_token_logits.view(-1, VOCAB_SIZE),
            targets.view(-1),
        )

        total_loss = lm_loss + AUX_LOSS_WEIGHT * aux_loss

        loss_end = now_raw_ns()

        # ---------------- Backward ----------------
        backward_start = now_raw_ns()

        engine.backward(total_loss)

        backward_end = now_raw_ns()

        # The current semantic trace is intentionally scoped to
        # MoE forward/backward. Optimizer-step tracing will be
        # added separately when framework-wide collective tracing
        # is introduced.
        enable_trace(False)

        # ---------------- Optimizer step ----------------
        step_start = now_raw_ns()

        engine.step()

        step_end = now_raw_ns()

        iter_end = now_raw_ns()
        # ===================== ITERATION END =====================

        if is_measured:
            # ----- Event timeline -----
            event_rows.append(
                [rank, measured_iter, "iter_start", iter_start]
            )
            event_rows.append(
                [rank, measured_iter, "forward_start", forward_start]
            )

            for event_name, event_time in engine.module.last_events:
                event_rows.append(
                    [rank, measured_iter, event_name, event_time]
                )

            event_rows.append(
                [rank, measured_iter, "forward_end", forward_end]
            )
            event_rows.append(
                [rank, measured_iter, "loss_start", loss_start]
            )
            event_rows.append(
                [rank, measured_iter, "loss_end", loss_end]
            )
            event_rows.append(
                [rank, measured_iter, "backward_start", backward_start]
            )
            event_rows.append(
                [rank, measured_iter, "backward_end", backward_end]
            )
            event_rows.append(
                [rank, measured_iter, "step_start", step_start]
            )
            event_rows.append(
                [rank, measured_iter, "step_end", step_end]
            )
            event_rows.append(
                [rank, measured_iter, "iter_end", iter_end]
            )

            # ----- Per-iteration summary -----
            row = {
                "rank": rank,
                "iter": measured_iter,
                "lm_loss": float(lm_loss.detach()),
                "aux_loss": float(aux_loss.detach()),
                "total_loss": float(total_loss.detach()),
                "iter_ms": (iter_end - iter_start) / 1e6,
                "forward_ms": (forward_end - forward_start) / 1e6,
                "loss_ms": (loss_end - loss_start) / 1e6,
                "backward_ms": (backward_end - backward_start) / 1e6,
                "step_ms": (step_end - step_start) / 1e6,
            }

            for layer_id, counts in enumerate(expert_counts):
                row[f"layer{layer_id}_exp_counts"] = str(
                    counts.tolist()
                )

            iter_rows.append(row)

            print(
                f"[rank {rank}] "
                f"iter={measured_iter:02d} "
                f"loss={row['total_loss']:.4f} "
                f"iter={row['iter_ms']:.2f}ms "
                f"fwd={row['forward_ms']:.2f}ms "
                f"bwd={row['backward_ms']:.2f}ms "
                f"step={row['step_ms']:.2f}ms "
                + " ".join(
                    f"L{layer_id}={counts.tolist()}"
                    for layer_id, counts in enumerate(expert_counts)
                ),
                flush=True,
            )

    # --------------------------------------------------------
    # 6. End-of-run checks + logs
    # --------------------------------------------------------
    dist.barrier()

    router_still_synced = check_router_sync(
        engine.module,
        world_size,
        rank,
        label="after_training",
        fail_on_mismatch=False,
    )

    write_logs(
        outdir=args.outdir,
        rank=rank,
        event_rows=event_rows,
        iter_rows=iter_rows,
        anchor_realtime_ns=anchor_realtime_ns,
        anchor_mono_raw_ns=anchor_mono_raw_ns,
    )

    # Flush DeepSpeed's in-memory semantic trace only after
    # measurement has completed.
    semantic_trace_path = (
        Path(args.outdir) / f"moe_semantic_trace_rank{rank}.csv"
    )
    save_csv(str(semantic_trace_path))

    print(
        f"[rank {rank}] semantic trace: {semantic_trace_path}",
        flush=True,
    )

    if rank == 0:
        print()
        print(
            "SUCCESS: Transformer-MoE training completed.",
            flush=True,
        )
        print(
            f"Logs: {args.outdir}/",
            flush=True,
        )

        if not router_still_synced:
            print(
                "WARNING: Router weights diverged across ranks after training.",
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
