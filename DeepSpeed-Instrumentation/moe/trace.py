import csv
import time


TRACE_ENABLED = False

_rank = -1
_iteration = -1
_collective_seq = 0
_events = []


EVENT_NAMES = {
    1: "moe_start",
    2: "router_start",
    3: "router_end",
    4: "dispatch_prep_start",
    5: "dispatch_prep_end",
    6: "dispatch_a2a_start",
    7: "dispatch_a2a_end",
    8: "expert_start",
    9: "expert_end",
    10: "return_a2a_start",
    11: "return_a2a_end",
    12: "combine_start",
    13: "combine_end",
    14: "moe_end",

    20: "bwd_return_a2a_start",
    21: "bwd_return_a2a_end",
    22: "bwd_dispatch_a2a_start",
    23: "bwd_dispatch_a2a_end",
    24: "bwd_unknown_a2a_start",
    25: "bwd_unknown_a2a_end",
    26: "expert_bwd_start",
    27: "expert_bwd_end",
    28: "router_bwd_start",
    29: "router_bwd_end",

    30: "grad_allreduce_nonexpert_start",
    31: "grad_allreduce_nonexpert_end",
    32: "grad_allreduce_expert_start",
    33: "grad_allreduce_expert_end",
    34: "grad_allreduce_unknown_start",
    35: "grad_allreduce_unknown_end",
}


def init_trace(rank: int):
    global _rank
    _rank = int(rank)


def enable_trace(enabled: bool = True):
    global TRACE_ENABLED
    TRACE_ENABLED = bool(enabled)


def set_iteration(iteration: int):
    global _iteration, _collective_seq
    _iteration = int(iteration)
    _collective_seq = 0


def next_collective_id() -> int:
    global _collective_seq
    cid = _collective_seq
    _collective_seq += 1
    return cid


def clear_trace():
    global _collective_seq
    _events.clear()
    _collective_seq = 0


def mark(
    event_id: int,
    layer_id: int = -1,
    numel: int = 0,
    element_size: int = 0,
    collective_id: int = -1,
    group_size: int = 0,
):
    if not TRACE_ENABLED:
        return

    t_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)

    _events.append(
        (
            t_ns,
            int(event_id),
            _rank,
            _iteration,
            int(layer_id),
            int(numel),
            int(element_size),
            int(collective_id),
            int(group_size),
        )
    )


def save_csv(path: str):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "t_mono_raw_ns",
            "event_id",
            "event",
            "rank",
            "iteration",
            "layer_id",
            "numel",
            "element_size",
            "bytes",
            "collective_id",
            "group_size",
        ])

        for (
            t_ns,
            event_id,
            rank,
            iteration,
            layer_id,
            numel,
            element_size,
            collective_id,
            group_size,
        ) in _events:
            writer.writerow([
                t_ns,
                event_id,
                EVENT_NAMES.get(event_id, f"unknown_{event_id}"),
                rank,
                iteration,
                layer_id,
                numel,
                element_size,
                numel * element_size,
                collective_id,
                group_size,
            ])
