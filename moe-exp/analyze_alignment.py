import argparse
import csv
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_trace(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["t"] = int(r["t_mono_raw_ns"])
        r["iteration"] = int(r["iteration"])
        r["layer_id"] = int(r["layer_id"])
        r["collective_id"] = int(r["collective_id"])
        r["bytes"] = int(r["bytes"])
    return rows


def read_counter(path, left_ns, right_ns, bin_ns):
    anchor = None
    edges = np.arange(
        left_ns, right_ns + 2 * bin_ns, bin_ns, dtype=np.int64
    )
    values = np.zeros(len(edges) - 1, dtype=np.float64)

    first_time = last_time = None
    previous_time = previous_value = None
    failures = 0
    gaps = 0

    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#"):
                m = re.search(
                    r"anchor_start realtime_ns=\d+ mono_raw_ns=(\d+)",
                    line,
                )
                if m:
                    anchor = int(m.group(1))
                continue

            if line.startswith("ta_ns,"):
                continue

            if anchor is None:
                raise RuntimeError(f"No clock anchor in {path}")

            ta, tb, value, ok = line.rstrip().split(",")
            if ok != "1":
                failures += 1
                previous_time = previous_value = None
                continue

            t = anchor + (int(ta) + int(tb)) // 2
            value = int(value)

            if first_time is None:
                first_time = t
            last_time = t

            if previous_time is not None:
                delta = value - previous_value
                if delta < 0:
                    raise RuntimeError(f"Counter decreased: {path}")
                if t <= previous_time:
                    raise RuntimeError(f"Non-increasing timestamps: {path}")

                # Assign each observed byte increment to its observation
                # timestamp. This is an approximation, not a packet timestamp.
                if left_ns <= t < right_ns:
                    index = int((t - left_ns) // bin_ns)
                    if 0 <= index < len(values):
                        values[index] += delta

                if t - previous_time > bin_ns:
                    gaps += 1

            previous_time, previous_value = t, value

    if first_time is None:
        raise RuntimeError(f"No valid samples: {path}")

    return edges, values, first_time, last_time, failures, gaps


def paired_collectives(rows, iteration):
    selected = [r for r in rows if r["iteration"] == iteration]
    starts = {}
    intervals = []

    for r in selected:
        name = r["event"]
        if not name.endswith("_start"):
            continue

        key = (
            name[:-6],
            r["layer_id"],
            r["collective_id"],
        )
        starts[key] = r

    for r in selected:
        name = r["event"]
        if not name.endswith("_end"):
            continue

        base = name[:-4]
        if not (
            "a2a" in base or "allreduce" in base
        ):
            continue

        key = (base, r["layer_id"], r["collective_id"])
        start = starts.get(key)
        if start is None:
            continue

        intervals.append({
            "name": base,
            "start": start["t"],
            "end": r["t"],
            "bytes": start["bytes"],
            "layer": r["layer_id"],
        })

    return sorted(intervals, key=lambda x: x["start"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default="logs_nic_alignment/node0")
    p.add_argument("--iteration", type=int, default=0)
    p.add_argument("--bin-us", type=int, default=1000)
    args = p.parse_args()

    base = Path(args.directory)
    rows = read_trace(base / "moe_semantic_trace_rank0.csv")
    selected = [r for r in rows if r["iteration"] == args.iteration]

    if not selected:
        raise RuntimeError("Iteration not found")

    # Covers all semantic events of the requested iteration.
    left = min(r["t"] for r in selected) - 2_000_000
    right = max(r["t"] for r in selected) + 2_000_000
    bin_ns = args.bin_us * 1000

    tx_edges, tx, tx_first, tx_last, tx_fail, tx_gaps = read_counter(
        base / "tx.csv", left, right, bin_ns
    )
    _, rx, rx_first, rx_last, rx_fail, rx_gaps = read_counter(
        base / "rx.csv", left, right, bin_ns
    )

    print(f"Iteration {args.iteration} coverage:")
    for label, first, last in [
        ("TX", tx_first, tx_last),
        ("RX", rx_first, rx_last),
    ]:
        print(
            f"  {label}: start margin={(left-first)/1e9:.3f}s, "
            f"end margin={(last-right)/1e9:.3f}s"
        )
        if first > left or last < right:
            raise RuntimeError(
                f"{label} does not cover the iteration. Do not plot."
            )

    print(
        f"Counter failures: TX={tx_fail}, RX={rx_fail}; "
        f"gaps > bin: TX={tx_gaps}, RX={rx_gaps}"
    )

    x = (tx_edges[:-1] - left) / 1e6
    tx_gbps = tx * 8 / (bin_ns / 1e9) / 1e9
    rx_gbps = rx * 8 / (bin_ns / 1e9) / 1e9

    intervals = paired_collectives(rows, args.iteration)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(15, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax.step(x, tx_gbps, where="post", label="NIC TX", linewidth=1.4)
    ax.step(x, rx_gbps, where="post", label="NIC RX", linewidth=1.4)
    ax.set_ylabel("Estimated throughput (Gbps)")
    ax.legend()
    ax.grid(alpha=0.2)

    kinds = {
        "dispatch_a2a": (0, "Dispatch A2A"),
        "return_a2a": (1, "Return A2A"),
        "bwd_return_a2a": (2, "Bwd Return A2A"),
        "bwd_dispatch_a2a": (3, "Bwd Dispatch A2A"),
        "grad_allreduce_nonexpert": (4, "Non-expert AllReduce"),
        "grad_allreduce_expert": (5, "Expert AllReduce"),
    }

    for interval in intervals:
        name = interval["name"]
        if name not in kinds:
            continue

        y, label = kinds[name]
        x0 = (interval["start"] - left) / 1e6
        x1 = (interval["end"] - left) / 1e6

        ax2.broken_barh(
            [(x0, max(x1 - x0, 0.001))],
            (y - 0.35, 0.7),
        )

        if name == "grad_allreduce_nonexpert":
            ax.axvspan(x0, x1, alpha=0.12)

        print(
            f"{label:24s} L={interval['layer']:2d} "
            f"start={x0:8.3f}ms end={x1:8.3f}ms "
            f"duration={x1-x0:7.3f}ms "
            f"tensor_bytes={interval['bytes']}"
        )

    ax2.set_yticks([v[0] for v in kinds.values()])
    ax2.set_yticklabels([v[1] for v in kinds.values()])
    ax2.set_ylim(-0.8, len(kinds) - 0.2)
    ax2.set_xlabel("Time since plot window start (ms)")
    ax2.grid(axis="x", alpha=0.2)

    fig.suptitle(
        f"Node0 Iteration {args.iteration}: NIC vs Collectives "
        f"({args.bin_us}us bins)"
    )
    fig.tight_layout()

    output = base / (
        f"alignment_iter{args.iteration}_bin{args.bin_us}us.png"
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)

    print(f"\nSaved: {output}")
    print(
        f"Observed in plot window: "
        f"TX={tx.sum():,.0f} B, RX={rx.sum():,.0f} B"
    )


if __name__ == "__main__":
    main()
