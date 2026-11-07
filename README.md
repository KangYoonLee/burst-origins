# Burst Origins

**Research code · v0.1 — CPU/Gloo MoE collective timing and NIC traffic measurement**

This repository is the initial codebase for studying **where traffic bursts originate in AI datacenter communication**. The present work focuses on rank-to-rank execution timing differences in expert-parallel Mixture-of-Experts (MoE) training, especially **Collective Entry Skew**, **Exit Skew**, and the relationship to NIC TX/RX traffic.

> **Research status:** This is an observational measurement prototype, not a finished network-control system. A large Entry Skew is **not**, by itself, evidence of network congestion, a traffic burst, or a causal training slowdown.

## Current experiment

- **Environment:** 4 CloudLab CPU hosts; PyTorch + DeepSpeed MoE; `torch.distributed`/Gloo over TCP.
- **D5K2:** `MOE_LAYERS=5`, `MOE_TOP_K=2`, 4 experts, one rank per host.
- **Measurements:** 20 measured iterations per run, five independent executions of **one** workload configuration; 5 layers × 2 forward AllToAll calls × 20 iterations = **200 collectives/run**, **1,000 total**.
- **Communication phases:** Dispatch A2A → local Expert FFN → Return A2A.
- **Collected:** application/collective entry-exit timestamps, NIC TX/RX byte counters, node clock-offset probes, run logs.
- **Observed exploratory pattern:** Entry Skew ≥ 1 ms and Exit Skew ≤ 0.5 ms in 429/1,000 forward collectives (42.9%). Pattern detection is not a performance threshold or causal proof.
- **Chain analysis:** same pattern in both consecutive collectives appeared in 40/500 intra-layer Dispatch→Return and 20/400 inter-layer Return→next Dispatch transitions. This is **co-occurrence**, not measured propagation/causal influence.

## Repository layout

```text
.
├── moe-exp/                       # workload, orchestrators, counter probe, plots
│   └── backward_tools/           # backward semantic-trace utilities
├── DeepSpeed-Instrumentation/     # instrumented DeepSpeed 0.19.7 sources
├── install_instrumentation.sh     # explicit, version-checked patch installer
├── scripts/                       # preflight / environment capture
├── docs/                          # experiment design, analysis and data protocol
└── third_party/                   # source attributions and license notices
```

## Quick start (CloudLab; four CPU nodes)

1. Create a Python environment containing compatible PyTorch and **DeepSpeed 0.19.7** at `~/moe-env`. Save the **environment's** package list with `bash scripts/capture_environment.sh`.
2. Confirm the actual NIC, CPU affinity and rendezvous settings. Scripts default to experimental hardware values (`enp65s0f0np0`, `10.10.1.1`) and are **not portable defaults**.
3. Inspect the instrumented DeepSpeed source changes. On **each node**, run `bash install_instrumentation.sh --apply`; it backs up installed DeepSpeed files before replacement. Do this only in a disposable research environment.
4. Build NIC probe: `cc -O2 -o moe-exp/counter_probe_paced moe-exp/counter_probe_paced.c`.
5. **Set workload parameters explicitly** on all ranks:
   ```bash
   export MOE_LAYERS=5 MOE_TOP_K=2
   export NIC_IFACE=enp65s0f0np0 MASTER_ADDR=10.10.1.1 MASTER_PORT=29610
   ```
6. On each of four nodes, run with a distinct rank (0 to 3):
   ```bash
   cd moe-exp
   ./run_measured_moe.sh <rank> 2 20 d5k2_run_new
   ```
   The original setup assumes compatible `~/moe-env` on all nodes. Launch ranks concurrently. See [Experiment setup](docs/EXPERIMENT_SETUP.md) for important safety and coverage checks.
7. Archive raw run data, clock-sync files and logs **outside Git**. Use `docs/DATA_STORAGE.md` for dataset recording and restore instructions.

**Important:** The executable's default configuration is **D2K1** unless `MOE_LAYERS` and `MOE_TOP_K` are exported. Read the configured environment in saved run logs before treating any output as D5K2.

## Research questions

1. What makes one rank ready to enter a collective later than its peers? (Compute imbalance, CPU scheduling, previous communication, collective runtime)
2. How does rank readiness skew affect NIC transmission timing and network burstiness? (not yet causally established)
3. Could collective readiness information improve future communication scheduling or load balancing?

## Data, reproducibility, and limits

Raw captures (including 100 µs counter probes), PCAPs and derived plots are excluded from Git to keep history compact and prevent accidental disclosure. The same source code **does not guarantee** reproduction of identical Skew/ECDF tail events due to CPU scheduling and network variability.

The uploaded environment snapshot captured system Python, **not** the `~/moe-env` virtual environment; record the latter before releasing the original CloudLab reservation. Full end-to-end rerun from this repo has **not** been verified.

See [Research protocol](docs/MEASUREMENT_PROTOCOL.md) and [Research roadmap](docs/RESEARCH_ROADMAP.md).

## Attribution

`DeepSpeed-Instrumentation` contains modified files from Microsoft DeepSpeed 0.19.7 (Apache-2.0) and code acknowledging adapted FairScale components (BSD). Original source copyright headers are preserved. See [Third-party notices](third_party/README.md). This repository does not assign a blanket license to unrelated source material.
