# Experimental setup (CloudLab CPU/Gloo)

## Known environment

- Four CloudLab CPU nodes; rank 0..3. Each node runs one distributed rank.
- Gloo TCP (not NCCL/RDMA). Measured NIC had interface name `enp65s0f0np0` in the source environment; verify when migrating.
- Installed DeepSpeed expected version: `0.19.7`; instrumented copies are under `DeepSpeed-Instrumentation/`.
- D5K2 means `MOE_LAYERS=5`, `MOE_TOP_K=2`, with 4 experts and expert parallel size 4. All processes should be given the same configuration.
- Execution was measured for 20 iterations per independent run; 5 historical runs were collected. `run_measured_moe.sh` default warmup count is 2.

## Reproducing new measurements

```bash
# On each node, in the cloned repo:
cd ~/burst-origins
bash scripts/preflight.sh
bash install_instrumentation.sh --apply  # only after manual review
cc -O2 -o moe-exp/counter_probe_paced moe-exp/counter_probe_paced.c

export MOE_LAYERS=5 MOE_TOP_K=2
export NIC_IFACE=enp65s0f0np0
export MASTER_ADDR=10.10.1.1 MASTER_PORT=29610
cd moe-exp
./run_measured_moe.sh 0 2 20 my_d5k2_r01   # node0 only; use rank 1/2/3 on peers
```

The shell scripts were authored for this specific lab experiment; they are not a general-purpose launch manager and do **not** orchestrate all four nodes automatically. Verify rendezvous reachability, CPU affinity, permissions, and clock sync procedure before launching. The installation script **replaces files within the local DeepSpeed environment** and creates backups; use disposable environments.

## Raw clock and instrumentation

- Application and NIC-probe times use Linux `CLOCK_MONOTONIC_RAW` on each host.
- Host clocks are not synchronized by this monotonic clock alone. Use pre/post cross-host offset measurements and store the correction model with all plots.
- NIC counter rate is estimated by finite differences and temporal interpolation across read intervals, not from wire packet timestamps. Track sample gaps and missing-coverage bins.
- Use `moe-exp/verify_capture.py` and inspect `probe_tx.log`, `probe_rx.log`, training logs, and actual per-rank imported source versions.

## Compatibility risks

1. The original installer is pinned to DeepSpeed `0.19.7`; applying it to another version is unsupported.
2. System `pip freeze` in the historical ENV snapshot is **not** the training virtualenv; capture the correct env.
3. Socket/NIC/port defaults are CloudLab-specific; change env variables and inspect script before migration.
4. A repeat execution is not guaranteed to yield identical extreme entry skew, even with fixed seeds.
