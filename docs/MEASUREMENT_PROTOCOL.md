# Timing and traffic measurement protocol

## What was measured

- **Collective entry / exit:** per-rank timestamps immediately around instrumented forward Dispatch and Return AllToAll calls in DeepSpeed.
- **Entry Skew:** `max_r(entry_r) - min_r(entry_r)` after aligning ranks to a common estimated time reference.
- **Exit Skew:** `max_r(exit_r) - min_r(exit_r)`.
- **Global collective span:** `max_r(exit_r) - min_r(entry_r)`, not identical to per-rank call duration or end-to-end iteration time.
- **NIC traffic:** sysfs `tx_bytes`/`rx_bytes` sampled by `counter_probe_paced.c` at a target period; estimated throughput from counter deltas.

## Alignment and interpretation

`CLOCK_MONOTONIC_RAW` is a per-host clock; pre/post offset estimation must be applied before cross-rank skew is computed. Uncertainty from software timestamping and clock drift remains, especially for small exit-skew differences. The byte-counter probe timestamps represent reads, not individual packet transmission timestamps. Counter derivative plots are time-binned approximations and must mask missing/gapped samples rather than filling zeros.

An interval spent *inside* a blocking collective does **not** establish that the NIC was continuously transmitting during that interval. A large Entry Skew or small Exit Skew does **not** prove network congestion, traffic burst, or a causal performance penalty.

## Exploratory analysis (2026-11-06)

- Workload: D5K2, four ranks, five independent runs, 20 measured iterations each.
- Forward collectives per run: 20 × 5 × 2 = 200; total 1,000.
- Flag criterion: Entry Skew ≥ 1.0 ms AND Exit Skew ≤ 0.5 ms.
- Flagged: 429/1,000 (42.9%) total, with counts: run01 86, run02 76, run03 96, run04 69, run05 102.
- Chain transitions within iterations: 100 Dispatch→Return / run and 80 Return→next Dispatch / run.
- The same flag occurred in both consecutive collectives for 40/500 and 20/400 transitions, respectively. This co-occurrence measure does not identify causal propagation.

Repeat experiments with threshold sensitivity checks and run-wise stratification; 1,000 nested observations should not be treated as 1,000 fully independent experimental repeats.
