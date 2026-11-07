# Raw data storage and provenance

Raw data MUST NOT be committed to this repository. Store original experiment archives in an approved private research storage location (e.g. institution-managed drive), with at least one independent backup.

## Historical archives

- `d5k2_5runs_analysis.zip`: 5-run original traces and counter captures used for 2026-11-06 analysis; historical Git SHA unknown because code predates this repository renewal.
- `cloudlab_RUNS_20261107.tar.gz`: 2026-11-07 CloudLab run directory snapshot (about 49 MB compressed). **Not yet confirmed off CloudLab; back it up before release.**
- `D5K2_5Run_Collective_Chain_Analysis.zip`: derived tables and figures; regenerate if needed, but keep the version presented at meetings.
- `baseline_d2k1_01/`: separate historical baseline data; preserve before deleting the source reservation.

## Naming convention

```text
<workload>_<YYYYMMDD>_runNN/
  node0/...
  node1/...
  node2/...
  node3/...
  sync/...
```

For each dataset record `git rev-parse HEAD`, input seed, config (`MOE_LAYERS`, `MOE_TOP_K`), hardware, network interface, Python/Torch/DeepSpeed versions, node roles, source hashes, count of iterations, warmup count, probe interval and clock alignment method.

## Back up and verify

```bash
sha256sum archive.tar.gz > archive.tar.gz.sha256
# After copying to remote storage/download:
sha256sum -c archive.tar.gz.sha256
```

Avoid deleting local or CloudLab originals until the external copy has been downloaded and checksum-verified. Code alone cannot recreate the same timing tail or packet events from an old run.
