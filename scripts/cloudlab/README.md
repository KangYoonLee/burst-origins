# CloudLab CPU/Gloo packet capture workflow

Scripts for the four-node D5K2 MoE measurement (CPU, PyTorch Gloo/TCP, DeepSpeed 0.19.7). These are experiment-specific; raw PCAPs, logs, CSVs and archives must stay outside Git.

* `bootstrap_moe_fresh.sh` installs dependencies, creates `~/moe-env`, clones `~/burst-origins`, builds the NIC probe and runs preflight.
* `experiment_control.sh` runs 4-rank Gloo smoke tests and pre/post cross-node raw-clock offset sampling.
* `capture_moe_node.sh` runs D5K2 training with NIC TX/RX probes, host `tcpdump` (`-s 192`) and local clock anchors, then validates and archives each node.

## Preparation

Verify the CloudLab NIC/IP configuration (historically experiment NIC `enp65s0f0np0` on `10.10.1.0/24`, control NIC `eno33np0`). The scripts assume four Linux nodes and the `~/burst-origins` / `~/moe-env` paths. On every node, run the bootstrap script (either from a downloaded copy in `~/` or from `scripts/cloudlab/` in a clone), then run the Gloo smoke test **simultaneously** with ranks 0..3. After it succeeds, apply `bash install_instrumentation.sh --apply` on every node; the installer modifies an otherwise disposable DeepSpeed 0.19.7 environment after creating backups.

On nodes 1..3, start `nohup bash scripts/cloudlab/experiment_control.sh server > ~/clock_sync_server.log 2>&1 < /dev/null &` from the repository root. On node0, set the current reservation's **control-network IPv4** addresses:

```bash
export CLOCK_PEER_1_IP=<node1_control_ipv4>
export CLOCK_PEER_2_IP=<node2_control_ipv4>
export CLOCK_PEER_3_IP=<node3_control_ipv4>
```

These addresses are deliberately **not hard-coded** in Git. Determine them with `ip -br -4 addr` on each host. In all commands below, run from `~/burst-origins`.

## Capture and backup

```bash
# On node0 just before training
bash scripts/cloudlab/experiment_control.sh pre d5k2_capture_r01

# On each of four nodes concurrently, substituting rank 0, 1, 2, 3
bash scripts/cloudlab/capture_moe_node.sh <rank> d5k2_capture_r01

# On node0 immediately after all four ranks complete
bash scripts/cloudlab/experiment_control.sh post d5k2_capture_r01
tar -czf ~/d5k2_capture_r01_clock_sync.tar.gz -C moe-exp/clock_sync d5k2_capture_r01
```

Use a **fresh run name** for each attempt. On every rank check `training return code: 0`, `VALIDATION: PASS (basic coverage checks)`, nonempty PCAP with data traffic, capture-drop logs and `ARCHIVE READY`. Download `~/d5k2_capture_r01_node0.tar.gz` through `node3.tar.gz` from the corresponding hosts, plus the node0 clock-sync archive. Check archive integrity before the reservation ends.

## Limitations

PCAP timestamps are host observations, **not physical-wire packet timing**. TSO/GSO/GRO can change observed TCP segment timing/size. With `-s 192`, packet data is truncated; original record length rather than captured length is needed for approximate volume. Application/ NIC probes use host-local `CLOCK_MONOTONIC_RAW`, whereas PCAP uses approximate `CLOCK_REALTIME`: local realtime/raw anchors and peer pre/post samples are used for **approximate** alignment, not guaranteed sub-ms precision or packet-to-collective attribution. Check probe sampling gaps and clock drift. This is CPU/Gloo/TCP, not GPU/NCCL/RDMA.
