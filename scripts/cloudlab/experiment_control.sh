#!/usr/bin/env bash
# Shared helper for 4-node rendezvous smoke, control-NIC clock server, and pre/post samples.
# Usage:
#   bash ~/experiment_control.sh smoke <rank 0..3>       (simultaneously on 4 nodes)
#   nohup bash ~/experiment_control.sh server > ~/clock_sync_server.log 2>&1 < /dev/null &  (ranks 1..3)
#   bash ~/experiment_control.sh pre  <run_name>         (node0 only, server running)
#   bash ~/experiment_control.sh post <run_name>         (node0 only, server running)
set -Eeuo pipefail
MODE="${1:?mode: smoke|server|pre|post}"
REPO="$HOME/burst-origins"
NIC="${NIC_IFACE:-enp65s0f0np0}"
case "$MODE" in
  smoke)
    RANK="${2:?rank required}"
    [[ "$RANK" =~ ^[0-3]$ ]] || { echo 'rank must be 0..3' >&2; exit 2; }
    cd "$REPO"
    export GLOO_SOCKET_IFNAME="$NIC"
    exec "$HOME/moe-env/bin/torchrun" \
      --nnodes=4 --nproc-per-node=1 --node-rank="$RANK" \
      --master-addr=10.10.1.1 --master-port=29610 \
      moe-exp/dist_test.py
    ;;
  server)
    CTRL_IP="$(ip -4 -o addr show eno33np0 | awk '{print $4}' | cut -d/ -f1)"
    [[ -n "$CTRL_IP" ]] || { echo 'Missing control NIC IP' >&2; exit 2; }
    echo "Listening clock-sync on control NIC $CTRL_IP:29790"
    exec python3 "$REPO/moe-exp/clock_sync_raw.py" server --bind "$CTRL_IP" --port 29790
    ;;
  pre|post)
    RUN="${2:?run name required}"
    [[ "$RUN" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'unsafe run name' >&2; exit 2; }
    OUT="$REPO/moe-exp/clock_sync/$RUN"
    mkdir -p "$OUT"
    # Control-network addresses vary across CloudLab reservations.
    # Set these on node0 before running pre/post clock calibration.
    for RANK in 1 2 3; do
      PEER_VAR="CLOCK_PEER_${RANK}_IP"
      IP="${!PEER_VAR:-}"
      if [[ -z "$IP" ]]; then
        echo "Set $PEER_VAR to node${RANK}'s control-NIC IPv4 address before $MODE" >&2
        exit 2
      fi
      echo "[$MODE] node$RANK $IP"
      python3 "$REPO/moe-exp/clock_sync_raw.py" client \
        --peer "$IP" --port 29790 --samples 120 --interval-ms 1 \
        --output "$OUT/node${RANK}_${MODE}.csv"
    done
    echo "Clock sync $MODE complete: $OUT"
    ;;
  *)
    echo 'Usage: experiment_control.sh smoke RANK | server | pre RUN | post RUN' >&2
    exit 2
    ;;
esac
