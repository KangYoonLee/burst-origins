#!/usr/bin/env bash
# Usage: bash ~/capture_moe_node.sh <rank:0-3> <run_name> [warmup=2] [iters=20]
# Run simultaneously on all four hosts after instrumentation and pre-sync.
set -Eeuo pipefail
RANK="${1:?rank 0..3 required}"
RUN="${2:?run name required}"
WARMUP="${3:-2}"
ITERS="${4:-20}"
[[ "$RANK" =~ ^[0-3]$ ]] || { echo 'rank must be 0..3' >&2; exit 2; }
[[ "$RUN" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'unsafe run name' >&2; exit 2; }
cd "$HOME/burst-origins/moe-exp"
export NIC_IFACE="${NIC_IFACE:-enp65s0f0np0}"
export MASTER_ADDR="${MASTER_ADDR:-10.10.1.1}"
export MASTER_PORT="${MASTER_PORT:-29610}"
export MOE_LAYERS=5 MOE_TOP_K=2
export GLOO_SOCKET_IFNAME="$NIC_IFACE"
BASE="$PWD/runs/$RUN/node$RANK"
if [[ -e "$BASE/packets.pcap" || -e "$BASE/events_rank${RANK}.csv" || -e "$BASE/tx.csv" || -e "$BASE/rx.csv" ]]; then
  echo "Refusing to overwrite previous capture in $BASE" >&2
  exit 2
fi
mkdir -p "$BASE"
cat > "$BASE/run_parameters.txt" <<PARAMS
run=$RUN
rank=$RANK
node=$(hostname)
when=$(date -Is)
NIC_IFACE=$NIC_IFACE
MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
MOE_LAYERS=$MOE_LAYERS
MOE_TOP_K=$MOE_TOP_K
warmup=$WARMUP
iters=$ITERS
PARAMS
(cd "$HOME/burst-origins" && git rev-parse HEAD) > "$BASE/git_commit.txt"
ethtool -k "$NIC_IFACE" > "$BASE/nic_offloads.txt" 2>&1 || true
ip -br -4 addr > "$BASE/ip_addresses.txt"

local_anchors() {
  local stage="$1"
  "$HOME/moe-env/bin/python" - "$BASE/local_clock_${stage}.csv" <<'PY'
import csv, sys, time
with open(sys.argv[1], 'w', newline='') as f:
    w=csv.writer(f)
    w.writerow(['realtime_mid_ns','mono_raw_ns','realtime_pair_width_ns'])
    for _ in range(64):
        a=time.clock_gettime_ns(time.CLOCK_REALTIME)
        b=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        c=time.clock_gettime_ns(time.CLOCK_REALTIME)
        w.writerow([(a+c)//2,b,c-a])
PY
}
local_anchors pre

# sudo -v prompts here if required, before background capture starts.
sudo -v
PIDFILE="$BASE/tcpdump.pid"
echo "[rank $RANK] starting packet capture on $NIC_IFACE (snaplen 192: headers/lengths only)"
sudo bash -c '
  echo "$$" > "$1"
  exec tcpdump -i "$2" -nn -s 192 -B 8192 -U -w "$3" "tcp and net 10.10.1.0/24"
' bash "$PIDFILE" "$NIC_IFACE" "$BASE/packets.pcap" > "$BASE/tcpdump.log" 2>&1 &
SUDO_WRAPPER_PID=$!
CAP_STOPPED=0
stop_capture() {
  if [[ "$CAP_STOPPED" == 1 ]]; then return; fi
  CAP_STOPPED=1
  if [[ -s "$PIDFILE" ]]; then
    CAP_PID="$(cat "$PIDFILE")"
    sudo -n kill -INT "$CAP_PID" 2>/dev/null || true
  fi
  wait "$SUDO_WRAPPER_PID" 2>/dev/null || true
  if [[ -f "$BASE/packets.pcap" ]]; then
    sudo -n chown "$(id -un):$(id -gn)" "$BASE/packets.pcap" || true
  fi
}
trap stop_capture EXIT
sleep 1
if [[ ! -s "$PIDFILE" ]] || ! sudo -n kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo 'ERROR: tcpdump did not start; inspect tcpdump.log' >&2
  cat "$BASE/tcpdump.log" >&2 || true
  exit 1
fi

echo "[rank $RANK] running actual D5K2 training + NIC byte probes"
RESULT=0
./run_measured_moe.sh "$RANK" "$WARMUP" "$ITERS" "$RUN" 180 100 || RESULT=$?
stop_capture
trap - EXIT
local_anchors post
printf '\n[rank %s] training return code: %d\n' "$RANK" "$RESULT"
tail -n 7 "$BASE/train.log" 2>/dev/null || true
printf '\n[rank %s] packet capture stats:\n' "$RANK"
tail -n 8 "$BASE/tcpdump.log" || true
ls -lh "$BASE/packets.pcap" "$BASE/tx.csv" "$BASE/rx.csv" 2>/dev/null || true
if [[ "$RESULT" -ne 0 ]]; then
  echo 'TRAINING FAILED -- raw traces preserved; inspect train.log' >&2
  exit "$RESULT"
fi
"$HOME/moe-env/bin/python" verify_capture.py --directory "$BASE" --rank "$RANK"
ARCHIVE="$HOME/${RUN}_node${RANK}.tar.gz"
tar -czf "$ARCHIVE" -C "runs/$RUN" "node$RANK"
echo "[rank $RANK] ARCHIVE READY: $ARCHIVE"
