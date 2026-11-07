#!/usr/bin/env bash
set -euo pipefail
# On EACH node, run in parallel:
#   ./run_measured_moe.sh <rank> [warmup=2] [iters=20] [run_name=baseline01] [max_capture_sec=120] [interval_us=100]
# Each node writes to ~/moe-burst/moe-exp/runs/<run_name>/node<rank>/

RANK="${1:?usage: $0 <rank> [warmup] [iters] [run_name] [max_capture_sec] [interval_us]}"
WARMUP="${2:-2}"
ITERS="${3:-20}"
RUN_NAME="${4:-baseline01}"
MAX_SEC="${5:-120}"
INTERVAL_US="${6:-100}"
[[ "$RANK" =~ ^[0-3]$ ]] || { echo 'rank must be 0..3' >&2; exit 2; }
[[ "$RUN_NAME" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo 'run_name must be alphanumeric, underscore, dot or hyphen' >&2; exit 2; }
cd "$(dirname "$(realpath "$0")")"
NIC="${NIC_IFACE:-enp65s0f0np0}"
TX_CPU="${TX_CPU:-6}"
RX_CPU="${RX_CPU:-7}"
BASE="$PWD/runs/$RUN_NAME/node$RANK"
if [[ -e "$BASE/tx.csv" || -e "$BASE/rx.csv" || -e "$BASE/events_rank$RANK.csv" ]]; then
  echo "Refusing to overwrite existing experiment: $BASE" >&2
  exit 2
fi
mkdir -p "$BASE"
[[ -r "/sys/class/net/$NIC/statistics/tx_bytes" ]] || { echo "Missing NIC stats for $NIC" >&2; exit 2; }
[[ -x ./counter_probe_paced ]] || { echo 'Compile ./counter_probe_paced first' >&2; exit 2; }
[[ -x ./run_actual_moe.sh ]] || { echo 'Missing executable ./run_actual_moe.sh' >&2; exit 2; }

printf 'rank=%s\nnic=%s\nwarmup=%s\niters=%s\nmax_capture_sec=%s\ntarget_interval_us=%s\n' \
  "$RANK" "$NIC" "$WARMUP" "$ITERS" "$MAX_SEC" "$INTERVAL_US" > "$BASE/measurement.conf"
{ date -Is; uname -a; ip -br addr show "$NIC"; ethtool -i "$NIC" || true; ethtool "$NIC" | grep -E 'Speed|Link detected' || true; sha256sum actual_moe_train.py run_actual_moe.sh counter_probe_paced.c; } > "$BASE/environment.txt" 2>&1

# Verify the DeepSpeed sources actually imported by this virtual environment.
"$HOME/moe-env/bin/python" - <<'PYDEEPSPEED' > "$BASE/imported_sources.txt" 2>&1
import hashlib
import deepspeed
import deepspeed.runtime.engine as engine
import deepspeed.moe.sharded_moe as sharded
import deepspeed.moe.trace as trace
for name, module in [('deepspeed',deepspeed),('engine',engine),('sharded_moe',sharded),('trace',trace)]:
    path=module.__file__
    print(name,path,hashlib.sha256(open(path,'rb').read()).hexdigest())
PYDEEPSPEED

TX_PID=''; RX_PID=''
stop_probes() {
  for pid in "$TX_PID" "$RX_PID"; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then kill -INT "$pid" 2>/dev/null || true; fi
  done
  for pid in "$TX_PID" "$RX_PID"; do
    if [[ -n "$pid" ]]; then wait "$pid" || true; fi
  done
}
trap 'stop_probes' EXIT
./counter_probe_paced "/sys/class/net/$NIC/statistics/tx_bytes" "$TX_CPU" "$MAX_SEC" "$BASE/tx.csv" "$INTERVAL_US" >"$BASE/probe_tx.log" 2>&1 &
TX_PID=$!
./counter_probe_paced "/sys/class/net/$NIC/statistics/rx_bytes" "$RX_CPU" "$MAX_SEC" "$BASE/rx.csv" "$INTERVAL_US" >"$BASE/probe_rx.log" 2>&1 &
RX_PID=$!
sleep 0.8

echo "[rank $RANK] training started, probes pids TX=$TX_PID RX=$RX_PID output=$BASE"
set +e
./run_actual_moe.sh "$RANK" "$WARMUP" "$ITERS" "$BASE" >"$BASE/train.log" 2>&1
TRAIN_RC=$?
set -e
sleep 0.8
stop_probes
# Verify the DeepSpeed sources actually imported by this virtual environment.
"$HOME/moe-env/bin/python" - <<'PYDEEPSPEED' > "$BASE/imported_sources.txt" 2>&1
import hashlib
import deepspeed
import deepspeed.runtime.engine as engine
import deepspeed.moe.sharded_moe as sharded
import deepspeed.moe.trace as trace
for name, module in [('deepspeed',deepspeed),('engine',engine),('sharded_moe',sharded),('trace',trace)]:
    path=module.__file__
    print(name,path,hashlib.sha256(open(path,'rb').read()).hexdigest())
PYDEEPSPEED

TX_PID=''; RX_PID=''
trap - EXIT
if [[ "$TRAIN_RC" -ne 0 ]]; then
  echo "[rank $RANK] TRAINING FAILED ($TRAIN_RC). See $BASE/train.log" >&2
  exit "$TRAIN_RC"
fi
[[ -s "$BASE/tx.csv" && -s "$BASE/rx.csv" ]] || { echo "Probe output missing" >&2; exit 1; }
echo "[rank $RANK] capture completed. Verify training coverage and sample gaps before plotting: $BASE"
