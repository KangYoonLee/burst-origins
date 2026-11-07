#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(dirname "$(realpath "$0")")")"
VENV_PY="${VENV_PY:-$HOME/moe-env/bin/python}"
echo "Repo: $PWD"
echo "NIC: ${NIC_IFACE:-enp65s0f0np0}"
echo "MOE_LAYERS: ${MOE_LAYERS:-2} (D5K2 requires 5)"
echo "MOE_TOP_K: ${MOE_TOP_K:-1} (D5K2 requires 2)"
[[ -x "$VENV_PY" ]] || { echo "ERROR: missing experiment virtualenv: $VENV_PY" >&2; exit 1; }
"$VENV_PY" - <<'PY'
import torch, deepspeed
print('PyTorch',torch.__version__, 'DeepSpeed',deepspeed.__version__)
assert deepspeed.__version__ == '0.19.7', 'Instrumentation pins DeepSpeed 0.19.7'
PY
NIC="${NIC_IFACE:-enp65s0f0np0}"
[[ -r "/sys/class/net/$NIC/statistics/tx_bytes" ]] || { echo "ERROR: NIC byte counter not readable: $NIC" >&2; exit 1; }
[[ -f moe-exp/counter_probe_paced.c ]] || { echo 'ERROR: counter probe source missing' >&2; exit 1; }
echo 'Preflight passed. You must still validate rendezvous, permissions, CPU affinity and synchronized launch.'
