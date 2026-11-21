#!/usr/bin/env bash
# Fresh 4-node CloudLab CPU/Gloo environment for burst-origins
# Run independently on each node: bash ~/bootstrap_moe_fresh.sh
set -Eeuo pipefail
export DEBIAN_FRONTEND=noninteractive
NIC="${NIC_IFACE:-enp65s0f0np0}"
if ! ip -4 addr show "$NIC" | grep -q '10\.10\.1\.'; then
  echo "ERROR: expected experiment NIC $NIC on 10.10.1.0/24. Check ip -br -4 addr." >&2
  exit 1
fi
echo "[1/6] Installing system tools"
sudo apt-get update -qq
sudo apt-get install -y -qq git python3-venv python3-pip python3-dev build-essential tcpdump ethtool ca-certificates

echo "[2/6] Cloning experiment repo"
if [[ ! -d "$HOME/burst-origins/.git" ]]; then
  git clone https://github.com/KangYoonLee/burst-origins.git "$HOME/burst-origins"
fi
cd "$HOME/burst-origins"
git rev-parse HEAD | tee "$HOME/burst-origins/BOOTSTRAP_COMMIT.txt"

echo "[3/6] Creating fresh venv"
if [[ ! -x "$HOME/moe-env/bin/python" ]]; then
  python3 -m venv "$HOME/moe-env"
fi
PY="$HOME/moe-env/bin/python"
"$PY" -m pip install -q --upgrade pip wheel setuptools

echo "[4/6] CPU PyTorch 2.6.0"
"$PY" -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.6.0'

echo "[5/6] DeepSpeed 0.19.7 + analysis dependencies"
"$PY" -m pip install 'deepspeed==0.19.7' 'numpy==1.26.4' pandas matplotlib

echo "[6/6] Build probe + preflight"
cc -O2 -o moe-exp/counter_probe_paced moe-exp/counter_probe_paced.c
bash scripts/preflight.sh
"$PY" - <<'PY'
import torch, deepspeed
print('BOOTSTRAP VERSIONS', torch.__version__, deepspeed.__version__)
PY
tcpdump --version | head -1
echo "BOOTSTRAP OK -- NEXT: Gloo smoke on all 4 nodes, THEN instrumentation patch"
