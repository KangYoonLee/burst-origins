#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_actual_moe.sh <rank> [warmup] [iters] [outdir]
#
# Example:
#   ./run_actual_moe.sh 0 1 2 smoke_actual

RANK="${1:?usage: $0 <rank> [warmup] [iters] [outdir]}"
WARMUP="${2:-2}"
ITERS="${3:-20}"
OUTDIR="${4:-actual_moe_run}"

cd "$(dirname "$(realpath "$0")")"
source "$HOME/moe-env/bin/activate"

export GLOO_SOCKET_IFNAME="${NIC_IFACE:-enp65s0f0np0}"
export TORCHDYNAMO_DISABLE=1

exec torchrun \
  --nnodes=4 \
  --nproc-per-node=1 \
  --node-rank="${RANK}" \
  --master-addr="${MASTER_ADDR:-10.10.1.1}" \
  --master-port="${MASTER_PORT:-29610}" \
  actual_moe_train.py \
  --warmup "${WARMUP}" \
  --iters "${ITERS}" \
  --outdir "${OUTDIR}"
