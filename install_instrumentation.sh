#!/usr/bin/env bash
# Run explicitly AFTER successful Gloo smoke test, separately on each node.
set -euo pipefail
if [[ "${1:-}" != "--apply" ]]; then
 echo "Usage: bash install_instrumentation.sh --apply"
 echo "This replaces three installed DeepSpeed source files after backup."
 exit 2
fi
cd "$(dirname "$(realpath "$0")")"
PY="$HOME/moe-env/bin/python"
DS_VER="$($PY -c 'import deepspeed; print(deepspeed.__version__)' 2>/dev/null | tail -1)"
if [[ "$DS_VER" != "0.19.7" ]]; then
 echo "Expected deepspeed 0.19.7; found $DS_VER. Refusing to patch." >&2; exit 1
fi
DS_DIR="$($PY -c 'import deepspeed, pathlib; print(pathlib.Path(deepspeed.__file__).parent)' 2>/dev/null | tail -1)"
[[ -f "$DS_DIR/runtime/engine.py" && -f "$DS_DIR/moe/sharded_moe.py" ]] || exit 1
BACKUP="${DEEPSPEED_BACKUP_DIR:-$PWD/deepspeed-backups}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP/runtime" "$BACKUP/moe"
cp -p "$DS_DIR/runtime/engine.py" "$BACKUP/runtime/engine.py"
cp -p "$DS_DIR/moe/sharded_moe.py" "$BACKUP/moe/sharded_moe.py"
TRACE_EXISTED=0
if [[ -e "$DS_DIR/moe/trace.py" ]]; then
 TRACE_EXISTED=1
 cp -p "$DS_DIR/moe/trace.py" "$BACKUP/moe/trace.py"
fi
rollback() {
 echo "[ERROR] Rolling back instrumentation files" >&2
 cp -p "$BACKUP/runtime/engine.py" "$DS_DIR/runtime/engine.py"
 cp -p "$BACKUP/moe/sharded_moe.py" "$DS_DIR/moe/sharded_moe.py"
 if [[ "$TRACE_EXISTED" == 1 ]]; then cp -p "$BACKUP/moe/trace.py" "$DS_DIR/moe/trace.py"; else rm -f "$DS_DIR/moe/trace.py"; fi
}
trap 'rollback' ERR
cp DeepSpeed-Instrumentation/moe/trace.py "$DS_DIR/moe/trace.py"
cp DeepSpeed-Instrumentation/moe/sharded_moe.py "$DS_DIR/moe/sharded_moe.py"
cp DeepSpeed-Instrumentation/runtime/engine.py "$DS_DIR/runtime/engine.py"
"$PY" -m py_compile "$DS_DIR/moe/trace.py" "$DS_DIR/moe/sharded_moe.py" "$DS_DIR/runtime/engine.py"
"$PY" - <<'PYSMOKE'
import deepspeed
from deepspeed.moe.trace import init_trace, enable_trace, mark
from deepspeed.moe.layer import MoE
print('DeepSpeed instrumentation import: OK', deepspeed.__version__)
PYSMOKE
trap - ERR
printf 'Instrumentation installed. Backup: %s\n' "$BACKUP"
