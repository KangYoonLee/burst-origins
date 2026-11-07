#!/usr/bin/env bash
set -euo pipefail
VENV_PY="${VENV_PY:-$HOME/moe-env/bin/python}"
[[ -x "$VENV_PY" ]] || { echo "Missing experiment venv python: $VENV_PY" >&2; exit 1; }
OUT="${1:-$(pwd)/local_environment_snapshot}"
mkdir -p "$OUT"
"$VENV_PY" --version > "$OUT/python_version.txt" 2>&1
"$VENV_PY" -m pip freeze > "$OUT/pip_freeze_experiment_venv.txt"
"$VENV_PY" - <<'PYENV' > "$OUT/key_versions.txt"
import sys
print('Python:', sys.version)
for name in ['torch','deepspeed','numpy','matplotlib']:
    try:
        pkg=__import__(name)
        print(name, getattr(pkg,'__version__','unknown'))
    except Exception as exc:
        print(name, 'NOT AVAILABLE:', exc)
PYENV
uname -a > "$OUT/uname.txt"
printf 'Saved to %s
' "$OUT"
