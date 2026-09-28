#!/usr/bin/env bash
# Invoke from node0. This launcher delegates process cleanup to Python.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python3 "$ROOT/run_experiment.py" "$@"
