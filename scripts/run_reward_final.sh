#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 configs/reward_final_<experiment>.json" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$ROOT/$CONFIG"
fi

if [[ ! -f "$CONFIG" ]]; then
  echo "Configuration not found: $CONFIG" >&2
  exit 2
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
cd "$ROOT"
exec "${PYTHON:-python3}" -u run_final_guidance.py --config "$CONFIG"