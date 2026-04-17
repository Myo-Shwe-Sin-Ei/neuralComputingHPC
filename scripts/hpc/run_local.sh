#!/usr/bin/env bash
# Local (non-SLURM) runner. Works on macOS / Linux workstations.
# Usage: bash scripts/hpc/run_local.sh
set -euo pipefail

export PYTHONUNBUFFERED=1
# Use Agg backend if DISPLAY is unset (e.g. CI / remote shell without X forwarding).
if [ -z "${DISPLAY:-}" ]; then
    export MPLBACKEND=Agg
fi

mkdir -p outputs outputs/logs models

python scripts/download_data.py

jupyter nbconvert --to notebook --execute neco_starter.ipynb \
    --output neco_starter.ipynb \
    --ExecutePreprocessor.timeout=3600

echo "Local run complete. See outputs/ and models/."
