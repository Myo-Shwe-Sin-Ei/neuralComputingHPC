#!/bin/bash
# ============================================================================
# SLURM job script: CPU-only variant (SVM grid search is CPU-parallel; MLP
# falls back to CPU).  Use when GPU queue is congested.
# Submit with:  sbatch scripts/hpc/run_svm.slurm
# Monitor:      squeue -u $USER
# Tail logs:    tail -f outputs/logs/slurm-<jobid>.out
# Cancel:       scancel <jobid>
# ============================================================================
#SBATCH --job-name=neco_svm
#SBATCH --output=outputs/logs/slurm-%j.out
#SBATCH --error=outputs/logs/slurm-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G

set -euo pipefail

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

mkdir -p outputs outputs/logs models

# Uncomment if your workflow requires the script to activate the venv itself:
# source venv/bin/activate

python scripts/download_data.py

jupyter nbconvert --to notebook --execute neco_starter.ipynb \
    --output neco_starter.ipynb \
    --ExecutePreprocessor.timeout=7200

echo "CPU job complete. Artifacts in outputs/ and models/."
