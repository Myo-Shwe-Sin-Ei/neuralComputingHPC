#!/bin/bash
# ============================================================================
# SLURM job script: MLP + SVM training (GPU)
# Submit with:  sbatch scripts/hpc/run_mlp.slurm
# Monitor:      squeue -u $USER
# Tail logs:    tail -f outputs/logs/slurm-<jobid>.out
# Cancel:       scancel <jobid>
# ============================================================================
#SBATCH --job-name=neco_mlp
#SBATCH --output=outputs/logs/slurm-%j.out
#SBATCH --error=outputs/logs/slurm-%j.err
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

mkdir -p outputs outputs/logs models

# Uncomment if your workflow requires the script to activate the venv itself:
# source venv/bin/activate

# Download data if missing (no-op if already present).
python scripts/download_data.py

# Run the main notebook end-to-end. Produces outputs/*.png, models/*, config.json.
jupyter nbconvert --to notebook --execute neco_starter.ipynb \
    --output neco_starter.ipynb \
    --ExecutePreprocessor.timeout=3600

echo "MLP/SVM training job complete. Artifacts in outputs/ and models/."
