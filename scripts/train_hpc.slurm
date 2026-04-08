#!/bin/bash

# ============================================================
# ISU Nova HPC – NMT Transformer training job
# Cluster:   Nova  (nova.its.iastate.edu)
# Partition: instruction  (for class accounts)
# ============================================================
# IMPORTANT: Replace the --account value below with YOUR class
#            account, e.g. s2026.COMS.474.1   (semester.DEPT.NUM.section)
# ============================================================

#SBATCH --job-name=nmt-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --ntasks-per-core=1
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=instruction
#SBATCH --account=s2026.se.4390.01

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs artifacts

# ── Storage: use /work/classtmp for large files ──
# Home is only 10 GB on Nova; keep caches elsewhere.
WORKDIR="/work/classtmp/$USER/nmt-train"
mkdir -p "$WORKDIR"
export KERAS_HOME="$WORKDIR/.keras"
mkdir -p "$KERAS_HOME"

export KERAS_BACKEND=tensorflow
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=2

# ── Modules ──
module purge
module load python

# ── Virtual environment (created once, reused on later runs) ──
VENV_DIR="$WORKDIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

python3 -m pip --no-cache-dir install --upgrade pip
python3 -m pip --no-cache-dir install -r requirements.txt

# ── Train ──
# Dataset should be pre-downloaded on the login node:
#   python3 scripts/prepare_dataset.py --output-dir data/spa-eng
srun python3 train.py --epochs 30 --dataset-file data/spa-eng/spa.txt --output-dir artifacts
