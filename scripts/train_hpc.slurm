#!/bin/bash
#SBATCH --job-name=nmt-train
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
##SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs artifacts

# Cache location for Keras downloads and metadata.
if [[ -n "${SCRATCH:-}" ]]; then
  export KERAS_HOME="$SCRATCH/.keras"
else
  export KERAS_HOME="$SLURM_SUBMIT_DIR/.keras"
fi
mkdir -p "$KERAS_HOME"

export KERAS_BACKEND=tensorflow
export PYTHONUNBUFFERED=1

module purge
module load python/3.11
# module load cuda/12.2

if [[ ! -d ".venv" ]]; then
  python -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

# Uses local dataset prepared on login node to avoid compute-node internet dependency.
srun python train.py --epochs 30 --dataset-file data/spa-eng/spa.txt --output-dir artifacts
