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
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
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

EPOCHS="${EPOCHS:-30}"
DATASET_FILE="${DATASET_FILE:-data/spa-eng/spa.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"
NUM_GPUS="${NUM_GPUS:-1}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11}"
EXPECTED_PYTHON="${EXPECTED_PYTHON:-3.11}"
REQUIRE_GPU="${REQUIRE_GPU:-1}"
AUTO_PUSH_TO_GIT="${AUTO_PUSH_TO_GIT:-1}"
GIT_BRANCH="${GIT_BRANCH:-main}"

# ── Modules ──
module purge
if ! module load "$PYTHON_MODULE"; then
  echo "Failed to load module $PYTHON_MODULE"
  exit 1
fi

ACTIVE_PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python module requested: $PYTHON_MODULE"
echo "Python runtime detected: $ACTIVE_PYTHON_VERSION"
if [[ "$ACTIVE_PYTHON_VERSION" != "$EXPECTED_PYTHON" ]]; then
  echo "Expected Python $EXPECTED_PYTHON but got $ACTIVE_PYTHON_VERSION"
  echo "Set EXPECTED_PYTHON or fix PYTHON_MODULE before running."
  exit 1
fi

# ── Virtual environment (created once, reused on later runs) ──
VENV_DIR="$WORKDIR/.venv"
VENV_PY_FILE="$VENV_DIR/.python-version"
if [[ -d "$VENV_DIR" ]]; then
  EXISTING_VENV_PYTHON=""
  if [[ -x "$VENV_DIR/bin/python3" ]]; then
    EXISTING_VENV_PYTHON=$($VENV_DIR/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
  elif [[ -f "$VENV_PY_FILE" ]]; then
    EXISTING_VENV_PYTHON=$(cat "$VENV_PY_FILE")
  fi

  if [[ -z "$EXISTING_VENV_PYTHON" ]]; then
    echo "Could not determine existing venv Python version. Rebuilding venv."
    rm -rf "$VENV_DIR"
  elif [[ "$EXISTING_VENV_PYTHON" != "$ACTIVE_PYTHON_VERSION" ]]; then
    echo "Rebuilding venv: found Python $EXISTING_VENV_PYTHON, need $ACTIVE_PYTHON_VERSION"
    rm -rf "$VENV_DIR"
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  python3 -m venv "$VENV_DIR"
  echo "$ACTIVE_PYTHON_VERSION" > "$VENV_PY_FILE"
fi
source "$VENV_DIR/bin/activate"

python3 -m pip --no-cache-dir install --upgrade pip
python3 -m pip --no-cache-dir install -r requirements.txt
# Force a GPU-capable TensorFlow install on Linux compute nodes.
# The plain 'tensorflow' wheel can be CPU-only in some environments.
python3 -m pip --no-cache-dir install --upgrade "tensorflow[and-cuda]>=2.16.0"

SITE_PACKAGES=$(python3 -c 'import site; print(site.getsitepackages()[0])')
CUDA_LIB_PATHS=(
  "$SITE_PACKAGES/nvidia/cublas/lib"
  "$SITE_PACKAGES/nvidia/cuda_runtime/lib"
  "$SITE_PACKAGES/nvidia/cuda_nvrtc/lib"
  "$SITE_PACKAGES/nvidia/cudnn/lib"
  "$SITE_PACKAGES/nvidia/cufft/lib"
  "$SITE_PACKAGES/nvidia/curand/lib"
  "$SITE_PACKAGES/nvidia/cusolver/lib"
  "$SITE_PACKAGES/nvidia/cusparse/lib"
  "$SITE_PACKAGES/nvidia/nccl/lib"
  "$SITE_PACKAGES/nvidia/nvjitlink/lib"
)

for lib_path in "${CUDA_LIB_PATHS[@]}"; do
  if [[ -d "$lib_path" ]]; then
    export LD_LIBRARY_PATH="$lib_path:${LD_LIBRARY_PATH:-}"
  fi
done
echo "LD_LIBRARY_PATH configured for TensorFlow CUDA runtime."

# Map Slurm GPU assignment into CUDA visibility when not already present.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -n "${SLURM_JOB_GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$SLURM_JOB_GPUS"
fi

echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

TF_GPU_COUNT=$(python3 -c 'import tensorflow as tf; print(len(tf.config.list_physical_devices("GPU")))')
TF_BUILT_WITH_CUDA=$(python3 -c 'import tensorflow as tf; print(tf.test.is_built_with_cuda())')
echo "TensorFlow built with CUDA: $TF_BUILT_WITH_CUDA"
echo "TensorFlow GPU devices detected: $TF_GPU_COUNT"
if [[ "$REQUIRE_GPU" == "1" && "$TF_GPU_COUNT" -eq 0 ]]; then
  echo "TensorFlow does not see any GPUs. Aborting to avoid CPU-only training."
  echo "Try a different python module (default is python/3.11) and recreate the venv."
  exit 1
fi

# ── Train ──
# Dataset should be pre-downloaded on the login node:
#   python3 scripts/prepare_dataset.py --output-dir data/spa-eng
echo "========== NMT Job Configuration =========="
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Partition: ${SLURM_JOB_PARTITION:-unknown}"
echo "Account: ${SLURM_JOB_ACCOUNT:-unknown}"
echo "Epochs: $EPOCHS"
echo "Requested GPUs for training: $NUM_GPUS"
echo "Dataset file: $DATASET_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Require GPU: $REQUIRE_GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU(s):"
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
  echo "GPU info unavailable: nvidia-smi not found"
fi
echo "==========================================="

srun python3 train.py --epochs "$EPOCHS" --dataset-file "$DATASET_FILE" --output-dir "$OUTPUT_DIR" --num-gpus "$NUM_GPUS"

if [[ "$AUTO_PUSH_TO_GIT" == "1" ]]; then
  echo "Training complete. Preparing automatic Git push from HPC..."

  RUN_STAMP=$(python3 - "$OUTPUT_DIR/metadata.json" <<'PY'
import json
import sys
from pathlib import Path

meta_path = Path(sys.argv[1])
try:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
except Exception:
    print("")
else:
    print(meta.get("trained_at", ""))
PY
)

  COMMIT_MSG="Model epoch ${EPOCHS} - ${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
  bash scripts/push_artifacts_to_git.sh \
    --branch "$GIT_BRANCH" \
    --with-model \
    --use-lfs \
    --message "$COMMIT_MSG"
fi
