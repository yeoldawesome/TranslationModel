#!/bin/bash
set -euo pipefail

# Usage examples:
# ./scripts/submit_and_watch_hpc.sh
# ./scripts/submit_and_watch_hpc.sh --epochs 30

# Default values for one-command use.
# CLI flags still override these values.
EMAIL="dnlong5@iastate.edu"
ACCOUNT="s2026.se.4390.01"
PARTITION="instruction"
EPOCHS="3"
DATASET_FILE="data/spa-eng/spa.txt"
OUTPUT_DIR="artifacts"
SCRIPT_PATH="scripts/train_hpc.slurm"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --email)
      EMAIL="$2"
      shift 2
      ;;
    --account)
      ACCOUNT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --dataset-file)
      DATASET_FILE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --script)
      SCRIPT_PATH="$2"
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [options]

Defaults in this file:
  EMAIL=dnlong5@iastate.edu
  ACCOUNT=s2026.se.4390.01
  PARTITION=instruction
  EPOCHS=3

Optional overrides:
  --email         Email address for Slurm notifications
  --account       Slurm account (e.g. s2026.se.4390.01)

Optional:
  --partition     Slurm partition (default: instruction)
  --epochs        Number of training epochs (default: 30)
  --dataset-file  Path to spa.txt (default: data/spa-eng/spa.txt)
  --output-dir    Training output directory (default: artifacts)
  --script        Slurm script path (default: scripts/train_hpc.slurm)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$EMAIL" ]]; then
  echo "EMAIL is empty. Set EMAIL in this file or pass --email." >&2
  exit 1
fi

if [[ -z "$ACCOUNT" ]]; then
  echo "ACCOUNT is empty. Set ACCOUNT in this file or pass --account." >&2
  exit 1
fi

mkdir -p logs

echo "Submitting training job..."
echo "  account: $ACCOUNT"
echo "  partition: $PARTITION"
echo "  epochs: $EPOCHS"
echo "  dataset: $DATASET_FILE"
echo "  output: $OUTPUT_DIR"
echo "  email: $EMAIL"

submit_output=$(sbatch \
  -A "$ACCOUNT" \
  -p "$PARTITION" \
  --mail-user="$EMAIL" \
  --mail-type=BEGIN,END,FAIL \
  --export=ALL,EPOCHS="$EPOCHS",DATASET_FILE="$DATASET_FILE",OUTPUT_DIR="$OUTPUT_DIR" \
  "$SCRIPT_PATH")

echo "$submit_output"
job_id=$(echo "$submit_output" | awk '{print $4}')

if [[ -z "$job_id" ]]; then
  echo "Could not parse job id from sbatch output" >&2
  exit 1
fi

log_file="logs/nmt-train-${job_id}.out"
err_file="logs/nmt-train-${job_id}.err"

echo "Job ID: $job_id"
echo "Live stdout: $log_file"
echo "Live stderr: $err_file"

echo "Waiting for log file to appear..."
for _ in $(seq 1 120); do
  if [[ -f "$log_file" ]]; then
    break
  fi
  sleep 1
done

if [[ ! -f "$log_file" ]]; then
  echo "Log file not created yet. You can monitor manually with: tail -f $log_file"
  exit 0
fi

echo "Streaming live logs (Ctrl+C to stop tail; job keeps running)..."
tail -f "$log_file"
