#!/bin/bash
set -euo pipefail

# Push training artifacts from HPC to GitHub.
#
# Default behavior:
# - Commits metadata + vocab files
# - Skips transformer_model.keras if it exceeds GitHub's 100MB limit
#
# Optional behavior:
# - --with-model: include model if <=100MB
# - --with-model --use-lfs: track model with Git LFS and include it
#
# Usage examples:
#   bash scripts/push_artifacts_to_git.sh
#   bash scripts/push_artifacts_to_git.sh --message "Update epoch 20 artifacts"
#   bash scripts/push_artifacts_to_git.sh --with-model --use-lfs

ARTIFACTS_DIR="artifacts"
BRANCH="main"
MESSAGE="Update training artifacts"
WITH_MODEL=0
USE_LFS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --artifacts-dir)
      ARTIFACTS_DIR="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --message)
      MESSAGE="$2"
      shift 2
      ;;
    --with-model)
      WITH_MODEL=1
      shift
      ;;
    --use-lfs)
      USE_LFS=1
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [options]

Options:
  --artifacts-dir <dir>   Artifacts directory (default: artifacts)
  --branch <name>         Git branch to push (default: main)
  --message <msg>         Commit message (default: "Update training artifacts")
  --with-model            Attempt to include transformer_model.keras
  --use-lfs               Use Git LFS for transformer_model.keras (requires git-lfs)
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ ! -d ".git" ]]; then
  echo "Run this from the repository root." >&2
  exit 1
fi

if [[ ! -d "$ARTIFACTS_DIR" ]]; then
  echo "Artifacts directory not found: $ARTIFACTS_DIR" >&2
  exit 1
fi

git fetch origin

git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

FILES_TO_ADD=(
  "$ARTIFACTS_DIR/metadata.json"
  "$ARTIFACTS_DIR/eng_vocab.json"
  "$ARTIFACTS_DIR/spa_vocab.json"
)

MODEL_FILE=""
if [[ -f "$ARTIFACTS_DIR/metadata.json" ]]; then
  METADATA_MODEL_NAME=$(python3 - "$ARTIFACTS_DIR/metadata.json" <<'PY'
import json
import sys
from pathlib import Path

meta_path = Path(sys.argv[1])
try:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
except Exception:
    print("")
else:
    print(meta.get("model_filename", ""))
PY
)
  if [[ -n "$METADATA_MODEL_NAME" && -f "$ARTIFACTS_DIR/$METADATA_MODEL_NAME" ]]; then
    MODEL_FILE="$ARTIFACTS_DIR/$METADATA_MODEL_NAME"
  fi
fi

if [[ -z "$MODEL_FILE" && -f "$ARTIFACTS_DIR/transformer_model.keras" ]]; then
  MODEL_FILE="$ARTIFACTS_DIR/transformer_model.keras"
fi

if [[ -z "$MODEL_FILE" ]]; then
  LATEST_MODEL=$(ls -1t "$ARTIFACTS_DIR"/*.keras 2>/dev/null | head -n 1 || true)
  if [[ -n "$LATEST_MODEL" ]]; then
    MODEL_FILE="$LATEST_MODEL"
  fi
fi

if [[ -f "$MODEL_FILE" ]]; then
  MODEL_SIZE_BYTES=$(wc -c < "$MODEL_FILE")
  MODEL_SIZE_MB=$((MODEL_SIZE_BYTES / 1024 / 1024))

  if [[ "$WITH_MODEL" -eq 1 ]]; then
    if [[ "$USE_LFS" -eq 1 ]]; then
      if ! command -v git-lfs >/dev/null 2>&1; then
        echo "git-lfs not found. Install/load git-lfs first, then retry with --use-lfs." >&2
        exit 1
      fi
      git lfs install
      git lfs track "${MODEL_FILE}"
      FILES_TO_ADD+=(".gitattributes")
      FILES_TO_ADD+=("$MODEL_FILE")
      echo "Including model via Git LFS: $MODEL_FILE (${MODEL_SIZE_MB}MB)"
    else
      if [[ "$MODEL_SIZE_BYTES" -gt 100000000 ]]; then
        echo "Model is ${MODEL_SIZE_MB}MB and exceeds GitHub 100MB limit."
        echo "Re-run with --use-lfs to include it via Git LFS, or omit model (current default)."
      else
        FILES_TO_ADD+=("$MODEL_FILE")
        echo "Including model: $MODEL_FILE (${MODEL_SIZE_MB}MB)"
      fi
    fi
  else
    echo "Skipping model file by default: $MODEL_FILE (${MODEL_SIZE_MB}MB)"
  fi
fi

for path in "${FILES_TO_ADD[@]}"; do
  if [[ -f "$path" ]]; then
    if [[ "$path" == *.keras ]]; then
      git add -f "$path"
    else
      git add "$path"
    fi
  fi
done

if git diff --cached --quiet; then
  echo "No artifact changes to commit."
  exit 0
fi

git commit -m "$MESSAGE"
git push origin "$BRANCH"

echo "Artifacts pushed to origin/$BRANCH"
