#!/usr/bin/env bash
# Sync run artifacts to HuggingFace Hub or S3.
# Usage: ./scripts/sync_results.sh runs/2026-05-10_stand_v1

set -euo pipefail

RUN_DIR=${1:?"Usage: $0 <run_dir>"}

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

if [[ -n "${HF_TOKEN:-}" && -n "${HF_REPO:-}" ]]; then
  echo "Uploading to HuggingFace Hub: ${HF_REPO}"
  python - <<EOF
from huggingface_hub import HfApi
api = HfApi(token="${HF_TOKEN}")
api.upload_folder(
    folder_path="${RUN_DIR}",
    repo_id="${HF_REPO}",
    repo_type="model",
    path_in_repo="$(basename ${RUN_DIR})",
)
print("Upload complete.")
EOF
else
  echo "HF_TOKEN or HF_REPO not set — skipping upload."
  echo "Set them in .env to enable automatic checkpoint sync."
fi
