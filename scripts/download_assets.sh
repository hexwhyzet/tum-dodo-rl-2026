#!/usr/bin/env bash
set -euo pipefail

# Load .env
set -a; source "$(dirname "$0")/../.env"; set +a

REPO="${HF_REPO}"
TOKEN="${HF_TOKEN}"

echo "Downloading assets from ${REPO}..."

hf download "${REPO}" \
    --repo-type=model \
    --token="${TOKEN}" \
    --local-dir=assets/robot/usd

echo "Download complete."
