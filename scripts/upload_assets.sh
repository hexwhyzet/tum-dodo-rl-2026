#!/usr/bin/env bash
set -euo pipefail

# Load .env
set -a; source "$(dirname "$0")/../.env"; set +a

REPO="${HF_REPO}"
TOKEN="${HF_TOKEN}"

echo "Uploading to ${REPO}..."

hf upload "${REPO}" assets/robot/usd assets/robot/usd \
    --repo-type=model \
    --token="${TOKEN}"

echo "Upload complete."
