#!/usr/bin/env bash
# Run training inside Docker.
# Usage: ./scripts/train.sh stand_v1 [extra hydra overrides]

set -euo pipefail

TASK=${1:-stand_v1}
shift || true  # remaining args passed to python

# Load env vars from .env if present
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

docker compose -f docker/docker-compose.yml run --rm train \
  /workspace/isaaclab_tasks/walking/train.py \
  "env=${TASK}" \
  "$@"
