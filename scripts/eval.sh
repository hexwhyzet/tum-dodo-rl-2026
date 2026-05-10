#!/usr/bin/env bash
# Evaluate a checkpoint and record video.
# Usage: ./scripts/eval.sh runs/2026-05-10_stand_v1/model.pt

set -euo pipefail

CHECKPOINT=${1:?"Usage: $0 <checkpoint_path>"}

if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

docker compose -f docker/docker-compose.yml run --rm \
  -e CHECKPOINT="${CHECKPOINT}" \
  eval
