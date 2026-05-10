#!/usr/bin/env bash
set -euo pipefail

REPO=${DOCKER_REPO:-ultravanish/dodo-rl}
TAG=${DOCKER_TAG:-latest}

# Load .env if present
if [[ -f .env ]]; then
  set -a; source .env; set +a
fi

if [[ -z "${NGC_API_KEY:-}" ]]; then
  echo "NGC_API_KEY is not set" >&2
  exit 1
fi

echo "==> Logging into NGC..."
docker login nvcr.io -u '$oauthtoken' -p "$NGC_API_KEY"

echo "==> Building $REPO:$TAG..."
docker build -f docker/Dockerfile -t "$REPO:$TAG" .

echo "==> Pushing $REPO:$TAG..."
docker push "$REPO:$TAG"

echo "==> Done: $REPO:$TAG"
