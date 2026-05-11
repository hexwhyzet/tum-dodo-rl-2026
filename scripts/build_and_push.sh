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

if [[ "$(uname -m)" == "arm64" || "$(uname -m)" == "aarch64" ]]; then
  echo "==> ARM detected, using buildx for linux/amd64..."
  docker buildx build --platform linux/amd64 -f docker/Dockerfile -t "$REPO:$TAG" \
    --cache-from type=registry,ref="$REPO:cache-amd64" \
    --cache-to type=registry,ref="$REPO:cache-amd64",mode=max \
    --push .
else
  echo "==> x86_64 detected, using plain docker build..."
  docker build -f docker/Dockerfile -t "$REPO:$TAG" .
  docker push "$REPO:$TAG"
fi

echo "==> Done: $REPO:$TAG"
