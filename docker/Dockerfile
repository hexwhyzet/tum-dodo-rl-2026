# Isaac Lab 2.3.2 pre-built image (includes Isaac Sim 5.x + Isaac Lab)
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab
FROM nvcr.io/nvidia/isaac-lab:2.3.2

# ── system deps ──────────────────────────────────────────────────────────────
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    tmux \
    rsync \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Claude Code ───────────────────────────────────────────────────────────────
RUN curl -fsSL https://claude.ai/install.sh | sh

# ── project deps ─────────────────────────────────────────────────────────────
COPY docker/requirements.txt /tmp/requirements.txt
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install --no-cache-dir -r /tmp/requirements.txt

# ── env vars ─────────────────────────────────────────────────────────────────
ENV DISPLAY=""
ENV OMNI_KIT_ALLOW_ROOT="1"

# Code is mounted at runtime via -v, not baked into the image
WORKDIR ${ISAACLAB_PATH}
