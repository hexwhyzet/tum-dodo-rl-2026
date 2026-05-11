# Isaac Lab 2.3.2 pre-built image (includes Isaac Sim 5.x + Isaac Lab)
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-lab
FROM nvcr.io/nvidia/isaac-lab:2.3.2

# ── system deps ──────────────────────────────────────────────────────────────
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    vim \
    tmux \
    rsync \
    && rm -rf /var/lib/apt/lists/*

# ── project deps ─────────────────────────────────────────────────────────────
COPY docker/requirements.txt /tmp/requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/requirements.txt

# ── Isaac Lab (editable install) ──────────────────────────────────────────────
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -e /isaac-lab/source/isaaclab

# ── copy project ─────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace


# ── env vars ─────────────────────────────────────────────────────────────────
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV ISAACLAB_PATH="/workspace/IsaacLab"
ENV DISPLAY=""
ENV OMNI_KIT_ALLOW_ROOT="1"

ENTRYPOINT ["/isaac-sim/python.sh"]
CMD ["--help"]
