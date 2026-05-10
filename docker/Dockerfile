# Isaac Sim 6.0.0-dev2 base image from NVIDIA NGC
# https://catalog.ngc.nvidia.com/orgs/nvidia/containers/isaac-sim
FROM nvcr.io/nvidia/isaac-sim:6.0.0-dev2

# ── system deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    vim \
    tmux \
    rsync \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# ── Isaac Lab 2.x ────────────────────────────────────────────────────────────
# Pin to a specific commit for reproducibility; update intentionally.
ARG ISAACLAB_COMMIT=main
RUN git clone https://github.com/isaac-sim/IsaacLab.git /opt/isaaclab \
    && cd /opt/isaaclab \
    && git checkout ${ISAACLAB_COMMIT}

WORKDIR /opt/isaaclab

# isaaclab.sh installs Isaac Lab into the Isaac Sim Python env
RUN ./isaaclab.sh --install

# ── project deps ─────────────────────────────────────────────────────────────
COPY docker/requirements.txt /tmp/requirements.txt
RUN /isaac-sim/python.sh -m pip install --no-cache-dir -r /tmp/requirements.txt

# ── copy project ─────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY . /workspace

RUN /isaac-sim/python.sh -m pip install --no-cache-dir -e /workspace

# ── env vars ─────────────────────────────────────────────────────────────────
ENV PYTHONPATH="/workspace:${PYTHONPATH}"
ENV ISAACLAB_PATH="/opt/isaaclab"
# Headless rendering
ENV DISPLAY=""
ENV OMNI_KIT_ALLOW_ROOT="1"

ENTRYPOINT ["/isaac-sim/python.sh"]
CMD ["--help"]
