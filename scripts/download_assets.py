"""Download large robot assets from HuggingFace Hub.

Usage:
    python scripts/download_assets.py
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download


def load_env():
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def main():
    load_env()
    token = os.environ.get("HF_TOKEN")
    repo = os.environ["HF_REPO"]

    snapshot_download(
        repo_id=repo,
        repo_type="model",
        local_dir="assets/robot/usd",
        allow_patterns="assets/robot/usd/**",
        token=token,
    )
    print("Download complete.")


if __name__ == "__main__":
    main()
