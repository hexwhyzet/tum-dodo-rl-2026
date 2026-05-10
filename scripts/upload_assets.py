"""Upload large robot assets to HuggingFace Hub.

Usage:
    python scripts/upload_assets.py
"""

import os
import requests
from pathlib import Path
from huggingface_hub import HfApi
from tqdm import tqdm


def load_env():
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def upload_file_with_progress(api: HfApi, local_path: Path, path_in_repo: str, repo_id: str):
    size = local_path.stat().st_size
    url = f"https://huggingface.co/{repo_id}/upload/main/{path_in_repo}"

    with open(local_path, "rb") as f:
        with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, desc=local_path.name[:50]) as bar:
            def reader():
                while True:
                    chunk = f.read(1024 * 64)
                    if not chunk:
                        break
                    bar.update(len(chunk))
                    yield chunk

            resp = requests.post(
                url,
                headers={"Authorization": f"Bearer {api.token}"},
                data=reader(),
                stream=True,
            )
            resp.raise_for_status()


def main():
    print("Starting...", flush=True)
    load_env()
    print("Env loaded", flush=True)
    token = os.environ["HF_TOKEN"]
    repo = os.environ["HF_REPO"]
    print(f"Token: {token[:4]}*** Repo: {repo}", flush=True)

    api = HfApi(token=token)
    print("API created", flush=True)
    try:
        api.create_repo(repo, repo_type="model", exist_ok=True)
        print(f"Repo ready: {repo}", flush=True)
    except Exception as e:
        print(f"create_repo skipped: {e}", flush=True)

    local_dir = Path("assets/robot/usd")
    files = sorted(f for f in local_dir.rglob("*") if f.is_file())
    print(f"Uploading {len(files)} files...")

    for i, file in enumerate(files, 1):
        path_in_repo = f"assets/robot/usd/{file.relative_to(local_dir)}"
        print(f"[{i}/{len(files)}] {file.name} ({file.stat().st_size / 1024**2:.1f} MB)")
        upload_file_with_progress(api, file, path_in_repo, repo)

    print("Upload complete.")


if __name__ == "__main__":
    main()
