"""Export trained policy to TorchScript and ONNX.

Usage:
    python scripts/export_policy.py --checkpoint runs/2026-05-10_stand_v1/model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--obs-dim", type=int, required=True, help="Observation vector size")
    return p.parse_args()


def export(checkpoint: Path, out_dir: Path, obs_dim: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    state_dict = torch.load(checkpoint, map_location="cpu")
    # RSL-RL saves under "model_state_dict"; adjust if different runner
    actor_state = {
        k.replace("actor.", ""): v
        for k, v in state_dict.get("model_state_dict", state_dict).items()
        if k.startswith("actor.")
    }

    # Reconstruct actor network — must match training config
    from isaaclab_tasks.walking.policy import build_actor
    actor = build_actor(obs_dim=obs_dim)
    actor.load_state_dict(actor_state)
    actor.eval()

    # TorchScript
    dummy = torch.zeros(1, obs_dim)
    scripted = torch.jit.trace(actor, dummy)
    ts_path = out_dir / "policy.pt"
    scripted.save(str(ts_path))
    print(f"TorchScript saved: {ts_path}")

    # ONNX
    onnx_path = out_dir / "policy.onnx"
    torch.onnx.export(
        actor,
        dummy,
        str(onnx_path),
        input_names=["obs"],
        output_names=["action"],
        opset_version=17,
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
    )
    print(f"ONNX saved: {onnx_path}")


if __name__ == "__main__":
    args = parse_args()
    out_dir = args.out_dir or args.checkpoint.parent / "exported"
    export(args.checkpoint, out_dir, args.obs_dim)
