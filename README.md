# dodo-rl

Robot locomotion training pipeline built on Isaac Sim 6.0.0-dev2 + Isaac Lab 2.x.

## Stack

- **Simulator**: Isaac Sim 6.0.0-dev2
- **RL Framework**: Isaac Lab 2.x
- **Algorithm**: PPO (via Isaac Lab's RSL-RL integration)
- **Logging**: TensorBoard + WandB
- **Config**: Hydra + OmegaConf
- **Infra**: Docker + Vast.ai

## Roadmap

| Stage | Task | Status |
|-------|------|--------|
| 1 | Standing policy | 🔲 |
| 2 | Balance with perturbations | 🔲 |
| 3 | Forward walking | 🔲 |
| 4 | Terrain randomization | 🔲 |
| 5 | Domain randomization | 🔲 |
| 6 | Sim-to-real | 🔲 |

## Quickstart

```bash
cp .env.example .env
# fill in WANDB_API_KEY etc.

# Local dev (no Isaac Sim required for config editing)
make install

# Train on Vast.ai
make vast-train TASK=stand_v1
```

## Project structure

```
assets/          robot URDF/USD and terrain meshes
configs/         Hydra configs (env, train, domain_rand)
isaaclab_tasks/  custom Isaac Lab environments
scripts/         shell scripts for train/eval/sync
docker/          Dockerfile and compose
runs/            experiment artifacts (gitignored except structure)
```

## Reproducing an experiment

```bash
git checkout <commit>
docker pull <image-from-runs/experiment/docker_image.txt>
make train CONFIG=runs/2026-05-10_stand_v1/config.yaml
```

## Experiment tracking

Each run saves to `runs/YYYY-MM-DD_<name>/`:
- `config.yaml` — full resolved Hydra config
- `git_commit.txt` — exact commit hash
- `docker_image.txt` — docker image digest
- `model.pt` — final checkpoint
- `metrics.csv` — reward components per iteration
- `videos/` — policy rollout videos
