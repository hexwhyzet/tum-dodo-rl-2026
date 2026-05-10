.PHONY: install train eval export sync build push-image help

# ── Local dev ─────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

lint:
	ruff check isaaclab_tasks scripts

# ── Docker ────────────────────────────────────────────────────────────────────

build:
	docker compose -f docker/docker-compose.yml build

push-image:
	@test -n "$(REGISTRY)" || (echo "Set REGISTRY=your.registry/dodo-rl" && exit 1)
	docker tag dodo-rl:dev $(REGISTRY):$(shell git rev-parse --short HEAD)
	docker push $(REGISTRY):$(shell git rev-parse --short HEAD)

# ── Training ──────────────────────────────────────────────────────────────────

train:
	@test -n "$(TASK)" || (echo "Usage: make train TASK=stand_v1" && exit 1)
	./scripts/train.sh $(TASK) $(EXTRA)

eval:
	@test -n "$(CHECKPOINT)" || (echo "Usage: make eval CHECKPOINT=runs/.../model.pt" && exit 1)
	./scripts/eval.sh $(CHECKPOINT)

export:
	@test -n "$(CHECKPOINT)" || (echo "Usage: make export CHECKPOINT=runs/.../model.pt OBS_DIM=45" && exit 1)
	python scripts/export_policy.py --checkpoint $(CHECKPOINT) --obs-dim $(OBS_DIM)

sync:
	@test -n "$(RUN)" || (echo "Usage: make sync RUN=runs/2026-05-10_stand_v1" && exit 1)
	./scripts/sync_results.sh $(RUN)

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  make install          Install Python deps locally (no Isaac Sim)"
	@echo "  make build            Build Docker image"
	@echo "  make train TASK=X     Run training in Docker"
	@echo "  make eval CHECKPOINT=X  Evaluate checkpoint"
	@echo "  make export CHECKPOINT=X OBS_DIM=N  Export to TorchScript + ONNX"
	@echo "  make sync RUN=X       Upload run artifacts to HuggingFace"
	@echo ""
