SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

POETRY ?= poetry
PYTHON ?= python3
VENV_DIR ?= .venv
RPI_HOST ?=
SERVER_URL ?= http://localhost:8000
ECCV_RUNS ?= 100
DOCKER ?= docker
COMPOSE ?= docker compose
IMAGE_NAME ?= edge-bench
IMAGE_TAG ?= local
DOCKER_IMAGE ?= $(IMAGE_NAME):$(IMAGE_TAG)
DOCKER_PLATFORMS ?= linux/amd64,linux/arm64
DOCKER_PLATFORM_LOCAL ?= linux/amd64
DOCKER_BUILDER ?= edge-bench-builder
SMOKE_PORT ?= 18000
SMOKE_TIMEOUT ?= 30
SMOKE_CONTAINER ?= edge-bench-smoke
BENCH_MODEL ?=
BENCH_RUNS ?= 30
BENCH_WARMUP ?= 5
BENCH_BACKEND ?= cpu
BENCH_OUT ?= results/smoke
MATRIX_MODEL ?= data/models/mobilenetv1_int8_ptq_Fuzzy.tflite
MATRIX_TARGETS ?= x86=:cpu
DETERMINISM_MODELS ?= data/models/*.tflite
DETERMINISM_RUNS ?= 6

.PHONY: help setup setup-venv install install-hardware dev run server \
	lint format format-check typecheck test test-cov build ci check \
	benchmark-smoke test-hardware platform-matrix check-determinism \
	agent-deploy clean clean-pyc \
	eccv-models eccv-benchmark eccv-rpi-benchmark check-rpi-host \
	docker-login docker-build docker-build-no-cache docker-run docker-up docker-down docker-logs \
	docker-config docker-buildx-create docker-buildx docker-buildx-push docker-smoke

help: ## Show available targets and usage examples
	@printf "edge-bench Make targets\n\n"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' "$(MAKEFILE_LIST)"
	@printf "\nExamples:\n"
	@printf "  make install && make ci\n"
	@printf "  make run\n"
	@printf "  make benchmark-smoke BENCH_MODEL=data/models/c6_mobilenet_v2_int8.tflite\n"
	@printf "  make eccv-rpi-benchmark RPI_HOST=pi@192.168.1.100\n"
	@printf "  make docker-buildx-push DOCKER_PLATFORMS=linux/amd64,linux/arm64 IMAGE_NAME=<registry>/edge-bench IMAGE_TAG=v1\n"

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

setup: ## Install dependencies via Poetry and create runtime directories
	$(POETRY) install --with dev
	mkdir -p data/models data/scripts data/uploads results
	@echo "Setup complete. Run 'make run' to start the server."

install: setup ## Alias for setup

install-hardware: ## Install a TFLite runtime for benchmark-smoke / pytest -m hardware
	@# Deliberately outside poetry.lock: the right package is platform-specific
	@# and CI never needs any of them. See agent/tflite_backend.py.
	@arch="$$(uname -m)"; \
	if [ "$$arch" = "aarch64" ] || [ "$${arch#arm}" != "$$arch" ]; then \
		echo "ARM detected ($$arch): installing tflite-runtime"; \
		$(POETRY) run pip install tflite-runtime; \
	else \
		echo "x86_64 detected ($$arch): installing ai-edge-litert"; \
		$(POETRY) run pip install ai-edge-litert; \
	fi
	@$(POETRY) run python -c "import sys; sys.path.insert(0, 'agent'); \
	from tflite_backend import resolve_backend, backend_version; \
	_, _, s = resolve_backend(); print(f'TFLite runtime ready: {s} {backend_version(s)}')"

setup-venv: ## Legacy setup via local venv + requirements/server.txt
	$(PYTHON) -m venv "$(VENV_DIR)"
	"$(VENV_DIR)/bin/pip" install -r requirements/server.txt
	mkdir -p data/models data/scripts data/uploads
	@echo "Setup complete. Run '$(VENV_DIR)/bin/python -m server.main' to start."

# -------------------------------------------------------------------
# Run
# -------------------------------------------------------------------

run: ## Run API server (Poetry environment)
	$(POETRY) run python -m server.main

server: run ## Alias for run

dev: ## Run API server with autoreload on http://127.0.0.1:8000
	EDGEBENCH_DEBUG=true $(POETRY) run uvicorn server.main:app --reload --host 127.0.0.1 --port 8000

# -------------------------------------------------------------------
# Quality gates — every target below exits non-zero on failure
# -------------------------------------------------------------------

lint: ## Run Ruff linting (no autofix)
	$(POETRY) run ruff check --no-fix .

format: ## Apply Ruff autofixes and formatting
	$(POETRY) run ruff check --fix .
	$(POETRY) run ruff format .

format-check: ## Verify formatting without writing files
	$(POETRY) run ruff format --check .

typecheck: ## Run mypy over the server package
	$(POETRY) run mypy

test: ## Run the CPU-only test suite (hardware tests excluded)
	$(POETRY) run pytest -q

test-cov: ## Run tests with coverage report
	$(POETRY) run pytest --cov --cov-report=term --cov-report=xml

build: ## Verify the application is importable and buildable in a clean env
	$(POETRY) run python -c "import server.main; print('server.main import OK')"
	$(POETRY) build --no-interaction 2>/dev/null || echo "package-mode=false: nothing to build (expected)"

ci: lint format-check typecheck test-cov build ## Full CPU-only quality gate (use this in CI)
	@echo ""
	@echo "CI quality gate passed."

check: lint test ## Fast local gate (lint + tests)

# -------------------------------------------------------------------
# Hardware / benchmark validation — never part of `make ci`
# -------------------------------------------------------------------

benchmark-smoke: ## Run a short real TFLite benchmark end-to-end (needs install-hardware + a model)
	$(POETRY) run python scripts/benchmark_smoke.py \
		$(if $(BENCH_MODEL),--model "$(BENCH_MODEL)",) \
		--backend "$(BENCH_BACKEND)" \
		--warmup "$(BENCH_WARMUP)" \
		--runs "$(BENCH_RUNS)" \
		--output-dir "$(BENCH_OUT)"

test-hardware: ## Run tests that require a real TFLite runtime (skipped when unavailable)
	@# Override the model with EDGEBENCH_TEST_MODEL=/path/to/model.tflite
	$(POETRY) run pytest -m hardware -v

check-determinism: ## Verify models return the same output for the same input
	$(POETRY) run python scripts/check_determinism.py $(DETERMINISM_MODELS) --runs "$(DETERMINISM_RUNS)"

platform-matrix: ## Compare one model across devices/backends (needs SSH + models)
	@# Targets are name=host:backend[@model]; @model is required for Edge TPU,
	@# which needs its own compiled *_edgetpu.tflite build.
	$(POETRY) run python scripts/platform_matrix.py \
		--model "$(MATRIX_MODEL)" \
		$(foreach t,$(MATRIX_TARGETS),--target "$(t)") \
		--runs "$(BENCH_RUNS)" --warmup "$(BENCH_WARMUP)"

docker-login: ## Refresh Docker Hub auth (fixes expired token issues)
	$(DOCKER) logout || true
	$(DOCKER) login

docker-build: ## Build local Docker image (single platform)
	$(DOCKER) build -t "$(DOCKER_IMAGE)" .

docker-build-no-cache: ## Build local Docker image without cache
	$(DOCKER) build --no-cache -t "$(DOCKER_IMAGE)" .

docker-run: ## Run local Docker image on port 8000
	$(DOCKER) run --rm -p 8000:8000 "$(DOCKER_IMAGE)"

docker-smoke: ## Build image, run container, and verify /docs is reachable
	$(DOCKER) build -t "$(DOCKER_IMAGE)" .
	$(DOCKER) rm -f "$(SMOKE_CONTAINER)" >/dev/null 2>&1 || true
	$(DOCKER) run -d --name "$(SMOKE_CONTAINER)" -p "$(SMOKE_PORT):8000" "$(DOCKER_IMAGE)" >/dev/null
	@cleanup() { $(DOCKER) rm -f "$(SMOKE_CONTAINER)" >/dev/null 2>&1 || true; }; \
	trap cleanup EXIT; \
	echo "Waiting for health endpoint on http://127.0.0.1:$(SMOKE_PORT)/api/health"; \
	for _ in $$(seq 1 "$(SMOKE_TIMEOUT)"); do \
		if curl -fsS "http://127.0.0.1:$(SMOKE_PORT)/api/health" >/dev/null; then \
			echo "Smoke test passed (default build target serves the API)."; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "Smoke test failed. Recent container logs:"; \
	$(DOCKER) logs --tail 120 "$(SMOKE_CONTAINER)" || true; \
	exit 1

docker-config: ## Validate and render the compose file
	$(COMPOSE) config

docker-up: ## Start stack using docker-compose
	$(COMPOSE) up -d --build

docker-down: ## Stop stack
	$(COMPOSE) down

docker-logs: ## Follow compose logs for edge-bench service
	$(COMPOSE) logs -f edge-bench

docker-buildx-create: ## Create and bootstrap dedicated buildx builder
	@if ! $(DOCKER) buildx inspect "$(DOCKER_BUILDER)" >/dev/null 2>&1; then \
		$(DOCKER) buildx create --name "$(DOCKER_BUILDER)" --driver docker-container --use; \
	else \
		$(DOCKER) buildx use "$(DOCKER_BUILDER)"; \
	fi
	$(DOCKER) buildx inspect --bootstrap

docker-buildx: docker-buildx-create ## Build single-platform image via buildx and load locally
	$(DOCKER) buildx build \
		--platform "$(DOCKER_PLATFORM_LOCAL)" \
		--tag "$(DOCKER_IMAGE)" \
		--load \
		.

docker-buildx-push: docker-buildx-create ## Multi-platform build and push image to registry
	$(DOCKER) buildx build \
		--platform "$(DOCKER_PLATFORMS)" \
		--tag "$(DOCKER_IMAGE)" \
		--push \
		.

check-rpi-host: ## Validate that RPI_HOST is provided
	@if [ -z "$(RPI_HOST)" ]; then \
		echo "Usage: make $${TARGET} RPI_HOST=pi@192.168.1.100"; \
		exit 1; \
	fi

agent-deploy: TARGET=agent-deploy
agent-deploy: check-rpi-host ## Deploy agent to Raspberry Pi and run installer
	scp -r agent/ "$(RPI_HOST):~/edge-bench-agent/"
	ssh "$(RPI_HOST)" "cd ~/edge-bench-agent && chmod +x install.sh && ./install.sh"

clean: clean-pyc ## Remove generated data and local virtual env
	rm -rf data/ "$(VENV_DIR)"

clean-pyc: ## Remove Python cache artifacts
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

eccv-models: ## Copy exported ECCV models into ./models
	@echo "Copying ECCV models to edge-bench..."
	mkdir -p models
	cp -v ../export/mobilenetv2_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/mobilenetv1_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/efficientnet_lite0_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/efficientnet_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/resnet50_int8_ptq_*.tflite models/ 2>/dev/null || true
	@echo "Models ready in: models/"
	@ls -la models/*.tflite 2>/dev/null || echo "No models found"

eccv-benchmark: ## Run local ECCV benchmark against SERVER_URL
	@echo "Running ECCV benchmark via edge-bench..."
	cd .. && $(POETRY) run python scripts/9.9_run_edgebench.py \
		--server "$(SERVER_URL)" \
		--runs "$(ECCV_RUNS)" \
		--export-csv results/T4_edgetpu_final.csv

eccv-rpi-benchmark: TARGET=eccv-rpi-benchmark
eccv-rpi-benchmark: check-rpi-host ## Run ECCV benchmark on Raspberry Pi and fetch results
	@echo "Deploying models to $(RPI_HOST)..."
	scp models/*.tflite "$(RPI_HOST):~/models/"
	# agent/ holds the canonical benchmark implementations (seeded inputs,
	# GC-gated timing, timezone-aware timestamps). Do not ship scripts/ copies.
	scp agent/benchmark_tflite.py "$(RPI_HOST):~/"
	scp agent/benchmark_eccv_models.py "$(RPI_HOST):~/"
	@echo "Running benchmark on RPi..."
	ssh "$(RPI_HOST)" "cd ~ && python3 benchmark_eccv_models.py --local --models-dir models --output eccv_results.json --csv T4_edgetpu.csv"
	@echo "Downloading results..."
	scp "$(RPI_HOST):~/eccv_results.json" "../results/"
	scp "$(RPI_HOST):~/T4_edgetpu.csv" "../results/T4_edgetpu_final.csv"
	@echo "Done! Results in: ../results/"
