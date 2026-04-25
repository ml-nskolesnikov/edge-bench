SHELL := /usr/bin/env bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help

POETRY ?= poetry
PYTHON ?= python3
VENV_DIR ?= .venv
RPI_HOST ?=
SERVER_URL ?= http://localhost:8000
ECCV_RUNS ?= 100

.PHONY: help setup setup-venv install server lint test check agent-deploy clean clean-pyc \
	eccv-models eccv-benchmark eccv-rpi-benchmark check-rpi-host

help: ## Show available targets and usage examples
	@printf "edge-bench Make targets\n\n"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' "$(MAKEFILE_LIST)"
	@printf "\nExamples:\n"
	@printf "  make setup\n"
	@printf "  make server\n"
	@printf "  make eccv-rpi-benchmark RPI_HOST=pi@192.168.1.100\n"

setup: ## Install dependencies via Poetry and create runtime directories
	$(POETRY) install --with dev
	mkdir -p data/models data/scripts data/uploads
	@echo "Setup complete. Run 'make server' to start."

setup-venv: ## Legacy setup via local venv + requirements/server.txt
	$(PYTHON) -m venv "$(VENV_DIR)"
	"$(VENV_DIR)/bin/pip" install -r requirements/server.txt
	mkdir -p data/models data/scripts data/uploads
	@echo "Setup complete. Run '$(VENV_DIR)/bin/python -m server.main' to start."

install: setup ## Alias for setup

server: ## Run API server (Poetry environment)
	$(POETRY) run python -m server.main

lint: ## Run Ruff linting
	$(POETRY) run ruff check .

test: ## Run tests
	$(POETRY) run pytest -v

check: lint test ## Run lint + tests

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
	scp scripts/benchmark_tflite.py "$(RPI_HOST):~/"
	scp scripts/benchmark_eccv_models.py "$(RPI_HOST):~/"
	@echo "Running benchmark on RPi..."
	ssh "$(RPI_HOST)" "cd ~ && python3 benchmark_eccv_models.py --local --models-dir models --output eccv_results.json --csv T4_edgetpu.csv"
	@echo "Downloading results..."
	scp "$(RPI_HOST):~/eccv_results.json" "../results/"
	scp "$(RPI_HOST):~/T4_edgetpu.csv" "../results/T4_edgetpu_final.csv"
	@echo "Done! Results in: ../results/"
