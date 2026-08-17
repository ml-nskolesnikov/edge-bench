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

help: ## Показать список целей и примеры использования
	@printf "edge-bench — цели Make\n\n"
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' "$(MAKEFILE_LIST)"
	@printf "\nПримеры:\n"
	@printf "  make install && make ci\n"
	@printf "  make run\n"
	@printf "  make benchmark-smoke BENCH_MODEL=data/models/c6_mobilenet_v2_int8.tflite\n"
	@printf "  make eccv-rpi-benchmark RPI_HOST=pi@192.168.1.100\n"
	@printf "  make docker-buildx-push DOCKER_PLATFORMS=linux/amd64,linux/arm64 IMAGE_NAME=<registry>/edge-bench IMAGE_TAG=v1\n"

# -------------------------------------------------------------------
# Установка
# -------------------------------------------------------------------

setup: ## Установить зависимости через Poetry и создать рабочие каталоги
	$(POETRY) install --with dev
	mkdir -p data/models data/scripts data/uploads results
	@echo "Установка завершена. Запустите 'make run', чтобы поднять сервер."

install: setup ## Псевдоним для setup

install-hardware: ## Установить TFLite runtime для benchmark-smoke / pytest -m hardware
	@# Намеренно вне poetry.lock: нужный пакет платформо-зависим,
	@# и CI он никогда не требуется. См. agent/tflite_backend.py.
	@arch="$$(uname -m)"; \
	if [ "$$arch" = "aarch64" ] || [ "$${arch#arm}" != "$$arch" ]; then \
		echo "Обнаружен ARM ($$arch): устанавливаю tflite-runtime"; \
		$(POETRY) run pip install tflite-runtime; \
	else \
		echo "Обнаружен x86_64 ($$arch): устанавливаю ai-edge-litert"; \
		$(POETRY) run pip install ai-edge-litert; \
	fi
	@$(POETRY) run python -c "import sys; sys.path.insert(0, 'agent'); \
	from tflite_backend import resolve_backend, backend_version; \
	_, _, s = resolve_backend(); print(f'TFLite runtime ready: {s} {backend_version(s)}')"

setup-venv: ## Устаревшая установка через локальный venv + requirements/server.txt
	$(PYTHON) -m venv "$(VENV_DIR)"
	"$(VENV_DIR)/bin/pip" install -r requirements/server.txt
	mkdir -p data/models data/scripts data/uploads
	@echo "Установка завершена. Запустите '$(VENV_DIR)/bin/python -m server.main'."

# -------------------------------------------------------------------
# Запуск
# -------------------------------------------------------------------

run: ## Запустить API-сервер (окружение Poetry)
	$(POETRY) run python -m server.main

server: run ## Псевдоним для run

dev: ## Запустить API-сервер с автоперезагрузкой на http://127.0.0.1:8000
	EDGEBENCH_DEBUG=true $(POETRY) run uvicorn server.main:app --reload --host 127.0.0.1 --port 8000

# -------------------------------------------------------------------
# Проверка — каждая цель ниже завершается с ненулевым кодом при ошибке
# -------------------------------------------------------------------

lint: ## Запустить линтинг Ruff (без автофикса)
	$(POETRY) run ruff check --no-fix .

format: ## Применить автофиксы и форматирование Ruff
	$(POETRY) run ruff check --fix .
	$(POETRY) run ruff format .

format-check: ## Проверить форматирование без изменения файлов
	$(POETRY) run ruff format --check .

typecheck: ## Запустить mypy по пакету server
	$(POETRY) run mypy

test: ## Прогнать тесты без hardware-зависимых (CPU-only)
	$(POETRY) run pytest -q

test-cov: ## Прогнать тесты с отчётом о покрытии
	$(POETRY) run pytest --cov --cov-report=term --cov-report=xml

build: ## Проверить, что приложение импортируется и собирается в чистом окружении
	$(POETRY) run python -c "import server.main; print('server.main import OK')"
	$(POETRY) build --no-interaction 2>/dev/null || echo "package-mode=false: собирать нечего (ожидаемо)"

ci: lint format-check typecheck test-cov build ## Полный CPU-only гейт качества (использовать в CI)
	@echo ""
	@echo "CI quality gate passed."

check: lint test ## Быстрая локальная проверка (lint + тесты)

# -------------------------------------------------------------------
# Hardware / бенчмарк — никогда не входит в `make ci`
# -------------------------------------------------------------------

benchmark-smoke: ## Прогнать короткий реальный TFLite-бенчмарк end-to-end (нужен install-hardware + модель)
	$(POETRY) run python scripts/benchmark_smoke.py \
		$(if $(BENCH_MODEL),--model "$(BENCH_MODEL)",) \
		--backend "$(BENCH_BACKEND)" \
		--warmup "$(BENCH_WARMUP)" \
		--runs "$(BENCH_RUNS)" \
		--output-dir "$(BENCH_OUT)"

test-hardware: ## Прогнать тесты, требующие реальный TFLite runtime (пропускаются при его отсутствии)
	@# Переопределить модель: EDGEBENCH_TEST_MODEL=/path/to/model.tflite
	$(POETRY) run pytest -m hardware -v

check-determinism: ## Проверить, что модели возвращают одинаковый выход на одинаковом входе
	$(POETRY) run python scripts/check_determinism.py $(DETERMINISM_MODELS) --runs "$(DETERMINISM_RUNS)"

platform-matrix: ## Сравнить одну модель по устройствам/бэкендам (нужны SSH и модели)
	@# Targets задаются как name=host:backend[@model]; @model обязателен для Edge TPU,
	@# которому нужна своя скомпилированная сборка *_edgetpu.tflite.
	$(POETRY) run python scripts/platform_matrix.py \
		--model "$(MATRIX_MODEL)" \
		$(foreach t,$(MATRIX_TARGETS),--target "$(t)") \
		--runs "$(BENCH_RUNS)" --warmup "$(BENCH_WARMUP)"

# -------------------------------------------------------------------
# Docker
# -------------------------------------------------------------------

docker-login: ## Обновить авторизацию Docker Hub (лечит просроченный токен)
	$(DOCKER) logout || true
	$(DOCKER) login

docker-build: ## Собрать локальный Docker-образ (одна платформа)
	$(DOCKER) build -t "$(DOCKER_IMAGE)" .

docker-build-no-cache: ## Собрать локальный Docker-образ без кэша
	$(DOCKER) build --no-cache -t "$(DOCKER_IMAGE)" .

docker-run: ## Запустить локальный Docker-образ на порту 8000
	$(DOCKER) run --rm -p 8000:8000 "$(DOCKER_IMAGE)"

docker-smoke: ## Собрать образ, запустить контейнер и проверить доступность /docs
	$(DOCKER) build -t "$(DOCKER_IMAGE)" .
	$(DOCKER) rm -f "$(SMOKE_CONTAINER)" >/dev/null 2>&1 || true
	$(DOCKER) run -d --name "$(SMOKE_CONTAINER)" -p "$(SMOKE_PORT):8000" "$(DOCKER_IMAGE)" >/dev/null
	@cleanup() { $(DOCKER) rm -f "$(SMOKE_CONTAINER)" >/dev/null 2>&1 || true; }; \
	trap cleanup EXIT; \
	echo "Ожидаю health endpoint на http://127.0.0.1:$(SMOKE_PORT)/api/health"; \
	for _ in $$(seq 1 "$(SMOKE_TIMEOUT)"); do \
		if curl -fsS "http://127.0.0.1:$(SMOKE_PORT)/api/health" >/dev/null; then \
			echo "Smoke test пройден (сервис по умолчанию отдаёт API)."; \
			exit 0; \
		fi; \
		sleep 1; \
	done; \
	echo "Smoke test провален. Последние логи контейнера:"; \
	$(DOCKER) logs --tail 120 "$(SMOKE_CONTAINER)" || true; \
	exit 1

docker-config: ## Провалидировать и отрендерить compose-файл
	$(COMPOSE) config

docker-up: ## Поднять стек через docker-compose
	$(COMPOSE) up -d --build

docker-down: ## Остановить стек
	$(COMPOSE) down

docker-logs: ## Логи сервиса edge-bench в реальном времени
	$(COMPOSE) logs -f edge-bench

docker-buildx-create: ## Создать и подготовить отдельный buildx builder
	@if ! $(DOCKER) buildx inspect "$(DOCKER_BUILDER)" >/dev/null 2>&1; then \
		$(DOCKER) buildx create --name "$(DOCKER_BUILDER)" --driver docker-container --use; \
	else \
		$(DOCKER) buildx use "$(DOCKER_BUILDER)"; \
	fi
	$(DOCKER) buildx inspect --bootstrap

docker-buildx: docker-buildx-create ## Собрать одноплатформенный образ через buildx и загрузить локально
	$(DOCKER) buildx build \
		--platform "$(DOCKER_PLATFORM_LOCAL)" \
		--tag "$(DOCKER_IMAGE)" \
		--load \
		.

docker-buildx-push: docker-buildx-create ## Мультиплатформенная сборка и push образа в реестр
	$(DOCKER) buildx build \
		--platform "$(DOCKER_PLATFORMS)" \
		--tag "$(DOCKER_IMAGE)" \
		--push \
		.

# -------------------------------------------------------------------
# Прочее
# -------------------------------------------------------------------

check-rpi-host: ## Проверить, что задан RPI_HOST
	@if [ -z "$(RPI_HOST)" ]; then \
		echo "Использование: make $${TARGET} RPI_HOST=pi@192.168.1.100"; \
		exit 1; \
	fi

agent-deploy: TARGET=agent-deploy
agent-deploy: check-rpi-host ## Развернуть агент на Raspberry Pi и запустить инсталлятор
	scp -r agent/ "$(RPI_HOST):~/edge-bench-agent/"
	ssh "$(RPI_HOST)" "cd ~/edge-bench-agent && chmod +x install.sh && ./install.sh"

clean: clean-pyc ## Удалить сгенерированные данные и локальное виртуальное окружение
	rm -rf data/ "$(VENV_DIR)"

clean-pyc: ## Удалить артефакты кэша Python
	rm -rf __pycache__ .pytest_cache .ruff_cache
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

eccv-models: ## Скопировать экспортированные ECCV-модели в ./models
	@echo "Копирую ECCV-модели в edge-bench..."
	mkdir -p models
	cp -v ../export/mobilenetv2_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/mobilenetv1_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/efficientnet_lite0_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/efficientnet_int8_ptq_*.tflite models/ 2>/dev/null || true
	cp -v ../export/resnet50_int8_ptq_*.tflite models/ 2>/dev/null || true
	@echo "Модели готовы в: models/"
	@ls -la models/*.tflite 2>/dev/null || echo "Модели не найдены"

eccv-benchmark: ## Прогнать локальный ECCV-бенчмарк против SERVER_URL
	@echo "Запускаю ECCV-бенчмарк через edge-bench..."
	cd .. && $(POETRY) run python scripts/9.9_run_edgebench.py \
		--server "$(SERVER_URL)" \
		--runs "$(ECCV_RUNS)" \
		--export-csv results/T4_edgetpu_final.csv

eccv-rpi-benchmark: TARGET=eccv-rpi-benchmark
eccv-rpi-benchmark: check-rpi-host ## Прогнать ECCV-бенчмарк на Raspberry Pi и забрать результаты
	@echo "Разворачиваю модели на $(RPI_HOST)..."
	scp models/*.tflite "$(RPI_HOST):~/models/"
	# agent/ содержит эталонные реализации бенчмарка (детерминированные входы,
	# GC-gated таймеры, timezone-aware таймстампы). Копии из scripts/ не использовать.
	scp agent/benchmark_tflite.py "$(RPI_HOST):~/"
	scp agent/benchmark_eccv_models.py "$(RPI_HOST):~/"
	@echo "Запускаю бенчмарк на RPi..."
	ssh "$(RPI_HOST)" "cd ~ && python3 benchmark_eccv_models.py --local --models-dir models --output eccv_results.json --csv T4_edgetpu.csv"
	@echo "Скачиваю результаты..."
	scp "$(RPI_HOST):~/eccv_results.json" "../results/"
	scp "$(RPI_HOST):~/T4_edgetpu.csv" "../results/T4_edgetpu_final.csv"
	@echo "Готово! Результаты в: ../results/"
