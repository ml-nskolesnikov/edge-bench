# Stage order matters: Docker builds the LAST stage when no --target is given,
# so `server` must stay at the bottom. `bench` is opt-in via --target.

FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN python -m pip install --upgrade pip \
    && pip install poetry==2.2.1 poetry-plugin-export==1.9.0

# poetry.lock must be copied alongside pyproject.toml: without it `poetry
# export` re-resolves dependencies at build time and the image stops being
# reproducible. The FastAPI pin in particular is load-bearing (see README).
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --only main --without-hashes -o requirements.txt \
    && pip install --ignore-installed --prefix=/install -r requirements.txt


# ---------------------------------------------------------------------------
# Shared application layer: dependencies, code and the non-root user.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS app

LABEL org.opencontainers.image.title="edge-bench" \
      org.opencontainers.image.description="Remote ML benchmarking for Raspberry Pi + Coral Edge TPU" \
      org.opencontainers.image.source="https://github.com/ml-nskolesnikov/edge-bench" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EDGEBENCH_HOST=0.0.0.0 \
    EDGEBENCH_PORT=8000 \
    EDGEBENCH_DATABASE_PATH=/app/data/edgebench.db \
    EDGEBENCH_MODELS_DIR=/app/data/models \
    EDGEBENCH_UPLOAD_DIR=/app/data/uploads \
    EDGEBENCH_SCRIPTS_DIR=/app/data/scripts

WORKDIR /app

# curl is needed by HEALTHCHECK; no build toolchain reaches the final image.
RUN apt-get update \
    && apt-get install --no-install-recommends -y curl \
    && rm -rf /var/lib/apt/lists/*

# Runtime Python dependencies exported from the Poetry lock (main group only —
# ruff, mypy and pytest never reach the production image).
COPY --from=builder /install /usr/local

# Application code.
COPY server/ server/
COPY agent/ agent/

# Run as non-root. UID is build-time configurable so a bind-mounted ./data
# owned by the host user stays writable (see docker-compose.yml).
ARG APP_UID=10001
ARG APP_GID=10001
RUN groupadd --gid "${APP_GID}" appuser \
    && useradd --create-home --uid "${APP_UID}" --gid "${APP_GID}" appuser \
    && mkdir -p /app/data/models /app/data/uploads /app/data/scripts /app/results \
    && chown -R appuser:appuser /app


# ---------------------------------------------------------------------------
# Optional benchmark image — NOT the default target.
#
#   docker build --target bench -t edge-bench:bench .
#   docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/results:/app/results" \
#       edge-bench:bench --runs 30
#
# The server image deliberately ships no inference runtime: the server never
# runs inference, the agent does. This target exists for hardware validation
# on hosts without a usable system Python.
#
# There is no CUDA layer here on purpose — edge-bench measures TFLite on CPU
# and Coral Edge TPU. Edge TPU inside a container additionally needs
# --device /dev/bus/usb and libedgetpu on the host.
# ---------------------------------------------------------------------------
FROM app AS bench

USER root

# ai-edge-litert is the maintained x86_64 TFLite runtime (see
# agent/tflite_backend.py for the full resolution order).
RUN pip install --no-cache-dir ai-edge-litert numpy psutil

COPY scripts/benchmark_smoke.py scripts/benchmark_smoke.py

RUN chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "scripts/benchmark_smoke.py"]
CMD ["--runs", "30"]


# ---------------------------------------------------------------------------
# Default target: the API server. Must remain the last stage in this file.
# ---------------------------------------------------------------------------
FROM app AS server

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${EDGEBENCH_PORT}/api/health" || exit 1

# uvicorn installs its own SIGTERM handler, so PID 1 shuts down gracefully.
CMD ["python", "-m", "server.main"]
