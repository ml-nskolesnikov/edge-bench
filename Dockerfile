FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Use Poetry lockfile as the single source of truth for runtime deps.
RUN python -m pip install --upgrade pip \
    && pip install poetry==2.2.1 poetry-plugin-export==1.9.0

COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt --only main --without-hashes -o requirements.txt \
    && pip install --ignore-installed --prefix=/install -r requirements.txt


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Runtime Python dependencies exported from Poetry lock.
COPY --from=builder /install /usr/local

# Application code.
COPY server/ server/
COPY agent/ agent/

# Run as non-root.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data/models /app/data/uploads /app/data/scripts \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "-m", "server.main"]
