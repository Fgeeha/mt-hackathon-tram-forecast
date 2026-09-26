FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_DEV=1 POLARS_MAX_THREADS=1 \
    TRAM_ROOT=/app PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# libgomp нужен lightgbm (импортируется только в пайплайне, но ставится как зависимость)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project
COPY src ./src
COPY artifacts ./artifacts
RUN uv sync --frozen

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -fs localhost:8000/api/health || exit 1
CMD ["sh", "-c", "uvicorn tram_forecast.api:app --host 0.0.0.0 --port 8000 --workers ${WORKERS:-4} --no-access-log"]
