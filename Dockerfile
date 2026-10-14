# Tollgate gateway image (Python 3.12, spec §2).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv
WORKDIR /app

# Dependencies first for layer caching.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY gateway ./gateway
COPY config ./config
COPY data ./data
COPY scripts ./scripts
COPY alembic.ini ./
COPY docker/gateway-entrypoint.sh /usr/local/bin/gateway-entrypoint.sh
RUN uv sync --frozen --no-dev && chmod +x /usr/local/bin/gateway-entrypoint.sh

RUN useradd --system --uid 10001 tollgate
USER tollgate
EXPOSE 8000
ENTRYPOINT ["gateway-entrypoint.sh"]
