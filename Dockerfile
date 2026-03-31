# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Builds the equity-signals wheel so the runtime stage has no build tools.
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tooling only — not present in runtime image.
RUN pip install --no-cache-dir build

# Copy source and build the wheel.
COPY . .
RUN python -m build --wheel --outdir /build/dist

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

# Non-root user for security.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy wheel from builder.
COPY --from=builder /build/dist/*.whl /app/

# Install the package and its runtime extras.
# fastapi, uvicorn, gunicorn, anthropic are declared in pyproject.toml extras.
RUN pip install --no-cache-dir /app/*.whl "fastapi[standard]" gunicorn anthropic

# Copy application code (app/ is not part of the installable wheel).
COPY app/ ./app/
COPY gunicorn.conf.py ./gunicorn.conf.py

# Drop to non-root.
USER appuser

# PORT is injected by the platform at runtime — never hardcode.
EXPOSE $PORT

CMD ["gunicorn", "app.main:app", "-c", "gunicorn.conf.py"]
