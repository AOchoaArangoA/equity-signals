"""app.main — FastAPI application entry point.

Creates the FastAPI app with lifespan startup/shutdown hooks, registers
routers, and configures OpenAPI metadata.

Run locally::

    uvicorn app.main:app --reload --port 8000

Or via gunicorn::

    gunicorn app.main:app -c gunicorn.conf.py
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from app.core.config import get_settings
from app.routers import health, orders, signals

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Startup and shutdown hooks.

    Startup validates that all required API keys are present and logs the
    active environment.  Shutdown logs a clean termination message.
    """
    settings = get_settings()

    # Validate required keys — fail fast before accepting any traffic.
    missing: list[str] = []
    if not settings.anthropic_api_key:
        missing.append("ANTHROPIC_API_KEY")
    if not settings.api_key:
        missing.append("API_KEY")

    if missing:
        logger.error("Missing required environment variables: %s", missing)
        raise RuntimeError(f"Missing required environment variables: {missing}")

    logger.info("=" * 60)
    logger.info("equity-signals API starting up")
    logger.info("Environment : %s", settings.environment)
    logger.info("Log level   : %s", settings.log_level)
    logger.info("=" * 60)

    yield

    logger.info("equity-signals API shutting down")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Equity Signals API",
    version="0.1.0",
    description=(
        "REST API for the equity-signals pipeline.\n\n"
        "Runs mean-reversion Z-score signal scans on a pre-built Russell 2000 "
        "universe filtered by mid-cap, sector, ROE, and price-to-book criteria.\n\n"
        "**Authentication**: all `/api/v1/*` endpoints require an `X-API-Key` header."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(signals.router)
app.include_router(orders.router)
