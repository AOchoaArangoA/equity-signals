"""app.core.config — application-level settings.

Extends the equity-signals pipeline configuration with FastAPI/service
credentials.  Always access settings via :func:`get_settings` — never
instantiate :class:`Settings` directly.

Required environment variables (no defaults):
    ANTHROPIC_API_KEY   — Anthropic API key for LLM calls
    API_KEY             — Secret for ``X-API-Key`` header authentication

Optional:
    ENVIRONMENT         — ``"development"`` | ``"production"`` (default: ``"development"``)
    LOG_LEVEL           — Python logging level string (default: ``"INFO"``)
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-level settings for the FastAPI layer.

    Pipeline credentials (Alpaca, yfinance, FMP) are read directly by the
    ``equity_signals`` package via its own ``Settings`` class.  This class
    only adds the keys that are exclusive to the API layer.
    """

    # ------------------------------------------------------------------
    # Required — must be present in environment / .env
    # ------------------------------------------------------------------
    anthropic_api_key: str = Field(description="Anthropic API key for LLM service calls.")
    api_key: str = Field(description="Secret token for X-API-Key header authentication.")

    # ------------------------------------------------------------------
    # Optional — sensible defaults for development
    # ------------------------------------------------------------------
    environment: str = Field(
        default="development",
        description="Deployment environment: 'development' or 'production'.",
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level (DEBUG, INFO, WARNING, ERROR).",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Allow extra fields so the equity_signals pipeline vars in .env
        # don't cause validation errors here.
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application :class:`Settings` singleton.

    The instance is created once and reused for the lifetime of the process.
    Override in tests by clearing the cache::

        get_settings.cache_clear()
    """
    return Settings()
