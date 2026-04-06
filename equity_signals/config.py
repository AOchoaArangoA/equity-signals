"""Application configuration loaded from environment variables via pydantic-settings.

All keys are read from a `.env` file (or the process environment). Copy
`.env.example` to `.env` and fill in your values before running the package.

Required keys (no default — must be set):
    ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL

Optional keys (sensible defaults provided):
    FMP_API_KEY             — only needed for FMP price endpoints (not fundamentals)
    FMP_CHUNK_SIZE, FMP_MAX_WORKERS, FMP_CACHE_TTL_DAYS
    YFINANCE_MAX_WORKERS, YFINANCE_CACHE_TTL_DAYS
    ALPACA_FFILL_LIMIT
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralised settings for the equity-signals pipeline."""

    # ------------------------------------------------------------------
    # Required credentials
    # ------------------------------------------------------------------
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_base_url: str

    # ------------------------------------------------------------------
    # FMP credentials (optional — only needed for FMP price endpoints)
    # /key-metrics-ttm and fundamentals now sourced from yfinance.
    # ------------------------------------------------------------------
    fmp_api_key: str | None = Field(
        default=None,
        description="FMP API key. Required only for FMP price/OHLCV endpoints.",
    )

    # ------------------------------------------------------------------
    # FMP loader tuning
    # ------------------------------------------------------------------
    fmp_chunk_size: int = Field(default=50, description="Tickers per FMP batch request.")
    fmp_max_workers: int = Field(default=5, description="ThreadPoolExecutor workers for FMP.")
    fmp_cache_ttl_days: int = Field(default=7, description="Days before FMP parquet cache expires.")

    # ------------------------------------------------------------------
    # yfinance loader tuning
    # ------------------------------------------------------------------
    yfinance_max_workers: int = Field(
        default=8,
        description="ThreadPoolExecutor workers for yfinance fundamentals fetching.",
    )
    yfinance_cache_ttl_days: int = Field(
        default=7,
        description="Days before yfinance parquet cache expires.",
    )

    # ------------------------------------------------------------------
    # Alpaca loader tuning
    # ------------------------------------------------------------------
    alpaca_ffill_limit: int = Field(
        default=5,
        description="Max consecutive missing trading days to forward-fill in OHLCV data.",
    )

    # ------------------------------------------------------------------
    # Telegram notifications (optional)
    # ------------------------------------------------------------------
    telegram_bot_token: str | None = Field(
        default=None,
        description="Telegram bot token from @BotFather. Required for notifications.",
    )
    telegram_chat_id: str | None = Field(
        default=None,
        description="Telegram chat or channel ID to send messages to.",
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
