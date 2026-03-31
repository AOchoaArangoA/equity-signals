"""app.schemas.requests — API request models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SignalRequest(BaseModel):
    """Parameters for a mean-reversion signal scan.

    All fields are optional — defaults mirror the pipeline's own defaults so
    a bare ``POST /api/v1/signals`` with an empty body produces a sensible
    result.

    Attributes:
        sectors: GICS sectors to include.  Empty list = all sectors.
        pb_percentile: Keep the cheapest ``pb_percentile`` % of tickers by
            price-to-book ratio within each sector.
        midcap_min: Minimum market capitalisation in USD (inclusive).
        midcap_max: Maximum market capitalisation in USD (inclusive).
        top_n: Number of top-ranked value tickers to compute signals for.
        window: SMA lookback window in trading days.
        z_entry: Absolute Z-score threshold to enter a long signal.
        days: OHLCV lookback in calendar days.
    """

    sectors: list[str] = Field(
        default=[],
        description="GICS sectors to include. Empty = all sectors.",
        examples=[["Technology", "Industrials"]],
    )
    pb_percentile: float = Field(
        default=30.0,
        ge=1.0,
        le=100.0,
        description="Keep cheapest pb_percentile % within each sector.",
    )
    midcap_min: float = Field(
        default=300_000_000,
        gt=0,
        description="Minimum market cap in USD.",
    )
    midcap_max: float = Field(
        default=2_000_000_000,
        gt=0,
        description="Maximum market cap in USD.",
    )
    top_n: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top tickers to generate signals for.",
    )
    window: int = Field(
        default=20,
        ge=5,
        le=200,
        description="SMA lookback window in trading days.",
    )
    z_entry: float = Field(
        default=1.5,
        gt=0,
        description="Z-score magnitude to trigger a long signal.",
    )
    days: int = Field(
        default=60,
        ge=10,
        le=365,
        description="OHLCV lookback in calendar days.",
    )
