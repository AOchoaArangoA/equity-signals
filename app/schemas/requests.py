"""app.schemas.requests — API request models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class UniverseRequest(BaseModel):
    """Parameters for generating the investable universe.

    Attributes:
        index_top_pct: Top % of Russell 2000 by index weight to consider.
        midcap_min: Minimum market cap in USD (inclusive).
        midcap_max: Maximum market cap in USD (inclusive).
        sectors: GICS sectors to include. Empty = all sectors.
        pb_percentile: Keep the cheapest pb_percentile % by P/B within each sector.
    """

    index_top_pct: float = Field(
        default=5.0,
        gt=0,
        le=20.0,
        description=(
            "Top % of Russell 2000 by index weight. "
            "5% ≈ 100 tickers (~30 s), 10% ≈ 200 tickers (~60 s), "
            "20% ≈ 400 tickers (~120 s). Capped at 20% to prevent timeouts."
        ),
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


class SignalRequest(BaseModel):
    """Parameters for a mean-reversion signal scan on an explicit ticker list.

    The caller decides which tickers to analyse — this endpoint is completely
    independent of ``/universe`` and accepts any valid US equity symbols.

    Attributes:
        tickers: Required list of uppercase ticker symbols to compute signals for.
        window: SMA lookback window in trading days.
        z_entry: Absolute Z-score threshold to enter a long signal.
        z_exit: Absolute Z-score threshold to return to neutral.
        days: OHLCV lookback in calendar days.
    """

    tickers: list[str] = Field(
        min_length=1,
        description="Ticker symbols to analyse, e.g. ['AAPL', 'MSFT'].",
        examples=[["AAPL", "MSFT", "ASIX"]],
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
        description="Z-score magnitude to trigger a long entry signal.",
    )
    z_exit: float = Field(
        default=0.5,
        gt=0,
        description="Z-score magnitude below which signal returns to neutral.",
    )
    days: int = Field(
        default=60,
        ge=10,
        le=365,
        description="OHLCV lookback in calendar days.",
    )
