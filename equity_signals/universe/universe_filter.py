"""universe_filter — filter and rank a raw ticker universe into a value signal.

:class:`UniverseFilter` applies a four-stage pipeline to a list of ticker
symbols:

1. **Market-cap filter** — keeps only mid-cap stocks whose market cap falls
   within ``[midcap_min, midcap_max]``.
2. **Sector filter** — retains only tickers that belong to one of the
   configured ``sectors`` (pass-all when ``sectors`` is empty).
3. **ROE filter** — drops tickers with ``roe <= 0`` or missing ROE to avoid
   value traps (cheap stocks with deteriorating or negative returns).
4. **Intra-sector P/B ranking** — ranks tickers by price-to-book ratio within
   each sector (ascending, so low P/B = value) and keeps the top
   ``pb_percentile`` percent per sector as the final *value signal*.

All filter stages are fully vectorized using pandas operations.  Every stage
logs its input and output size at ``INFO`` level for observability.

Typical usage::

    from equity_signals.universe.universe_filter import UniverseFilter, FilterConfig

    config = FilterConfig(
        midcap_min=300_000_000,
        midcap_max=2_000_000_000,
        sectors=["Technology", "Industrials"],
        pb_percentile=30,
    )
    uf = UniverseFilter(config)
    df = uf.run(["AAPL", "MSFT", "GE", "F", "XOM"])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from equity_signals.data.yfinance_loader import YFinanceLoader

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_DEFAULT_MIDCAP_MIN: float = 300_000_000   # $300 M
_DEFAULT_MIDCAP_MAX: float = 2_000_000_000  # $2 B
_DEFAULT_PB_PERCENTILE: int = 30

OUTPUT_COLS: list[str] = [
    "ticker",
    "market_cap",
    "pb_ratio",
    "roe",
    "sector",
    "pb_rank_sector",
    "value_signal",
]


# ---------------------------------------------------------------------------
# FilterConfig
# ---------------------------------------------------------------------------


@dataclass
class FilterConfig:
    """Immutable configuration for :class:`UniverseFilter`.

    Args:
        midcap_min: Minimum market capitalisation in USD (inclusive).
            Defaults to ``$300 M``.
        midcap_max: Maximum market capitalisation in USD (inclusive).
            Defaults to ``$2 B``.
        sectors: List of GICS sector strings to retain.  Tickers whose sector
            does not appear in this list are dropped.  An empty list means
            *all sectors are accepted*.
        pb_percentile: Percentage of tickers per sector to retain after
            intra-sector P/B ranking.  ``30`` means the cheapest 30 % by P/B
            within each sector are kept.  Must be in the range ``(0, 100]``.

    Raises:
        ValueError: If *pb_percentile* is not in ``(0, 100]``.

    Example::

        cfg = FilterConfig(
            midcap_min=300_000_000,
            midcap_max=2_000_000_000,
            sectors=["Industrials", "Energy"],
            pb_percentile=30,
        )
    """

    midcap_min: float = _DEFAULT_MIDCAP_MIN
    midcap_max: float = _DEFAULT_MIDCAP_MAX
    sectors: list[str] = field(default_factory=list)
    pb_percentile: int = _DEFAULT_PB_PERCENTILE

    def __post_init__(self) -> None:
        if not (0 < self.pb_percentile <= 100):
            raise ValueError(
                f"pb_percentile must be in (0, 100], got {self.pb_percentile}"
            )


# ---------------------------------------------------------------------------
# UniverseFilter
# ---------------------------------------------------------------------------


class UniverseFilter:
    """Filter and rank a raw ticker universe into a value signal DataFrame.

    The filter pipeline runs in four deterministic, fully-vectorized stages:

    1. **Market-cap filter** — drops tickers outside
       ``[config.midcap_min, config.midcap_max]``.  Tickers with a missing
       ``market_cap`` are also dropped at this stage.
    2. **Sector filter** — if ``config.sectors`` is non-empty, drops tickers
       whose ``sector`` is not in the list.  Tickers with a ``None`` sector are
       dropped when a sector filter is active.
    3. **ROE filter** — drops tickers with ``roe <= 0`` or ``NaN`` to exclude
       value traps (companies with negative or zero return on equity).
    4. **P/B ranking** — within each surviving sector, tickers are ranked by
       ``pb_ratio`` ascending (lower P/B = better value).  The cheapest
       ``config.pb_percentile`` percent are kept and flagged as the value
       signal.  Tickers with missing ``pb_ratio`` are ranked last.

    Args:
        config: :class:`FilterConfig` instance describing all filter parameters.
        loader: Optional :class:`~equity_signals.data.yfinance_loader.YFinanceLoader`
            instance.  A default loader is created if not supplied.

    Example::

        uf = UniverseFilter(FilterConfig())
        df = uf.run(["AAPL", "MSFT", "GE", "CAT", "XOM"])
    """

    def __init__(
        self,
        config: FilterConfig | None = None,
        *,
        loader: YFinanceLoader | None = None,
    ) -> None:
        self._config: FilterConfig = config or FilterConfig()
        self._loader: YFinanceLoader = loader or YFinanceLoader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, tickers: list[str]) -> pd.DataFrame:
        """Execute the full filter pipeline on *tickers* and return ranked results.

        Args:
            tickers: Raw list of uppercase ticker symbols to evaluate.

        Returns:
            DataFrame sorted by sector then ``pb_rank_sector`` ascending, with
            columns defined by :data:`OUTPUT_COLS`:

            +-------------------+-----------------------------------------------+
            | Column            | Description                                   |
            +===================+===============================================+
            | ticker            | Ticker symbol (str)                           |
            +-------------------+-----------------------------------------------+
            | market_cap        | Market cap in USD (float)                     |
            +-------------------+-----------------------------------------------+
            | pb_ratio          | Price-to-book ratio (float or NaN)            |
            +-------------------+-----------------------------------------------+
            | roe               | Return on equity (float or NaN)               |
            +-------------------+-----------------------------------------------+
            | sector            | GICS sector string (str)                      |
            +-------------------+-----------------------------------------------+
            | pb_rank_sector    | Rank within sector, 1 = lowest P/B (int)      |
            +-------------------+-----------------------------------------------+
            | value_signal      | ``True`` for tickers in the top percentile    |
            +-------------------+-----------------------------------------------+

            An empty DataFrame with these columns is returned when no tickers
            survive all filters.

        Raises:
            Any unhandled exception propagated from
            :class:`~equity_signals.data.yfinance_loader.YFinanceLoader`
            (per-ticker errors are caught and returned as NaN rows internally).
        """
        if not tickers:
            logger.warning("UniverseFilter.run — called with empty ticker list")
            return pd.DataFrame(columns=OUTPUT_COLS)

        logger.info("UniverseFilter.run — input: %d tickers", len(tickers))

        # Stage 0 — fetch fundamentals
        df = self._loader.get_fundamentals(tickers)
        if df.empty:
            logger.warning("UniverseFilter.run — YFinanceLoader returned no data")
            return pd.DataFrame(columns=OUTPUT_COLS)

        logger.info(
            "NaN counts: market_cap=%d, pb_ratio=%d, roe=%d, sector=%d",
            df["market_cap"].isna().sum(),
            df["pb_ratio"].isna().sum(),
            df["roe"].isna().sum(),
            df["sector"].isna().sum(),
        )

        # Stage 1 — market-cap filter
        n_before = len(df)
        df = self._apply_midcap_filter(df)
        logger.info(
            "After midcap filter: %d tickers remaining (dropped %d)",
            len(df),
            n_before - len(df),
        )
        if df.empty:
            return pd.DataFrame(columns=OUTPUT_COLS)

        # Stage 2 — sector filter
        n_before = len(df)
        df = self._apply_sector_filter(df)
        logger.info(
            "After sector filter: %d tickers remaining (dropped %d)",
            len(df),
            n_before - len(df),
        )
        if df.empty:
            return pd.DataFrame(columns=OUTPUT_COLS)

        # Stage 3 — ROE filter (value trap exclusion)
        n_before = len(df)
        df = self._apply_roe_filter(df)
        logger.info(
            "After ROE filter: %d tickers remaining (dropped %d)",
            len(df),
            n_before - len(df),
        )
        if df.empty:
            return pd.DataFrame(columns=OUTPUT_COLS)

        # Stage 4 — intra-sector P/B ranking and percentile cut
        n_before = len(df)
        df = self._apply_pb_ranking(df)
        logger.info(
            "After P/B ranking: %d tickers remaining (dropped %d)",
            len(df),
            n_before - len(df),
        )

        return (
            df[OUTPUT_COLS]
            .sort_values(["sector", "pb_rank_sector"])
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Private filter stages
    # ------------------------------------------------------------------

    def _apply_midcap_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows whose market_cap is missing or outside the configured range.

        Args:
            df: DataFrame from :meth:`~equity_signals.data.yfinance_loader.YFinanceLoader.get_fundamentals`.

        Returns:
            Filtered copy of *df*.
        """
        before = len(df)
        df = df.dropna(subset=["market_cap"])
        dropped_null = before - len(df)
        if dropped_null:
            logger.debug(
                "Midcap filter — dropped %d ticker(s) with missing market_cap",
                dropped_null,
            )

        mask = (
            (df["market_cap"] >= self._config.midcap_min)
            & (df["market_cap"] <= self._config.midcap_max)
        )
        return df[mask].copy()

    def _apply_sector_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop rows whose sector is not in ``config.sectors``.

        When ``config.sectors`` is empty the method is a no-op.

        Args:
            df: Market-cap filtered DataFrame.

        Returns:
            Filtered copy of *df*.
        """
        if not self._config.sectors:
            return df

        df = df.dropna(subset=["sector"])
        return df[df["sector"].isin(self._config.sectors)].copy()

    def _apply_roe_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop value traps: tickers with missing or non-positive ROE.

        A stock with ``roe <= 0`` is earning nothing (or destroying capital)
        on its book value.  Including such stocks in a value signal based on
        low P/B creates a value trap — the stock appears cheap but the
        underlying business is impaired.

        Args:
            df: Sector-filtered DataFrame.

        Returns:
            Filtered copy of *df* containing only rows where ``roe > 0``.
        """
        before = len(df)
        df = df.dropna(subset=["roe"])
        dropped_null = before - len(df)
        if dropped_null:
            logger.debug(
                "ROE filter — dropped %d ticker(s) with missing ROE",
                dropped_null,
            )

        mask = df["roe"] > 0
        dropped_nonpositive = (~mask).sum()
        if dropped_nonpositive:
            logger.debug(
                "ROE filter — dropped %d ticker(s) with ROE <= 0",
                dropped_nonpositive,
            )

        return df[mask].copy()

    def _apply_pb_ranking(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rank tickers by P/B intra-sector and flag the cheapest percentile.

        Ranking is ascending (rank 1 = lowest P/B = best value).  Tickers with
        a missing ``pb_ratio`` receive the highest (worst) rank within their
        sector via ``na_option="bottom"``.

        The per-sector rank threshold is computed with vectorized
        :meth:`pandas.Series.clip` and :meth:`pandas.Series.round` operations
        so no Python-level loops are used.

        The ``value_signal`` column is ``True`` for every ticker whose rank
        falls within the top ``config.pb_percentile`` percent of its sector.
        At least one ticker per sector always receives ``value_signal=True``.

        Args:
            df: ROE-filtered DataFrame.

        Returns:
            Copy of *df* with ``pb_rank_sector`` (int) and ``value_signal``
            (bool) columns added.
        """
        df = df.copy()

        # Vectorized intra-sector ascending rank; ties share the lowest rank.
        df["pb_rank_sector"] = (
            df.groupby("sector")["pb_ratio"]
            .rank(method="min", ascending=True, na_option="bottom")
            .astype(int)
        )

        # Vectorized percentile threshold per sector.
        # clip(lower=1) guarantees at least one survivor per sector.
        sector_sizes: pd.Series = df.groupby("sector")["pb_rank_sector"].transform("max")
        threshold: pd.Series = (
            (sector_sizes * self._config.pb_percentile / 100)
            .clip(lower=1)
            .round()
            .astype(int)
        )

        df["value_signal"] = df["pb_rank_sector"] <= threshold

        return df
