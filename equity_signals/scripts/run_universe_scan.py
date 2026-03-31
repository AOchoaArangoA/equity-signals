#!/usr/bin/env python3
"""run_universe_scan — structural equity universe analysis (Steps 1–3).

Downloads the Russell 2000 constituent list, fetches fundamental data via
yfinance, and applies the four-stage UniverseFilter pipeline (mid-cap →
sector → ROE → intra-sector P/B ranking).  Results are written to
``output/universe_YYYYMMDD.parquet`` and ``.csv``.

This script covers the *slow, structural* part of the pipeline.  Fundamentals
change quarterly, so it is designed to run **monthly or on-demand**.  Its
output is consumed by :mod:`equity_signals.scripts.run_signal_scan`.

Usage::

    equity-universe-scan
    equity-universe-scan --index-top-pct 10 --sectors Technology Industrials
    equity-universe-scan --index-top-pct 100   # full Russell 2000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from equity_signals.universe.ticker_loader import TickerLoader
from equity_signals.universe.universe_filter import FilterConfig, UniverseFilter

OUTPUT_DIR: Path = Path("output")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Structural equity universe analysis. "
            "Downloads the Russell 2000, fetches fundamentals via yfinance, "
            "and applies mid-cap / sector / ROE / P/B filters."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-top-pct",
        type=float,
        default=20.0,
        metavar="PCT",
        help=(
            "Use only the top PCT%% of Russell 2000 tickers by index weight. "
            "Pass 100 for the full universe (~2 000 tickers)."
        ),
    )
    parser.add_argument(
        "--midcap-min",
        type=float,
        default=300_000_000,
        metavar="USD",
        help="Minimum market capitalisation in USD (inclusive).",
    )
    parser.add_argument(
        "--midcap-max",
        type=float,
        default=2_000_000_000,
        metavar="USD",
        help="Maximum market capitalisation in USD (inclusive).",
    )
    parser.add_argument(
        "--sectors",
        nargs="*",
        default=[],
        metavar="SECTOR",
        help="GICS sectors to include. Omit for all sectors.",
    )
    parser.add_argument(
        "--pb-percentile",
        type=float,
        default=30.0,
        metavar="PCT",
        help="Keep the cheapest PCT%% of tickers by P/B ratio within each sector.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic (importable)
# ---------------------------------------------------------------------------


def run(
    index_top_pct: float = 20.0,
    midcap_min: float = 300_000_000,
    midcap_max: float = 2_000_000_000,
    sectors: list[str] | None = None,
    pb_percentile: float = 30.0,
) -> pd.DataFrame:
    """Run the structural universe scan and return the filtered DataFrame.

    Also writes ``output/universe_YYYYMMDD.parquet`` and ``.csv``.

    Args:
        index_top_pct: Percentage of Russell 2000 to use by index weight.
        midcap_min: Minimum market cap in USD.
        midcap_max: Maximum market cap in USD.
        sectors: GICS sectors to include.  ``None`` or empty = all sectors.
        pb_percentile: P/B percentile cutoff per sector.

    Returns:
        Filtered universe DataFrame with columns
        ``ticker, market_cap, pb_ratio, roe, sector, pb_rank_sector,
        value_signal``.

    Raises:
        SystemExit: On unrecoverable errors (ticker download failure,
            filter pipeline crash, or save failure).
    """
    if sectors is None:
        sectors = []

    today_str = date.today().strftime("%Y%m%d")
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Universe scan — %s", date.today().isoformat())
    logger.info(
        "Config — index_top_pct: %.0f%%, midcap: [%.0fM, %.0fM], "
        "sectors: %s, pb_percentile: %.0f%%",
        index_top_pct,
        midcap_min / 1e6,
        midcap_max / 1e6,
        sectors or "(all)",
        pb_percentile,
    )
    logger.info("=" * 60)

    # ---- Step 1: load tickers ------------------------------------------
    try:
        loader = TickerLoader()
        tickers = (
            loader.get_top_pct(index_top_pct)
            if index_top_pct < 100.0
            else loader.get_russell2000()
        )
    except Exception as exc:
        logger.error("Failed to load ticker universe: %s", exc, exc_info=True)
        sys.exit(1)

    logger.info("Step 1 complete — %d tickers loaded", len(tickers))

    # ---- Step 2: filter ------------------------------------------------
    config = FilterConfig(
        midcap_min=midcap_min,
        midcap_max=midcap_max,
        sectors=sectors,
        pb_percentile=int(pb_percentile),
    )
    try:
        df = UniverseFilter(config).run(tickers)
    except Exception as exc:
        logger.error("Filter pipeline failed: %s", exc, exc_info=True)
        sys.exit(1)

    value_count = int(df["value_signal"].sum()) if not df.empty else 0
    logger.info(
        "Step 2 complete — %d survivors, %d with value_signal=True",
        len(df),
        value_count,
    )

    # ---- Step 3: save --------------------------------------------------
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        parquet_path = OUTPUT_DIR / f"universe_{today_str}.parquet"
        csv_path = OUTPUT_DIR / f"universe_{today_str}.csv"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)
    except Exception as exc:
        logger.error("Failed to save output: %s", exc, exc_info=True)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info(
        "Universe scan complete: %d tickers → %s  (%.1fs)",
        value_count,
        parquet_path,
        elapsed,
    )
    logger.info("=" * 60)
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
    )
    args = _parse_args()
    run(
        index_top_pct=args.index_top_pct,
        midcap_min=args.midcap_min,
        midcap_max=args.midcap_max,
        sectors=args.sectors or [],
        pb_percentile=args.pb_percentile,
    )


if __name__ == "__main__":
    main()
