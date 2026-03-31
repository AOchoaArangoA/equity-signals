#!/usr/bin/env python3
"""run_signal_scan — weekly mean-reversion signal generation (Step 4).

Reads the most recent ``output/universe_*.parquet`` produced by
:mod:`equity_signals.scripts.run_universe_scan`, selects the top-N tickers by
``pb_rank_sector`` where ``value_signal=True``, fetches their OHLCV history
via :class:`~equity_signals.data.alpaca_loader.AlpacaLoader` (with automatic
yfinance fallback), and computes Z-score mean-reversion signals.

This script covers the *fast, weekly* part of the pipeline.  It does **not**
import :class:`~equity_signals.universe.ticker_loader.TickerLoader` or
:class:`~equity_signals.data.yfinance_loader.YFinanceLoader` — it consumes
the pre-built universe parquet as its only structural input.

Usage::

    equity-signal-scan
    equity-signal-scan --top-n 10 --window 20 --z-entry 2.0
    equity-signal-scan --days 90
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

from equity_signals.data.alpaca_loader import AlpacaLoader
from equity_signals.strategies.mean_reversion import MeanReversionStrategy
from equity_signals.universe.universe_store import load_latest_universe

OUTPUT_DIR: Path = Path("output")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Weekly mean-reversion signal scan. "
            "Reads the latest universe parquet, fetches OHLCV, "
            "and computes Z-score signals for the top-N value tickers."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        metavar="N",
        help="Number of top-ranked value tickers to analyse.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=20,
        metavar="DAYS",
        help="SMA/EMA lookback window in trading days.",
    )
    parser.add_argument(
        "--z-entry",
        type=float,
        default=1.5,
        metavar="Z",
        help="Absolute Z-score threshold to enter a long signal.",
    )
    parser.add_argument(
        "--z-exit",
        type=float,
        default=0.5,
        metavar="Z",
        help="Absolute Z-score threshold to return to neutral.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        metavar="DAYS",
        help="OHLCV lookback in calendar days.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core logic (importable)
# ---------------------------------------------------------------------------


def run(
    top_n: int = 5,
    window: int = 20,
    z_entry: float = 1.5,
    z_exit: float = 0.5,
    days: int = 60,
) -> pd.DataFrame:
    """Run the signal scan and return the signals DataFrame.

    Also writes ``output/mean_reversion_signals_YYYYMMDD.csv``.

    Args:
        top_n: Number of top-ranked value tickers to analyse.
        window: MA lookback window in trading days.
        z_entry: Z-score entry threshold (long signal when ``z < -z_entry``).
        z_exit: Z-score exit threshold (neutral when ``|z| <= z_exit``).
        days: OHLCV lookback in calendar days.

    Returns:
        Signals DataFrame with columns
        ``ticker, date, close, ma, std, z_score, signal, strategy``.

    Raises:
        SystemExit: If no universe file is found or an unrecoverable error
            occurs during OHLCV fetch or signal computation.
    """
    today_str = date.today().strftime("%Y%m%d")
    t0 = time.perf_counter()

    logger.info("=" * 60)
    logger.info("Signal scan — %s", date.today().isoformat())
    logger.info(
        "Config — top_n: %d, window: %d, z_entry: %.2f, z_exit: %.2f, days: %d",
        top_n, window, z_entry, z_exit, days,
    )
    logger.info("=" * 60)

    # ---- Load universe -------------------------------------------------
    try:
        universe = load_latest_universe()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    logger.info("Universe loaded — %d tickers total", len(universe))

    # ---- Select top-N tickers -----------------------------------------
    value_df = universe[universe["value_signal"] == True]  # noqa: E712
    if value_df.empty:
        logger.warning("No tickers with value_signal=True in universe — nothing to do")
        sys.exit(0)

    top_tickers: list[str] = (
        value_df.nsmallest(top_n, "pb_rank_sector")["ticker"].tolist()
    )
    logger.info(
        "Top %d tickers by pb_rank_sector: %s",
        len(top_tickers),
        top_tickers,
    )

    # ---- Fetch OHLCV (Alpaca primary, yfinance fallback) ---------------
    try:
        prices = AlpacaLoader().get_ohlcv(top_tickers, days=days)
    except Exception as exc:
        logger.error("OHLCV fetch failed: %s", exc, exc_info=True)
        sys.exit(1)

    if prices.empty:
        logger.error("No price data returned for any ticker — aborting signal scan")
        sys.exit(1)

    # ---- Compute signals -----------------------------------------------
    try:
        strategy = MeanReversionStrategy(window=window, z_entry=z_entry, z_exit=z_exit)
        signals = strategy.compute(prices)
    except Exception as exc:
        logger.error("Signal computation failed: %s", exc, exc_info=True)
        sys.exit(1)

    # ---- Print tail and save -------------------------------------------
    print(
        signals[["ticker", "date", "close", "z_score", "signal"]]
        .tail(top_n)
        .to_string(index=False)
    )

    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        signals_path = OUTPUT_DIR / f"mean_reversion_signals_{today_str}.csv"
        signals.to_csv(signals_path, index=False)
    except Exception as exc:
        logger.error("Failed to save signals: %s", exc, exc_info=True)
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    logger.info("=" * 60)
    logger.info(
        "Signal scan complete: %d signals → %s  (%.1fs)",
        len(signals),
        signals_path,
        elapsed,
    )
    logger.info("=" * 60)
    return signals


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
        top_n=args.top_n,
        window=args.window,
        z_entry=args.z_entry,
        z_exit=args.z_exit,
        days=args.days,
    )


if __name__ == "__main__":
    main()
