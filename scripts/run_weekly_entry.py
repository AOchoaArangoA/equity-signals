#!/usr/bin/env python3
"""run_weekly_entry — enter long positions for active mean-reversion signals.

Reads the most recent signal CSV from ``output/``, identifies tickers where
the latest ``signal == 1`` (price has fallen significantly below its moving
average), skips any already held as open positions, and submits market-buy
orders for the remainder.

Run weekly, after ``equity-signal-scan`` has produced fresh signals.

Usage::

    python scripts/run_weekly_entry.py
    python scripts/run_weekly_entry.py --notional 1000   # $ per position
    python scripts/run_weekly_entry.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from glob import glob
from pathlib import Path

import pandas as pd

from equity_signals.execution import AlpacaTrader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path("output")
_DEFAULT_NOTIONAL = 1_000.0   # USD per position


def _load_latest_signals() -> pd.DataFrame:
    """Return the most recent ``mean_reversion_signals_*.csv`` from output/."""
    pattern = str(_OUTPUT_DIR / "mean_reversion_signals_*.csv")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(
            "No signal file found in output/. Run equity-signal-scan first."
        )
    path = files[-1]
    logger.info("Loading signals from %s", path)
    return pd.read_csv(path)


def _entry_tickers(signals: pd.DataFrame) -> list[str]:
    """Return tickers whose latest signal is 1 (active long entry)."""
    latest = (
        signals.sort_values("date", ascending=True)
        .groupby("ticker")
        .last()
        .reset_index()
    )
    active = latest[latest["signal"].fillna(0).astype(int) == 1]
    return active["ticker"].tolist()


def _latest_close(signals: pd.DataFrame, ticker: str) -> float | None:
    """Return the most recent close price for *ticker*, or None."""
    rows = signals[signals["ticker"] == ticker].sort_values("date")
    if rows.empty:
        return None
    return float(rows.iloc[-1]["close"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enter long positions for active mean-reversion signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--notional",
        type=float,
        default=_DEFAULT_NOTIONAL,
        metavar="USD",
        help="Dollar amount to invest per position.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log which orders would be submitted without actually sending them.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info(
        "Weekly entry check%s — notional $%.0f/position",
        " (DRY RUN)" if args.dry_run else "",
        args.notional,
    )
    logger.info("=" * 60)

    # ---- Load signals --------------------------------------------------
    try:
        signals = _load_latest_signals()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    candidates = _entry_tickers(signals)
    if not candidates:
        logger.info("No tickers with signal=1 — nothing to enter")
        sys.exit(0)

    logger.info("Signal=1 tickers: %s", candidates)

    # ---- Load open positions (skip already held) -----------------------
    trader = AlpacaTrader()
    try:
        positions = trader.get_open_positions()
    except Exception as exc:
        logger.error("Failed to fetch positions: %s", exc, exc_info=True)
        sys.exit(1)

    held = {p["ticker"] for p in positions}
    to_enter = [t for t in candidates if t not in held]
    skipped = [t for t in candidates if t in held]

    if skipped:
        logger.info("Skipping (already held): %s", skipped)
    if not to_enter:
        logger.info("All signal=1 tickers already held — nothing to enter")
        sys.exit(0)

    logger.info("Entering: %s", to_enter)

    # ---- Submit buys ---------------------------------------------------
    entries = 0
    for ticker in to_enter:
        close = _latest_close(signals, ticker)
        if close is None or close <= 0:
            logger.warning("Cannot compute qty for %s — missing close price, skipping", ticker)
            continue

        qty = round(args.notional / close, 4)
        logger.info(
            "ENTER %s — close=%.2f, notional=$%.0f → qty=%.4f",
            ticker, close, args.notional, qty,
        )

        if not args.dry_run:
            try:
                result = trader.submit_market_buy(ticker, int(qty))
                logger.info(
                    "  Order submitted: id=%s status=%s",
                    result["order_id"], result["status"],
                )
                entries += 1
            except Exception as exc:
                logger.error("  Failed to enter %s: %s", ticker, exc)
        else:
            entries += 1

    elapsed = time.perf_counter() - t0
    logger.info(
        "Entry check complete — %d entr%s submitted in %.1fs%s",
        entries,
        "y" if entries == 1 else "ies",
        elapsed,
        " (dry run)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
