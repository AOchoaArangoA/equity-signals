#!/usr/bin/env python3
"""daily_exit_check — close positions whose mean-reversion signal has exited.

Reads the most recent mean-reversion signal CSV from ``output/``, cross-
references with open Alpaca positions, and submits market-sell orders for
any position where the latest ``signal == 0`` (Z-score has reverted to
the neutral band).

Run daily, preferably at market open, after the previous night's signal scan
has been written.

Usage::

    python scripts/daily_exit_check.py
    python scripts/daily_exit_check.py --dry-run   # log only, no orders
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
    df = pd.read_csv(path)
    return df


def _latest_signal_per_ticker(signals: pd.DataFrame) -> dict[str, int]:
    """Return the most recent signal value for each ticker."""
    latest = (
        signals.sort_values("date", ascending=True)
        .groupby("ticker")
        .last()
        .reset_index()
    )
    return dict(zip(latest["ticker"], latest["signal"].fillna(0).astype(int)))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exit positions whose mean-reversion signal has turned neutral.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log which orders would be submitted without actually sending them.",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("Daily exit check%s", " (DRY RUN)" if args.dry_run else "")
    logger.info("=" * 60)

    # ---- Load signals --------------------------------------------------
    try:
        signals = _load_latest_signals()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)

    latest = _latest_signal_per_ticker(signals)
    logger.info("Loaded signals for %d tickers", len(latest))

    # ---- Load open positions -------------------------------------------
    trader = AlpacaTrader()
    try:
        positions = trader.get_open_positions()
    except Exception as exc:
        logger.error("Failed to fetch positions: %s", exc, exc_info=True)
        sys.exit(1)

    if not positions:
        logger.info("No open positions — nothing to do")
        sys.exit(0)

    logger.info("%d open position(s): %s", len(positions), [p["ticker"] for p in positions])

    # ---- Submit exits --------------------------------------------------
    exits = 0
    for position in positions:
        ticker = position["ticker"]
        signal = latest.get(ticker, 0)

        if signal == 0:
            logger.info(
                "EXIT %s — signal=0, qty=%.4f, unrealized=%.2f%%",
                ticker, position["qty"], position["unrealized_pct"] * 100,
            )
            if not args.dry_run:
                try:
                    result = trader.submit_market_sell(ticker, int(position["qty"]))
                    logger.info(
                        "  Order submitted: id=%s status=%s",
                        result["order_id"], result["status"],
                    )
                    exits += 1
                except Exception as exc:
                    logger.error("  Failed to exit %s: %s", ticker, exc)
            else:
                exits += 1
        else:
            logger.info("HOLD %s — signal=%d, keeping position", position["ticker"], signal)

    elapsed = time.perf_counter() - t0
    logger.info(
        "Exit check complete — %d exit(s) submitted in %.1fs%s",
        exits, elapsed,
        " (dry run)" if args.dry_run else "",
    )


if __name__ == "__main__":
    main()
