#!/usr/bin/env python3
"""run_signals — download OHLCV and compute Z-score signals for the watchlist.

Reads ``config/watchlist.json``, fetches price data (Alpaca → yfinance fallback),
computes mean-reversion Z-scores, and writes the latest signal per ticker to
``output/signals_latest.json``.  This file is the handoff consumed by
``run_entry.py`` and ``run_exit.py``.

Usage::

    python scripts/run_signals.py
    python scripts/run_signals.py --days 90 --window 30 --z-entry 2.0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("run_signals")

_REPO_ROOT = Path(__file__).parent.parent
_WATCHLIST = _REPO_ROOT / "config" / "watchlist.json"
_OUTPUT_DIR = _REPO_ROOT / "output"
_SIGNALS_LATEST = _OUTPUT_DIR / "signals_latest.json"


def load_watchlist() -> dict:
    if not _WATCHLIST.exists():
        log.error("config/watchlist.json not found")
        sys.exit(1)
    return json.loads(_WATCHLIST.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute Z-score signals for watchlist tickers.")
    parser.add_argument("--days",    type=int,   default=60,  help="OHLCV lookback days (default: 60)")
    parser.add_argument("--window",  type=int,   default=20,  help="Rolling MA window (default: 20)")
    parser.add_argument("--z-entry", type=float, default=1.5, help="Z-score entry threshold (default: 1.5)")
    args = parser.parse_args()

    config = load_watchlist()
    tickers: list[str] = config["tickers"]
    strategy = config.get("strategy", {})
    days   = args.days   or strategy.get("days",    60)
    window = args.window or strategy.get("window",  20)
    z_entry = args.z_entry or strategy.get("z_entry", 1.5)

    result: dict = {
        "run_date": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tickers": {},
        "errors": [],
    }

    # ── Fetch OHLCV ──────────────────────────────────────────────────────────
    try:
        from equity_signals.data.alpaca_loader import AlpacaLoader
        prices = AlpacaLoader().get_ohlcv(tickers, days=days)
        log.info("OHLCV: %d rows for %d tickers", len(prices),
                 prices.index.get_level_values("ticker").nunique())
    except Exception as exc:
        log.warning("Alpaca failed (%s) — falling back to yfinance", exc)
        try:
            from equity_signals.data.yfinance_loader import fetch_ohlcv
            prices = fetch_ohlcv(tickers, days=days)
            log.info("yfinance fallback: %d rows", len(prices))
        except Exception as exc2:
            log.error("Both price sources failed: %s", exc2)
            result["errors"].append(str(exc2))
            _emit(result)
            sys.exit(1)

    # ── Compute signals ───────────────────────────────────────────────────────
    try:
        from equity_signals.strategies.mean_reversion import MeanReversionStrategy
        signals = MeanReversionStrategy(window=window, z_entry=z_entry).compute(prices)
    except Exception as exc:
        log.error("Signal computation failed: %s", exc)
        result["errors"].append(str(exc))
        _emit(result)
        sys.exit(1)

    # ── Latest row per ticker ─────────────────────────────────────────────────
    latest = (
        signals.groupby("ticker").last().reset_index()
    )
    for _, row in latest.iterrows():
        result["tickers"][row["ticker"]] = {
            "close":   round(float(row["close"]),   4),
            "ma":      round(float(row["ma"]),      4) if row["ma"] == row["ma"] else None,
            "std":     round(float(row["std"]),     4) if row["std"] == row["std"] else None,
            "z_score": round(float(row["z_score"]), 4) if row["z_score"] == row["z_score"] else None,
            "signal":  int(row["signal"]),
        }

    log.info("Signals computed for %d tickers", len(result["tickers"]))

    # ── Persist and emit ──────────────────────────────────────────────────────
    _OUTPUT_DIR.mkdir(exist_ok=True)
    _SIGNALS_LATEST.write_text(json.dumps(result, indent=2))
    log.info("Saved → %s", _SIGNALS_LATEST)

    _emit(result)


def _emit(result: dict) -> None:
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
