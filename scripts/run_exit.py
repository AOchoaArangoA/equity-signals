#!/usr/bin/env python3
"""run_exit — check open positions and submit exits when conditions are met.

Exit conditions (evaluated in order):
  1. Stop-loss: ``unrealized_pct <= -stop_loss`` → exit immediately.
  2. Z-score reversal: ``z_score > -z_exit`` → price has mean-reverted, take profit.

Z-scores are read from ``output/signals_latest.json`` when fresh (< 4 hours).
If stale or missing, the script fetches the last 25 days of OHLCV and
computes the Z-score inline before deciding.

Usage::

    python scripts/run_exit.py                  # live — submits sells
    python scripts/run_exit.py --dry-run        # log only, no orders
    python scripts/run_exit.py --z-exit 0.3     # tighter exit band
    python scripts/run_exit.py --stop-loss 0.05 # tighter stop
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("run_exit")

_REPO_ROOT = Path(__file__).parent.parent
_OUTPUT_DIR = _REPO_ROOT / "output"
_SIGNALS_LATEST = _OUTPUT_DIR / "signals_latest.json"
_SIGNALS_MAX_AGE_H = 4
_INLINE_DAYS = 25


def _load_cached_signals() -> dict | None:
    """Return cached signals dict if fresh, else None."""
    if not _SIGNALS_LATEST.exists():
        return None
    data = json.loads(_SIGNALS_LATEST.read_text())
    run_date = datetime.fromisoformat(data["run_date"].replace("Z", "+00:00"))
    age_h = (datetime.now(timezone.utc) - run_date).total_seconds() / 3600
    if age_h > _SIGNALS_MAX_AGE_H:
        log.info("signals_latest.json is %.1f h old — will compute inline", age_h)
        return None
    return data.get("tickers", {})


def _compute_z_inline(ticker: str, window: int = 20) -> float | None:
    """Fetch last _INLINE_DAYS of OHLCV and return the latest Z-score."""
    try:
        from equity_signals.data.alpaca_loader import AlpacaLoader
        prices = AlpacaLoader().get_ohlcv([ticker], days=_INLINE_DAYS)
    except Exception:
        try:
            from equity_signals.data.yfinance_loader import fetch_ohlcv
            prices = fetch_ohlcv([ticker], days=_INLINE_DAYS)
        except Exception as exc:
            log.warning("Inline price fetch failed for %s: %s", ticker, exc)
            return None

    try:
        from equity_signals.strategies.mean_reversion import MeanReversionStrategy
        signals = MeanReversionStrategy(window=window).compute(prices)
        ticker_rows = signals[signals["ticker"] == ticker]
        if ticker_rows.empty:
            return None
        return float(ticker_rows.iloc[-1]["z_score"])
    except Exception as exc:
        log.warning("Inline Z-score failed for %s: %s", ticker, exc)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check open positions and exit if conditions met.")
    parser.add_argument("--z-exit",    type=float, default=0.5,  help="Exit when z_score > -z_exit (default: 0.5)")
    parser.add_argument("--stop-loss", type=float, default=0.07, help="Stop-loss fraction (default: 0.07)")
    parser.add_argument("--window",    type=int,   default=20,   help="MA window for inline Z-score (default: 20)")
    parser.add_argument("--dry-run",   action="store_true",      help="Log exits without submitting orders")
    args = parser.parse_args()

    result: dict = {
        "run_date":          datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dry_run":           args.dry_run,
        "positions_checked": 0,
        "exits_triggered":   [],
        "positions_held":    [],
        "errors":            [],
    }

    # ── Open positions ────────────────────────────────────────────────────────
    try:
        from equity_signals.execution.alpaca_trader import AlpacaTrader
        trader = AlpacaTrader()
        positions = trader.get_open_positions()
    except Exception as exc:
        log.error("Failed to fetch positions: %s", exc)
        result["errors"].append(str(exc))
        _emit(result)
        sys.exit(1)

    if not positions:
        log.info("No open positions — nothing to check")
        _emit(result)
        return

    result["positions_checked"] = len(positions)
    log.info("%d open position(s): %s", len(positions), [p["ticker"] for p in positions])

    cached = _load_cached_signals()

    # ── Evaluate each position ────────────────────────────────────────────────
    for pos in positions:
        ticker       = pos["ticker"]
        qty          = pos["qty"]
        unreal_pct   = pos["unrealized_pct"]  # decimal fraction from Alpaca

        # 1. Stop-loss
        stop_hit = unreal_pct <= -args.stop_loss

        # 2. Z-score
        if cached and ticker in cached:
            z = cached[ticker].get("z_score")
            z_source = "cached"
        else:
            z = _compute_z_inline(ticker, window=args.window)
            z_source = "inline"

        z_reverted = z is not None and z > -args.z_exit

        should_exit  = stop_hit or z_reverted
        exit_reasons = []
        if stop_hit:
            exit_reasons.append(f"stop_loss (unrealized={unreal_pct:.2%})")
        if z_reverted:
            exit_reasons.append(f"z_exit (z={z:.3f} > -{args.z_exit}, source={z_source})")

        z_str = f"{z:.3f}" if z is not None else "N/A"
        log.info(
            "%s %s | unreal=%.2f%% z=%s (%s) | %s",
            "EXIT" if should_exit else "HOLD",
            ticker,
            unreal_pct * 100,
            z_str,
            z_source,
            " · ".join(exit_reasons) if exit_reasons else "holding",
        )

        if should_exit:
            exit_record: dict = {
                "ticker":         ticker,
                "qty":            qty,
                "unrealized_pct": round(unreal_pct, 6),
                "z_score":        round(z, 4) if z is not None else None,
                "z_source":       z_source,
                "exit_reasons":   exit_reasons,
            }
            if args.dry_run:
                exit_record["status"] = "dry_run"
            else:
                try:
                    order = trader.submit_market_sell(ticker, int(qty))
                    exit_record.update({"status": "submitted", **order})
                    log.info("SELL submitted: %s", order)
                except Exception as exc:
                    log.error("Sell failed for %s: %s", ticker, exc)
                    exit_record["status"] = "error"
                    exit_record["error"]  = str(exc)
                    result["errors"].append({"ticker": ticker, "error": str(exc)})

            result["exits_triggered"].append(exit_record)
        else:
            result["positions_held"].append({
                "ticker":         ticker,
                "qty":            qty,
                "unrealized_pct": round(unreal_pct, 6),
                "z_score":        round(z, 4) if z is not None else None,
            })

    # ── Telegram notification ─────────────────────────────────────────────────
    try:
        from equity_signals.notifications.telegram import TelegramNotifier
        notifier = TelegramNotifier()
        run_date_display = result["run_date"].replace("T", " ").replace("Z", " UTC")

        if result["exits_triggered"]:
            for ex in result["exits_triggered"]:
                reason_str = " · ".join(ex.get("exit_reasons", []))
                z_str = f"{ex['z_score']:.2f}" if ex.get("z_score") is not None else "N/A"
                msg = (
                    f"<b>🔴 EXIT — {ex['ticker']}</b>\n"
                    f"Reason: {reason_str}\n"
                    f"Z-score: {z_str}\n"
                    f"Unrealized P&amp;L: {ex['unrealized_pct']:.1%}\n"
                    f"Qty sold: {ex['qty']}"
                )
                notifier.send(msg)
        else:
            held_tickers = ", ".join(p["ticker"] for p in result["positions_held"])
            notifier.send(
                f"<b>✅ Positions OK — {run_date_display}</b>\n"
                f"{result['positions_checked']} position(s) checked, all within limits\n"
                f"Tickers: {held_tickers}"
            )
    except Exception as exc:
        log.warning("Telegram notification failed: %s", exc)

    _emit(result)


def _emit(result: dict) -> None:
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
