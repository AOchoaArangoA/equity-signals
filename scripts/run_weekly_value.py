#!/usr/bin/env python3
"""run_weekly_value — weekly entry scan combining Russell 2000 value filter
with manual watchlist and mean-reversion Z-score confluence.

Flow:
  1. Load Russell 2000 → apply UniverseFilter (P/B value signal)
  2. Merge manual watchlist tickers (fetch fundamentals for any not in universe)
  3. Compute Z-scores for all candidates (Alpaca → yfinance fallback)
  4. Score candidates: combined_score = pb_rank * 0.4 + |z_score| * 0.6
     Eligibility: z_score < -z_entry AND roe > 0
  5. Select top-N, skip already-held positions
  6. Execute entries (market or limit for extended hours)
  7. Send Telegram summary
  8. Save output/weekly_value_YYYYMMDD.json

Usage::

    python scripts/run_weekly_value.py               # live paper trade
    python scripts/run_weekly_value.py --dry-run     # preview only
    python scripts/run_weekly_value.py --extended-hours
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("run_weekly_value")

_REPO_ROOT   = Path(__file__).parent.parent
_WATCHLIST   = _REPO_ROOT / "config" / "watchlist.json"
_OUTPUT_DIR  = _REPO_ROOT / "output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_watchlist() -> dict:
    if not _WATCHLIST.exists():
        log.error("config/watchlist.json not found")
        sys.exit(1)
    return json.loads(_WATCHLIST.read_text())


def _tg_send(message: str) -> None:
    try:
        from equity_signals.notifications.telegram import TelegramNotifier
        TelegramNotifier().send(message)
    except Exception as exc:
        log.warning("Telegram notification failed: %s", exc)


def _emit(result: dict) -> None:
    print(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_russell_universe(
    pb_percentile: int,
    midcap_min: float,
    midcap_max: float,
    sectors: list[str],
) -> "pd.DataFrame":
    """Load Russell 2000 and apply value filter. Returns universe_df."""
    import pandas as pd
    from equity_signals.universe.ticker_loader import TickerLoader
    from equity_signals.universe.universe_filter import FilterConfig, UniverseFilter

    log.info("Loading Russell 2000 tickers…")
    all_tickers = TickerLoader().get_russell2000()
    log.info("Russell 2000: %d tickers loaded", len(all_tickers))

    cfg = FilterConfig(
        midcap_min=midcap_min,
        midcap_max=midcap_max,
        sectors=sectors,
        pb_percentile=pb_percentile,
    )
    df = UniverseFilter(cfg).run(all_tickers)
    passing = df[df["value_signal"] == True]
    log.info(
        "Russell 2000: %d → %d after value filter (pb_percentile=%d)",
        len(all_tickers), len(passing), pb_percentile,
    )
    return passing.copy()


def step_merge_watchlist(
    universe_df: "pd.DataFrame",
    watchlist_tickers: list[str],
) -> "pd.DataFrame":
    """Add watchlist tickers missing from universe; fetch their fundamentals."""
    import pandas as pd
    from equity_signals.data.yfinance_loader import YFinanceLoader

    universe_set   = set(universe_df["ticker"].tolist())
    missing        = [t for t in watchlist_tickers if t not in universe_set]
    already_in     = [t for t in watchlist_tickers if t in universe_set]

    if already_in:
        log.info("Watchlist tickers already in universe: %s", already_in)

    if not missing:
        log.info("All watchlist tickers already in universe — nothing to add")
        return universe_df

    log.info("Fetching fundamentals for watchlist additions: %s", missing)
    try:
        loader = YFinanceLoader()
        fund_df = loader.get_fundamentals(missing)
    except Exception as exc:
        log.warning("Could not fetch fundamentals for watchlist tickers: %s", exc)
        fund_df = pd.DataFrame()

    rows = []
    for ticker in missing:
        row = fund_df[fund_df["ticker"] == ticker].iloc[0] if (
            not fund_df.empty and ticker in fund_df["ticker"].values
        ) else None

        rows.append({
            "ticker":         ticker,
            "market_cap":     float(row["market_cap"]) if row is not None and row["market_cap"] == row["market_cap"] else None,
            "pb_ratio":       float(row["pb_ratio"])   if row is not None and row["pb_ratio"] == row["pb_ratio"]   else None,
            "roe":            float(row["roe"])         if row is not None and row["roe"] == row["roe"]             else None,
            "sector":         str(row["sector"])        if row is not None and row["sector"] == row["sector"]       else "Manual",
            "pb_rank_sector": 999,   # manual — no intra-sector rank
            "value_signal":   True,
        })

    added_df = pd.DataFrame(rows)
    log.info("Watchlist added: %d tickers — %s", len(rows), missing)
    return pd.concat([universe_df, added_df], ignore_index=True)


def step_compute_signals(
    tickers: list[str],
    days: int,
    window: int,
    z_entry: float,
) -> "pd.DataFrame":
    """Fetch OHLCV and compute Z-scores. Returns latest-row DataFrame per ticker."""
    import pandas as pd

    log.info("Fetching OHLCV for %d tickers (days=%d)…", len(tickers), days)
    try:
        from equity_signals.data.alpaca_loader import AlpacaLoader
        prices = AlpacaLoader().get_ohlcv(tickers, days=days)
        log.info("Alpaca: %d rows for %d tickers", len(prices),
                 prices.index.get_level_values("ticker").nunique())
    except Exception as exc:
        log.warning("Alpaca failed (%s) — falling back to yfinance", exc)
        try:
            from equity_signals.data.yfinance_loader import fetch_ohlcv
            prices = fetch_ohlcv(tickers, days=days)
            log.info("yfinance fallback: %d rows", len(prices))
        except Exception as exc2:
            log.error("Both price sources failed: %s", exc2)
            raise

    from equity_signals.strategies.mean_reversion import MeanReversionStrategy
    signals = MeanReversionStrategy(window=window, z_entry=z_entry).compute(prices)
    latest  = signals.groupby("ticker").last().reset_index()
    log.info("Signals computed for %d tickers", len(latest))
    return latest


def step_score_and_select(
    universe_df:  "pd.DataFrame",
    signals_df:   "pd.DataFrame",
    z_entry:      float,
    top_n:        int,
) -> "tuple[list[dict], list[str]]":
    """Merge, score, filter, and return (candidates, no_signal_list)."""
    import pandas as pd

    merged = universe_df.merge(
        signals_df[["ticker", "close", "z_score"]],
        on="ticker",
        how="left",
    )

    # Eligibility: z_score must be negative and past threshold; roe > 0
    eligible = merged[
        (merged["z_score"] < -z_entry) &
        (merged["roe"].fillna(0) > 0)
    ].copy()

    no_signal = merged[
        ~merged["ticker"].isin(eligible["ticker"])
    ]["ticker"].tolist()

    log.info(
        "After z_score filter (z < -%.1f, roe > 0): %d candidates with signal",
        z_entry, len(eligible),
    )

    if eligible.empty:
        return [], no_signal

    # Combined score: lower is better
    eligible["combined_score"] = (
        eligible["pb_rank_sector"].fillna(999) * 0.4 +
        eligible["z_score"].abs() * 0.6
    )
    eligible = eligible.sort_values("combined_score")

    top = eligible.head(top_n)
    log.info(
        "Top %d selected: %s",
        len(top), top["ticker"].tolist(),
    )

    candidates = top[[
        "ticker", "pb_ratio", "pb_rank_sector", "roe", "sector", "close", "z_score", "combined_score"
    ]].to_dict("records")

    return candidates, no_signal


def step_execute(
    candidates:    list[dict],
    held:          set[str],
    cash:          float,
    position_pct:  float,
    extended_hours: bool,
    dry_run:       bool,
    trader,
) -> list[dict]:
    """Submit entry orders and return order records."""
    to_enter = [c for c in candidates if c["ticker"] not in held]
    skipped  = [c["ticker"] for c in candidates if c["ticker"] in held]
    if skipped:
        log.info("Skipping (already held): %s", skipped)

    if not to_enter:
        log.info("All top candidates already held — no entries")
        return []

    size_per = cash * position_pct / len(to_enter)
    entries  = []

    for c in to_enter:
        ticker = c["ticker"]
        try:
            price = trader.get_current_price(ticker)
            qty   = math.floor(size_per / price)
            if qty <= 0:
                log.warning("SKIP %s — qty=0 (price=$%.2f > size=$%.2f)", ticker, price, size_per)
                continue

            record: dict = {
                "ticker":          ticker,
                "qty":             qty,
                "price":           round(price, 4),
                "estimated_value": round(qty * price, 2),
                "pb_ratio":        c.get("pb_ratio"),
                "pb_rank_sector":  int(c.get("pb_rank_sector", 999)),
                "z_score":         round(c.get("z_score", 0), 4),
                "order_type":      "limit_extended" if extended_hours else "market",
            }

            if dry_run:
                record["status"] = "dry_run"
                log.info("DRY RUN — BUY %s qty=%d ~$%.2f", ticker, qty, qty * price)
            elif extended_hours:
                limit_price = round(price * 1.001, 2)
                order = trader.submit_limit_buy(ticker, qty, limit_price=limit_price)
                record.update({"limit_price": limit_price, "status": "submitted", **order})
                log.info("LIMIT BUY %s qty=%d limit=$%.2f → %s", ticker, qty, limit_price, order)
            else:
                order = trader.submit_market_buy(ticker, qty)
                record.update({"status": "submitted", **order})
                log.info("MARKET BUY %s qty=%d → %s", ticker, qty, order)

            entries.append(record)

        except Exception as exc:
            log.error("Order failed for %s: %s", ticker, exc)
            entries.append({"ticker": ticker, "status": "error", "error": str(exc)})

    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Weekly value + mean-reversion confluence entry scan.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dry-run",        action="store_true", help="Preview without submitting orders")
    parser.add_argument("--extended-hours", action="store_true", help="Use limit orders for extended hours")
    parser.add_argument("--pb-percentile",  type=int,   default=30,   help="P/B percentile cutoff for value filter")
    parser.add_argument("--z-entry",        type=float, default=1.5,  help="Z-score entry threshold (negative)")
    parser.add_argument("--position-pct",   type=float, default=0.20, help="Fraction of cash per position")
    parser.add_argument("--top-n",          type=int,   default=5,    help="Max number of entries per run")
    args = parser.parse_args()

    config   = _load_watchlist()
    strategy = config.get("strategy", {})
    universe_cfg = config.get("universe", {})
    watchlist_tickers: list[str] = config.get("tickers", [])

    pb_percentile = args.pb_percentile or universe_cfg.get("pb_percentile", 30)
    midcap_min    = universe_cfg.get("midcap_min", 300_000_000)
    midcap_max    = universe_cfg.get("midcap_max", 2_000_000_000)
    sectors       = universe_cfg.get("sectors", [])
    z_entry       = args.z_entry or strategy.get("z_entry", 1.5)
    days          = strategy.get("days", 60)
    window        = strategy.get("window", 20)

    run_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    result: dict = {
        "run_date":           run_date,
        "dry_run":            args.dry_run,
        "universe_size":      0,
        "watchlist_added":    [],
        "candidates":         0,
        "confluence_signals": 0,
        "entries":            [],
        "no_signal":          [],
        "errors":             [],
    }

    # ── Step 1: Russell 2000 universe ─────────────────────────────────────────
    try:
        universe_df = step_russell_universe(pb_percentile, midcap_min, midcap_max, sectors)
        result["universe_size"] = len(universe_df)
        n_russell = len(universe_df)
    except Exception as exc:
        log.error("Universe filter failed: %s", exc)
        result["errors"].append(f"universe: {exc}")
        _emit(result)
        sys.exit(1)

    # ── Step 2: Merge watchlist ────────────────────────────────────────────────
    try:
        universe_set_before = set(universe_df["ticker"].tolist())
        combined_df = step_merge_watchlist(universe_df, watchlist_tickers)
        watchlist_added = [t for t in watchlist_tickers if t not in universe_set_before]
        result["watchlist_added"] = watchlist_added
        n_watchlist = len(watchlist_added)
    except Exception as exc:
        log.error("Watchlist merge failed: %s", exc)
        result["errors"].append(f"watchlist_merge: {exc}")
        combined_df  = universe_df
        watchlist_added = []
        n_watchlist = 0

    all_tickers = combined_df["ticker"].tolist()

    # ── Step 3: Compute signals ────────────────────────────────────────────────
    try:
        signals_df = step_compute_signals(all_tickers, days, window, z_entry)
    except Exception as exc:
        log.error("Signal computation failed: %s", exc)
        result["errors"].append(f"signals: {exc}")
        _emit(result)
        sys.exit(1)

    # ── Step 4 & 5: Score and select ──────────────────────────────────────────
    candidates, no_signal = step_score_and_select(combined_df, signals_df, z_entry, args.top_n)
    result["candidates"]         = len(candidates)
    result["confluence_signals"] = len(candidates)
    result["no_signal"]          = no_signal

    # ── Step 6: Open positions ─────────────────────────────────────────────────
    try:
        from equity_signals.execution.alpaca_trader import AlpacaTrader
        trader    = AlpacaTrader()
        cash      = trader.get_available_cash()
        positions = trader.get_open_positions()
        held      = {p["ticker"] for p in positions}
        log.info("Cash: $%.2f | Held: %s", cash, sorted(held))
    except Exception as exc:
        log.error("Alpaca connection failed: %s", exc)
        result["errors"].append(f"alpaca: {exc}")
        _emit(result)
        sys.exit(1)

    # ── Step 7: Execute entries ────────────────────────────────────────────────
    entries = step_execute(
        candidates, held, cash, args.position_pct,
        args.extended_hours, args.dry_run, trader,
    )
    result["entries"] = entries

    # ── Save output ────────────────────────────────────────────────────────────
    _OUTPUT_DIR.mkdir(exist_ok=True)
    date_tag   = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path   = _OUTPUT_DIR / f"weekly_value_{date_tag}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    log.info("Saved → %s", out_path)

    # ── Step 8: Telegram ───────────────────────────────────────────────────────
    _send_telegram(result, n_russell, n_watchlist, candidates, no_signal, args.dry_run)

    _emit(result)


def _send_telegram(
    result: dict,
    n_russell: int,
    n_watchlist: int,
    candidates: list[dict],
    no_signal: list[str],
    dry_run: bool,
) -> None:
    run_date_display = result["run_date"].replace("T", " ").replace("Z", " UTC")
    mode_tag = " (DRY RUN)" if dry_run else ""

    entry_lines = []
    for e in result["entries"]:
        status = "✅" if e.get("status") == "submitted" else ("🔲" if e.get("status") == "dry_run" else "❌")
        pb_str = f"P/B={e['pb_ratio']:.2f}" if e.get("pb_ratio") else "P/B=N/A"
        entry_lines.append(
            f"{status} {e['ticker']}  {pb_str}  z={e['z_score']:+.2f}  "
            f"rank={e['pb_rank_sector']}  → {'BUY' if e.get('status') != 'error' else 'ERR'} "
            f"{e.get('qty', '?')} @ ~${e.get('price', '?'):.2f}"
        )

    no_signal_str = ", ".join(no_signal[:10])
    if len(no_signal) > 10:
        no_signal_str += f" (+{len(no_signal) - 10} more)"

    entries_block = "\n".join(entry_lines) if entry_lines else "  (none)"
    msg = (
        f"<b>📈 Weekly Value Scan{mode_tag} — {run_date_display}</b>\n\n"
        f"<b>Universe:</b> {n_russell} Russell 2000 + {n_watchlist} watchlist\n"
        f"<b>Value candidates:</b> {result['universe_size'] + n_watchlist}\n"
        f"<b>Confluence signals:</b> {result['confluence_signals']}\n\n"
        f"<b>Entries:</b>\n<pre>{entries_block}</pre>\n\n"
        f"<b>No signal:</b> {no_signal_str if no_signal_str else 'none'}"
    )
    _tg_send(msg)


if __name__ == "__main__":
    main()
