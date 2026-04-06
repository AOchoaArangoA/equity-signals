"""
test_mvp_local.py — Prueba local del ciclo completo MVP

Simula el flujo end-to-end sin ejecutar órdenes reales:
    1. Carga universo desde archivo o genera uno pequeño
    2. Descarga OHLCV para top N tickers
    3. Calcula señales de mean reversion
    4. Aplica regla de confluencia
    5. Simula ejecución (dry_run=True por defecto)
    6. Muestra reporte completo en consola

Uso:
    python test_mvp_local.py                          # dry run completo
    python test_mvp_local.py --tickers ASIX SCL WS    # tickers manuales
    python test_mvp_local.py --execute                # ejecuta órdenes reales paper
    python test_mvp_local.py --check-exits            # simula chequeo de salida
"""

import argparse
import json
import sys
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_mvp")


# ── helpers de display ────────────────────────────────────────────────────────

def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


def ok(msg: str) -> None:
    print(f"  ✔  {msg}")


def warn(msg: str) -> None:
    print(f"  ⚠  {msg}")


def fail(msg: str) -> None:
    print(f"  ✘  {msg}")


# ── paso 1: universo ──────────────────────────────────────────────────────────

def load_universe(tickers_override: list[str] | None, top_n: int) -> list[dict]:
    """Carga el universo desde output/ o usa tickers manuales."""

    if tickers_override:
        section("PASO 1 — Universo (manual)")
        universe = [
            {"ticker": t, "market_cap": None, "pb_ratio": None,
             "roe": None, "sector": "Manual", "pb_rank_sector": i + 1,
             "value_signal": True}
            for i, t in enumerate(tickers_override)
        ]
        ok(f"{len(universe)} tickers provistos manualmente: {tickers_override}")
        return universe

    section("PASO 1 — Universo (desde archivo o scan)")

    candidates = sorted(Path("output").glob("universe_*.parquet")) if Path("output").exists() else []

    if candidates:
        import pandas as pd
        path = candidates[-1]
        df = pd.read_parquet(path)
        df_top = (
            df[df["value_signal"] == True]
            .nsmallest(top_n, "pb_rank_sector")
        )
        universe = df_top.to_dict("records")
        ok(f"Cargado desde {path.name} — {len(df)} tickers totales")
        ok(f"Top {top_n} seleccionados:")
        for r in universe:
            print(f"     {r['ticker']:6s}  P/B={r['pb_ratio']:.2f}  "
                  f"ROE={r['roe']:.1%}  sector={r['sector']}")
        return universe

    # fallback: universo hardcoded para pruebas
    warn("No se encontró output/universe_*.parquet — usando universo hardcoded")
    universe = [
        {"ticker": "ASIX", "market_cap": 646_562_240, "pb_ratio": 0.793,
         "roe": 0.062, "sector": "Basic Materials", "pb_rank_sector": 1, "value_signal": True},
        {"ticker": "SCL",  "market_cap": 1_122_946_944, "pb_ratio": 0.900,
         "roe": 0.039, "sector": "Basic Materials", "pb_rank_sector": 2, "value_signal": True},
        {"ticker": "WS",   "market_cap": 1_402_498_048, "pb_ratio": 1.247,
         "roe": 0.104, "sector": "Basic Materials", "pb_rank_sector": 3, "value_signal": True},
    ]
    for r in universe:
        print(f"     {r['ticker']:6s}  P/B={r['pb_ratio']:.2f}  sector={r['sector']}")
    return universe


# ── paso 2: OHLCV ─────────────────────────────────────────────────────────────

def fetch_prices(tickers: list[str], days: int) -> "pd.DataFrame":
    section("PASO 2 — Descarga de precios (OHLCV)")
    try:
        from equity_signals.data.alpaca_loader import AlpacaLoader
        log.info("Intentando Alpaca...")
        df = AlpacaLoader().get_ohlcv(tickers, days=days)
        ok(f"Alpaca: {len(df)} filas para {df.index.get_level_values('ticker').nunique()} tickers")
        return df
    except Exception as e:
        warn(f"Alpaca falló: {e}")
        warn("Fallback a yfinance...")
        from equity_signals.data.yfinance_loader import fetch_ohlcv
        df = fetch_ohlcv(tickers, days=days)
        ok(f"yfinance: {len(df)} filas para {df.index.get_level_values('ticker').nunique()} tickers")
        return df


# ── paso 3: señales ───────────────────────────────────────────────────────────

def compute_signals(prices: "pd.DataFrame", window: int, z_entry: float) -> "pd.DataFrame":
    section("PASO 3 — Mean Reversion (Z-score)")
    from equity_signals.strategies.mean_reversion import MeanReversionStrategy
    strategy = MeanReversionStrategy(window=window, z_entry=z_entry)
    signals = strategy.compute(prices)

    latest = (
        signals.groupby("ticker")
        .last()
        .reset_index()[["ticker", "close", "ma", "z_score", "signal"]]
    )

    ok(f"Señales calculadas (última fecha por ticker):")
    print(f"\n  {'ticker':8s} {'close':>8s} {'MA20':>8s} {'z_score':>9s} {'signal':>7s}")
    print(f"  {'─'*8} {'─'*8} {'─'*8} {'─'*9} {'─'*7}")
    for _, row in latest.iterrows():
        ma_str = f"{row['ma']:.2f}" if row['ma'] is not None and row['ma'] == row['ma'] else "  —"
        z_str  = f"{row['z_score']:.3f}" if row['z_score'] == row['z_score'] else "  —"
        flag = "LONG" if row["signal"] == 1 else "—"
        print(f"  {row['ticker']:8s} {row['close']:>8.2f} {ma_str:>8s} {z_str:>9s} {flag:>7s}")

    return signals


# ── paso 4: confluencia ───────────────────────────────────────────────────────

def apply_confluence(universe: list[dict], signals: "pd.DataFrame",
                     z_threshold: float, pb_rank_max: int,
                     force_entry: bool = False) -> list[dict]:
    section("PASO 4 — Regla de confluencia (P/B + Z-score)")
    import pandas as pd

    latest_z = (
        signals.groupby("ticker")["z_score"].last().reset_index()
        .rename(columns={"z_score": "latest_z"})
    )
    universe_df = pd.DataFrame(universe)
    merged = universe_df.merge(latest_z, on="ticker", how="left")

    if force_entry:
        warn("--force-entry activo: ignorando confluencia — todos los tickers entran")

    entries = []
    for _, row in merged.iterrows():
        pb_ok = row["pb_rank_sector"] <= pb_rank_max
        z_ok  = row.get("latest_z", float("nan")) < -z_threshold
        z_display = f"{row['latest_z']:.3f}" if row["latest_z"] == row["latest_z"] else "N/A"
        confluent = force_entry or (pb_ok and z_ok)

        status = "ENTRADA" if confluent else "—"
        reason = []
        if not pb_ok:
            reason.append(f"pb_rank={int(row['pb_rank_sector'])} > {pb_rank_max}")
        if not z_ok:
            reason.append(f"z={z_display} >= -{z_threshold}")

        reason_str = " · ".join(reason) if reason else "confluencia ok"
        print(f"  {row['ticker']:6s}  pb_rank={int(row['pb_rank_sector'])}  "
              f"z={z_display:>8s}  → {status:7s}  ({reason_str})")

        if confluent:
            entries.append({
                "ticker": row["ticker"],
                "pb_rank_sector": int(row["pb_rank_sector"]),
                "pb_ratio": row.get("pb_ratio"),
                "latest_z": row["latest_z"],
            })

    print()
    if entries:
        ok(f"{len(entries)} ticker(s) con señal de ENTRADA: {[e['ticker'] for e in entries]}")
    else:
        warn("Ningún ticker cumple la regla de confluencia esta semana")

    return entries


# ── paso 5: ejecución ─────────────────────────────────────────────────────────

def execute_entries(entries: list[dict], dry_run: bool,
                    position_pct: float = 0.20,
                    extended_hours: bool = False) -> list[dict]:
    section(f"PASO 5 — Ejecución {'(DRY RUN)' if dry_run else '(REAL paper trade)'}")
    orders = []

    if not entries:
        warn("Sin entradas — no hay órdenes que ejecutar")
        return orders

    try:
        from equity_signals.execution.alpaca_trader import AlpacaTrader
        trader = AlpacaTrader()
        cash = trader.get_available_cash()
        ok(f"Cash disponible en Alpaca paper: ${cash:,.2f}")

        size_per_trade = cash * position_pct / len(entries)

        for entry in entries:
            ticker = entry["ticker"]
            try:
                price = trader.get_current_price(ticker)
                qty = int(size_per_trade // price)
                estimated_value = qty * price

                if qty <= 0:
                    warn(f"{ticker}: qty=0 (precio ${price:.2f} > tamaño ${size_per_trade:.2f})")
                    continue

                section_mode = "LIMIT extended hours" if extended_hours else "MARKET"
                print(f"\n  {ticker}: qty={qty}  precio~${price:.2f}  "
                      f"valor~${estimated_value:.2f}  tipo={section_mode}")

                if dry_run:
                    warn(f"DRY RUN — orden NO enviada a Alpaca")
                    orders.append({
                        "ticker": ticker, "qty": qty,
                        "estimated_value": estimated_value,
                        "status": "dry_run",
                        "order_type": section_mode
                    })
                elif extended_hours:
                    # Limit order al precio actual — compatible con after/pre-market
                    order = trader.submit_limit_buy(
                        ticker, qty,
                        limit_price=round(price * 1.001, 2),  # 0.1% sobre precio actual
                        extended_hours=True
                    )
                    ok(f"Limit order enviada (extended hours): {order}")
                    orders.append({
                        "ticker": ticker, "qty": qty,
                        "limit_price": round(price * 1.001, 2),
                        "estimated_value": estimated_value,
                        "status": "submitted",
                        "order_type": "limit_extended",
                        "order_id": order.get("order_id")
                    })
                else:
                    order = trader.submit_market_buy(ticker, qty)
                    ok(f"Market order enviada: {order}")
                    orders.append({
                        "ticker": ticker, "qty": qty,
                        "estimated_value": estimated_value,
                        "status": "submitted",
                        "order_type": "market",
                        "order_id": order.get("order_id")
                    })
            except Exception as e:
                fail(f"{ticker}: {e}")
                orders.append({"ticker": ticker, "status": "error", "error": str(e)})

    except ImportError:
        warn("AlpacaTrader no disponible — mostrando plan de órdenes estimado")
        for entry in entries:
            print(f"  PLAN: BUY {entry['ticker']}  z={entry['latest_z']:.3f}  "
                  f"pb_rank={entry['pb_rank_sector']}")
            orders.append({"ticker": entry["ticker"], "status": "no_trader"})

    except Exception as e:
        fail(f"Error conectando a Alpaca: {e}")

    return orders


# ── paso 6: chequeo de salida ─────────────────────────────────────────────────

def check_exits(signals: "pd.DataFrame", z_exit: float, stop_loss_pct: float) -> None:
    section("CHEQUEO DE SALIDA — posiciones abiertas")
    try:
        from equity_signals.execution.alpaca_trader import AlpacaTrader
        trader = AlpacaTrader()
        positions = trader.get_open_positions()

        if not positions:
            warn("No hay posiciones abiertas en Alpaca paper")
            return

        print(f"\n  {'ticker':8s} {'qty':>6s} {'unreal%':>8s} {'z_score':>9s} {'acción':>10s}")
        print(f"  {'─'*8} {'─'*6} {'─'*8} {'─'*9} {'─'*10}")

        for pos in positions:
            ticker = pos["ticker"]
            unreal_pct = pos["unrealized_pct"]

            ticker_signals = signals[signals["ticker"] == ticker] if signals is not None else None
            z = ticker_signals["z_score"].iloc[-1] if ticker_signals is not None and len(ticker_signals) > 0 else float("nan")

            stop_hit  = unreal_pct <= -stop_loss_pct
            z_reverted = z == z and z > -z_exit

            if stop_hit:
                action = "SALIR (stop)"
            elif z_reverted:
                action = "SALIR (z exit)"
            else:
                action = "MANTENER"

            z_str = f"{z:.3f}" if z == z else "N/A"
            print(f"  {ticker:8s} {pos['qty']:>6}  {unreal_pct:>7.1%}  {z_str:>9s}  {action:>10s}")

    except ImportError:
        warn("AlpacaTrader no disponible — no se pueden leer posiciones")
    except Exception as e:
        fail(f"Error leyendo posiciones: {e}")


# ── reporte final ─────────────────────────────────────────────────────────────

def print_summary(universe: list[dict], entries: list[dict],
                  orders: list[dict], dry_run: bool) -> None:
    section("RESUMEN")
    result = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "dry_run": dry_run,
        "universe_candidates": len(universe),
        "confluence_signals": len(entries),
        "orders": orders,
    }
    print(json.dumps(result, indent=2, default=str))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Test MVP local — equity-signals")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers manuales (omite UniverseFilter)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Cuántos tickers tomar del universo (default: 5)")
    parser.add_argument("--window", type=int, default=20,
                        help="Ventana SMA (default: 20)")
    parser.add_argument("--z-entry", type=float, default=1.5,
                        help="Umbral de entrada Z-score (default: 1.5)")
    parser.add_argument("--z-exit", type=float, default=0.5,
                        help="Umbral de salida Z-score (default: 0.5)")
    parser.add_argument("--stop-loss", type=float, default=0.07,
                        help="Stop-loss porcentaje (default: 0.07)")
    parser.add_argument("--pb-rank-max", type=int, default=3,
                        help="Máximo pb_rank_sector para confluencia (default: 3)")
    parser.add_argument("--days", type=int, default=60,
                        help="Días de OHLCV a descargar (default: 60)")
    parser.add_argument("--position-pct", type=float, default=0.20,
                        help="% del cash por posición (default: 0.20)")
    parser.add_argument("--execute", action="store_true",
                        help="Ejecutar órdenes reales en Alpaca paper")
    parser.add_argument("--check-exits", action="store_true",
                        help="Solo chequear posiciones abiertas y condiciones de salida")
    parser.add_argument("--force-entry", action="store_true",
                        help="Fuerza entrada ignorando regla de confluencia (solo para testing)")
    parser.add_argument("--extended-hours", action="store_true",
                        help="Usa limit order al último precio para operar fuera de horario")
    args = parser.parse_args()

    dry_run = not args.execute

    print(f"\n{'═' * 60}")
    print(f"  equity-signals — MVP test local")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  modo: {'DRY RUN' if dry_run else 'PAPER TRADE REAL'}")
    print(f"{'═' * 60}")

    universe = load_universe(args.tickers, args.top_n)
    tickers = [u["ticker"] for u in universe]

    if not tickers:
        fail("Universo vacío — abortando")
        sys.exit(1)

    try:
        prices = fetch_prices(tickers, args.days)
    except Exception as e:
        fail(f"No se pudo obtener precios: {e}")
        sys.exit(1)

    signals = compute_signals(prices, args.window, args.z_entry)

    if args.check_exits:
        check_exits(signals, args.z_exit, args.stop_loss)
        return

    entries = apply_confluence(
        universe, signals, args.z_entry, args.pb_rank_max,
        force_entry=args.force_entry
    )
    orders = execute_entries(
        entries, dry_run, args.position_pct,
        extended_hours=args.extended_hours
    )
    print_summary(universe, entries, orders, dry_run)


if __name__ == "__main__":
    main()