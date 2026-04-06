# equity-signals — codebase guide for Claude

## Architecture: three interfaces, one core package

```
equity_signals/          ← Core package (quantitative logic only)
scripts/                 ← CLI interface (batch / scheduled runs)
app/                     ← HTTP interface (FastAPI)
```

The rule is strict: **quantitative logic lives in `equity_signals/` only**.
Scripts and API orchestrate it — they never duplicate it.

---

## Layer 1 — Core package (`equity_signals/`)

```
equity_signals/
├── config.py                    pydantic-settings singleton (settings)
├── exceptions.py                custom exception hierarchy
├── cli.py                       equity-signals CLI entry point
├── data/
│   ├── yfinance_loader.py       fundamentals via yfinance (free)
│   │   └── fetch_ohlcv()        module-level OHLCV helper for fallback
│   ├── alpaca_loader.py         OHLCV via Alpaca (yfinance fallback built-in)
│   └── fmp_loader.py            FMP profile endpoint (optional)
├── universe/
│   ├── ticker_loader.py         Russell 2000 from iShares IWM CSV
│   ├── universe_filter.py       4-stage filter: midcap→sector→ROE→P/B rank
│   └── universe_store.py        load_latest_universe() — /tmp/ first, output/ second
├── strategies/
│   ├── base.py                  BaseStrategy ABC
│   └── mean_reversion.py        Z-score mean-reversion (vectorised, long-only)
├── execution/
│   ├── __init__.py              re-exports AlpacaTrader
│   └── alpaca_trader.py         paper trading only (paper=True hardcoded)
│       ├── get_open_positions()
│       ├── submit_market_buy(ticker, qty)
│       └── submit_market_sell(ticker, qty)
└── scripts/
    ├── run_universe_scan.py     importable run() + main() entry point
    └── run_signal_scan.py       importable run() + main() entry point
```

### Key design constraints
- `equity_signals/` never imports from `app/`
- `equity_signals/execution/` always `paper=True` — no live trading
- All cache I/O uses the caller-supplied `cache_dir` (default `.cache/`, API uses `/tmp/`)

---

## Layer 2 — CLI (`scripts/`)

| Script | Purpose | Frequency |
|--------|---------|-----------|
| `run_weekly_scan.py` | **DEPRECATED** — runs universe + signals sequentially | — |
| `equity-universe-scan` | Universe scan (fundamentals, slow) | Monthly |
| `equity-signal-scan` | Signal scan (prices + Z-score) | Weekly |
| `scripts/run_weekly_entry.py` | Enter long positions for signal=1 tickers | Weekly |
| `scripts/daily_exit_check.py` | Exit positions where signal=0 | Daily |

### Typical weekly workflow
```bash
# 1. Rebuild universe (monthly or when fundamentals are stale)
equity-universe-scan --index-top-pct 10

# 2. Generate signals (every week)
equity-signal-scan --top-n 5

# 3. Enter new positions
python scripts/run_weekly_entry.py --notional 1000

# 4. Check exits daily at market open
python scripts/daily_exit_check.py
```

Both execution scripts accept `--dry-run` to log orders without submitting them.

---

## Layer 3 — HTTP API (`app/`)

```
app/
├── main.py                      FastAPI app + lifespan hooks
├── core/config.py               app-level settings (ANTHROPIC_API_KEY, API_KEY)
├── routers/
│   ├── health.py                GET  /health                (no auth)
│   ├── signals.py               POST /api/v1/universe       (X-API-Key)
│   │                            POST /api/v1/signals        (X-API-Key)
│   └── orders.py                GET  /api/v1/positions      (X-API-Key)
│                                POST /api/v1/orders/exit/{ticker}
├── schemas/
│   ├── requests.py              UniverseRequest, SignalRequest
│   └── responses.py             UniverseResponse, SignalResponse,
│                                Position, OrderConfirmation, TickerSignal, UniverseTicker
└── services/
    ├── universe_service.py      delegates to TickerLoader + UniverseFilter
    ├── signal_engine.py         delegates to AlpacaLoader + MeanReversionStrategy
    └── llm_service.py           Anthropic SDK wrapper (unused in endpoints)
```

### Endpoint reference

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/health` | — | Liveness check |
| POST | `/api/v1/universe` | X-API-Key | Build investable universe (slow, ~30–120 s) |
| POST | `/api/v1/signals` | X-API-Key | Compute signals for any ticker list (fast) |
| GET | `/api/v1/positions` | X-API-Key | List open Alpaca paper positions |
| POST | `/api/v1/orders/exit/{ticker}` | X-API-Key | Market-sell a position |

### Typical API workflow (Pattern A — pipeline)
```bash
# Step 1: build universe
UNIVERSE=$(curl -s -X POST .../api/v1/universe \
  -H "X-API-Key: $API_KEY" \
  -d '{"index_top_pct": 5}')

# Step 2: extract top 5 tickers, run signals
TICKERS=$(echo $UNIVERSE | jq '[.tickers[:5][].ticker]')
curl -X POST .../api/v1/signals \
  -H "X-API-Key: $API_KEY" \
  -d "{\"tickers\": $TICKERS}"
```

### Typical API workflow (Pattern B — ad-hoc)
```bash
curl -X POST .../api/v1/signals \
  -H "X-API-Key: $API_KEY" \
  -d '{"tickers": ["AAPL", "MSFT", "ASIX"]}'
```

---

## Shared execution layer

`AlpacaTrader` is imported from `equity_signals.execution` in **both** scripts and the API:

```python
# In scripts:
from equity_signals.execution import AlpacaTrader

# In app/routers/orders.py:
from equity_signals.execution import AlpacaTrader
```

No logic is duplicated. The paper-trading flag (`paper=True`) is enforced inside
`alpaca_trader.py` and cannot be overridden by callers.

---

## Environment variables

```
# Required for execution scripts and orders API
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# Required for FastAPI app
API_KEY=          # X-API-Key header secret
ANTHROPIC_API_KEY= # LLM service (llm_service.py, not yet wired to endpoints)

# Optional
FMP_API_KEY=      # FMP profile endpoint only
LOG_LEVEL=INFO
ENVIRONMENT=development
```

---

## Running tests

```bash
.venv/bin/python -m pytest tests/ -q
# Expected: 95 passed
```

## Running the API locally

```bash
pip install -e ".[api]"
uvicorn app.main:app --reload --port 8000
# Docs: http://localhost:8000/docs
```
