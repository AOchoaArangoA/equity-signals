"""alpaca_trader — order execution via the Alpaca paper-trading API.

:class:`AlpacaTrader` wraps ``alpaca-py``'s :class:`~alpaca.trading.TradingClient`
to provide a minimal, well-typed interface for the equity-signals execution
layer.  It is the *only* place in the codebase that touches the Alpaca
trading API — both CLI scripts and the FastAPI layer import it from here.

.. warning::
    ``paper=True`` is hardcoded until live trading is explicitly enabled.
    Do not change this flag without a full review of risk controls.

Typical usage::

    from equity_signals.execution import AlpacaTrader

    trader = AlpacaTrader()
    positions = trader.get_open_positions()
    order = trader.submit_market_sell("AAPL", qty=10)
"""

from __future__ import annotations

import logging

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestTradeRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from equity_signals.config import settings

logger = logging.getLogger(__name__)


class AlpacaTrader:
    """Submits and queries orders via the Alpaca paper-trading API.

    .. warning::
        ``paper=True`` is hardcoded.  Live trading is not enabled.

    Args:
        api_key: Alpaca API key.  Defaults to ``settings.alpaca_api_key``.
        secret_key: Alpaca secret key.  Defaults to ``settings.alpaca_secret_key``.

    Example::

        trader = AlpacaTrader()
        positions = trader.get_open_positions()
        order = trader.submit_market_sell("AAPL", qty=5)
    """

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self._api_key = api_key or settings.alpaca_api_key
        self._secret_key = secret_key or settings.alpaca_secret_key
        # paper=True is intentional — do not remove without risk-control review.
        self._client = TradingClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
            paper=True,
        )
        self._data_client = StockHistoricalDataClient(
            api_key=self._api_key,
            secret_key=self._secret_key,
        )
        logger.info("AlpacaTrader initialised (paper=True)")

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_available_cash(self) -> float:
        """Return available cash in the paper account."""
        account = self._client.get_account()
        return float(account.cash)

    def get_current_price(self, ticker: str) -> float:
        """Return latest trade price for *ticker*."""
        request = StockLatestTradeRequest(symbol_or_symbols=ticker)
        trade = self._data_client.get_stock_latest_trade(request)
        return float(trade[ticker].price)

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[dict]:
        """Return all currently open positions.

        Returns:
            List of dicts with keys ``ticker``, ``qty``, ``market_value``,
            ``unrealized_pct`` (as a decimal fraction, e.g. 0.05 = 5 %).
            Empty list if no positions are open.
        """
        positions = self._client.get_all_positions()
        result = [
            {
                "ticker": str(p.symbol),
                "qty": float(p.qty),
                "market_value": float(p.market_value),
                "unrealized_pct": float(p.unrealized_plpc),
            }
            for p in positions
        ]
        logger.info("AlpacaTrader — %d open position(s)", len(result))
        return result

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def submit_market_buy(self, ticker: str, qty: int) -> dict:
        """Submit a market buy order for *qty* shares of *ticker*.

        Returns:
            Dict with ``order_id`` and ``status`` keys.
        """
        logger.info("AlpacaTrader — market BUY %s qty=%d", ticker, qty)
        order = self._client.submit_order(MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        ))
        result = {"order_id": str(order.id), "status": str(order.status)}
        logger.info("AlpacaTrader — order accepted: id=%s status=%s",
                    result["order_id"], result["status"])
        return result

    def close_position(self, ticker: str) -> dict:
        """Close the entire open position for *ticker*.

        If a pending sell order already exists for the symbol (``held_for_orders``
        equals the full position qty), returns that order's details immediately
        instead of submitting a duplicate — Alpaca would reject it with
        ``insufficient qty available`` (code 40310000).

        Otherwise cancels any pending buy orders for the symbol, then submits
        a market-sell for the full position via the native close-position endpoint.

        Returns:
            Dict with ``order_id``, ``status``, and optional ``already_pending``
            key when a sell order was already in flight.
        """
        from alpaca.trading.enums import QueryOrderStatus
        from alpaca.trading.requests import GetOrdersRequest

        logger.info("AlpacaTrader — close_position %s", ticker)

        # Check for existing open orders on this ticker
        open_orders = self._client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker])
        )
        sell_orders = [o for o in open_orders if o.side.value == "sell"]
        if sell_orders:
            o = sell_orders[0]
            result = {
                "order_id":       str(o.id),
                "status":         str(o.status.value),
                "already_pending": True,
            }
            logger.info(
                "AlpacaTrader — sell already pending for %s: id=%s status=%s",
                ticker, result["order_id"], result["status"],
            )
            return result

        # Cancel any pending buy orders so shares are freed up
        buy_orders = [o for o in open_orders if o.side.value == "buy"]
        for o in buy_orders:
            self._client.cancel_order_by_id(str(o.id))
            logger.info("AlpacaTrader — cancelled pending buy order %s for %s", o.id, ticker)

        order = self._client.close_position(ticker)
        result = {"order_id": str(order.id), "status": str(order.status)}
        logger.info("AlpacaTrader — close_position accepted: id=%s status=%s",
                    result["order_id"], result["status"])
        return result

    def submit_market_sell(self, ticker: str, qty: int) -> dict:
        """Submit a market sell order for *qty* shares of *ticker*.

        Returns:
            Dict with ``order_id`` and ``status`` keys.
        """
        logger.info("AlpacaTrader — market SELL %s qty=%d", ticker, qty)
        order = self._client.submit_order(MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        ))
        result = {"order_id": str(order.id), "status": str(order.status)}
        logger.info("AlpacaTrader — order accepted: id=%s status=%s",
                    result["order_id"], result["status"])
        return result

    def submit_limit_buy(
        self,
        ticker: str,
        qty: int,
        limit_price: float,
        extended_hours: bool = True,
    ) -> dict:
        """Submit a limit buy order, supports extended hours.

        Returns:
            Dict with ``order_id`` and ``status`` keys.
        """
        logger.info(
            "AlpacaTrader — limit BUY %s qty=%d limit=%.4f extended_hours=%s",
            ticker, qty, limit_price, extended_hours,
        )
        order = self._client.submit_order(LimitOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.BUY,
            limit_price=limit_price,
            time_in_force=TimeInForce.DAY,
            extended_hours=extended_hours,
        ))
        result = {"order_id": str(order.id), "status": str(order.status)}
        logger.info("AlpacaTrader — limit order accepted: id=%s status=%s",
                    result["order_id"], result["status"])
        return result
