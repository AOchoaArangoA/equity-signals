"""equity_signals.execution — order execution layer.

Provides :class:`~equity_signals.execution.alpaca_trader.AlpacaTrader`,
the single entry point for submitting and querying orders.  Imported by
both CLI scripts and the FastAPI layer — no logic is duplicated.
"""

from equity_signals.execution.alpaca_trader import AlpacaTrader

__all__ = ["AlpacaTrader"]
