"""ExchangeAdapter interface: subscribe, normalize, snapshot, liq_formula.

Binance USDT-M is the first implementation (trading_system.collectors.binance);
Bybit/OKX plug in at stage 5 by implementing the same interface.
"""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from trading_system.core.schema import BookSnapshot, Record, Side


@dataclass(frozen=True, slots=True)
class RawMessage:
    """A raw exchange payload plus local receive time (UTC ns)."""

    stream: str  # exchange-native stream tag, e.g. "btcusdt@depth@100ms"
    payload: bytes | str
    ts_recv: int


class LiquidationFormula(abc.ABC):
    """Exchange-specific liquidation price model.

    v1 is a flat maintenance margin rate; exchanges with margin brackets
    override with their tier tables.
    """

    @abc.abstractmethod
    def liq_price(
        self,
        entry: float,
        leverage: float,
        side: Side,
        *,
        symbol: str | None = None,
        qty: float | None = None,
    ) -> float:
        """Liquidation price for an isolated position opened at `entry`."""

    @abc.abstractmethod
    def maint_margin_rate(self, symbol: str, notional_usd: float) -> float:
        """Maintenance margin rate applicable to a notional (bracket lookup)."""


class ExchangeAdapter(abc.ABC):
    """One exchange = one adapter. All outputs are unified-schema records."""

    name: str

    @abc.abstractmethod
    def subscribe(
        self, symbols: Sequence[str], streams: Sequence[str]
    ) -> AsyncIterator[RawMessage]:
        """Yield raw websocket messages for symbols/streams (reconnects inside)."""

    @abc.abstractmethod
    def normalize(self, raw: RawMessage) -> list[Record]:
        """Parse one raw message into zero or more unified-schema records."""

    @abc.abstractmethod
    async def snapshot(self, symbol: str, depth: int = 1000) -> BookSnapshot:
        """Fetch a REST order book snapshot."""

    @abc.abstractmethod
    def liq_formula(self) -> LiquidationFormula:
        """The exchange's liquidation price model."""
