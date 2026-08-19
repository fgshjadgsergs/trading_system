"""Reference strategies used by M8 tests and demo reports.

All are bar-driven and emit market orders toward a target position; the
pending signed quantity is subtracted so in-flight orders are not re-sent
while latency delays their fills.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from trading_system.backtest.engine import Bar, Context, Order, OrderType
from trading_system.core.schema import Side

_MIN_QTY = 1e-9


def _orders_toward(target_qty: float, ctx: Context) -> list[Order]:
    delta = target_qty - (ctx.position_qty + ctx.pending_qty_signed)
    if abs(delta) < _MIN_QTY:
        return []
    side = Side.BUY if delta > 0 else Side.SELL
    return [Order(side=side, qty=abs(delta), order_type=OrderType.MARKET)]


class TargetPositionStrategy:
    """Follows a precomputed per-bar target array: position = targets[bar.index] * qty."""

    def __init__(self, targets: np.ndarray, qty: float = 1.0) -> None:
        self.targets = np.asarray(targets, dtype=float)
        self.qty = qty

    def on_bar(self, bar: Bar, ctx: Context) -> list[Order]:
        if bar.index >= len(self.targets):
            return []
        return _orders_toward(float(self.targets[bar.index]) * self.qty, ctx)


class RandomStrategy:
    """Null strategy: seeded random entries/exits, independent of the data."""

    def __init__(self, seed: int, p_trade: float = 0.3, qty: float = 0.05) -> None:
        self.rng = np.random.default_rng(seed)
        self.p_trade = p_trade
        self.qty = qty
        self.target = 0.0

    def on_bar(self, bar: Bar, ctx: Context) -> list[Order]:
        if self.rng.random() < self.p_trade:
            self.target = float(self.rng.integers(-1, 2))  # -1, 0 or +1
        return _orders_toward(self.target * self.qty, ctx)


class MACrossStrategy:
    """Long when fast SMA > slow SMA, short otherwise (after warmup)."""

    def __init__(self, fast: int = 8, slow: int = 30, qty: float = 0.5) -> None:
        if not 0 < fast < slow:
            raise ValueError("need 0 < fast < slow")
        self.fast = fast
        self.slow = slow
        self.qty = qty
        self.closes: deque[float] = deque(maxlen=slow)

    def on_bar(self, bar: Bar, ctx: Context) -> list[Order]:
        self.closes.append(bar.close)
        if len(self.closes) < self.slow:
            return []
        closes = list(self.closes)
        fast_ma = sum(closes[-self.fast :]) / self.fast
        slow_ma = sum(closes) / self.slow
        target = self.qty if fast_ma > slow_ma else -self.qty
        return _orders_toward(target, ctx)
