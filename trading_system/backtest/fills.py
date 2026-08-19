"""Fill models: taker pricing with spread+impact, strict-cross limit fills, latency.

Market (taker) orders execute at ``mid +/- half_spread +/- impact``. The impact
model is deliberately simple and fully documented here:

    impact_bps = coef_bps * order_notional_usd / recent_volume_usd, capped at cap_bps

where ``recent_volume_usd`` is the USD volume traded over a rolling window
ending strictly before the fill (pre-print state, so the fill never sees the
print that triggers it). When an L2 book side is available the fill instead
walks the book and pays the VWAP of the consumed levels.

Limit orders fill only when a subsequent trade print passes STRICTLY through
the limit price (print < limit for buys, print > limit for sells) — touching
the limit is not enough. Partial fills are allowed, capped pro-rata to each
print's size.

Latency: an order placed at ``t`` becomes active at ``t + latency`` with
latency drawn uniformly from ``[latency_ms_min, latency_ms_max]`` by a seeded
rng; an order can never see or act on prints before its activation.

Fees are a fraction of fill notional: ``taker_fee`` for market fills,
``maker_fee`` for limit fills.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from trading_system.core.schema import Side
from trading_system.core.timeutils import NS_PER_MS

BPS = 1e-4


def impact_bps(
    order_notional_usd: float,
    recent_volume_usd: float,
    coef_bps: float,
    cap_bps: float,
) -> float:
    """Impact in bps: linear in order notional relative to recent traded volume.

    Zero-volume windows are charged the cap (nothing traded recently, so a
    taker order is assumed to move the price the most).
    """
    if order_notional_usd <= 0.0 or coef_bps <= 0.0:
        return 0.0
    if recent_volume_usd <= 0.0:
        return cap_bps
    return min(cap_bps, coef_bps * order_notional_usd / recent_volume_usd)


def market_fill_price(side: Side, mid: float, half_spread_bps: float, imp_bps: float) -> float:
    """Taker price: mid worsened by half-spread plus impact, direction-aware."""
    adj = (half_spread_bps + imp_bps) * BPS
    return mid * (1.0 + adj) if side is Side.BUY else mid * (1.0 - adj)


def walk_book(levels: Sequence[tuple[float, float]], qty: float) -> float:
    """VWAP of consuming ``qty`` coins across (price, qty) levels, best first.

    Any remainder beyond the provided depth executes at the worst level's
    price (conservative: the book snapshot is all we know).
    """
    if qty <= 0.0:
        raise ValueError("qty must be positive")
    if not levels:
        raise ValueError("empty book side")
    remaining = qty
    cost = 0.0
    worst = levels[0][0]
    for price, level_qty in levels:
        worst = price
        take = min(remaining, level_qty)
        cost += take * price
        remaining -= take
        if remaining <= 0.0:
            break
    if remaining > 0.0:
        cost += remaining * worst
    return cost / qty


def limit_crossed(side: Side, limit_price: float, print_price: float) -> bool:
    """Strict pass-through: the print must trade beyond the limit, not at it."""
    if side is Side.BUY:
        return print_price < limit_price
    return print_price > limit_price


def fee_usd(notional_usd: float, maker: bool, maker_fee: float, taker_fee: float) -> float:
    """Fee as a fraction of fill notional."""
    return abs(notional_usd) * (maker_fee if maker else taker_fee)


class LatencyModel:
    """Uniform order latency in [min_ms, max_ms], drawn from a seeded rng."""

    def __init__(self, min_ms: float, max_ms: float, rng: np.random.Generator) -> None:
        if min_ms < 0 or max_ms < min_ms:
            raise ValueError("need 0 <= latency_ms_min <= latency_ms_max")
        self._min = float(min_ms)
        self._max = float(max_ms)
        self._rng = rng

    def draw_ns(self) -> int:
        if self._max == self._min:
            return int(self._min * NS_PER_MS)
        return int(self._rng.uniform(self._min, self._max) * NS_PER_MS)
