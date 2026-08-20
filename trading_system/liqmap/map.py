"""Liquidation map core: allocate ΔOI to leverage-implied liq prices, consume
zones traversed by price, decay stale heat. Mass accounting is exact:

    ΣH == contributed - consumed - decayed - removed

`consumed` is heat taken by the traversed price path, `removed` is heat taken
out by negative ΔOI (positions closed voluntarily) — together they are the
checklist's "снятое".

Weights over the leverage grid are w = f(context) from day one; the static v1
simply ignores the context argument (stage-3 calibrators plug in here).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from trading_system.core.liquidation import liq_price
from trading_system.core.schema import Side
from trading_system.liqmap.buckets import PriceBuckets, rebucket


@dataclass(frozen=True, slots=True)
class Context:
    """Market context for dynamic weights (multi-TF features from M3)."""

    ts: int
    features: dict[str, float] = field(default_factory=dict)


WeightFn = Callable[[Context | None], np.ndarray]
# (entry, leverage, side[, qty in coins]) -> liquidation price. The 4-argument
# form lets bracket tables pick the maintenance tier from position size;
# 3-argument callables (flat MMR) keep working.
LiqPriceFn = Callable[..., float]


class StaticWeights:
    """v1 weights: a fixed distribution over the leverage grid."""

    def __init__(self, weights: np.ndarray) -> None:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be a non-negative 1-d vector with positive sum")
        self._w = w / w.sum()

    def __call__(self, context: Context | None = None) -> np.ndarray:
        return self._w


def default_liq_price_fn(mmr: float = 0.005) -> LiqPriceFn:
    return lambda entry, lev, side: liq_price(entry, lev, side, mmr)


def _takes_qty(fn: LiqPriceFn) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # builtins without introspection
        return False
    return len(params) >= 4


class LiqMap:
    """Heat H (USD) per price bucket, split by liquidation side.

    `long` heat sits BELOW price (longs liquidate on the way down), `short`
    heat sits above. One instance per (exchange, symbol).
    """

    def __init__(
        self,
        leverage_grid: list[float],
        buckets: PriceBuckets,
        weight_fn: WeightFn,
        liq_price_fn: LiqPriceFn | None = None,
        long_share: float = 0.5,
        decay_half_life_s: float = 86_400.0,
    ) -> None:
        if not 0 <= long_share <= 1:
            raise ValueError("long_share in [0, 1]")
        self.leverage_grid = np.asarray(leverage_grid, dtype=float)
        self.buckets = buckets
        self.weight_fn = weight_fn
        self.liq_price_fn = liq_price_fn or default_liq_price_fn()
        self._fn_takes_qty = _takes_qty(self.liq_price_fn)
        self.long_share = long_share
        self.decay_half_life_s = decay_half_life_s
        self.heat: dict[Side, dict[int, float]] = {Side.BUY: {}, Side.SELL: {}}
        self.contributed = 0.0
        self.consumed = 0.0
        self.decayed = 0.0
        self.removed = 0.0

    # -- mass -----------------------------------------------------------------
    def total_heat(self) -> float:
        return sum(sum(h.values()) for h in self.heat.values())

    def mass_balance_error(self) -> float:
        return abs(
            self.total_heat() - (self.contributed - self.consumed - self.decayed - self.removed)
        )

    # -- update ---------------------------------------------------------------
    def allocate(
        self,
        d_oi_usd: float,
        price: float,
        context: Context | None = None,
        long_share: float | None = None,
    ) -> None:
        """Distribute a ΔOI increment across sides x leverage grid.

        Positive ΔOI adds heat at implied liquidation prices; negative ΔOI
        removes heat proportionally from both sides (voluntary closes).
        `long_share` overrides the instance default for this increment —
        the hook for time-varying shares derived from the ratio streams.
        """
        if d_oi_usd == 0.0:
            return
        if d_oi_usd < 0.0:
            self._remove_proportional(-d_oi_usd)
            return
        ls = self.long_share if long_share is None else long_share
        if not 0 <= ls <= 1:
            raise ValueError("long_share in [0, 1]")
        w = self.weight_fn(context)
        if len(w) != len(self.leverage_grid):
            raise ValueError("weight vector length != leverage grid length")
        for side, share in ((Side.BUY, ls), (Side.SELL, 1.0 - ls)):
            if share == 0.0:
                continue
            side_heat = self.heat[side]
            for lev, wl in zip(self.leverage_grid, w, strict=True):
                if wl == 0.0:
                    continue
                amount = d_oi_usd * share * wl
                if self._fn_takes_qty:
                    # bracket tables pick the maintenance tier from position size
                    lp = self.liq_price_fn(price, float(lev), side, amount / price)
                else:
                    lp = self.liq_price_fn(price, float(lev), side)
                if lp <= 0.0:
                    continue
                idx = self.buckets.index(lp)
                side_heat[idx] = side_heat.get(idx, 0.0) + amount
                self.contributed += amount

    def _remove_proportional(self, amount_usd: float) -> None:
        total = self.total_heat()
        if total <= 0.0:
            return
        factor = min(amount_usd / total, 1.0)
        removed = 0.0
        for side_heat in self.heat.values():
            for idx in list(side_heat):
                delta = side_heat[idx] * factor
                side_heat[idx] -= delta
                removed += delta
                if side_heat[idx] <= 1e-15:
                    del side_heat[idx]
        self.removed += removed

    def consume(self, path_lo: float, path_hi: float) -> float:
        """Zero heat in every bucket the price path [lo, hi] touched.

        A long pool at price p triggers once price trades down to p, a short
        pool once price trades up to p — both mean p in [lo, hi].
        """
        if path_lo > path_hi:
            raise ValueError("path_lo > path_hi")
        taken = 0.0
        for side_heat in self.heat.values():
            for idx in list(side_heat):
                # buckets are half-open [lo, hi): a bucket ending exactly at
                # path_lo was never traversed, so the overlap test is strict
                if self.buckets.hi(idx) > path_lo and self.buckets.lo(idx) <= path_hi:
                    taken += side_heat.pop(idx)
        self.consumed += taken
        return taken

    def decay(self, dt_s: float) -> float:
        """Exponential half-life decay of all heat; returns the decayed mass."""
        if dt_s < 0:
            raise ValueError("dt_s must be >= 0")
        keep = 0.5 ** (dt_s / self.decay_half_life_s)
        lost = 0.0
        for side_heat in self.heat.values():
            for idx in list(side_heat):
                delta = side_heat[idx] * (1.0 - keep)
                side_heat[idx] -= delta
                lost += delta
                if side_heat[idx] <= 1e-15:
                    del side_heat[idx]
        self.decayed += lost
        return lost

    def step(
        self,
        bar_low: float,
        bar_high: float,
        bar_close: float,
        d_oi_usd: float,
        dt_s: float,
        context: Context | None = None,
        long_share: float | None = None,
    ) -> None:
        """One bar update: consume the traversed path, allocate new OI, decay."""
        self.consume(bar_low, bar_high)
        self.allocate(d_oi_usd, bar_close, context, long_share=long_share)
        self.decay(dt_s)

    def rebucket_to(self, new_buckets: PriceBuckets) -> None:
        for side in self.heat:
            self.heat[side] = rebucket(self.heat[side], self.buckets, new_buckets)
        self.buckets = new_buckets

    # -- views ----------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """Dense arrays (prices, long heat, short heat) over the occupied range."""
        idxs = sorted(set(self.heat[Side.BUY]) | set(self.heat[Side.SELL]))
        if not idxs:
            return {"prices": np.array([]), "long": np.array([]), "short": np.array([])}
        lo, hi = idxs[0], idxs[-1]
        rng = np.arange(lo, hi + 1)
        return {
            "prices": np.array([self.buckets.center(int(i)) for i in rng]),
            "long": np.array([self.heat[Side.BUY].get(int(i), 0.0) for i in rng]),
            "short": np.array([self.heat[Side.SELL].get(int(i), 0.0) for i in rng]),
        }

    def top_pools(self, k: int = 5) -> list[tuple[float, float, Side]]:
        """Largest pools as (bucket center price, heat, side), descending."""
        pools = [
            (self.buckets.center(idx), heat, side)
            for side, side_heat in self.heat.items()
            for idx, heat in side_heat.items()
        ]
        return sorted(pools, key=lambda p: -p[1])[:k]
