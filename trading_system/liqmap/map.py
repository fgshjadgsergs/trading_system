"""Liquidation map core: allocate ΔOI to leverage-implied liq prices, consume
zones traversed by price, decay stale heat. Mass accounting is exact:

    ΣH == contributed - consumed - decayed - removed

`consumed` is heat taken by the traversed price path, `removed` is heat taken
out by negative ΔOI (positions closed voluntarily) — together they are the
checklist's "снятое". `dropped` counts ΔOI⁺ that never became heat (slices
whose liquidation price is <= 0, e.g. 1x longs) and sits OUTSIDE the identity:
contributed + dropped == the positive ΔOI fed in. All accumulators carry
Neumaier compensation, so the identity holds to ~1 ulp of the true sums.

Weights over the leverage grid are w = f(context) from day one; the static v1
simply ignores the context argument (stage-3 calibrators plug in here).
"""

from __future__ import annotations

import inspect
import math
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
        if w.ndim != 1 or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be a finite non-negative 1-d vector with positive sum")
        self._w = w / w.sum()

    def __call__(self, context: Context | None = None) -> np.ndarray:
        return self._w


def default_liq_price_fn(mmr: float = 0.005) -> LiqPriceFn:
    return lambda entry, lev, side: liq_price(entry, lev, side, mmr)


def _neumaier_add(value: float, carry: float, x: float) -> tuple[float, float]:
    """One Neumaier-compensated addition step: returns (new_value, new_carry).

    The pair represents the exact sum as value + carry; `value` alone is the
    ordinary float accumulator (public attribute), `carry` holds the rounding
    error recovered at each step.
    """
    s = value + x
    if abs(value) >= abs(x):
        carry += (value - s) + x
    else:
        carry += (x - s) + value
    return s, carry


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
        typical_account_usd: float | None = None,
        blur_sigma0_bps: float | None = None,
        blur_sigma1: float = 0.0,
        fractional_edge_consume: bool = False,
        close_out_fraction: float = 1.0,
    ) -> None:
        """`typical_account_usd` (M1, opt-in): with a 4-argument (bracket)
        liq_price_fn, the maintenance tier is picked for a REPRESENTATIVE
        account of this USD size instead of the aggregate slice notional.
        The tier qty then does not depend on the weight vector (so heat
        targets are w-independent and bit-identical to the fast mirror);
        the admissible-account clamp inside the bracket engine still
        applies. Default None keeps the old slice-notional behavior.

        `blur_sigma0_bps` (R1, opt-in): Gaussian blur of each slice around
        its liquidation price. The kernel width in price units is

            sigma_price = (sigma0 + sigma1 * |price - lp| / price * 1e4)
                          * price / 1e4

        i.e. `blur_sigma0_bps` is in bps of the current price and
        `blur_sigma1` adds bps of width per bp of distance between the
        entry price and lp. Support is ±3 sigma (in buckets), capped at 41
        buckets; cell weights ∝ exp(-0.5 * (offset / sigma_b)^2) at bucket
        centers. Cells on the wrong side of the current price (long heat at
        or above price, short heat at or below) are unphysical and are cut,
        the rest renormalize; if everything is cut the whole slice lands in
        index(lp) as without blur. Default None = no blur (point mass)."""
        if not 0 <= long_share <= 1:
            raise ValueError("long_share in [0, 1]")
        if not decay_half_life_s > 0:  # `not >` rejects NaN too; +inf = no decay
            raise ValueError("decay_half_life_s must be positive")
        if typical_account_usd is not None and not (
            typical_account_usd > 0 and math.isfinite(typical_account_usd)
        ):
            raise ValueError("typical_account_usd must be positive and finite (or None)")
        if blur_sigma0_bps is not None and not (
            blur_sigma0_bps >= 0 and math.isfinite(blur_sigma0_bps)
        ):
            raise ValueError("blur_sigma0_bps must be >= 0 and finite (or None)")
        if not (blur_sigma1 >= 0 and math.isfinite(blur_sigma1)):
            raise ValueError("blur_sigma1 must be >= 0 and finite")
        self.leverage_grid = np.asarray(leverage_grid, dtype=float)
        # plain-float copy of the grid: allocate() iterates Python floats, so
        # no np.float64 leaks into liq_price_fn / the heat dict (N1)
        self._grid_list = [float(g) for g in self.leverage_grid]
        self.buckets = buckets
        self.weight_fn = weight_fn
        self.liq_price_fn = liq_price_fn or default_liq_price_fn()
        self._fn_takes_qty = _takes_qty(self.liq_price_fn)
        self.long_share = long_share
        self.decay_half_life_s = decay_half_life_s
        self.typical_account_usd = typical_account_usd
        self.blur_sigma0_bps = blur_sigma0_bps
        self.blur_sigma1 = blur_sigma1
        self.fractional_edge_consume = fractional_edge_consume
        if not 0.0 <= close_out_fraction <= 1.0:
            raise ValueError("close_out_fraction in [0, 1]")
        # share of a negative ΔOI that removes heat. 1.0 (default) assumes
        # every voluntarily closed position was carrying heat somewhere on the
        # map; in practice much of the closing flow is positions far from
        # liquidation (or already-consumed ones), and charging the full amount
        # against the map drains levels the price never reached
        self.close_out_fraction = close_out_fraction
        self.heat: dict[Side, dict[int, float]] = {Side.BUY: {}, Side.SELL: {}}
        # public accumulators stay plain readable/writable floats; each has a
        # private Neumaier carry (N3) so the mass identity holds to ~1 ulp of
        # the true sums even after 1e5+ operations
        self.contributed = 0.0
        self.consumed = 0.0
        self.decayed = 0.0
        self.removed = 0.0
        self.dropped = 0.0  # ΔOI⁺ skipped because lp <= 0 (M5); not in the balance
        self._contributed_carry = 0.0
        self._consumed_carry = 0.0
        self._decayed_carry = 0.0
        self._removed_carry = 0.0
        self._dropped_carry = 0.0

    # -- mass -----------------------------------------------------------------
    def total_heat(self) -> float:
        return sum(sum(h.values()) for h in self.heat.values())

    def mass_balance_error(self) -> float:
        return abs(
            self.total_heat()
            - (
                (self.contributed + self._contributed_carry)
                - (self.consumed + self._consumed_carry)
                - (self.decayed + self._decayed_carry)
                - (self.removed + self._removed_carry)
            )
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
        if not math.isfinite(d_oi_usd):
            raise ValueError("d_oi_usd must be finite")
        if d_oi_usd == 0.0:
            return
        if d_oi_usd < 0.0:
            self._remove_proportional(-d_oi_usd * self.close_out_fraction)
            return
        if not (math.isfinite(price) and price > 0):
            raise ValueError("price must be positive and finite")
        ls = self.long_share if long_share is None else long_share
        if not 0 <= ls <= 1:
            raise ValueError("long_share in [0, 1]")
        w = np.asarray(self.weight_fn(context), dtype=float)
        if len(w) != len(self.leverage_grid):
            raise ValueError("weight vector length != leverage grid length")
        if not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("weight vector must be finite and non-negative")
        for side, share in ((Side.BUY, ls), (Side.SELL, 1.0 - ls)):
            if share == 0.0:
                continue
            side_heat = self.heat[side]
            for lev, wl in zip(self._grid_list, w.tolist(), strict=True):
                if wl == 0.0:
                    continue
                amount = d_oi_usd * share * wl
                if self._fn_takes_qty:
                    # bracket tables pick the maintenance tier from position
                    # size: the aggregate slice notional by default, or the
                    # representative typical account when configured (M1)
                    qty_usd = (
                        amount if self.typical_account_usd is None else self.typical_account_usd
                    )
                    lp = self.liq_price_fn(price, lev, side, qty_usd / price)
                else:
                    lp = self.liq_price_fn(price, lev, side)
                if lp <= 0.0:
                    # slice can never liquidate (e.g. 1x longs): count it (M5)
                    self.dropped, self._dropped_carry = _neumaier_add(
                        self.dropped, self._dropped_carry, amount
                    )
                    continue
                if self.blur_sigma0_bps is None:
                    idx = self.buckets.index(lp)
                    side_heat[idx] = side_heat.get(idx, 0.0) + amount
                else:
                    self._spread_blurred(side_heat, side, lp, price, amount)
                self.contributed, self._contributed_carry = _neumaier_add(
                    self.contributed, self._contributed_carry, amount
                )

    def _blur_cells(self, lp: float, price: float, side: Side) -> tuple[list[int], list[float]]:
        """R1 kernel support and normalized weights for a slice at lp.

        See __init__ for the sigma formula. Returns ([], []) when the kernel
        degenerates or every cell is cut by the side trim — the caller then
        falls back to the point allocation at index(lp).
        """
        dist_bps = abs(price - lp) / price * 1e4
        sigma_price = (self.blur_sigma0_bps + self.blur_sigma1 * dist_bps) * price / 1e4
        sigma_b = sigma_price / self.buckets.bucket_size
        if not sigma_b > 0.0:
            return [], []
        half = min(int(math.ceil(3.0 * sigma_b)), 20)  # width cap: 41 buckets
        idx0 = self.buckets.index(lp)
        cells: list[int] = []
        weights: list[float] = []
        for off in range(-half, half + 1):
            idx = idx0 + off
            center = self.buckets.center(idx)
            # a Gaussian tail past the current price is unphysical: long heat
            # must stay strictly below price, short heat strictly above
            if side is Side.BUY:
                if center >= price:
                    continue
            elif center <= price:
                continue
            cells.append(idx)
            weights.append(math.exp(-0.5 * (off / sigma_b) ** 2))
        total = math.fsum(weights)
        if not total > 0.0:
            return [], []
        return cells, [wt / total for wt in weights]

    def _spread_blurred(
        self, side_heat: dict[int, float], side: Side, lp: float, price: float, amount: float
    ) -> None:
        """Distribute `amount` over the blur kernel; conserves mass exactly.

        The largest-weight cell takes the exact remainder of the sequential
        split, so the added parts always sum to `amount` (and never leave a
        negative crumb in a low-weight tail cell).
        """
        cells, weights = self._blur_cells(lp, price, side)
        if not cells:
            idx = self.buckets.index(lp)  # fully cut or degenerate kernel
            side_heat[idx] = side_heat.get(idx, 0.0) + amount
            return
        j_max = max(range(len(weights)), key=weights.__getitem__)
        remaining = amount
        for j, (idx, wt) in enumerate(zip(cells, weights, strict=True)):
            if j == j_max:
                continue
            part = amount * wt
            side_heat[idx] = side_heat.get(idx, 0.0) + part
            remaining -= part
        idx = cells[j_max]
        side_heat[idx] = side_heat.get(idx, 0.0) + remaining

    def _remove_proportional(self, amount_usd: float) -> None:
        total = self.total_heat()
        if total <= 0.0:
            return
        factor = min(amount_usd / total, 1.0)
        removed, carry = 0.0, 0.0
        for side_heat in self.heat.values():
            for idx in list(side_heat):
                delta = side_heat[idx] * factor
                side_heat[idx] -= delta
                removed, carry = _neumaier_add(removed, carry, delta)
                if side_heat[idx] <= 1e-15:
                    # как в decay: остаток-пыль зачисляется в removed точно
                    removed, carry = _neumaier_add(removed, carry, side_heat[idx])
                    del side_heat[idx]
        self.removed, self._removed_carry = _neumaier_add(self.removed, self._removed_carry, removed)
        self.removed, self._removed_carry = _neumaier_add(self.removed, self._removed_carry, carry)

    def _consume_amount(self, idx: int, path_lo: float, path_hi: float, h: float) -> float:
        """How much of bucket idx's heat the path takes.

        Full-bucket mode: everything. Fractional-edge mode: interior buckets
        fully, edge buckets pro rata to interval overlap (heat assumed uniform
        within a bucket) — a bar that clipped a bucket's corner should not
        wipe pools the price never reached.
        """
        if not self.fractional_edge_consume:
            return h
        lo, hi = self.buckets.lo(idx), self.buckets.hi(idx)
        if path_lo <= lo and hi <= path_hi:
            return h
        width = hi - lo
        if width <= 0.0:
            return h
        overlap = min(hi, path_hi) - max(lo, path_lo)
        frac = min(max(overlap / width, 0.0), 1.0)
        return h * frac

    def consume(self, path_lo: float, path_hi: float) -> float:
        """Zero heat in every bucket the price path [lo, hi] touched.

        A long pool at price p triggers once price trades down to p, a short
        pool once price trades up to p — both mean p in [lo, hi]. With
        fractional_edge_consume, partially traversed edge buckets lose only
        the traversed share of their heat.
        """
        if not path_lo <= path_hi:  # `not <=` also rejects NaN bounds (broken bar)
            raise ValueError("path_lo > path_hi (or NaN path bound)")
        taken, carry = 0.0, 0.0

        def take(side_heat: dict[int, float], idx: int) -> tuple[float, float]:
            part = self._consume_amount(idx, path_lo, path_hi, side_heat[idx])
            rest = side_heat[idx] - part
            if rest <= 0.0:
                side_heat.pop(idx)
            else:
                side_heat[idx] = rest
            return _neumaier_add(taken, carry, part)

        # N2: candidate index range of the path, extended ±1 to cover index()'s
        # ulp correction at exact bucket edges; no bucket outside it can pass
        # the overlap predicates (bucket grid is monotone in the index)
        i_lo = self.buckets.index(path_lo) - 1
        i_hi = self.buckets.index(path_hi) + 1
        width = i_hi - i_lo + 1
        for side_heat in self.heat.values():
            if width < len(side_heat):
                # narrow path over a big map: probe candidates by membership,
                # with THE SAME half-open predicates as the full scan
                for idx in range(i_lo, i_hi + 1):
                    if (
                        idx in side_heat
                        and self.buckets.hi(idx) > path_lo
                        and self.buckets.lo(idx) <= path_hi
                    ):
                        taken, carry = take(side_heat, idx)
            else:
                for idx in list(side_heat):
                    # buckets are half-open [lo, hi): a bucket ending exactly at
                    # path_lo was never traversed, so the overlap test is strict
                    if self.buckets.hi(idx) > path_lo and self.buckets.lo(idx) <= path_hi:
                        taken, carry = take(side_heat, idx)
        self.consumed, self._consumed_carry = _neumaier_add(self.consumed, self._consumed_carry, taken)
        self.consumed, self._consumed_carry = _neumaier_add(self.consumed, self._consumed_carry, carry)
        return taken

    def decay(self, dt_s: float) -> float:
        """Exponential half-life decay of all heat; returns the decayed mass."""
        if not dt_s >= 0:  # `not >=` also rejects NaN, which would poison all heat
            raise ValueError("dt_s must be >= 0 (and not NaN)")
        keep = 0.5 ** (dt_s / self.decay_half_life_s)
        lost, carry = 0.0, 0.0
        for side_heat in self.heat.values():
            for idx in list(side_heat):
                delta = side_heat[idx] * (1.0 - keep)
                side_heat[idx] -= delta
                lost, carry = _neumaier_add(lost, carry, delta)
                if side_heat[idx] <= 1e-15:
                    # пыль уходит в decayed, а не испаряется молча: без этого
                    # инвариант массы подтекает по <=1e-15 на каждое удаление
                    lost, carry = _neumaier_add(lost, carry, side_heat[idx])
                    del side_heat[idx]
        self.decayed, self._decayed_carry = _neumaier_add(self.decayed, self._decayed_carry, lost)
        self.decayed, self._decayed_carry = _neumaier_add(self.decayed, self._decayed_carry, carry)
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
