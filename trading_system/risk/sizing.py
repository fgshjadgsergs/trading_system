"""Volatility-target position sizing.

position_usd = equity * target_daily_vol / realized_daily_vol, where realized
daily vol comes from an EWMA estimator over per-bar returns, capped by
max_position_usd (configs `risk` section) and rounded down to the exchange
quantity step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SizeResult:
    """Sizing decision: USD notional, coin qty (step-rounded) and flags."""

    position_usd: float
    qty: float
    capped: bool
    vol_used: float
    reason: str


class EwmaVol:
    """EWMA volatility estimator over per-bar returns, reported as daily vol.

    var <- (1 - alpha) * var + alpha * ret^2 with alpha = 1 - 0.5^(1/halflife).
    Daily vol scales the per-bar vol by sqrt(bars_per_day).
    """

    def __init__(self, halflife_bars: float, bars_per_day: float) -> None:
        if halflife_bars <= 0 or bars_per_day <= 0:
            raise ValueError("halflife_bars and bars_per_day must be positive")
        self.alpha = 1.0 - 0.5 ** (1.0 / halflife_bars)
        self.bars_per_day = bars_per_day
        self._var: float | None = None
        self.n = 0

    def update(self, ret: float) -> float:
        """Feed one per-bar return; returns the updated daily vol.

        Non-finite returns (NaN/inf from a broken feed) are ignored entirely:
        one bad tick must not poison the estimator forever.
        """
        r = float(ret)
        if not math.isfinite(r):
            return self.daily_vol
        r2 = r * r
        self._var = r2 if self._var is None else (1.0 - self.alpha) * self._var + self.alpha * r2
        self.n += 1
        return self.daily_vol

    @property
    def daily_vol(self) -> float:
        """Current daily volatility estimate; 0.0 before any update."""
        if self._var is None:
            return 0.0
        return math.sqrt(self._var * self.bars_per_day)

    def reset(self) -> None:
        self._var = None
        self.n = 0


def round_qty_to_step(qty: float, step: float) -> float:
    """Round a quantity DOWN to the exchange step size (never oversize)."""
    if step <= 0:
        raise ValueError("step must be positive")
    if qty <= 0:
        return 0.0
    steps = math.floor(qty / step + 1e-9)  # epsilon absorbs float dust like 0.30000000004/0.1
    return round(steps * step, 12)


def vol_target_position_usd(
    equity: float,
    target_daily_vol: float,
    realized_daily_vol: float,
    max_position_usd: float,
    vol_floor: float = 1e-8,
) -> tuple[float, bool, str]:
    """Raw vol-target notional with cap and zero-vol guard.

    Returns (position_usd, capped, reason). A vol at/below vol_floor means the
    estimator has no reliable information — the guard sizes to zero instead of
    exploding toward infinity (the conservative failure mode for a risk module).
    """
    if (
        not math.isfinite(equity)
        or not math.isfinite(target_daily_vol)
        or equity <= 0
        or target_daily_vol <= 0
    ):
        # NaN compares False against <= 0, so finiteness is checked explicitly:
        # garbage equity must size to zero, never to NaN or the cap.
        return 0.0, False, "non-positive equity or target vol"
    if not math.isfinite(realized_daily_vol) or realized_daily_vol <= vol_floor:
        return 0.0, False, "zero-vol guard: no reliable vol estimate"
    raw = equity * target_daily_vol / realized_daily_vol
    if raw > max_position_usd:
        return max_position_usd, True, "capped at max_position_usd"
    return raw, False, "vol-target"


class VolTargetSizer:
    """EWMA estimator + vol-target formula + cap + step rounding in one object.

    All parameters map to the configs `risk` section; qty_step is the exchange
    LOT_SIZE step for the traded symbol.
    """

    def __init__(
        self,
        target_daily_vol: float,
        max_position_usd: float,
        qty_step: float,
        halflife_bars: float = 60.0,
        bars_per_day: float = 1440.0,
        vol_floor: float = 1e-8,
    ) -> None:
        self.target_daily_vol = target_daily_vol
        self.max_position_usd = max_position_usd
        self.qty_step = qty_step
        self.vol_floor = vol_floor
        self.estimator = EwmaVol(halflife_bars=halflife_bars, bars_per_day=bars_per_day)

    def update(self, ret: float) -> float:
        """Feed one per-bar return into the vol estimator."""
        return self.estimator.update(ret)

    def size(self, equity: float, price: float) -> SizeResult:
        """Sizing decision at the current vol estimate for a given mark price."""
        vol = self.estimator.daily_vol
        usd, capped, reason = vol_target_position_usd(
            equity, self.target_daily_vol, vol, self.max_position_usd, self.vol_floor
        )
        qty = round_qty_to_step(usd / price, self.qty_step) if price > 0 and usd > 0 else 0.0
        return SizeResult(position_usd=usd, qty=qty, capped=capped, vol_used=vol, reason=reason)
