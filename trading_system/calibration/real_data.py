"""Fast causal heat builder over REAL bar data — the bridge from the lake to
the stage-3 calibrators.

LiqMap replays are exact but too slow inside a calibrator loop (every weight
candidate is a full replay). This vectorized builder mirrors LiqMap semantics
on a fixed bucket grid: allocate ΔOI⁺ at leverage-implied liquidation prices
(per-bar long shares), consume buckets swept by the bar's [low, high] path,
remove proportionally on ΔOI⁻, decay exponentially. Row t is the snapshot at
bar t's close and uses information up to bar t only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from trading_system.core.liquidation import liq_price
from trading_system.core.schema import Side

HeatBuilder = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class BarArrays:
    """Aligned per-bar inputs for the builder (row t = bar t)."""

    ts: np.ndarray  # bar close, UTC ns
    close: np.ndarray
    low: np.ndarray
    high: np.ndarray
    d_oi_usd: np.ndarray  # signed; NaN-free (warm-up filled by caller policy)
    long_share: np.ndarray  # in [0, 1]
    atr: np.ndarray


def bars_to_arrays(bars: pl.DataFrame, oi_fallback_frac: float = 0.05) -> BarArrays:
    """Bar frame -> arrays; warm-up bars without ΔOI get frac * quote_volume."""
    d_oi = bars["d_oi_usd"].to_numpy() if "d_oi_usd" in bars.columns else np.full(bars.height, np.nan)
    d_oi = np.asarray(d_oi, dtype=float)
    fallback = bars["quote_volume"].to_numpy() * oi_fallback_frac
    d_oi = np.where(np.isnan(d_oi), fallback, d_oi)
    if "long_share" in bars.columns:
        ls = np.nan_to_num(bars["long_share"].to_numpy().astype(float), nan=0.5)
    else:
        ls = np.full(bars.height, 0.5)
    atr = bars["atr"].to_numpy().astype(float) if "atr" in bars.columns else np.full(bars.height, np.nan)
    return BarArrays(
        ts=bars["ts_close"].to_numpy().astype(np.int64),
        close=bars["close"].to_numpy().astype(float),
        low=bars["low"].to_numpy().astype(float),
        high=bars["high"].to_numpy().astype(float),
        d_oi_usd=d_oi,
        long_share=np.clip(ls, 0.0, 1.0),
        atr=atr,
    )


def bucket_grid(arr: BarArrays, atr_fraction: float = 0.1, span_frac: float = 0.36) -> np.ndarray:
    """Fixed bucket edges covering the whole leverage grid's reach.

    Bucket size = median ATR * atr_fraction (the map's own rule); the span
    stretches `span_frac` beyond the traded range so 3x pools stay on-grid.
    """
    size = float(np.nanmedian(arr.atr)) * atr_fraction
    if not size > 0:
        raise ValueError("ATR-derived bucket size must be positive")
    lo = float(arr.low.min()) * (1.0 - span_frac)
    hi = float(arr.high.max()) * (1.0 + span_frac)
    i0 = int(np.floor(lo / size))
    i1 = int(np.ceil(hi / size))
    return np.arange(i0, i1 + 1) * size


def make_real_heat_builder(
    arr: BarArrays,
    leverage_grid: np.ndarray,
    bucket_edges: np.ndarray,
    bar_s: float,
    decay_half_life_s: float = 86_400.0,
    mmr: float = 0.005,
    liq_fn: Callable[[float, float, Side, float], float] | None = None,
) -> HeatBuilder:
    """build(w) -> (n_bars, n_buckets) heat matrices under mixture w.

    `liq_fn(entry, lev, side, qty)` (e.g. bracket tables) overrides flat mmr;
    its qty argument is approximated with the equal-weight slice size, so the
    bucket targets stay independent of the candidate w (this keeps the
    precompute out of the calibrator loop; tier drift across candidates is a
    second-order effect).
    """
    n = len(arr.ts)
    grid = np.asarray(leverage_grid, dtype=float)
    k = len(grid)
    edges = np.asarray(bucket_edges, dtype=float)
    nb = len(edges) - 1
    decay = 0.5 ** (bar_s / decay_half_life_s)

    # bucket -1 == no allocation: LiqMap drops lp <= 0 (e.g. 1x longs that can
    # never liquidate above zero), so the fast builder must not park that mass
    # in the bottom edge bucket
    bucket_of = np.empty((n, k, 2), dtype=int)
    for t in range(n):
        entry = arr.close[t]
        for j in range(k):
            for si, side in enumerate((Side.BUY, Side.SELL)):
                if liq_fn is not None:
                    share = arr.long_share[t] if si == 0 else 1.0 - arr.long_share[t]
                    qty = max(arr.d_oi_usd[t], 0.0) * share / (k * max(entry, 1e-12))
                    lp = liq_fn(entry, float(grid[j]), side, max(qty, 1e-12))
                else:
                    lp = liq_price(entry, float(grid[j]), side, mmr)
                if lp > 0.0 and np.isfinite(lp):
                    b = int(np.searchsorted(edges, lp, side="right") - 1)
                    bucket_of[t, j, si] = min(max(b, 0), nb - 1)
                else:
                    bucket_of[t, j, si] = -1

    cross_lo = np.empty(n, dtype=int)
    cross_hi = np.empty(n, dtype=int)
    for t in range(n):
        cross_lo[t] = max(int(np.searchsorted(edges, arr.low[t], side="right") - 1), 0)
        cross_hi[t] = min(int(np.searchsorted(edges, arr.high[t], side="right") - 1), nb - 1)

    def build(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        wm = np.tile(w, (n, 1)) if w.ndim == 1 else w
        if wm.shape != (n, k):
            raise ValueError(f"weights must be ({k},) or ({n}, {k})")
        wm = wm / np.maximum(wm.sum(axis=1, keepdims=True), 1e-300)
        heat = np.empty((n, nb))
        row = np.zeros(nb)
        for t in range(n):
            # bar order mirrors LiqMap.step: consume path -> allocate -> decay
            if cross_hi[t] >= cross_lo[t]:
                row[cross_lo[t] : cross_hi[t] + 1] = 0.0
            usd = arr.d_oi_usd[t]
            if usd > 0.0:
                shares = (arr.long_share[t], 1.0 - arr.long_share[t])
                for j in range(k):
                    contrib = usd * wm[t, j]
                    for si in (0, 1):
                        b = bucket_of[t, j, si]
                        if b >= 0:
                            row[b] += contrib * shares[si]
            elif usd < 0.0:
                total = row.sum()
                if total > 0.0:
                    row *= max(0.0, 1.0 + usd / total)  # proportional close-out
            row *= decay
            heat[t] = row
        return heat

    return build
