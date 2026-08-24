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

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from trading_system.core.liquidation import liq_price
from trading_system.core.schema import Side

HeatBuilder = Callable[[np.ndarray], np.ndarray]

LONG, SHORT = 0, 1
"""Side axis order of the builder's precompute and of split-sides heat:
index 0 = long heat (pools BELOW price; real liquidation prints of longs
carry order side SELL), index 1 = short heat (pools above; prints side BUY).
"""


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
    vwap: np.ndarray | None = None  # bar VWAP entry prices (M4); None when unknown


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
    close = bars["close"].to_numpy().astype(float)
    vwap: np.ndarray | None = None
    if "vwap_bar" in bars.columns:
        vwap = bars["vwap_bar"].to_numpy().astype(float)
    elif "quote_volume" in bars.columns and "volume" in bars.columns:
        vol = bars["volume"].to_numpy().astype(float)
        qv = bars["quote_volume"].to_numpy().astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(vol > 0, qv / np.where(vol > 0, vol, 1.0), close)
    if vwap is not None:
        vwap = np.where(np.isfinite(vwap) & (vwap > 0), vwap, close)
    return BarArrays(
        ts=bars["ts_close"].to_numpy().astype(np.int64),
        close=close,
        low=bars["low"].to_numpy().astype(float),
        high=bars["high"].to_numpy().astype(float),
        d_oi_usd=d_oi,
        long_share=np.clip(ls, 0.0, 1.0),
        atr=atr,
        vwap=vwap,
    )


def bucket_grid(arr: BarArrays, atr_fraction: float = 0.1, span_frac: float = 0.36) -> np.ndarray:
    """Fixed bucket edges covering the whole leverage grid's reach.

    Bucket size = median ATR * atr_fraction (the map's own rule); the span
    stretches `span_frac` beyond the traded range so 3x pools stay on-grid.
    See `exact_bucket_grid` (N5) for a span built from the actually computed
    liquidation prices instead of a blanket margin.
    """
    size = float(np.nanmedian(arr.atr)) * atr_fraction
    if not size > 0:
        raise ValueError("ATR-derived bucket size must be positive")
    lo = float(arr.low.min()) * (1.0 - span_frac)
    hi = float(arr.high.max()) * (1.0 + span_frac)
    i0 = int(np.floor(lo / size))
    i1 = int(np.ceil(hi / size))
    return np.arange(i0, i1 + 1) * size


def _entry_prices(arr: BarArrays, entry: str) -> np.ndarray:
    """Per-bar allocation entry prices: bar close (default) or bar VWAP (M4)."""
    if entry == "close":
        return arr.close
    if entry == "vwap":
        if arr.vwap is None:
            raise ValueError("entry='vwap' requires a vwap array (vwap_bar or quote/volume columns)")
        return arr.vwap
    raise ValueError(f"entry must be 'close' or 'vwap', got {entry!r}")


def slice_liq_prices(
    arr: BarArrays,
    leverage_grid: np.ndarray,
    mmr: float = 0.005,
    liq_fn: Callable[[float, float, Side, float], float] | None = None,
    typical_account_usd: float | None = None,
    entry: str = "close",
) -> np.ndarray:
    """(n, k, 2) liquidation prices per (bar, leverage rung, position side).

    Side axis: 0 = long, 1 = short (module constants LONG/SHORT). Slices that
    never liquidate (lp <= 0 or non-finite) are returned as NaN. Mirrors
    LiqMap.allocate's per-slice lp computation, including the typical-account
    tier qty (M1) when `typical_account_usd` is set.
    """
    n = len(arr.ts)
    grid = np.asarray(leverage_grid, dtype=float)
    k = len(grid)
    entries = _entry_prices(arr, entry)
    lps = np.full((n, k, 2), np.nan)
    for t in range(n):
        e = entries[t]
        for j in range(k):
            for si, side in enumerate((Side.BUY, Side.SELL)):
                if liq_fn is not None:
                    if typical_account_usd is not None:
                        qty = typical_account_usd / e
                    else:
                        share = arr.long_share[t] if si == LONG else 1.0 - arr.long_share[t]
                        qty = max(arr.d_oi_usd[t], 0.0) * share / (k * max(e, 1e-12))
                    lp = liq_fn(e, float(grid[j]), side, max(qty, 1e-12))
                else:
                    lp = liq_price(e, float(grid[j]), side, mmr)
                if lp > 0.0 and np.isfinite(lp):
                    lps[t, j, si] = lp
    return lps


def _kernel_sigma_buckets(
    entry: float, lp: float, bucket_size: float, sigma0_bps: float, sigma1: float
) -> float:
    """Blur kernel width in buckets — the same formula as LiqMap (R1)."""
    dist_bps = abs(entry - lp) / entry * 1e4
    sigma_price = (sigma0_bps + sigma1 * dist_bps) * entry / 1e4
    return sigma_price / bucket_size


def exact_bucket_grid(
    arr: BarArrays,
    leverage_grid: np.ndarray,
    atr_fraction: float = 0.1,
    mmr: float = 0.005,
    liq_fn: Callable[[float, float, Side, float], float] | None = None,
    typical_account_usd: float | None = None,
    blur_sigma0_bps: float | None = None,
    blur_sigma1: float = 0.0,
    entry: str = "close",
) -> np.ndarray:
    """N5: bucket edges whose span is built from the ACTUALLY computed
    liquidation prices (two-pass: precompute lp -> min/max over finite lps
    ∪ the traded [low, high] range), plus the blur kernel's reach when blur
    is active — instead of `bucket_grid`'s blanket ±36% margin that leaves
    edge buckets collecting clamped (fictitious) mass.
    """
    size = float(np.nanmedian(arr.atr)) * atr_fraction
    if not size > 0:
        raise ValueError("ATR-derived bucket size must be positive")
    lps = slice_liq_prices(
        arr, leverage_grid, mmr=mmr, liq_fn=liq_fn,
        typical_account_usd=typical_account_usd, entry=entry,
    )
    finite = lps[np.isfinite(lps)]
    lo = float(arr.low.min())
    hi = float(arr.high.max())
    if finite.size:
        lo = min(lo, float(finite.min()))
        hi = max(hi, float(finite.max()))
    margin_buckets = 1  # slack for edge-of-bucket rounding of the span bounds
    if blur_sigma0_bps is not None:
        # kernel support is capped at +-20 buckets around the lp bucket (R1)
        entries = _entry_prices(arr, entry)
        sig_max = 0.0
        for t in range(len(arr.ts)):
            for lp in lps[t].ravel():
                if np.isfinite(lp):
                    sig_max = max(
                        sig_max,
                        _kernel_sigma_buckets(entries[t], lp, size, blur_sigma0_bps, blur_sigma1),
                    )
        margin_buckets += min(int(math.ceil(3.0 * sig_max)), 20)
    i0 = int(np.floor(lo / size)) - margin_buckets
    i1 = int(np.floor(hi / size)) + 1 + margin_buckets
    edges = np.arange(i0, i1 + 1) * size
    b = np.searchsorted(edges, finite, side="right") - 1
    if finite.size and not ((b >= 0) & (b < len(edges) - 1)).all():
        raise ValueError("exact_bucket_grid: computed liquidation price fell off the grid span")
    return edges


def make_real_heat_builder(
    arr: BarArrays,
    leverage_grid: np.ndarray,
    bucket_edges: np.ndarray,
    bar_s: float,
    decay_half_life_s: float = 86_400.0,
    mmr: float = 0.005,
    liq_fn: Callable[[float, float, Side, float], float] | None = None,
    typical_account_usd: float | None = None,
    blur_sigma0_bps: float | None = None,
    blur_sigma1: float = 0.0,
    entry: str = "close",
    split_sides: bool = False,
    fractional_edge_consume: bool = False,
    close_out_fraction: float = 1.0,
) -> HeatBuilder:
    """build(w) -> (n_bars, n_buckets) heat matrices under mixture w.

    `liq_fn(entry, lev, side, qty)` (e.g. bracket tables) overrides flat mmr;
    its qty argument is approximated with the equal-weight slice size, so the
    bucket targets stay independent of the candidate w (this keeps the
    precompute out of the calibrator loop; tier drift across candidates is a
    second-order effect). With `typical_account_usd` (M1, mirrors LiqMap),
    the tier qty is the representative account size instead — exactly
    typical_account_usd / entry, bit-identical to the map's choice.

    `blur_sigma0_bps`/`blur_sigma1` (R1) mirror LiqMap's Gaussian blur: the
    kernel offsets/weights are precomputed per (bar, rung, side) with the
    same sigma formula, side trim against the entry price and width cap.

    `entry` (M4): "close" (default) or "vwap" — the per-bar allocation entry
    price used for the lp precompute.

    `split_sides` (M8): build(w) returns (n_bars, 2, n_buckets) with the side
    axis 0 = long heat (pools below price; long liquidation prints have order
    side SELL), 1 = short heat — see module constants LONG/SHORT. Default
    False keeps the glued (n_bars, n_buckets) shape.
    """
    n = len(arr.ts)
    grid = np.asarray(leverage_grid, dtype=float)
    k = len(grid)
    edges = np.asarray(bucket_edges, dtype=float)
    nb = len(edges) - 1
    decay = 0.5 ** (bar_s / decay_half_life_s)
    entries = _entry_prices(arr, entry)
    lps = slice_liq_prices(
        arr, grid, mmr=mmr, liq_fn=liq_fn,
        typical_account_usd=typical_account_usd, entry=entry,
    )

    # bucket -1 == no allocation: LiqMap drops lp <= 0 (e.g. 1x longs that can
    # never liquidate above zero); off-span lps are dropped too (N5) — the old
    # clamp into the edge buckets made them fictitiously hot
    bucket_of = np.full((n, k, 2), -1, dtype=int)
    for t in range(n):
        for j in range(k):
            for si in (LONG, SHORT):
                lp = lps[t, j, si]
                if np.isfinite(lp):
                    b = int(np.searchsorted(edges, lp, side="right") - 1)
                    if 0 <= b < nb:
                        bucket_of[t, j, si] = b

    blur_of: dict[tuple[int, int, int], tuple[np.ndarray, np.ndarray]] | None = None
    if blur_sigma0_bps is not None:
        size = float(edges[1] - edges[0])
        centers = (edges[:-1] + edges[1:]) / 2.0
        blur_of = {}
        for t in range(n):
            price = entries[t]
            for j in range(k):
                for si in (LONG, SHORT):
                    b0 = bucket_of[t, j, si]
                    if b0 < 0:
                        continue
                    lp = lps[t, j, si]
                    sigma_b = _kernel_sigma_buckets(price, lp, size, blur_sigma0_bps, blur_sigma1)
                    cells, weights = [], []
                    if sigma_b > 0.0:
                        half = min(int(math.ceil(3.0 * sigma_b)), 20)  # cap: 41 buckets
                        for off in range(-half, half + 1):
                            b = b0 + off
                            if not 0 <= b < nb:
                                continue  # off-grid kernel cell: drop (N5)
                            center = centers[b]
                            # side trim: no long heat at/above price, no short
                            # heat at/below (same rule as LiqMap._blur_cells)
                            if si == LONG:
                                if center >= price:
                                    continue
                            elif center <= price:
                                continue
                            cells.append(b)
                            weights.append(math.exp(-0.5 * (off / sigma_b) ** 2))
                    total = math.fsum(weights)
                    if total > 0.0:
                        blur_of[(t, j, si)] = (
                            np.asarray(cells, dtype=int),
                            np.asarray(weights) / total,
                        )
                    else:  # fully cut or degenerate: point mass at the lp bucket
                        blur_of[(t, j, si)] = (
                            np.array([b0], dtype=int),
                            np.array([1.0]),
                        )

    cross_lo = np.empty(n, dtype=int)
    cross_hi = np.empty(n, dtype=int)
    for t in range(n):
        cross_lo[t] = max(int(np.searchsorted(edges, arr.low[t], side="right") - 1), 0)
        cross_hi[t] = min(int(np.searchsorted(edges, arr.high[t], side="right") - 1), nb - 1)
    if fractional_edge_consume:
        # traversed share of the two edge buckets (mirrors LiqMap._consume_amount;
        # heat assumed uniform within a bucket)
        w_lo = edges[cross_lo + 1] - edges[cross_lo]
        w_hi = edges[cross_hi + 1] - edges[cross_hi]
        frac_lo = np.clip(
            (np.minimum(arr.high, edges[cross_lo + 1]) - arr.low) / np.maximum(w_lo, 1e-300),
            0.0,
            1.0,
        )
        frac_hi = np.clip(
            (arr.high - np.maximum(arr.low, edges[cross_hi])) / np.maximum(w_hi, 1e-300),
            0.0,
            1.0,
        )

    def build(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        wm = np.tile(w, (n, 1)) if w.ndim == 1 else w
        if wm.shape != (n, k):
            raise ValueError(f"weights must be ({k},) or ({n}, {k})")
        wm = wm / np.maximum(wm.sum(axis=1, keepdims=True), 1e-300)
        heat = np.empty((n, 2, nb)) if split_sides else np.empty((n, nb))
        rows = np.zeros((2, nb)) if split_sides else np.zeros(nb)
        for t in range(n):
            # bar order mirrors LiqMap.step: consume path -> allocate -> decay
            if cross_hi[t] >= cross_lo[t]:
                a, b = cross_lo[t], cross_hi[t]
                if not fractional_edge_consume:
                    if split_sides:
                        rows[:, a : b + 1] = 0.0
                    else:
                        rows[a : b + 1] = 0.0
                elif a == b:
                    # single-bucket path: both edge formulas reduce to (high-low)/w
                    if split_sides:
                        rows[:, a] *= 1.0 - frac_lo[t]
                    else:
                        rows[a] *= 1.0 - frac_lo[t]
                else:
                    if split_sides:
                        rows[:, a] *= 1.0 - frac_lo[t]
                        rows[:, b] *= 1.0 - frac_hi[t]
                        if b > a + 1:
                            rows[:, a + 1 : b] = 0.0
                    else:
                        rows[a] *= 1.0 - frac_lo[t]
                        rows[b] *= 1.0 - frac_hi[t]
                        if b > a + 1:
                            rows[a + 1 : b] = 0.0
            usd = arr.d_oi_usd[t]
            if usd > 0.0:
                shares = (arr.long_share[t], 1.0 - arr.long_share[t])
                for j in range(k):
                    contrib = usd * wm[t, j]
                    for si in (LONG, SHORT):
                        target = rows[si] if split_sides else rows
                        if blur_of is not None:
                            cw = blur_of.get((t, j, si))
                            if cw is not None:
                                target[cw[0]] += contrib * shares[si] * cw[1]
                        else:
                            b = bucket_of[t, j, si]
                            if b >= 0:
                                target[b] += contrib * shares[si]
            elif usd < 0.0:
                total = rows.sum()
                if total > 0.0:
                    # proportional close-out, scaled by close_out_fraction
                    # (mirrors LiqMap: not every closed position was carrying
                    # heat, so charging the full ΔOI⁻ drains untouched levels)
                    rows *= max(0.0, 1.0 + usd * close_out_fraction / total)
            rows *= decay
            heat[t] = rows
        return heat

    return build
