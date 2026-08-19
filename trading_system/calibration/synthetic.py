"""Synthetic ground truth for calibration tests and demo reports.

Simulates position cohorts opened bar-by-bar with a KNOWN leverage mixture,
computes their liquidation prices via trading_system.core.liquidation.liq_price
and emits liquidation events when the price path crosses them. A matching
causal heat-builder callback (with consume-on-cross and exponential decay)
turns any candidate weight vector into heat matrices, so calibrators can be
scored against the known truth. Everything is seeded and offline.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from trading_system.core.liquidation import liq_price
from trading_system.core.schema import POLARS_SCHEMAS, Side
from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S

EXCHANGE = "binance_usdm"
BAR_NS = NS_PER_MIN


@dataclass
class SyntheticWorld:
    """Bars, cohorts' known mixture, resulting liquidations and bucket grid."""

    symbol: str
    ts: np.ndarray  # (n,) bar-close timestamps, UTC ns
    prices: np.ndarray  # (n,) close prices
    entry_notional: np.ndarray  # (n,) USD of positions opened per bar
    leverage_grid: np.ndarray  # (K,)
    true_weights: np.ndarray  # (n, K) mixture actually used per bar
    context: np.ndarray  # (n, d) context features
    bucket_edges: np.ndarray  # (n_buckets + 1,)
    liquidations: pl.DataFrame  # core "liquidation" schema
    liq_leverage_idx: np.ndarray  # (n_liqs,) true leverage rung per event
    atr: float
    mmr: float
    long_share: float


def ou_log_prices(
    n: int, s0: float, sigma: float = 0.01, phi: float = 0.995, seed: int = 42
) -> np.ndarray:
    """Mean-reverting log-price path: wide swings sweep liquidation bands."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[0] = 0.0
    eps = rng.normal(0.0, sigma, n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + eps[t]
    return s0 * np.exp(x)


def regime_context(
    n: int, period: int, n_noise: int = 2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """(context matrix, regime id array): block regimes + seeded noise features."""
    rng = np.random.default_rng(seed)
    regime = (np.arange(n) // period) % 2
    noise = rng.normal(0.0, 1.0, size=(n, n_noise))
    ctx = np.hstack([regime[:, None].astype(float), noise])
    return ctx, regime


def _pos_side(sd: int) -> Side:
    return Side.BUY if sd == 1 else Side.SELL


def make_world(
    n_bars: int = 1600,
    s0: float = 50_000.0,
    sigma: float = 0.01,
    phi: float = 0.995,
    seed: int = 42,
    symbol: str = "BTCUSDT",
    leverage_grid: tuple[float, ...] = (10.0, 20.0, 50.0),
    static_weights: tuple[float, ...] = (0.25, 0.45, 0.30),
    regime_period: int | None = None,
    regime_weights: tuple[tuple[float, ...], tuple[float, ...]] | None = None,
    notional_per_bar: float = 1_000_000.0,
    mmr: float = 0.005,
    long_share: float = 0.5,
    bucket_width_frac: float = 0.004,
    start_ts: int = 1_755_600_000 * NS_PER_S,
) -> SyntheticWorld:
    """Simulate cohorts with a known leverage mixture and emit liquidations.

    With `regime_period` set, the mixture alternates between the two
    `regime_weights` rows in blocks and the first context column carries the
    regime id; otherwise the mixture is `static_weights` everywhere and the
    context features are pure noise (context-free world).
    """
    grid = np.asarray(leverage_grid, dtype=float)
    k = len(grid)
    prices = ou_log_prices(n_bars, s0, sigma=sigma, phi=phi, seed=seed)
    ts = start_ts + (np.arange(n_bars, dtype=np.int64) + 1) * BAR_NS

    if regime_period is not None:
        if regime_weights is None:
            raise ValueError("regime_weights required when regime_period is set")
        ctx, regime = regime_context(n_bars, regime_period, seed=seed + 1)
        wa = np.asarray(regime_weights[0], float)
        wb = np.asarray(regime_weights[1], float)
        true_w = np.where(regime[:, None] == 0, wa[None, :], wb[None, :])
    else:
        ctx, _ = regime_context(n_bars, max(n_bars, 2), seed=seed + 1)
        ctx = ctx[:, 1:]  # drop the constant regime column: context-free world
        true_w = np.tile(np.asarray(static_weights, float), (n_bars, 1))
    true_w = true_w / true_w.sum(axis=1, keepdims=True)

    rng = np.random.default_rng(seed + 2)
    entry_notional = notional_per_bar * rng.lognormal(0.0, 0.3, n_bars)

    # one cohort per (bar, leverage rung, position side); flat preallocation
    n_cohorts = n_bars * k * 2
    c_liq = np.empty(n_cohorts)
    c_usd = np.empty(n_cohorts)
    c_side = np.empty(n_cohorts, dtype=int)  # +1 long, -1 short
    c_lev = np.empty(n_cohorts, dtype=int)
    for t in range(n_bars):
        for j in range(k):
            for si, sd in enumerate((1, -1)):
                i = (t * k + j) * 2 + si
                c_liq[i] = liq_price(prices[t], grid[j], _pos_side(sd), mmr)
                c_usd[i] = entry_notional[t] * true_w[t, j] * (
                    long_share if sd == 1 else 1.0 - long_share
                )
                c_side[i] = sd
                c_lev[i] = j

    lo = min(prices.min(), c_liq.min()) * 0.999
    hi = max(prices.max(), c_liq.max()) * 1.001
    width = s0 * bucket_width_frac
    n_buckets = int(np.ceil((hi - lo) / width))
    bucket_edges = lo + width * np.arange(n_buckets + 1)

    alive = np.zeros(n_cohorts, dtype=bool)
    rows: list[dict] = []
    lev_of_liq: list[int] = []
    for t in range(n_bars):
        if t > 0:
            p_lo = min(prices[t - 1], prices[t])
            p_hi = max(prices[t - 1], prices[t])
            hit = alive & (c_liq >= p_lo) & (c_liq <= p_hi)
            if hit.any():
                ev_ts = int(ts[t] - BAR_NS // 2)
                for i in np.flatnonzero(hit):
                    rows.append(
                        {
                            "exchange": EXCHANGE,
                            "symbol": symbol,
                            "ts_event": ev_ts,
                            "ts_recv": ev_ts + 5_000_000,
                            "price": float(c_liq[i]),
                            "qty": float(c_usd[i] / c_liq[i]),
                            "qty_usd": float(c_usd[i]),
                            # order side closing the position: SELL(-1) for longs
                            "side": -int(c_side[i]),
                        }
                    )
                    lev_of_liq.append(int(c_lev[i]))
                alive[hit] = False
        alive[t * k * 2 : (t + 1) * k * 2] = True
    liqs = pl.DataFrame(rows, schema=POLARS_SCHEMAS["liquidation"], orient="row")

    ret = np.abs(np.diff(np.log(prices), prepend=np.log(prices[0])))
    atr = float(np.median(ret) * s0) or width
    return SyntheticWorld(
        symbol=symbol,
        ts=ts,
        prices=prices,
        entry_notional=entry_notional,
        leverage_grid=grid,
        true_weights=true_w,
        context=ctx,
        bucket_edges=bucket_edges,
        liquidations=liqs,
        liq_leverage_idx=np.asarray(lev_of_liq, dtype=int),
        atr=atr,
        mmr=mmr,
        long_share=long_share,
    )


def realized_mixture(world: SyntheticWorld) -> np.ndarray:
    """USD share of emitted liquidations per leverage rung (exact bookkeeping)."""
    if world.liquidations.is_empty():
        return np.full(len(world.leverage_grid), np.nan)
    usd = world.liquidations["qty_usd"].to_numpy()
    out = np.bincount(world.liq_leverage_idx, weights=usd, minlength=len(world.leverage_grid))
    return out / out.sum()


def make_heat_builder(
    world: SyntheticWorld, decay_half_life_bars: float = 200.0
) -> Callable[[np.ndarray], np.ndarray]:
    """Causal heat builder: build_heat(w) -> (n_bars, n_buckets) matrices.

    Mass w[k] * notional goes to the bucket of liq_price(entry_t, L_k, side)
    when a cohort opens; buckets swept by the price path are consumed (zeroed,
    mirroring the simulator's crossing rule) and everything decays
    exponentially. Row t therefore reflects the surviving cohort mass under
    mixture w. Accepts a (K,) global vector or an (n_bars, K) per-bar matrix;
    rows only ever use information up to bar t (causal).
    """
    n = len(world.ts)
    k = len(world.leverage_grid)
    edges = world.bucket_edges
    nb = len(edges) - 1
    decay = 0.5 ** (1.0 / decay_half_life_bars)

    bucket_of = np.empty((n, k, 2), dtype=int)
    for t in range(n):
        for j in range(k):
            for si, sd in enumerate((1, -1)):
                lp = liq_price(world.prices[t], world.leverage_grid[j], _pos_side(sd), world.mmr)
                b = int(np.searchsorted(edges, lp, side="right") - 1)
                bucket_of[t, j, si] = min(max(b, 0), nb - 1)
    # buckets whose interval intersects [p_lo, p_hi] are consumed at bar t —
    # the same rule that liquidates cohorts in the simulator
    cross_lo = np.ones(n, dtype=int)
    cross_hi = np.zeros(n, dtype=int)
    for t in range(1, n):
        p_lo = min(world.prices[t - 1], world.prices[t])
        p_hi = max(world.prices[t - 1], world.prices[t])
        cross_lo[t] = max(int(np.searchsorted(edges, p_lo, side="right") - 1), 0)
        cross_hi[t] = min(int(np.searchsorted(edges, p_hi, side="right") - 1), nb - 1)
    shares = np.array([world.long_share, 1.0 - world.long_share])

    def build(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        wm = np.tile(w, (n, 1)) if w.ndim == 1 else w
        if wm.shape != (n, k):
            raise ValueError(f"weights must be ({k},) or ({n}, {k})")
        wm = wm / np.maximum(wm.sum(axis=1, keepdims=True), 1e-300)
        heat = np.empty((n, nb))
        row = np.zeros(nb)
        for t in range(n):
            row *= decay
            if cross_hi[t] >= cross_lo[t]:
                row[cross_lo[t] : cross_hi[t] + 1] = 0.0
            usd = world.entry_notional[t]
            for j in range(k):
                for si in (0, 1):
                    row[bucket_of[t, j, si]] += usd * wm[t, j] * shares[si]
            heat[t] = row
        return heat

    return build


def kl_divergence(p: np.ndarray, q: np.ndarray, floor: float = 1e-9) -> float:
    """KL(p || q) with flooring so exact zeros stay finite."""
    p = np.maximum(np.asarray(p, float), floor)
    q = np.maximum(np.asarray(q, float), floor)
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum(p * np.log(p / q)))
