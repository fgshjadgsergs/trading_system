"""M4 report figures: candles with heat overlay, vertical heat slice."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from trading_system.core.schema import records_to_frame
from trading_system.core.synth import synth_open_interest, synth_trades
from trading_system.features.bars import time_bars
from trading_system.features.indicators import with_atr
from trading_system.features.joins import join_open_interest
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.history import HeatHistory
from trading_system.liqmap.map import LiqMap, StaticWeights
from trading_system.viz.style import PALETTE, apply_style, save_fig


def build_demo_map(seed: int = 42) -> tuple[pl.DataFrame, LiqMap, HeatHistory]:
    """Bars + a liq map driven bar-by-bar over synthetic data."""
    trades = records_to_frame(synth_trades(n=30_000, mean_gap_ms=200.0, seed=seed), "trade")
    start = int(trades["ts_event"].min())
    oi = records_to_frame(
        synth_open_interest(n=2_000, step_s=7, start_ts=start, seed=seed), "open_interest"
    )
    bars = with_atr(join_open_interest(time_bars(trades, "1m"), oi), period=14)
    atr = float(bars["atr"].drop_nulls().median())
    grid = [3, 5, 10, 20, 25, 50, 75, 100, 125]
    lm = LiqMap(
        leverage_grid=grid,
        buckets=PriceBuckets.from_atr(atr, 0.1),
        weight_fn=StaticWeights(np.array([1, 2, 4, 6, 5, 4, 2, 2, 1], dtype=float)),
        decay_half_life_s=86_400.0,
    )
    hist = HeatHistory(lm)
    for row in bars.iter_rows(named=True):
        d_oi = row["d_oi_usd"]
        if d_oi is None:  # warm-up bars before the first OI point
            d_oi = row["quote_volume"] * 0.05
        lm.step(
            bar_low=row["low"],
            bar_high=row["high"],
            bar_close=row["close"],
            d_oi_usd=d_oi,  # signed: negative ΔOI removes heat proportionally
            dt_s=(row["ts_close"] - row["ts_open"]) / 1e9,
        )
        hist.record(row["ts_close"])
    return bars, lm, hist


def heat_overlay(
    bars: pl.DataFrame,
    hist: HeatHistory,
    name: str = "m4_heat_overlay",
    out_dir: Path | None = None,
) -> Path:
    """Candles with the H(time x price) heatmap painted behind them."""
    apply_style()
    ts, prices, H = hist.matrix()
    fig, ax = plt.subplots(figsize=(14, 8))
    if H.size:
        ax.imshow(
            np.log1p(H),
            aspect="auto",
            origin="lower",
            extent=(0, len(ts), float(prices[0]), float(prices[-1])),
            cmap=PALETTE["heat"],
            alpha=0.75,
        )
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    x = np.arange(len(o))
    up = c >= o
    ax.vlines(x + 0.5, lo, h, color="#cfd8dc", lw=0.7, zorder=3)
    ax.bar(
        x + 0.5,
        np.abs(c - o),
        bottom=np.minimum(o, c),
        width=0.8,
        color=np.where(up, PALETTE["long"], PALETTE["short"]),
        zorder=4,
    )
    ax.set_xlim(0, len(x))
    ax.set_title("Candles + liquidation heat H (log scale)")
    ax.set_xlabel("bar #")
    ax.set_ylabel("price")
    return save_fig(fig, name, out_dir)


def heat_slice(
    lm: LiqMap, name: str = "m4_heat_slice", out_dir: Path | None = None
) -> Path:
    """Vertical slice of H at the current moment, long vs short."""
    apply_style()
    snap = lm.snapshot()
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(snap["prices"], snap["long"], color=PALETTE["long"], label="long pools", height=lm.buckets.bucket_size * 0.9)
    ax.barh(snap["prices"], -snap["short"], color=PALETTE["short"], label="short pools", height=lm.buckets.bucket_size * 0.9)
    ax.axvline(0, color=PALETTE["neutral"], lw=1)
    ax.set_title("H slice at t (USD per bucket)")
    ax.set_xlabel("heat, USD (short ← → long)")
    ax.set_ylabel("price")
    ax.legend()
    return save_fig(fig, name, out_dir)


def demo_reports(out_dir: Path, seed: int = 42) -> list[Path]:
    bars, lm, hist = build_demo_map(seed=seed)
    return [heat_overlay(bars, hist, out_dir=out_dir), heat_slice(lm, out_dir=out_dir)]
