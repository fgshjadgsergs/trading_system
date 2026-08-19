"""M7 report figure: signal triggers over price with the pools they aim at."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from trading_system.core.schema import records_to_frame
from trading_system.core.synth import synth_trades
from trading_system.features.bars import time_bars
from trading_system.features.indicators import with_atr
from trading_system.profile.swings import equal_extremes, fractal_swings
from trading_system.signals.detectors import s1_magnet, s2_sweep_reversal, s3_filter
from trading_system.viz.style import PALETTE, apply_style, save_fig


def signals_chart(
    bars: pl.DataFrame,
    events: pl.DataFrame,
    pools: pl.DataFrame,
    name: str = "m7_signals",
    out_dir: Path | None = None,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    x = np.arange(len(o))
    ax.vlines(x + 0.5, lo, h, color="#90a4ae", lw=0.7)
    ax.bar(
        x + 0.5,
        np.abs(c - o) + 1e-9,
        bottom=np.minimum(o, c),
        width=0.8,
        color=np.where(c >= o, PALETTE["long"], PALETTE["short"]),
    )
    for p in pools.iter_rows(named=True):
        ax.axhline(p["price"], color=PALETTE["accent"], lw=1, alpha=0.6)
    idx_of_ts = {int(t): i for i, t in enumerate(bars["ts_close"].to_list())}
    markers = {"s1": ("o", PALETTE["accent"]), "s2": ("D", "#6a1b9a")}
    for ev in events.iter_rows(named=True):
        xi = idx_of_ts.get(ev["ts"])
        if xi is None:
            continue
        m, color = markers[ev["signal"]]
        face = "none" if ev.get("blocked") else color
        ax.scatter(xi + 0.5, ev["price"], marker=m, s=90, facecolors=face, edgecolors=color, zorder=6)
        ax.annotate(
            ("↑" if ev["side"] > 0 else "↓") + ev["signal"],
            (xi + 0.5, ev["price"]),
            textcoords="offset points",
            xytext=(4, 8),
            fontsize=8,
            color=color,
        )
    ax.set_title("Signal triggers over price (hollow = vetoed by S3); pool levels dashed")
    return save_fig(fig, name, out_dir)


def demo_reports(out_dir: Path, seed: int = 42) -> list[Path]:
    trades = records_to_frame(synth_trades(n=30_000, mean_gap_ms=200.0, seed=seed), "trade")
    bars = with_atr(time_bars(trades, "5m"), period=14)
    span = float(bars["high"].max() - bars["low"].min())
    sw = fractal_swings(bars, n=2)
    clusters = equal_extremes(sw, eps=span / 100)
    pools = pl.DataFrame(
        {
            "price": [float(bars["low"].min()) + span * f for f in (0.15, 0.5, 0.85)],
            "heat_usd": [4e6, 1e6, 3e6],
            "touched_ts": pl.Series([None, None, None], dtype=pl.Int64),
        }
    )
    ev1 = s1_magnet(bars, pools, k_atr=3.0)
    ev2 = (
        s2_sweep_reversal(bars, clusters, return_bars=3)
        if clusters.height
        else ev1.head(0)
    )
    events = pl.concat([ev1, ev2]) if ev2.height else ev1
    zones = pl.DataFrame(
        {
            "lo": [float(bars["low"].min()) + span * 0.45],
            "hi": [float(bars["low"].min()) + span * 0.55],
            "heat_usd": [5e6],
        }
    )
    events = s3_filter(events, zones, dense_quantile=0.5)
    return [signals_chart(bars, events, pools, out_dir=out_dir)]
