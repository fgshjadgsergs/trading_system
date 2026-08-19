"""M5 report figures: profile beside candles, marked swings, level-weight distribution."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from trading_system.core.schema import records_to_frame
from trading_system.core.synth import synth_trades
from trading_system.features.bars import time_bars
from trading_system.profile.swings import equal_extremes, fractal_swings, level_weights
from trading_system.profile.volume_profile import hvn_lvn, profile, value_area
from trading_system.viz.style import PALETTE, apply_style, save_fig


def _candles(ax, bars: pl.DataFrame) -> None:
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    x = np.arange(len(o))
    up = c >= o
    ax.vlines(x + 0.5, lo, h, color="#90a4ae", lw=0.7)
    ax.bar(
        x + 0.5,
        np.abs(c - o) + 1e-9,
        bottom=np.minimum(o, c),
        width=0.8,
        color=np.where(up, PALETTE["long"], PALETTE["short"]),
    )
    ax.set_xlim(0, len(x))


def profile_beside_candles(
    bars: pl.DataFrame,
    prof: pl.DataFrame,
    name: str = "m5_profile_candles",
    out_dir: Path | None = None,
) -> Path:
    apply_style()
    fig, (ax_c, ax_p) = plt.subplots(
        1, 2, figsize=(14, 8), sharey=True, gridspec_kw={"width_ratios": [4, 1]}
    )
    _candles(ax_c, bars)
    va = value_area(prof)
    nodes = hvn_lvn(prof)
    ax_p.barh(
        prof["price"], prof["volume_usd"], color=PALETTE["neutral"], height=prof["price"].diff().median() or 1.0
    )
    for kind, color in (("hvn", PALETTE["long"]), ("lvn", PALETTE["short"])):
        sel = nodes.filter(pl.col("node") == kind)
        ax_p.barh(sel["price"], sel["volume_usd"], color=color, height=prof["price"].diff().median() or 1.0)
    for y, ls in ((va.poc, "-"), (va.val, "--"), (va.vah, "--")):
        ax_c.axhline(y, color=PALETTE["accent"], ls=ls, lw=1)
        ax_p.axhline(y, color=PALETTE["accent"], ls=ls, lw=1)
    ax_c.set_title("Candles + POC / Value Area")
    ax_p.set_title("Volume profile (HVN green / LVN red)")
    return save_fig(fig, name, out_dir)


def swings_chart(
    bars: pl.DataFrame,
    swings: pl.DataFrame,
    clusters: pl.DataFrame,
    name: str = "m5_swings_levels",
    out_dir: Path | None = None,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(14, 8))
    _candles(ax, bars)
    idx_of_ts = {int(t): i for i, t in enumerate(bars["ts_open"].to_list())}
    for row in swings.iter_rows(named=True):
        x = idx_of_ts.get(row["ts_open"], None)
        if x is None:
            continue
        marker, color = ("v", PALETTE["short"]) if row["kind"] == "high" else ("^", PALETTE["long"])
        ax.scatter(x + 0.5, row["price"], marker=marker, color=color, s=60, zorder=5)
    for row in clusters.iter_rows(named=True):
        ax.axhline(row["price"], color=PALETTE["accent"], lw=1.2, alpha=0.8)
    ax.set_title("Fractal swings (▲ lows / ▼ highs) + equal-extreme stop clusters")
    return save_fig(fig, name, out_dir)


def weight_distribution(
    weighted: pl.DataFrame, name: str = "m5_level_weights", out_dir: Path | None = None
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(weighted["weight"].to_numpy(), bins=20, ax=ax, color=PALETTE["neutral"])
    ax.set_title("Level weight distribution (age-decayed)")
    ax.set_xlabel("weight")
    return save_fig(fig, name, out_dir)


def demo_reports(out_dir: Path, seed: int = 42) -> list[Path]:
    trades = records_to_frame(synth_trades(n=40_000, mean_gap_ms=200.0, seed=seed), "trade")
    bars = time_bars(trades, "5m")
    price_span = float(bars["high"].max() - bars["low"].min())
    prof = profile(trades, bin_size=price_span / 60)
    sw = fractal_swings(bars, n=2)
    clusters = equal_extremes(sw, eps=price_span / 200)
    now = int(bars["ts_close"].max())
    weighted = level_weights(
        clusters if clusters.height else sw.rename({"ts_confirmed": "ts_last"}).with_columns(pl.lit(1).cast(pl.Int64).alias("count")),
        now_ts=now,
    )
    return [
        profile_beside_candles(bars, prof, out_dir=out_dir),
        swings_chart(bars, sw, clusters, out_dir=out_dir),
        weight_distribution(weighted, out_dir=out_dir),
    ]
