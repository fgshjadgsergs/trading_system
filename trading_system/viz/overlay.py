"""Combined overlay chart: candles + liquidation heat + signal markers + profile.

mplfinance/plotly draw candlesticks; analytics stays in seaborn. This module
uses matplotlib primitives directly so heat, profile and markers can share one
price axis with the candles.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from trading_system.viz.style import PALETTE, apply_style, save_fig


def draw_candles(ax: plt.Axes, bars: pl.DataFrame) -> None:
    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    x = np.arange(len(o))
    ax.vlines(x + 0.5, lo, h, color="#90a4ae", lw=0.7, zorder=3)
    ax.bar(
        x + 0.5,
        np.abs(c - o) + 1e-9,
        bottom=np.minimum(o, c),
        width=0.8,
        color=np.where(c >= o, PALETTE["long"], PALETTE["short"]),
        zorder=4,
    )
    ax.set_xlim(0, len(x))


def overlay_chart(
    bars: pl.DataFrame,
    heat: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    events: pl.DataFrame | None = None,
    profile: pl.DataFrame | None = None,
    levels: pl.DataFrame | None = None,
    name: str = "overlay",
    out_dir: Path | None = None,
    title: str = "Candles + liquidation map + signals",
) -> Path:
    """One picture per checklist M10: свечи + карта + сигналы (+ профиль сбоку).

    heat: (ts, prices, H[m, n]) from liqmap.HeatHistory.matrix(); events: the
    M7 event frame; profile: M5 profile frame; levels: (price, ...) lines.
    """
    apply_style()
    if profile is not None and profile.height:
        fig, (ax, ax_p) = plt.subplots(
            1, 2, figsize=(15, 8), sharey=True, gridspec_kw={"width_ratios": [5, 1]}
        )
    else:
        fig, ax = plt.subplots(figsize=(14, 8))
        ax_p = None
    n_bars = bars.height
    if heat is not None:
        ts, prices, H = heat
        if H.size:
            ax.imshow(
                np.log1p(H),
                aspect="auto",
                origin="lower",
                extent=(0, min(n_bars, H.shape[1]), float(prices[0]), float(prices[-1])),
                cmap=PALETTE["heat"],
                alpha=0.7,
                zorder=1,
            )
    draw_candles(ax, bars)
    if levels is not None:
        for row in levels.iter_rows(named=True):
            ax.axhline(row["price"], color=PALETTE["accent"], lw=1, alpha=0.7, zorder=2)
    if events is not None and events.height:
        idx_of_ts = {int(t): i for i, t in enumerate(bars["ts_close"].to_list())}
        style = {"s1": ("o", PALETTE["accent"]), "s2": ("D", "#6a1b9a")}
        for ev in events.iter_rows(named=True):
            xi = idx_of_ts.get(ev["ts"])
            if xi is None:
                continue
            marker, color = style.get(ev["signal"], ("x", PALETTE["neutral"]))
            blocked = bool(ev.get("blocked", False))
            ax.scatter(
                xi + 0.5,
                ev["price"],
                marker=marker,
                s=90,
                facecolors="none" if blocked else color,
                edgecolors=color,
                zorder=6,
            )
    if ax_p is not None:
        step = profile["price"].diff().median() or 1.0
        ax_p.barh(profile["price"], profile["volume_usd"], color=PALETTE["neutral"], height=step)
        ax_p.set_title("volume profile")
    ax.set_title(title)
    ax.set_xlabel("bar #")
    ax.set_ylabel("price")
    return save_fig(fig, name, out_dir)
