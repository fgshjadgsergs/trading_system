"""Terminal-style liquidation map render: dark ground, teal→yellow bands,
dollar labels on the strongest standing pools — the look of liquidity-map
trading terminals, drawn from the same causal HeatHistory the model trades on.

The right section of the panel projects the FINAL snapshot forward: standing
pools extend as bands past the last candle, exactly how terminals show "what
is still out there". The projection is presentation only — no model state.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.colors import LinearSegmentedColormap

from trading_system.liqmap.history import HeatHistory
from trading_system.viz.style import save_fig

GROUND = "#0d1117"
PANEL = "#0d1117"
GRID = "#1d2733"
TEXT = "#9fb0bd"
UP = "#26a69a"
DOWN = "#ef5350"
LABEL_HOT = "#ffe95c"

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "liq_terminal",
    [
        (0.00, "#0d1117"),
        (0.25, "#0e3438"),
        (0.55, "#0f7f74"),
        (0.80, "#2fd6b0"),
        (1.00, "#ffee58"),
    ],
)


def _fmt_usd(v: float) -> str:
    for div, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= div:
            return f"${v / div:.1f}{suffix}".replace(".0", "")
    return f"${v:.0f}"


def terminal_heat_overlay(
    bars: pl.DataFrame,
    hist: HeatHistory,
    name: str = "terminal_heat",
    out_dir: Path | None = None,
    title: str = "",
    project_frac: float = 0.18,
    top_labels: int = 6,
    y_pad_frac: float = 0.10,
    q_lo: float = 0.55,
    q_hi: float = 0.995,
) -> Path:
    """Candles over banded liquidation heat, terminal-style, saved as png.

    y range = the traded range padded by `y_pad_frac` of the last close, so
    near-leverage bands frame the price; the deepest low-leverage pools fall
    off-screen by design (as in the reference terminals).

    Colour maps log1p(heat) between the `q_lo` and `q_hi` quantiles of the
    non-zero heat, not between 0 and the max: on dense grids (a minute chart
    holds thousands of small pools) a full-range log scale lifts every weak
    pool into the bright half and the panel turns into a solid wash. Pass
    q_lo=0.0, q_hi=1.0 for the plain full-range scale.
    """
    ts, prices, H = hist.matrix()
    n_bars = bars.height
    n_cols = H.shape[1] if H.size else n_bars
    n_proj = max(1, int(n_cols * project_frac))
    if H.size:
        H_ext = np.hstack([H, np.tile(H[:, -1:], (1, n_proj))])
    else:  # пустая карта: рисуем один пустой ряд с невырожденным extent
        H_ext = np.zeros((1, n_cols + n_proj))
        c0 = float(bars["close"][0])
        prices = np.array([c0 * 0.999, c0 * 1.001])

    last_close = float(bars["close"][-1])
    pad = last_close * y_pad_frac
    y_lo = float(bars["low"].min()) - pad
    y_hi = float(bars["high"].max()) + pad

    fig, ax = plt.subplots(figsize=(15, 8))
    fig.patch.set_facecolor(GROUND)
    ax.set_facecolor(PANEL)
    shown = np.log1p(H_ext)
    nz = shown[shown > 0.0]
    vmin, vmax = (0.0, None)
    if nz.size and (q_lo > 0.0 or q_hi < 1.0):
        vmin = float(np.quantile(nz, q_lo))
        vmax = float(np.quantile(nz, q_hi))
        if not vmax > vmin:  # вырожденный случай (все пулы одинаковы)
            vmin, vmax = 0.0, None
    ax.imshow(
        shown,
        aspect="auto",
        origin="lower",
        extent=(0, n_cols + n_proj, float(prices[0]), float(prices[-1])),
        cmap=HEAT_CMAP,
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )

    o = bars["open"].to_numpy()
    h = bars["high"].to_numpy()
    lo = bars["low"].to_numpy()
    c = bars["close"].to_numpy()
    x = np.arange(n_bars)
    up = c >= o
    ax.vlines(x + 0.5, lo, h, color=np.where(up, UP, DOWN), lw=0.8, zorder=3)
    ax.bar(
        x + 0.5,
        np.abs(c - o) + last_close * 1e-5,
        bottom=np.minimum(o, c),
        width=0.75,
        color=np.where(up, UP, DOWN),
        zorder=4,
    )
    ax.axhline(last_close, color="#c5d0d9", lw=0.7, ls=(0, (4, 3)), alpha=0.7, zorder=5)
    ax.annotate(
        f"{last_close:,.2f}".replace(",", " "),
        (n_cols + n_proj, last_close),
        xytext=(-4, 3),
        textcoords="offset points",
        ha="right",
        color="#0d1117",
        fontsize=8,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.25", "fc": "#ef5350" if not up[-1] else UP, "ec": "none"},
        zorder=7,
    )

    # dollar labels on the strongest STANDING pools (final snapshot)
    if len(hist):
        labeled = 0
        taken_prices: list[float] = []
        min_gap = (y_hi - y_lo) * 0.035
        for price, heat in hist.pools_at(len(hist) - 1, k=top_labels * 4):
            if not y_lo <= price <= y_hi or labeled >= top_labels:
                continue
            if any(abs(price - p) < min_gap for p in taken_prices) or abs(
                price - last_close
            ) < min_gap:
                continue
            taken_prices.append(price)
            ax.annotate(
                _fmt_usd(heat),
                (n_cols + n_proj, price),
                xytext=(-4, 2),
                textcoords="offset points",
                ha="right",
                color=LABEL_HOT if labeled < 2 else TEXT,
                fontsize=8,
                fontweight="bold" if labeled < 2 else "normal",
                zorder=7,
            )
            labeled += 1

    ax.set_xlim(0, n_cols + n_proj)
    ax.set_ylim(y_lo, y_hi)
    ax.yaxis.tick_right()
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, lw=0.4, alpha=0.5)
    n_ticks = 6
    tick_idx = np.linspace(0, n_bars - 1, n_ticks).astype(int)
    ax.set_xticks(tick_idx + 0.5)
    ax.set_xticklabels(
        [
            datetime.fromtimestamp(int(bars["ts_open"][int(i)]) / 1e9, tz=UTC).strftime(
                "%m-%d %H:%M"
            )
            for i in tick_idx
        ],
        color=TEXT,
        fontsize=8,
    )
    ax.set_title(
        title or "Liquidation map · log heat (teal → yellow = hotter)",
        color=TEXT,
        fontsize=10,
        loc="left",
    )
    return save_fig(fig, name, out_dir)
