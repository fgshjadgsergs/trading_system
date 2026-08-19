"""M2 demo figures: one-hour book depth heatmap; spread and +/-0.5% depth lines.

All figures are generated offline from seeded synthetic data and saved as png
via trading_system.viz.style.save_fig.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from trading_system.book.replay import BookReplayer
from trading_system.book.synth_local import mean_reverting_book_stream
from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S
from trading_system.viz.style import PALETTE, apply_style, save_fig


def _grid_to_matrix(
    grid: pl.DataFrame, tick: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Long (ts, side, price, qty) grid -> (matrix[price, time], ts axis, price axis).

    Matrix cells hold log1p(qty); prices are quantized to `tick`.
    """
    ts_vals = grid["ts"].unique().sort().to_numpy()
    p_min = float(grid["price"].min())
    p_max = float(grid["price"].max())
    n_prices = int(round((p_max - p_min) / tick)) + 1
    prices = p_min + tick * np.arange(n_prices)
    mat = np.zeros((n_prices, len(ts_vals)), dtype=np.float64)
    t_index = {int(t): i for i, t in enumerate(ts_vals)}
    p_arr = grid["price"].to_numpy()
    q_arr = grid["qty"].to_numpy()
    t_arr = grid["ts"].to_numpy()
    rows = np.rint((p_arr - p_min) / tick).astype(np.int64)
    cols = np.array([t_index[int(t)] for t in t_arr], dtype=np.int64)
    mat[rows, cols] = np.log1p(q_arr)
    return mat, ts_vals, prices


def book_heatmap(
    grid: pl.DataFrame, tick: float, name: str, out_dir: Path | None = None
) -> Path:
    """Time x price heatmap of book depth, brightness log(1+qty), via imshow."""
    apply_style()
    mat, ts_vals, prices = _grid_to_matrix(grid, tick)
    t0 = int(ts_vals[0])
    minutes = (ts_vals - t0) / NS_PER_MIN
    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(
        mat,
        origin="lower",
        aspect="auto",
        cmap=PALETTE["heat"],
        extent=(float(minutes[0]), float(minutes[-1]), float(prices[0]), float(prices[-1])),
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="log(1 + depth qty)")
    ax.set_xlabel("minutes from start")
    ax.set_ylabel("price")
    ax.set_title("M2: order book depth heatmap, one hour (top-of-book band)")
    ax.grid(False)
    return save_fig(fig, name, out_dir)


def spread_depth_figure(
    metrics: pl.DataFrame, depth_pct: float, name: str, out_dir: Path | None = None
) -> Path:
    """Spread over time (top) and bid/ask depth within +/-depth_pct (bottom)."""
    apply_style()
    t0 = int(metrics["ts"].min())
    minutes = ((metrics["ts"] - t0) / NS_PER_MIN).to_numpy()
    spread = metrics["spread"].to_numpy()
    bid_depth = metrics["bid_depth"].to_numpy()
    ask_depth = metrics["ask_depth"].to_numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    sns.lineplot(x=minutes, y=spread, ax=ax1, color=PALETTE["neutral"])
    ax1.set_ylabel("spread")
    ax1.set_title("M2: spread over time")
    sns.lineplot(x=minutes, y=bid_depth, ax=ax2, color=PALETTE["long"], label="bid depth")
    sns.lineplot(x=minutes, y=ask_depth, ax=ax2, color=PALETTE["short"], label="ask depth")
    ax2.set_ylabel(f"qty within ±{depth_pct:.2%} of mid")
    ax2.set_xlabel("minutes from start")
    ax2.set_title(f"M2: depth within ±{depth_pct:.2%} of mid")
    ax2.legend()
    fig.tight_layout()
    return save_fig(fig, name, out_dir)


def demo_reports(out_dir: Path, seed: int = 42) -> list[Path]:
    """Generate all M2 checklist figures from synthetic data; returns png paths."""
    tick = 0.5
    # ~1 hour of diffs at ~100ms cadence
    snapshot, diffs = mean_reverting_book_stream(n_diffs=36_000, seed=seed, tick=tick, n_levels=50)
    replayer = BookReplayer(snapshot, diffs)
    grid = replayer.sample_grid(interval_ns=10 * NS_PER_S, n=50)
    metrics = replayer.sample_metrics(interval_ns=5 * NS_PER_S, depth_pct=0.005)
    return [
        book_heatmap(grid, tick=tick, name="m2_book_heatmap_1h", out_dir=out_dir),
        spread_depth_figure(
            metrics, depth_pct=0.005, name="m2_spread_depth_time", out_dir=out_dir
        ),
    ]
