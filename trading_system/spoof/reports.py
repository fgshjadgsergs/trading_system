"""M6 report figures: stability-score book heatmap, lifetime distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from matplotlib import colormaps

from trading_system.core.timeutils import NS_PER_S
from trading_system.spoof.lifecycle import LevelJournal
from trading_system.spoof.metrics import large_level_lifetimes
from trading_system.spoof.score import journal_scores
from trading_system.spoof.synth import labeled_day
from trading_system.viz.style import PALETTE, apply_style, save_fig


def build_demo_journal(seed: int = 42) -> tuple[LevelJournal, pl.DataFrame, pl.DataFrame]:
    """Journal + scored episodes/grid over the synthetic labeled day."""
    states, trades, _truth = labeled_day(seed=seed)
    journal = LevelJournal(large_k=3.0, iceberg_refill_ms=300).run(states, trades)
    episodes_scored, grid_scored = journal_scores(journal)
    return journal, episodes_scored, grid_scored


def stability_heatmap(
    grid_scored: pl.DataFrame,
    name: str = "m6_stability_heatmap",
    out_dir: Path | None = None,
    time_bin_s: float = 1.0,
) -> Path:
    """Book heatmap (time x price) colored by level stability score."""
    apply_style()
    bin_ns = int(time_bin_s * NS_PER_S)
    ts0 = int(grid_scored["ts"].min())
    cells = (
        grid_scored.with_columns(((pl.col("ts") - ts0) // bin_ns).alias("tbin"))
        .group_by("tbin", "price")
        .agg(pl.col("score").mean())
    )
    prices = np.sort(grid_scored["price"].unique().to_numpy())
    n_bins = int(cells["tbin"].max()) + 1
    row = {p: i for i, p in enumerate(prices)}
    M = np.full((len(prices), n_bins), np.nan)
    for tbin, price, score in cells.iter_rows():
        M[row[price], int(tbin)] = score
    cmap = colormaps["RdYlGn"].with_extremes(bad="#eceff1")
    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(
        np.ma.masked_invalid(M),
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=(0, n_bins * time_bin_s, float(prices[0]), float(prices[-1])),
    )
    fig.colorbar(im, ax=ax, label="stability score (0 = spoof-like, 1 = stable)")
    ax.set_title("L2 levels colored by stability score")
    ax.set_xlabel("time, s")
    ax.set_ylabel("price")
    return save_fig(fig, name, out_dir)


def lifetime_distributions(
    episodes: pl.DataFrame,
    name: str = "m6_lifetimes_exec_vs_cancel",
    out_dir: Path | None = None,
) -> Path:
    """Seaborn lifetime distributions of large levels: executed vs canceled."""
    apply_style()
    pops = large_level_lifetimes(episodes)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(
        data={
            "lifetime, s": (pops["lifetime_ms"] / 1_000.0).to_numpy(),
            "outcome": pops["outcome"].to_numpy(),
        },
        x="lifetime, s",
        hue="outcome",
        hue_order=["executed", "canceled"],
        palette={"executed": PALETTE["long"], "canceled": PALETTE["short"]},
        log_scale=True,
        element="step",
        stat="count",
        common_norm=False,
        bins=24,
        ax=ax,
    )
    ax.set_title("Large-level lifetimes: executed vs canceled")
    return save_fig(fig, name, out_dir)


def demo_reports(out_dir: Path, seed: int = 42) -> list[Path]:
    """All M6 checklist figures from seeded synthetic data."""
    _journal, episodes_scored, grid_scored = build_demo_journal(seed=seed)
    return [
        stability_heatmap(grid_scored, out_dir=out_dir),
        lifetime_distributions(episodes_scored, out_dir=out_dir),
    ]
