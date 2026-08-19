"""Seaborn analytic templates: distributions, calibration curves, event studies
with confidence bands, correlation heatmaps. Every template returns a saved png."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from trading_system.viz.style import PALETTE, apply_style, save_fig


def dist_plot(
    values: np.ndarray,
    name: str,
    out_dir: Path | None = None,
    title: str = "",
    xlabel: str = "",
    bins: int = 40,
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(np.asarray(values), bins=bins, kde=True, ax=ax, color=PALETTE["neutral"])
    ax.set_title(title or name)
    ax.set_xlabel(xlabel)
    return save_fig(fig, name, out_dir)


def calibration_curve(
    predicted: np.ndarray,
    realized: np.ndarray,
    name: str,
    out_dir: Path | None = None,
    n_bins: int = 10,
    title: str = "Calibration",
) -> Path:
    """Predicted vs realized by prediction decile, with the identity line."""
    apply_style()
    predicted = np.asarray(predicted, dtype=float)
    realized = np.asarray(realized, dtype=float)
    order = np.argsort(predicted)
    chunks = np.array_split(order, n_bins)
    xs = np.array([predicted[c].mean() for c in chunks if len(c)])
    ys = np.array([realized[c].mean() for c in chunks if len(c)])
    fig, ax = plt.subplots(figsize=(8, 8))
    lims = (min(xs.min(), ys.min()), max(xs.max(), ys.max()))
    ax.plot(lims, lims, ls="--", color=PALETTE["neutral"], lw=1)
    sns.lineplot(x=xs, y=ys, marker="o", ax=ax, color=PALETTE["accent"])
    ax.set_xlabel("predicted (bin mean)")
    ax.set_ylabel("realized (bin mean)")
    ax.set_title(title)
    return save_fig(fig, name, out_dir)


def event_study_plot(
    paths: np.ndarray,
    name: str,
    out_dir: Path | None = None,
    ci: float = 0.95,
    n_boot: int = 500,
    seed: int = 42,
    title: str = "Event study",
    baseline: np.ndarray | None = None,
) -> Path:
    """Mean forward path around events with a bootstrap CI band.

    paths: (n_events, horizon) matrix of aligned forward returns/prices.
    """
    apply_style()
    paths = np.asarray(paths, dtype=float)
    rng = np.random.default_rng(seed)
    mean = np.nanmean(paths, axis=0)
    boots = np.empty((n_boot, paths.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, paths.shape[0], paths.shape[0])
        boots[b] = np.nanmean(paths[idx], axis=0)
    alpha = (1 - ci) / 2
    lo_b, hi_b = np.quantile(boots, [alpha, 1 - alpha], axis=0)
    x = np.arange(paths.shape[1])
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x, lo_b, hi_b, alpha=0.25, color=PALETTE["accent"], label=f"{int(ci*100)}% CI")
    ax.plot(x, mean, color=PALETTE["accent"], lw=2, label="event mean")
    if baseline is not None:
        ax.plot(x, np.asarray(baseline), color=PALETTE["neutral"], ls="--", lw=1.5, label="baseline")
    ax.axhline(0, color="#b0bec5", lw=0.8)
    ax.set_xlabel("bars after event")
    ax.set_title(title)
    ax.legend()
    return save_fig(fig, name, out_dir)


def corr_heatmap(
    frame: pd.DataFrame, name: str, out_dir: Path | None = None, title: str = "Correlations"
) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(frame.corr(), annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title(title)
    return save_fig(fig, name, out_dir)
