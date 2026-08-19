"""Stage 3 report figures: event studies with CI bands, capture ladder,
calibration curve. All from seeded synthetic data; every figure is saved as
png via trading_system.viz.style.save_fig.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from trading_system.calibration.event_studies import (
    lvn_study,
    magnet_study,
    mean_path_ci,
    reversal_study,
)
from trading_system.calibration.synthetic import make_heat_builder, make_world
from trading_system.calibration.weights import (
    RollingCalibrator,
    StaticWeightCalibrator,
    calibration_curve,
    compare_ladder,
)
from trading_system.viz.style import PALETTE, apply_style, save_fig

BAR_NS = 60 * 1_000_000_000


def dip_reversal_series(
    n: int = 1500, period: int = 100, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Price series with planted dip-and-reverse episodes.

    Returns (prices, atr, event_idx): every `period` bars the price sells off
    for 5 bars and then mean-reverts sharply — the planted 'pool touch ->
    reversal' effect the event study should detect.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 8.0, n)
    events = []
    for c in range(period, n - 40, period):
        steps[c - 5 : c] -= 30.0  # approach: 5 bars down
        steps[c : c + 20] += 12.0  # sharp reversal
        events.append(c)
    prices = 50_000.0 + np.cumsum(steps)
    atr = np.full(n, 60.0)
    return prices, atr, np.asarray(events, dtype=int)


def lvn_series(
    n: int = 3000, zone: tuple[float, float] = (49_800.0, 50_200.0), seed: int = 42
) -> np.ndarray:
    """Mean-reverting walk that accelerates inside the low-volume zone."""
    rng = np.random.default_rng(seed)
    p = np.empty(n)
    p[0] = 50_600.0
    for t in range(1, n):
        scale = 90.0 if zone[0] <= p[t - 1] <= zone[1] else 30.0
        p[t] = p[t - 1] + 0.01 * (50_000.0 - p[t - 1]) + rng.normal(0.0, scale)
    return p


def _band_plot(ax, x, mean, lo, hi, color: str, label: str) -> None:
    sns.lineplot(x=x, y=mean, ax=ax, color=color, label=label)
    ax.fill_between(x, lo, hi, color=color, alpha=0.2)


def fig_reversal_paths(seed: int) -> plt.Figure:
    prices, atr, events = dip_reversal_series(seed=seed)
    res = reversal_study(prices, atr, events, k_atr=1.0, horizon=30, seed=seed)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(res.horizon + 1)
    for paths, clusters, color, label in (
        (res.event_paths, res.event_clusters, PALETTE["long"], "pool touch"),
        (res.control_paths, res.control_clusters, PALETTE["neutral"], "base rate"),
    ):
        mean, lo, hi = mean_path_ci(paths, clusters, n_boot=300, seed=seed)
        _band_plot(ax, x, mean, lo, hi, color, label)
    ax.axhline(res.k_atr, ls="--", color=PALETTE["accent"], lw=1, label="k*ATR threshold")
    ax.set_xlabel("bars after event")
    ax.set_ylabel("move against approach (ATR)")
    s = res.stats
    ax.set_title(
        "Event study: reversal after touching a top-decile pool "
        f"(P={s.event_rate:.2f} vs {s.control_rate:.2f}, p={s.p_value:.3f})"
    )
    ax.legend()
    return fig


def fig_magnet_curve(seed: int) -> plt.Figure:
    world = make_world(n_bars=1200, seed=seed, static_weights=(0.60, 0.10, 0.30))
    build = make_heat_builder(world, decay_half_life_bars=200.0)
    heat = build(world.true_weights[0])
    res = magnet_study(
        world.prices,
        np.full(len(world.prices), world.atr * 5),
        heat,
        world.bucket_edges,
        horizon=30,
        stride=2,
        seed=seed,
    )
    centers = (res.bin_edges_atr[:-1] + res.bin_edges_atr[1:]) / 2
    ok = ~np.isnan(res.p_reach)
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.lineplot(x=centers[ok], y=res.p_reach[ok], marker="o", ax=ax, color=PALETTE["long"])
    ax.fill_between(centers[ok], res.ci_low[ok], res.ci_high[ok], color=PALETTE["long"], alpha=0.2)
    for xc, p, n in zip(centers[ok], res.p_reach[ok], res.n_samples[ok], strict=True):
        ax.annotate(f"n={n}", (xc, p), textcoords="offset points", xytext=(0, 8), fontsize=8)
    ax.set_xlabel("distance to pool (ATR)")
    ax.set_ylabel(f"P(reach within {res.horizon} bars)")
    ax.set_title("Magnet effect: probability of reaching a top-decile pool vs distance")
    return fig


def fig_lvn_paths(seed: int) -> plt.Figure:
    prices = lvn_series(seed=seed)
    res = lvn_study(prices, np.array([[49_800.0, 50_200.0]]), horizon=20, seed=seed)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(res.horizon + 1)
    for paths, clusters, color, label in (
        (res.event_abs_paths, res.event_clusters, PALETTE["short"], "LVN entry"),
        (res.control_abs_paths, res.control_clusters, PALETTE["neutral"], "elsewhere"),
    ):
        mean, lo, hi = mean_path_ci(paths, clusters, n_boot=300, seed=seed)
        _band_plot(ax, x, mean, lo, hi, color, label)
    ax.set_xlabel("bars after entry")
    ax.set_ylabel("mean |log return|")
    s = res.stats
    ax.set_title(
        f"LVN behavior: traversal speed in low-volume nodes (effect={s.effect:.4f}, "
        f"p={s.p_value:.3f})"
    )
    ax.legend()
    return fig


def _ladder(seed: int):
    world = make_world(
        n_bars=1200,
        seed=seed,
        regime_period=150,
        regime_weights=((0.02, 0.08, 0.90), (0.90, 0.08, 0.02)),
        bucket_width_frac=0.002,
    )
    build = make_heat_builder(world, decay_half_life_bars=25.0)
    n = len(world.ts)
    train = (int(world.ts[0]), int(world.ts[int(n * 0.60)]))
    test = (int(world.ts[int(n * 0.65)]), int(world.ts[-1]) + 1)
    cal = StaticWeightCalibrator(
        n_weights=3, seed=seed, flow_weight=4.0, n_candidates=16, refine_sweeps=1
    )
    roll = RollingCalibrator(cal, train_window_ns=500 * BAR_NS, refit_every_ns=250 * BAR_NS)
    res = compare_ladder(
        build,
        world.ts,
        world.bucket_edges,
        world.liquidations,
        world.prices,
        train,
        test,
        n_weights=3,
        context=world.context,
        context_labels=world.true_weights,
        calibrator=cal,
        rolling=roll,
        tolerance=0.02,
    )
    return world, build, res, test


def fig_ladder_bars(seed: int) -> tuple[plt.Figure, object]:
    world, build, res, test = _ladder(seed)
    order = ["naive", "static", "rolling", "contextual"]
    names = [r for r in order if r in res.capture]
    vals = [res.capture[r] for r in names]
    colors = [
        PALETTE["accent"] if r == res.selected else
        (PALETTE["neutral"] if r == "naive" else PALETTE["long"])
        for r in names
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=names, y=vals, hue=names, palette=colors, legend=False, ax=ax)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (i, v), ha="center", va="bottom")
    ax.set_ylabel("OOS capture rate")
    ax.set_title(
        f"Weight-calibration ladder, OOS capture (selected: {res.selected}; "
        f"complexity without OOS gain is rolled back)"
    )
    return fig, (world, build, res, test)


def fig_calibration_curve(world, build, res, test, seed: int) -> plt.Figure:
    heat = build(res.static_weights)
    pred, real = calibration_curve(
        heat, world.ts, world.bucket_edges, world.liquidations, n_bins=10, ts_range=test
    )
    deciles = np.arange(1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    width = 0.4
    axes[0].bar(deciles - width / 2, pred, width, label="predicted (heat mass)",
                color=PALETTE["neutral"])
    axes[0].bar(deciles + width / 2, real, width, label="realized (liq USD)",
                color=PALETTE["long"])
    axes[0].set_xlabel("heat decile (10 = hottest)")
    axes[0].set_ylabel("share of intensity")
    axes[0].set_title("Liquidation intensity by heat decile")
    axes[0].legend()
    lim = max(pred.max(), real.max()) * 1.1
    sns.scatterplot(x=pred, y=real, ax=axes[1], color=PALETTE["short"], s=60)
    axes[1].plot([0, lim], [0, lim], ls="--", color=PALETTE["grid"])
    axes[1].set_xlabel("predicted share")
    axes[1].set_ylabel("realized share")
    axes[1].set_title("Calibration: predicted vs realized")
    fig.suptitle("Map calibration curve (OOS)", y=1.02)
    return fig


def demo_reports(out_dir: Path | str, seed: int = 42) -> list[Path]:
    """Generate every stage-3 figure from synthetic data; returns saved paths."""
    apply_style()
    out = Path(out_dir)
    paths = [
        save_fig(fig_reversal_paths(seed), "cal_event_reversal_paths", out),
        save_fig(fig_magnet_curve(seed), "cal_event_magnet_curve", out),
        save_fig(fig_lvn_paths(seed), "cal_event_lvn_paths", out),
    ]
    fig, ctx = fig_ladder_bars(seed)
    paths.append(save_fig(fig, "cal_ladder_capture_bars", out))
    world, build, res, test = ctx
    paths.append(
        save_fig(fig_calibration_curve(world, build, res, test, seed), "cal_calibration_curve", out)
    )
    return paths
