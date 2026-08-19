"""M11 demo reports: fire-drill incident timeline and PnL envelope.

Both figures come from one deterministic run_drill(seed) — the same scripted
scenario the tests assert on: the depth stream goes silent, agg_trade takes a
gap burst and live PnL drifts out of the backtest envelope. The timeline shows,
per component, when the fault was injected, when the first alert fired and
when a human would plausibly have noticed; the exam passes when every alert
lands before its human marker. Split note: the M9 crash-drill incident log
lives in trading_system.risk.reports; this module owns the M11 monitoring
fire-drill figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trading_system.monitoring.drill import (
    DEPTH_BREAK_S,
    FRESHNESS_LIMITS_S,
    GAP_LEN_S,
    GAP_TIMES_S,
    DrillResult,
    run_drill,
)
from trading_system.viz.style import PALETTE, apply_style, save_fig

_KIND_STYLE = {
    "break": (PALETTE["short"], "x", "fault injected"),
    "alert": (PALETTE["accent"], "^", "first alert"),
    "human": (PALETTE["neutral"], "o", "human would notice"),
}
_COMPONENTS = ("depth", "gaps", "pnl")


def fig_drill_timeline(res: DrillResult, out_dir: Path) -> Path:
    """Freshness ages over time + break/alert/human lanes per component."""
    fig, (ax_age, ax_ev) = plt.subplots(
        2, 1, figsize=(13, 7.5), height_ratios=[1.6, 1], constrained_layout=True
    )

    colors = {"depth": PALETTE["short"], "agg_trade": PALETTE["neutral"], "mark_price": PALETTE["long"]}
    for stream, ages in res.freshness_ages.items():
        c = colors.get(stream, PALETTE["neutral"])
        ax_age.plot(res.freshness_t_s, ages, lw=1.4, color=c, label=stream)
        ax_age.axhline(FRESHNESS_LIMITS_S[stream], color=c, lw=0.9, ls=":", alpha=0.7)
    for g in GAP_TIMES_S:
        ax_age.axvspan(g - GAP_LEN_S, g, color=PALETTE["accent"], alpha=0.18)
    ax_age.axvline(DEPTH_BREAK_S, color=PALETTE["short"], lw=1.6, ls="--")
    ax_age.annotate(
        "depth stream killed",
        (DEPTH_BREAK_S, ax_age.get_ylim()[1] * 0.92),
        fontsize=9,
        color=PALETTE["short"],
        ha="left",
    )
    ax_age.set_ylabel("stream age, s")
    ax_age.set_title(
        "M11 fire drill — stream freshness under injected faults "
        "(dotted: per-stream limits, shaded: gap burst)"
    )
    ax_age.legend(loc="upper left")

    lane_y = {c: len(_COMPONENTS) - 1 - i for i, c in enumerate(_COMPONENTS)}
    seen_kinds: set[str] = set()
    for e in res.events:
        if e.component not in lane_y:
            continue
        color, marker, label = _KIND_STYLE[e.kind]
        ax_ev.scatter(
            e.t_s,
            lane_y[e.component],
            s=90,
            color=color,
            marker=marker,
            zorder=3,
            label=label if e.kind not in seen_kinds else None,
        )
        seen_kinds.add(e.kind)
    for c, y in lane_y.items():
        ax_ev.axhline(y, color=PALETTE["grid"], lw=0.8, zorder=0)
        alert = res.first(c, "alert")
        human = res.first(c, "human")
        if alert is not None and human is not None:
            ax_ev.annotate(
                f"lead {human.t_s - alert.t_s:.0f}s",
                (float(np.sqrt(alert.t_s * human.t_s)), y + 0.16),
                ha="center",
                fontsize=8.5,
                color=PALETTE["long"],
            )
            ax_ev.plot(
                [alert.t_s, human.t_s], [y, y], color=PALETTE["long"], lw=2.2, alpha=0.55
            )
    ax_ev.set_yticks([lane_y[c] for c in _COMPONENTS], _COMPONENTS)
    ax_ev.set_ylim(-0.6, len(_COMPONENTS) - 0.2)
    ax_ev.set_xscale("log")  # stream faults live at ~10^2 s, PnL drift at ~10^4 s
    ax_ev.set_xlabel("seconds since drill start (log scale)")
    verdict = "PASSED" if res.passed else "FAILED"
    ax_ev.set_title(
        f"incident lanes — every alert must precede its human marker: {verdict} "
        f"(suppressed repeats: {res.suppressed})"
    )
    ax_ev.legend(loc="upper right", fontsize=8.5)
    return save_fig(fig, "m11_drill_timeline", out_dir)


def fig_pnl_envelope(res: DrillResult, out_dir: Path) -> Path:
    """Cumulative live PnL vs the backtest expectation envelope."""
    tracker = res.pnl_tracker
    assert tracker is not None and tracker.start_ts is not None
    ts = tracker.start_ts + (res.pnl_t_days * 86_400.0 * 1e9).astype(np.int64)
    expected, lower, upper = tracker.envelope(ts)

    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    hours = res.pnl_t_days * 24.0
    ax.fill_between(
        hours, lower, upper, color=PALETTE["neutral"], alpha=0.15,
        label=f"backtest envelope (±{tracker.z_threshold:g}σ)",
    )
    ax.plot(hours, expected, color=PALETTE["neutral"], lw=1.2, ls="--", label="expected")
    ax.plot(hours, res.pnl_cum, color=PALETTE["short"], lw=1.6, label="live cum PnL")
    for kind in ("break", "alert", "human"):
        e = res.first("pnl", kind)
        if e is None:
            continue
        color, marker, label = _KIND_STYLE[kind]
        ax.axvline(e.t_s / 3600.0, color=color, lw=1.4, ls=":")
        ax.annotate(
            label,
            (e.t_s / 3600.0, ax.get_ylim()[0]),
            xytext=(4, 10),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
            rotation=90,
            va="bottom",
        )
    ax.set_xlabel("hours since drill start")
    ax.set_ylabel("cumulative PnL, USD")
    ax.set_title("M11 fire drill — live PnL diverges from the backtest envelope; alert beats the eye")
    ax.legend(loc="lower left")
    return save_fig(fig, "m11_pnl_envelope", out_dir)


def demo_reports(out_dir: Path, seed: int = 42) -> list[Path]:
    """Generate the M11 checklist figures from one deterministic drill run."""
    apply_style()
    res = run_drill(seed=seed)
    return [fig_drill_timeline(res, out_dir), fig_pnl_envelope(res, out_dir)]
