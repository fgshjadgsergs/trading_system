"""M10: overlay chart, seaborn templates, stage report builder."""

from __future__ import annotations

import numpy as np
import polars as pl

from trading_system.core.schema import records_to_frame
from trading_system.core.synth import synth_trades
from trading_system.core.timeutils import NS_PER_S
from trading_system.features.bars import time_bars
from trading_system.viz.overlay import overlay_chart
from trading_system.viz.report import build_report
from trading_system.viz.templates import (
    calibration_curve,
    corr_heatmap,
    dist_plot,
    event_study_plot,
)

T0 = 1_755_600_000 * NS_PER_S


def _bars() -> pl.DataFrame:
    trades = records_to_frame(synth_trades(n=15_000, mean_gap_ms=200.0, seed=3), "trade")
    return time_bars(trades, "5m")


def test_overlay_all_layers(tmp_path):
    bars = _bars()
    n = bars.height
    prices = np.linspace(float(bars["low"].min()), float(bars["high"].max()), 40)
    H = np.abs(np.random.default_rng(42).normal(0, 1, (40, n)))
    events = pl.DataFrame(
        {
            "ts": [int(bars["ts_close"][n // 2])],
            "signal": ["s1"],
            "side": [1],
            "price": [float(bars["close"][n // 2])],
            "target": [float(bars["close"][n // 2]) * 1.01],
            "meta": [1e6],
        },
        schema_overrides={"side": pl.Int8},
    )
    prof = pl.DataFrame(
        {"price": prices.tolist(), "volume_usd": np.abs(np.sin(prices)).tolist()}
    )
    levels = pl.DataFrame({"price": [float(bars["close"].median())]})
    p = overlay_chart(
        bars,
        heat=(bars["ts_close"].to_numpy(), prices, H),
        events=events,
        profile=prof,
        levels=levels,
        name="test_overlay",
        out_dir=tmp_path,
    )
    assert p.exists() and p.stat().st_size > 20_000


def test_overlay_minimal(tmp_path):
    p = overlay_chart(_bars(), name="test_overlay_min", out_dir=tmp_path)
    assert p.exists() and p.stat().st_size > 5_000


def test_templates_render(tmp_path):
    rng = np.random.default_rng(42)
    p1 = dist_plot(rng.normal(0, 1, 500), "t_dist", out_dir=tmp_path, title="dist")
    pred = rng.uniform(0, 1, 400)
    real = pred + rng.normal(0, 0.1, 400)
    p2 = calibration_curve(pred, real, "t_cal", out_dir=tmp_path)
    paths_matrix = rng.normal(0.001, 0.01, (60, 20)).cumsum(axis=1)
    p3 = event_study_plot(paths_matrix, "t_es", out_dir=tmp_path, baseline=np.zeros(20))
    import pandas as pd

    p4 = corr_heatmap(pd.DataFrame(rng.normal(0, 1, (200, 4)), columns=list("abcd")), "t_corr", out_dir=tmp_path)
    for p in (p1, p2, p3, p4):
        assert p.exists() and p.stat().st_size > 5_000


def test_event_study_ci_contains_mean(tmp_path):
    rng = np.random.default_rng(1)
    paths = rng.normal(0.5, 0.05, (200, 10))
    p = event_study_plot(paths, "t_es_ci", out_dir=tmp_path)
    assert p.exists()


def test_build_report(tmp_path):
    fig_path = dist_plot(np.arange(100.0), "fig_a", out_dir=tmp_path)
    index = build_report("stage-test", [(fig_path, "Каптион A")], out_root=tmp_path)
    assert index.exists()
    text = index.read_text(encoding="utf-8")
    assert "Каптион A" in text and "fig_a.png" in text
    assert (tmp_path / "stage-test" / "fig_a.png").exists()
