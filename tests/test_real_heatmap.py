"""real_heatmap.py: the offline half (lake -> figures) works on any lake."""

from __future__ import annotations

from trading_system.core.config import load_config
from trading_system.core.io import write_batch
from trading_system.core.schema import records_to_frame
from trading_system.core.synth import (
    synth_liquidations,
    synth_open_interest,
    synth_ratios,
    synth_trades,
)


def test_build_heatmap_from_lake(tmp_path):
    from scripts.real_heatmap import build_heatmap

    lake = tmp_path / "lake"
    out = tmp_path / "reports"
    trades = synth_trades(n=20_000, symbol="ETHUSDT", s0=4_500.0, mean_gap_ms=200.0, seed=8)
    start = trades[0].ts_event
    write_batch(lake, "trade", records_to_frame(trades, "trade"))
    write_batch(
        lake,
        "open_interest",
        records_to_frame(
            synth_open_interest(symbol="ETHUSDT", start_ts=start, n=800, price=4_500.0, seed=8),
            "open_interest",
        ),
    )
    write_batch(
        lake,
        "ratio",
        records_to_frame(synth_ratios(symbol="ETHUSDT", start_ts=start, n=50, seed=8), "ratio"),
    )
    write_batch(
        lake,
        "liquidation",
        records_to_frame(synth_liquidations(trades, rate=0.001, seed=8), "liquidation"),
    )
    paths = build_heatmap(lake, "ETHUSDT", load_config(), out, timeframe="1m")
    assert len(paths) == 3  # overlay, slice, heat-vs-liqs
    for p in paths:
        assert p.exists() and p.stat().st_size > 5_000
        assert "ethusdt_real_heat" in p.name
