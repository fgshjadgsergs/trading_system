"""real_heatmap.py: the offline half (lake -> figures) works on any lake."""

from __future__ import annotations

import numpy as np

from trading_system.core.config import load_config
from trading_system.core.io import write_batch
from trading_system.core.schema import Kline, records_to_frame
from trading_system.core.synth import (
    synth_liquidations,
    synth_open_interest,
    synth_ratios,
    synth_trades,
)
from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S


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


def test_build_heatmap_falls_back_to_klines(tmp_path):
    """No aggTrades in the lake -> the map builds from kline bars alone."""
    from scripts.real_heatmap import build_heatmap

    lake = tmp_path / "lake"
    out = tmp_path / "reports"
    rng = np.random.default_rng(11)
    start = 1_755_600_000 * NS_PER_S
    price = 4_500.0
    rows = []
    for i in range(600):
        o = price
        price *= float(np.exp(rng.normal(0, 0.001)))
        hi, lo = max(o, price) * 1.0005, min(o, price) * 0.9995
        vol = float(rng.lognormal(3, 0.5))
        rows.append(
            Kline(
                exchange="binance_usdm",
                symbol="ETHUSDT",
                ts_open=start + i * NS_PER_MIN,
                # exchange convention: close = open + step - 1ms
                ts_close=start + (i + 1) * NS_PER_MIN - 1_000_000,
                open=o,
                high=hi,
                low=lo,
                close=price,
                volume=vol,
                quote_volume=vol * price,
                taker_buy_volume=vol * 0.5,
                taker_buy_quote_volume=vol * price * 0.5,
                n_trades=100,
                closed=True,
            )
        )
    write_batch(lake, "kline", records_to_frame(rows, "kline"))
    write_batch(
        lake,
        "open_interest",
        records_to_frame(
            synth_open_interest(symbol="ETHUSDT", start_ts=start, n=800, price=4_500.0, seed=11),
            "open_interest",
        ),
    )
    paths = build_heatmap(lake, "ETHUSDT", load_config(), out, timeframe="1m")
    assert len(paths) == 2  # no liquidation stream -> overlay + slice only
    for p in paths:
        assert p.exists() and p.stat().st_size > 5_000
