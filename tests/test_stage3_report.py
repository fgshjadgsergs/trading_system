"""stage3_report: real-data heat builder consistency + Gate A verdict on known truth."""

from __future__ import annotations

import numpy as np
import polars as pl

from trading_system.calibration.real_data import (
    bars_to_arrays,
    bucket_grid,
    make_real_heat_builder,
)
from trading_system.calibration.synthetic import make_world
from trading_system.core.schema import records_to_frame
from trading_system.core.synth import synth_open_interest, synth_trades
from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S
from trading_system.features import join_open_interest, time_bars, with_atr
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.history import HeatHistory
from trading_system.liqmap.map import LiqMap, StaticWeights

CFG = {
    "liqmap": {
        "leverage_grid": [10.0, 20.0, 50.0],
        "bucket_atr_fraction": 0.5,
        "decay_half_life_s": 86_400.0,
        "long_share_default": 0.5,
        "maint_margin_rate_flat": 0.005,
    }
}


def _world_bars(world) -> pl.DataFrame:
    closes = world.prices
    lows = np.minimum(np.concatenate([[closes[0]], closes[:-1]]), closes)
    highs = np.maximum(np.concatenate([[closes[0]], closes[:-1]]), closes)
    return pl.DataFrame(
        {
            "ts_open": world.ts - NS_PER_MIN,
            "ts_close": world.ts,
            "close": closes,
            "low": lows,
            "high": highs,
            "open": np.concatenate([[closes[0]], closes[:-1]]),
            "quote_volume": np.full(len(closes), 1.0),
            "d_oi_usd": world.entry_notional,
            "atr": np.full(len(closes), world.atr),
        }
    )


def test_real_builder_matches_liqmap_replay():
    """The fast builder's row totals track an exact LiqMap replay."""
    trades = records_to_frame(synth_trades(n=8_000, seed=6), "trade")
    oi = records_to_frame(
        synth_open_interest(start_ts=int(trades["ts_event"].min()), n=400, seed=6),
        "open_interest",
    )
    bars = with_atr(join_open_interest(time_bars(trades, "1m"), oi), period=14)
    arr = bars_to_arrays(bars)
    edges = bucket_grid(arr, atr_fraction=0.5)
    grid = np.array([10.0, 25.0])
    w = np.array([0.6, 0.4])
    build = make_real_heat_builder(arr, grid, edges, bar_s=60.0, decay_half_life_s=86_400.0)
    heat = build(w)

    size = edges[1] - edges[0]
    lm = LiqMap(
        leverage_grid=list(grid),
        buckets=PriceBuckets(bucket_size=size),
        weight_fn=StaticWeights(w),
        decay_half_life_s=86_400.0,
    )
    hist = HeatHistory(lm)
    for row in bars.iter_rows(named=True):
        d_oi = row["d_oi_usd"]
        if d_oi is None:
            d_oi = row["quote_volume"] * 0.05
        lm.step(row["low"], row["high"], row["close"], d_oi, dt_s=60.0)
        hist.record(row["ts_close"])
    # totals per bar agree within a few percent (boundary conventions differ
    # at bucket edges; mass scale and dynamics must match)
    fast_totals = heat.sum(axis=1)
    exact_totals = np.array([hist.total_at(i) for i in range(len(hist))])
    mask = exact_totals > 0
    rel = np.abs(fast_totals[mask] - exact_totals[mask]) / exact_totals[mask]
    assert np.median(rel) < 0.05
    assert heat.shape == (bars.height, len(edges) - 1)
    # causality: row t never exceeds cumulative inflow up to t
    inflow = np.cumsum(np.maximum(arr.d_oi_usd, 0.0))
    assert np.all(fast_totals <= inflow + 1e-6)


def test_analyze_passes_gate_a_on_known_truth(tmp_path):
    """SyntheticWorld with a known mixture: calibrated map must beat naive."""
    from scripts.stage3_report import analyze

    world = make_world(n_bars=3_000, seed=7)
    bars = _world_bars(world)
    res = analyze(
        bars,
        world.liquidations,
        CFG,
        tmp_path,
        world.symbol,
        timeframe="1m",
        test_frac=0.3,
        embargo_days=0.05,
        n_candidates=16,
        seed=7,
    )
    assert res["capture"]["naive"] is not None
    assert res["gate_a"] is True, res["capture"]
    assert res["capture"]["static"] > res["capture"]["naive"]
    assert res["weights"] is not None
    for path, _ in res["figures"]:
        assert path.exists() and path.stat().st_size > 5_000


def test_analyze_without_truth_reports_no_verdict(tmp_path):
    from scripts.stage3_report import analyze

    world = make_world(n_bars=1_200, seed=8)
    bars = _world_bars(world)
    empty = world.liquidations.head(0)
    res = analyze(
        bars, empty, CFG, tmp_path, world.symbol,
        timeframe="1m", test_frac=0.3, embargo_days=0.05, seed=8,
    )
    assert res["gate_a"] is None
    assert res["capture"] == {}
    assert res["n_liq_train"] == 0 and res["n_liq_test"] == 0


def test_stage3_main_offline_smoke(tmp_path, monkeypatch):
    """main() with a prebuilt lake and --skip-download writes the report."""
    import scripts.stage3_report as s3
    from trading_system.core.io import write_batch
    from trading_system.core.schema import Kline

    lake = tmp_path / "lake"
    out = tmp_path / "reports"
    rng = np.random.default_rng(12)
    start = 1_755_600_000 * NS_PER_S
    price = 4_000.0
    rows = []
    for i in range(900):
        o = price
        price *= float(np.exp(rng.normal(0, 0.001)))
        vol = float(rng.lognormal(3, 0.5))
        rows.append(
            Kline(
                "binance_usdm", "ETHUSDT",
                start + i * NS_PER_MIN, start + (i + 1) * NS_PER_MIN,
                o, max(o, price) * 1.0004, min(o, price) * 0.9996, price,
                vol, vol * price, vol * 0.5, vol * price * 0.5, 50, True,
            )
        )
    write_batch(lake, "kline", records_to_frame(rows, "kline"))
    write_batch(
        lake,
        "open_interest",
        records_to_frame(
            synth_open_interest(symbol="ETHUSDT", start_ts=start, n=900, price=4_000.0, seed=12),
            "open_interest",
        ),
    )
    s3.main(
        [
            "--symbol", "ETHUSDT",
            "--lake", str(lake),
            "--out", str(out),
            "--timeframe", "1m",
            "--skip-download",
            "--embargo-days", "0.05",
        ]
    )
    index = out / "stage3-ethusdt" / "README.md"
    assert index.exists()
    text = index.read_text(encoding="utf-8")
    assert "Gate A" in text and "нет вердикта" in text  # no liquidation truth in this lake
