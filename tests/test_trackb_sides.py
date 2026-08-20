"""Track B2: causal long/short side shares from the ratio streams."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trading_system.core.schema import POLARS_SCHEMAS, records_to_frame
from trading_system.core.synth import synth_ratios, synth_trades
from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S
from trading_system.features.bars import time_bars
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.map import LiqMap, StaticWeights
from trading_system.liqmap.sides import join_long_share, long_share_series

T0 = 1_755_600_000 * NS_PER_S


def _ratio_row(ts: int, metric: str, share: float) -> dict:
    return {
        "exchange": "binance_usdm",
        "symbol": "BTCUSDT",
        "ts_event": ts,
        "ts_recv": ts,
        "metric": metric,
        "long_share": share,
        "short_share": 1.0 - share,
        "ratio": share / (1.0 - share),
    }


def _ratios(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=POLARS_SCHEMAS["ratio"])


def test_blend_renormalizes_over_available_metrics():
    rows = [
        _ratio_row(T0, "global_ls_account", 0.6),
        _ratio_row(T0 + NS_PER_MIN, "taker_ls", 0.4),
    ]
    series = long_share_series(_ratios(rows), blend={"global_ls_account": 0.4, "taker_ls": 0.3})
    # at T0 only the account metric exists -> pure 0.6
    assert series.filter(pl.col("ts_event") == T0)["long_share"][0] == pytest.approx(0.6)
    # at T0+1m both exist -> (0.4*0.6 + 0.3*0.4) / 0.7
    expected = (0.4 * 0.6 + 0.3 * 0.4) / 0.7
    assert series.filter(pl.col("ts_event") == T0 + NS_PER_MIN)["long_share"][0] == pytest.approx(expected)


def test_join_is_strictly_backward_and_clipped():
    trades = records_to_frame(synth_trades(n=3_000, seed=3), "trade")
    bars = time_bars(trades, "1m")
    cut = int(bars["ts_close"][2])
    rows = [
        _ratio_row(cut - 1, "global_ls_account", 0.99),  # published before bar-2 close
        _ratio_row(cut, "global_ls_account", 0.2),  # exactly at close -> next bar
    ]
    joined = join_long_share(bars, _ratios(rows), clip=(0.1, 0.9))
    assert joined["long_share"][0] == pytest.approx(0.5)  # before any ratio -> default
    assert joined["long_share"][2] == pytest.approx(0.9)  # 0.99 clipped to 0.9
    assert joined["long_share"][3] == pytest.approx(0.2)


def test_future_ratio_does_not_change_past_bars():
    trades = records_to_frame(synth_trades(n=5_000, seed=4), "trade")
    bars = time_bars(trades, "1m")
    ratios = records_to_frame(
        synth_ratios(start_ts=int(bars["ts_open"][0]), n=10, step_s=60, seed=4), "ratio"
    )
    base = join_long_share(bars, ratios)
    cut = int(bars["ts_close"][bars.height // 2])
    perturbed = ratios.with_columns(
        pl.when(pl.col("ts_event") >= cut).then(0.95).otherwise(pl.col("long_share")).alias("long_share")
    )
    redo = join_long_share(bars, perturbed)
    early = pl.col("ts_close") <= cut
    assert base.filter(early)["long_share"].to_list() == redo.filter(early)["long_share"].to_list()


def test_allocate_with_per_call_share():
    lm = LiqMap([10.0], PriceBuckets(10.0), StaticWeights(np.array([1.0])))
    lm.allocate(1_000_000.0, 50_000.0, long_share=0.7)
    snap = lm.snapshot()
    assert snap["long"].sum() == pytest.approx(700_000.0)
    assert snap["short"].sum() == pytest.approx(300_000.0)
    # step passes the share through; instance default untouched
    lm2 = LiqMap([10.0], PriceBuckets(10.0), StaticWeights(np.array([1.0])))
    lm2.step(49_000.0, 49_100.0, 50_000.0, 1_000_000.0, dt_s=0.0, long_share=0.25)
    snap2 = lm2.snapshot()
    assert snap2["long"].sum() == pytest.approx(250_000.0)
    assert lm2.long_share == 0.5
    with pytest.raises(ValueError):
        lm2.allocate(1.0, 100.0, long_share=1.5)


def test_empty_ratios_fall_back_to_default():
    trades = records_to_frame(synth_trades(n=2_000, seed=5), "trade")
    bars = time_bars(trades, "1m")
    joined = join_long_share(bars, _ratios([]).head(0), default=0.5)
    assert (joined["long_share"] == 0.5).all()
