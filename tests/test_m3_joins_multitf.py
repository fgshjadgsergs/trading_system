"""M3: anti-lookahead asof joins, indicators, multi-TF closed-bar semantics."""

from __future__ import annotations

import polars as pl
import pytest

from trading_system.core.schema import records_to_frame
from trading_system.core.synth import synth_open_interest, synth_trades
from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S
from trading_system.features.bars import time_bars
from trading_system.features.indicators import with_atr, with_vwap
from trading_system.features.joins import join_open_interest
from trading_system.features.multitf import build_multitf, join_context, tf_features


@pytest.fixture()
def trades() -> pl.DataFrame:
    return records_to_frame(synth_trades(n=30_000, mean_gap_ms=120.0, seed=5), "trade")


@pytest.fixture()
def oi(trades) -> pl.DataFrame:
    start = int(trades["ts_event"].min())
    return records_to_frame(
        synth_open_interest(n=1_000, step_s=7, start_ts=start, seed=5), "open_interest"
    )


def test_asof_join_is_strictly_backward(trades, oi):
    bars = time_bars(trades, "1m")
    joined = join_open_interest(bars, oi)
    # manual check on every bar: the joined OI is the last with ts_event < ts_close
    oi_sorted = oi.sort("ts_event")
    for row in joined.iter_rows(named=True):
        expected = oi_sorted.filter(pl.col("ts_event") < row["ts_close"])
        if expected.height:
            assert row["open_interest"] == expected["open_interest"][-1]


def test_no_lookahead_future_oi_does_not_change_past(trades, oi):
    bars = time_bars(trades, "1m")
    base = join_open_interest(bars, oi)
    cut = int(bars["ts_close"][bars.height // 2])
    # perturb every OI point at/after the cut — bars closing before must not move
    perturbed = oi.with_columns(
        pl.when(pl.col("ts_event") >= cut)
        .then(pl.col("open_interest") * 100.0)
        .otherwise(pl.col("open_interest"))
        .alias("open_interest")
    )
    redo = join_open_interest(bars, perturbed)
    early = pl.col("ts_close") <= cut
    assert base.filter(early)["open_interest"].to_list() == redo.filter(early)[
        "open_interest"
    ].to_list()


def test_oi_at_exact_bar_close_belongs_to_next_bar(trades, oi):
    bars = time_bars(trades, "1m").head(5)
    point_ts = int(bars["ts_close"][2])
    oi_one = oi.head(1).with_columns(
        pl.lit(point_ts, dtype=pl.Int64).alias("ts_event"),
        pl.lit(12345.0).alias("open_interest"),
    )
    joined = join_open_interest(bars, oi_one)
    assert joined["open_interest"][2] is None
    assert joined["open_interest"][3] == 12345.0


def test_vwap_and_atr_sane(trades):
    bars = with_atr(with_vwap(time_bars(trades, "5m")), period=14)
    assert (
        bars.select(
            ((pl.col("vwap_bar") >= pl.col("low")) & (pl.col("vwap_bar") <= pl.col("high")))
            .all()
        ).item()
    )
    atr = bars["atr"].drop_nulls()
    assert (atr > 0).all()
    tr = bars["tr"].drop_nulls()
    assert atr.max() <= tr.max() + 1e-9  # smoothed inside TR envelope


def test_vwap_session_anchor_resets(trades):
    bars = with_vwap(time_bars(trades, "5m"), session="1h")
    first_in_hour = bars.filter(pl.col("ts_open") % (3_600 * NS_PER_S) == 0)
    if first_in_hour.height:
        assert (
            first_in_hour.select(
                (pl.col("vwap_session") - pl.col("vwap_bar")).abs().max()
            ).item()
            < 1e-9
        )


def test_tf_features_columns_and_impulse(trades, oi):
    f = tf_features(trades, oi, "1m", zscore_window=24, impulse_k=3.0)
    assert {"vol_z", "taker_buy_share", "impulse", "d_oi_usd"} <= set(f.columns)
    share = f["taker_buy_share"].drop_nulls()
    assert ((share >= 0) & (share <= 1)).all()
    # plant an impulse: multiply one bar's trades 50x
    t = trades.with_columns(
        pl.when(
            pl.col("ts_event").is_between(
                int(trades["ts_event"].min()) + 40 * NS_PER_MIN,
                int(trades["ts_event"].min()) + 41 * NS_PER_MIN,
            )
        )
        .then(pl.col("qty_usd") * 50)
        .otherwise(pl.col("qty_usd"))
        .alias("qty_usd")
    )
    f2 = tf_features(t, oi, "1m", zscore_window=24, impulse_k=3.0)
    assert f2.filter(pl.col("impulse")).height >= 1


def test_zscore_baseline_excludes_current_bar(trades, oi):
    """A huge bar must not shrink its own z-score via the baseline."""
    t0 = int(trades["ts_event"].min())
    spiked = trades.with_columns(
        pl.when(
            pl.col("ts_event").is_between(t0 + 50 * NS_PER_MIN, t0 + 51 * NS_PER_MIN)
        )
        .then(pl.col("qty_usd") * 200)
        .otherwise(pl.col("qty_usd"))
        .alias("qty_usd")
    )
    f = tf_features(spiked, oi, "1m", zscore_window=24)
    spike_open = t0 + 50 * NS_PER_MIN - (t0 + 50 * NS_PER_MIN) % NS_PER_MIN
    z = f.filter(pl.col("ts_open") == spike_open)["vol_z"]
    assert z.item() > 10  # self-inclusive baseline would crush this


def test_multitf_only_closed_bars(trades, oi):
    base = time_bars(trades, "1m")
    mtf = build_multitf(trades, oi, ["1m", "5m"], zscore_window=12)
    joined = join_context(base, mtf, ["5m"])
    five = mtf.filter(pl.col("tf") == "5m").sort("ts_close")
    for row in joined.iter_rows(named=True):
        closed = five.filter(pl.col("ts_close") <= row["ts_close"])
        if closed.height and row["5m_quote_volume"] is not None:
            assert row["5m_quote_volume"] == closed["quote_volume"][-1]
        else:
            assert row["5m_quote_volume"] is None


def test_multitf_no_leak_from_forming_bar(trades, oi):
    """Bars of the 5m bucket in progress must see the PREVIOUS 5m stats."""
    base = time_bars(trades, "1m")
    mtf = build_multitf(trades, oi, ["5m"], zscore_window=12)
    joined = join_context(base, mtf, ["5m"])
    inside = joined.filter(pl.col("ts_close") % (5 * NS_PER_MIN) != 0)
    five = mtf.filter(pl.col("tf") == "5m")
    for row in inside.head(50).iter_rows(named=True):
        own_bucket_open = row["ts_open"] - row["ts_open"] % (5 * NS_PER_MIN)
        forming = five.filter(pl.col("ts_open") == own_bucket_open)
        if forming.height and row["5m_quote_volume"] is not None:
            assert row["5m_quote_volume"] != forming["quote_volume"][0] or (
                forming["ts_close"][0] <= row["ts_close"]
            )


def test_demo_reports(tmp_path):
    from trading_system.features.reports import demo_reports

    paths = demo_reports(tmp_path, seed=42)
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.stat().st_size > 5_000
