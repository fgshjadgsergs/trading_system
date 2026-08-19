"""M5: profile invariants, planted swings recovered exactly, level weights."""

from __future__ import annotations

import polars as pl
import pytest

from trading_system.core.schema import Side, Trade, records_to_frame
from trading_system.core.synth import synth_trades
from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S
from trading_system.profile.swings import equal_extremes, fractal_swings, level_weights
from trading_system.profile.volume_profile import (
    hvn_lvn,
    poc_price,
    profile,
    session_profiles,
    value_area,
)

T0 = 1_755_600_000 * NS_PER_S


@pytest.fixture()
def trades() -> pl.DataFrame:
    return records_to_frame(synth_trades(n=30_000, seed=9), "trade")


def _bars_from_path(path: list[tuple[float, float, float, float]]) -> pl.DataFrame:
    rows = []
    for i, (o, h, lo, c) in enumerate(path):
        rows.append(
            {
                "exchange": "binance_usdm",
                "symbol": "BTCUSDT",
                "ts_open": T0 + i * NS_PER_MIN,
                "ts_close": T0 + (i + 1) * NS_PER_MIN,
                "open": o,
                "high": h,
                "low": lo,
                "close": c,
            }
        )
    return pl.DataFrame(rows)


def test_value_area_invariants(trades):
    prof = profile(trades, bin_size=20.0)
    va = value_area(prof, pct=0.70)
    assert va.val <= va.poc <= va.vah  # POC inside VA
    assert va.share >= 0.70
    # not grossly overshooting: dropping the last added bin dips below 70%
    vols = prof["volume_usd"].to_numpy()
    largest_bin_share = vols.max() / vols.sum()
    assert va.share <= 0.70 + max(largest_bin_share, 0.05) + 1e-9
    assert poc_price(prof) == va.poc


def test_profile_total_volume_conserved(trades):
    prof = profile(trades, bin_size=15.0)
    assert prof["volume_usd"].sum() == pytest.approx(trades["qty_usd"].sum())
    assert prof["bin_lo"].is_sorted()


def test_profile_window_filter(trades):
    mid = int(trades["ts_event"].median())
    early = profile(trades, 20.0, ts_to=mid)
    late = profile(trades, 20.0, ts_from=mid)
    assert early["volume_usd"].sum() + late["volume_usd"].sum() == pytest.approx(
        trades["qty_usd"].sum()
    )


def test_session_profiles_partition(trades):
    sp = session_profiles(trades, 20.0)
    assert sp["session"].null_count() == 0
    assert sp["volume_usd"].sum() == pytest.approx(trades["qty_usd"].sum())


def test_hvn_lvn_planted():
    # heavy volume around 100, thin shelf at 110, heavy again at 120
    rows = []
    tid = 0
    for price, qty, reps in [(100.0, 5.0, 50), (110.0, 0.2, 5), (120.0, 4.0, 40), (105.0, 1.0, 10), (115.0, 1.0, 10)]:
        for _ in range(reps):
            tid += 1
            rows.append(
                Trade(
                    "binance_usdm",
                    "BTCUSDT",
                    T0 + tid * NS_PER_S,
                    T0 + tid * NS_PER_S,
                    price,
                    qty,
                    price * qty,
                    Side.BUY,
                    tid,
                )
            )
    prof = profile(records_to_frame(rows, "trade"), bin_size=5.0)
    nodes = hvn_lvn(prof, neighborhood=1)
    by_price = {row["price"]: row["node"] for row in nodes.iter_rows(named=True)}
    assert by_price[102.5] == "hvn"
    assert by_price[122.5] == "hvn"
    assert by_price[112.5] == "lvn"


def test_fractal_swings_planted_exactly():
    #             0    1    2    3    4    5    6    7    8
    highs = [10, 11, 15, 11, 10, 11, 18, 11, 10]
    lows = [9, 8, 9, 7, 9, 8, 9, 8, 9]
    bars = _bars_from_path(
        [(h - 0.5, h, lo, h - 0.5) for h, lo in zip(highs, lows, strict=True)]
    )
    sw = fractal_swings(bars, n=2)
    sw_h = sw.filter(pl.col("kind") == "high")
    sw_l = sw.filter(pl.col("kind") == "low")
    assert sw_h["price"].to_list() == [15.0, 18.0]
    assert sw_l["price"].to_list() == [7.0]
    # confirmation lags by n bars
    assert sw_h["ts_confirmed"].to_list() == [T0 + 5 * NS_PER_MIN, T0 + 9 * NS_PER_MIN]


def test_fractal_ties_do_not_fire():
    highs = [10, 12, 12, 12, 10]
    bars = _bars_from_path([(h - 1, h, h - 2, h - 1) for h in highs])
    sw = fractal_swings(bars, n=1)
    assert sw.filter(pl.col("kind") == "high").height == 0


def test_equal_extremes_with_eps():
    swings = pl.DataFrame(
        {
            "ts_open": [T0, T0 + NS_PER_MIN, T0 + 2 * NS_PER_MIN, T0 + 3 * NS_PER_MIN],
            "ts_confirmed": [T0 + NS_PER_MIN, T0 + 2 * NS_PER_MIN, T0 + 3 * NS_PER_MIN, T0 + 4 * NS_PER_MIN],
            "kind": ["high", "high", "high", "low"],
            "price": [100.0, 100.4, 150.0, 90.0],
        }
    )
    clusters = equal_extremes(swings, eps=0.5)
    assert clusters.height == 1  # only the two ~100 highs, lone swings don't cluster
    row = clusters.row(0, named=True)
    assert row["kind"] == "high"
    assert row["price"] == pytest.approx(100.2)
    assert row["count"] == 2
    # tighter eps splits them
    assert equal_extremes(swings, eps=0.1).height == 0


def test_level_weights_half_life():
    levels = pl.DataFrame(
        {"kind": ["high"], "price": [100.0], "count": [4], "ts_last": [T0]}
    )
    hl = 172_800.0
    w_now = level_weights(levels, now_ts=T0, half_life_s=hl)["weight"][0]
    w_later = level_weights(
        levels, now_ts=T0 + int(hl * NS_PER_S), half_life_s=hl
    )["weight"][0]
    assert w_now == pytest.approx(4.0)
    assert w_later == pytest.approx(2.0)


def test_swings_on_synth_never_use_future(trades):
    """A swing's confirmation ts is at least n bars after its bar."""
    from trading_system.features.bars import time_bars

    bars = time_bars(trades, "1m")
    sw = fractal_swings(bars, n=3)
    assert (
        sw.select(((pl.col("ts_confirmed") - pl.col("ts_open")) >= 3 * NS_PER_MIN).all()).item()
    )


def test_demo_reports(tmp_path):
    from trading_system.profile.reports import demo_reports

    paths = demo_reports(tmp_path, seed=42)
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.stat().st_size > 5_000


def test_va_poc_visual_reference_shape(trades):
    """Rough analogue of the manual TradingView check: unimodal synth data
    puts the POC near the densest traded price and VA covers it."""
    prof = profile(trades, bin_size=10.0)
    va = value_area(prof)
    densest = float(prof.sort("volume_usd", descending=True)["price"][0])
    assert va.val <= densest <= va.vah
    inside = prof.filter((pl.col("price") >= va.val) & (pl.col("price") <= va.vah))
    assert inside["volume_usd"].sum() / prof["volume_usd"].sum() == pytest.approx(
        va.share, rel=1e-9
    )
