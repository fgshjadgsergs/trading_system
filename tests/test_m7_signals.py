"""M7: synthetic scenarios with known answers — each detector fires exactly once
in the designated bar; detectors are stateless and lookahead-free."""

from __future__ import annotations

import polars as pl
import pytest

from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S
from trading_system.signals.detectors import s1_magnet, s2_sweep_reversal, s3_filter

T0 = 1_755_600_000 * NS_PER_S


def make_bars(ohlc: list[tuple[float, float, float, float]], atr: float = 10.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_open": [T0 + i * NS_PER_MIN for i in range(len(ohlc))],
            "ts_close": [T0 + (i + 1) * NS_PER_MIN for i in range(len(ohlc))],
            "open": [r[0] for r in ohlc],
            "high": [r[1] for r in ohlc],
            "low": [r[2] for r in ohlc],
            "close": [r[3] for r in ohlc],
            "atr": [atr] * len(ohlc),
        }
    )


def pools_frame(rows: list[tuple[float, float, int | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "price": [r[0] for r in rows],
            "heat_usd": [r[1] for r in rows],
            "touched_ts": pl.Series([r[2] for r in rows], dtype=pl.Int64),
        }
    )


# -- S1 magnet ----------------------------------------------------------------


def test_s1_fires_once_when_pool_enters_range():
    # price walks down toward a big pool at 60; k*ATR = 30
    closes = [130.0, 120.0, 100.0, 95.0, 90.0]
    bars = make_bars([(c + 1, c + 2, c - 2, c) for c in closes], atr=10.0)
    pools = pools_frame([(60.0, 5e6, None), (200.0, 4e6, None)])
    ev = s1_magnet(bars, pools, k_atr=3.0, min_heat_share=0.25)
    assert ev.height == 1  # exactly once, despite staying in range afterwards
    row = ev.row(0, named=True)
    # first close within k*ATR=30 of the pool at 60 is close=90 (bar index 4)
    assert row["ts"] == T0 + 5 * NS_PER_MIN
    assert row["side"] == -1  # pool below -> magnet pulls down -> short
    assert row["target"] == 60.0


def test_s1_ignores_touched_pools():
    closes = [90.0, 88.0, 86.0]
    bars = make_bars([(c + 1, c + 2, c - 2, c) for c in closes], atr=10.0)
    touched_at = T0 + 1  # touched before all bar closes
    pools = pools_frame([(80.0, 5e6, touched_at)])
    assert s1_magnet(bars, pools, k_atr=3.0).height == 0


def test_s1_pool_touched_midway_stops_being_target():
    closes = [200.0, 90.0, 88.0]
    bars = make_bars([(c + 1, c + 2, c - 2, c) for c in closes], atr=10.0)
    pools = pools_frame([(80.0, 5e6, T0 + 2 * NS_PER_MIN)])  # dies after bar 2 close... touched at bar-2 close
    ev = s1_magnet(bars, pools, k_atr=3.0)
    # in range at bar 2 close (ts = T0+2m) but touched_ts <= ts kills it there; bar 1 close=90 at ts T0+2m?
    # bar closes: ts T0+1m (200), T0+2m (90), T0+3m (88). Pool touched at T0+2m -> only bar at T0+1m may fire (out of range).
    assert ev.height == 0


def test_s1_small_pool_not_a_magnet():
    closes = [90.0]
    bars = make_bars([(c, c + 1, c - 1, c) for c in closes], atr=10.0)
    pools = pools_frame([(85.0, 1e5, None), (300.0, 9e6, None)])  # in-range pool is 1% of heat
    assert s1_magnet(bars, pools, k_atr=3.0, min_heat_share=0.25).height == 0


def test_s1_prefix_consistency():
    closes = [130.0, 120.0, 100.0, 95.0, 90.0, 85.0, 70.0]
    bars = make_bars([(c + 1, c + 2, c - 2, c) for c in closes], atr=10.0)
    pools = pools_frame([(60.0, 5e6, None)])
    full = s1_magnet(bars, pools, k_atr=3.0)
    for cut in range(1, len(closes) + 1):
        pref = s1_magnet(bars.head(cut), pools, k_atr=3.0)
        expected = full.filter(pl.col("ts") <= T0 + cut * NS_PER_MIN)
        assert pref.equals(expected)


# -- S2 sweep-reversal --------------------------------------------------------

LEVEL = pl.DataFrame({"kind": ["high"], "price": [110.0], "count": [3]})


def test_s2_fires_once_at_designated_bar():
    #        0: below   1: pierce (high 112)   2: reversal close 99 < level & < low[1]
    ohlc = [
        (100.0, 105.0, 98.0, 104.0),
        (104.0, 112.0, 103.0, 108.0),
        (108.0, 109.0, 98.5, 99.0),
        (99.0, 101.0, 97.0, 100.0),
    ]
    bars = make_bars(ohlc)
    ev = s2_sweep_reversal(bars, LEVEL, return_bars=3)
    assert ev.height == 1
    row = ev.row(0, named=True)
    assert row["ts"] == T0 + 3 * NS_PER_MIN  # bar index 2
    assert row["side"] == -1
    assert row["meta"] == 110.0


def test_s2_no_return_no_signal():
    ohlc = [
        (100.0, 105.0, 98.0, 104.0),
        (104.0, 112.0, 103.0, 111.0),  # pierce and hold above
        (111.0, 115.0, 110.5, 114.0),
        (114.0, 118.0, 113.0, 117.0),
    ]
    assert s2_sweep_reversal(make_bars(ohlc), LEVEL, return_bars=3).height == 0


def test_s2_return_without_structure_shift_no_signal():
    # closes back under level but NOT below the sweep bar's low (103)
    ohlc = [
        (100.0, 105.0, 98.0, 104.0),
        (104.0, 112.0, 103.0, 108.0),
        (108.0, 110.0, 104.0, 106.0),  # close 106 < 110 but > low[1]=103
        (106.0, 108.0, 104.5, 107.0),
    ]
    assert s2_sweep_reversal(make_bars(ohlc), LEVEL, return_bars=3).height == 0


def test_s2_mirrored_low_sweep():
    level = pl.DataFrame({"kind": ["low"], "price": [90.0], "count": [2]})
    ohlc = [
        (95.0, 97.0, 91.0, 96.0),
        (96.0, 97.0, 88.0, 92.0),  # pierce below 90, high 97
        (92.0, 99.0, 91.5, 98.0),  # close 98 > 90 and > high[1]=97 -> long signal
    ]
    ev = s2_sweep_reversal(make_bars(ohlc), level, return_bars=3)
    assert ev.height == 1
    assert ev["side"][0] == 1


def test_s2_stateless_same_input_same_output():
    ohlc = [
        (100.0, 105.0, 98.0, 104.0),
        (104.0, 112.0, 103.0, 108.0),
        (108.0, 109.0, 98.5, 99.0),
    ]
    bars = make_bars(ohlc)
    a = s2_sweep_reversal(bars, LEVEL)
    b = s2_sweep_reversal(bars, LEVEL)
    assert a.equals(b)


def test_s2_prefix_consistency():
    ohlc = [
        (100.0, 105.0, 98.0, 104.0),
        (104.0, 112.0, 103.0, 108.0),
        (108.0, 109.0, 98.5, 99.0),
        (99.0, 111.5, 97.0, 100.0),
        (100.0, 113.0, 99.0, 108.0),
        (108.0, 109.0, 95.0, 96.0),
    ]
    bars = make_bars(ohlc)
    full = s2_sweep_reversal(bars, LEVEL, return_bars=3)
    for cut in range(1, len(ohlc) + 1):
        pref = s2_sweep_reversal(bars.head(cut), LEVEL, return_bars=3)
        expected = full.filter(pl.col("ts") <= T0 + cut * NS_PER_MIN)
        assert pref.equals(expected), f"prefix {cut} diverges"


# -- S3 filter ----------------------------------------------------------------


def _one_event(price: float, target: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [T0],
            "signal": ["s1"],
            "side": [1 if target > price else -1],
            "price": [price],
            "target": [target],
            "meta": [0.0],
        },
        schema_overrides={"side": pl.Int8},
    )


def test_s3_blocks_path_through_dense_zone():
    zones = pl.DataFrame(
        {"lo": [104.0, 130.0], "hi": [106.0, 132.0], "heat_usd": [9e6, 1e5]}
    )
    ev = s3_filter(_one_event(100.0, 110.0), zones, dense_quantile=0.5)
    assert ev["blocked"][0]
    assert ev["block_zone"][0] == pytest.approx(105.0)


def test_s3_passes_when_zone_off_path_or_sparse():
    zones = pl.DataFrame({"lo": [130.0], "hi": [132.0], "heat_usd": [9e6]})
    ev = s3_filter(_one_event(100.0, 110.0), zones, dense_quantile=0.5)
    assert not ev["blocked"][0]
    sparse = pl.DataFrame(
        {"lo": [104.0, 200.0], "hi": [106.0, 210.0], "heat_usd": [1e3, 9e9]}
    )
    ev2 = s3_filter(_one_event(100.0, 110.0), sparse, dense_quantile=0.9)
    assert not ev2["blocked"][0]


def test_s3_empty_events_ok():
    ev = s3_filter(
        _one_event(100.0, 110.0).head(0),
        pl.DataFrame({"lo": [1.0], "hi": [2.0], "heat_usd": [1.0]}),
    )
    assert ev.height == 0 and "blocked" in ev.columns


def test_demo_reports(tmp_path):
    from trading_system.signals.reports import demo_reports

    paths = demo_reports(tmp_path, seed=42)
    assert len(paths) == 1
    assert paths[0].exists() and paths[0].stat().st_size > 5_000
