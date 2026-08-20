"""M3 stress: pathological time/volume bars, ATR edge cases, sigma=0 volume
z-scores, huge-magnitude CVD/VWAP precision, asof-join boundary contracts,
multi-TF closed-bar causality and a 30-day 1m->1d full stack.

Sizes scale with env STRESS_SCALE (default 1). Perf assertions are loose
sanity floors; measured figures are printed as [stress-perf] lines (-s).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import polars as pl
import pytest

from trading_system.core.schema import POLARS_SCHEMAS
from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S, TIMEFRAME_NS
from trading_system.features.bars import time_bars, volume_bars, with_cvd
from trading_system.features.indicators import with_atr, with_vwap
from trading_system.features.joins import asof_join_backward, join_open_interest
from trading_system.features.multitf import build_multitf, join_context, tf_features

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))

EX = "binance_usdm"
SYM = "BTCUSDT"
T0 = 1_755_600_000 * NS_PER_S  # minute-aligned UTC ns
NS_PER_DAY = 86_400 * NS_PER_S


def _scaled(n: int, lo: int = 100) -> int:
    return max(lo, int(n * SCALE))


def _perf(msg: str) -> None:
    print(f"[stress-perf] {msg}")


def trade_frame(ts, price, qty, side) -> pl.DataFrame:
    """Vectorized trade-schema frame (fast path around record dataclasses)."""
    ts = np.asarray(ts, dtype=np.int64)
    price = np.asarray(price, dtype=np.float64)
    qty = np.asarray(qty, dtype=np.float64)
    n = len(ts)
    return pl.DataFrame(
        {
            "exchange": np.full(n, EX),
            "symbol": np.full(n, SYM),
            "ts_event": ts,
            "ts_recv": ts,
            "price": price,
            "qty": qty,
            "qty_usd": price * qty,
            "side": np.asarray(side, dtype=np.int8),
            "trade_id": np.arange(1, n + 1, dtype=np.int64),
        },
        schema=POLARS_SCHEMAS["trade"],
    )


def oi_frame(ts_values, oi_values) -> pl.DataFrame:
    n = len(ts_values)
    return pl.DataFrame(
        {
            "exchange": [EX] * n,
            "symbol": [SYM] * n,
            "ts_event": list(ts_values),
            "ts_recv": list(ts_values),
            "open_interest": list(oi_values),
            "open_interest_usd": [v * 100.0 for v in oi_values],
        },
        schema=POLARS_SCHEMAS["open_interest"],
    )


@pytest.fixture(scope="module")
def month_trades() -> pl.DataFrame:
    """~1e6 trades spread over 30 days (scaled by STRESS_SCALE)."""
    n = _scaled(1_000_000, lo=20_000)
    rng = np.random.default_rng(42)
    mean_gap = 30 * NS_PER_DAY // n
    gaps = np.maximum(1, rng.exponential(mean_gap, n)).astype(np.int64)
    ts = T0 + np.cumsum(gaps)
    price = 50_000.0 * np.exp(np.cumsum(rng.normal(0.0, 2e-5, n)))
    qty = np.round(rng.lognormal(-3.0, 1.2, n), 6) + 1e-4
    side = np.where(rng.random(n) < 0.5, 1, -1)
    return trade_frame(ts, price, qty, side)


class TestTimeBarsPathological:
    def test_all_trades_in_one_timestamp(self):
        n = _scaled(10_000, lo=1_000)
        rng = np.random.default_rng(1)
        price = 100.0 + rng.standard_normal(n).round(4)
        t = trade_frame(np.full(n, T0 + 7), price, np.ones(n), np.ones(n))
        bars = time_bars(t, "1m")
        assert bars.height == 1
        b = bars.row(0, named=True)
        assert b["open"] == price[0] and b["close"] == price[-1]  # trade_id order
        assert b["high"] == price.max() and b["low"] == price.min()
        assert b["n_trades"] == n and b["volume"] == pytest.approx(float(n))

    def test_single_trade(self):
        bars = time_bars(trade_frame([T0 + 5], [123.0], [2.0], [-1]), "1m")
        assert bars.height == 1
        b = bars.row(0, named=True)
        assert (b["open"], b["high"], b["low"], b["close"]) == (123.0,) * 4
        assert b["ts_open"] == T0 and b["ts_close"] == T0 + NS_PER_MIN
        assert b["delta"] == -2.0

    def test_week_gap_yields_no_synthetic_bars(self):
        # contract: empty buckets are skipped, never filled with flat bars
        week = 7 * NS_PER_DAY
        t = trade_frame([T0, T0 + week], [100.0, 200.0], [1.0, 1.0], [1, 1])
        bars = time_bars(t, "1m")
        assert bars.height == 2
        assert bars["ts_open"].to_list() == [T0, T0 + week]

    def test_non_monotonic_input_is_sorted_internally(self):
        # contract: time_bars sorts by (ts_event, trade_id); shuffled input
        # must produce the identical frame, not garbage and not an error
        n = 5_000
        rng = np.random.default_rng(3)
        ts = T0 + np.sort(rng.integers(0, 60 * NS_PER_MIN, n))
        t = trade_frame(ts, 100 + rng.standard_normal(n), np.ones(n), np.ones(n))
        shuffled = t.sample(fraction=1.0, shuffle=True, seed=9)
        assert time_bars(shuffled, "1m").equals(time_bars(t, "1m"))

    def test_throughput_one_million_trades(self, month_trades):
        n = month_trades.height
        t0 = time.perf_counter()
        bars = time_bars(month_trades, "1m")
        dt = time.perf_counter() - t0
        rate = n / dt
        _perf(f"time_bars 1m on {n:,} trades: {rate:,.0f} trades/s, {bars.height:,} bars")
        assert bars["volume"].sum() == pytest.approx(month_trades["qty"].sum(), rel=1e-9)
        assert bars.select((pl.col("high") >= pl.col("low")).all()).item()
        assert bars.select((pl.col("ts_close") - pl.col("ts_open") == NS_PER_MIN).all()).item()
        assert rate > 30_000


class TestVolumeBars:
    def test_threshold_exactly_equals_trade_volume(self):
        # qty_usd == threshold: the trade alone closes its bar
        ts = [T0 + i * NS_PER_S for i in range(4)]
        t = trade_frame(ts, [100.0] * 4, [10.0] * 4, [1] * 4)  # qty_usd = 1000 each
        vb = volume_bars(t, 1000.0)
        assert vb.height == 4
        assert vb["quote_volume"].to_list() == [1000.0] * 4
        assert vb["ts_close"].to_list() == [t + 1 for t in ts]  # last trade ts + 1

    def test_trade_ten_times_threshold_is_not_split(self):
        th = 1000.0
        t = trade_frame(
            [T0, T0 + NS_PER_S, T0 + 2 * NS_PER_S],
            [100.0] * 3,
            [5.0, 100.0, 4.0],  # 0.5x, 10x, 0.4x of threshold
            [1] * 3,
        )
        vb = volume_bars(t, th)
        assert vb.height == 2
        assert vb["quote_volume"][0] == pytest.approx(10.5 * th)  # giant closes bar 1
        assert vb["quote_volume"][1] == pytest.approx(0.4 * th)  # trailing partial
        # a leading giant closes a single-trade bar
        vb2 = volume_bars(trade_frame([T0], [100.0], [100.0], [1]), th)
        assert vb2.height == 1 and vb2["n_trades"][0] == 1
        assert vb2["quote_volume"][0] == pytest.approx(10 * th)

    def test_zero_volume_trades(self):
        # all-zero: never reaches the threshold -> one trailing partial bar
        ts = [T0 + i * NS_PER_S for i in range(6)]
        z = trade_frame(ts, [100.0] * 6, [0.0] * 6, [1] * 6)
        vb = volume_bars(z, 1000.0)
        assert vb.height == 1 and vb["quote_volume"][0] == 0.0
        # zeros interleaved with real volume do not open extra bars
        mix = trade_frame(
            ts, [100.0] * 6, [0.0, 10.0, 0.0, 0.0, 10.0, 0.0], [1] * 6
        )  # 1000 usd at #2 and #5
        vbm = volume_bars(mix, 1000.0)
        assert vbm.height == 3  # close at #2, close at #5, trailing zero trade
        assert vbm["quote_volume"].sum() == pytest.approx(2000.0)

    def test_conservation_on_one_million_trades(self, month_trades):
        th = 5_000_000.0
        t0 = time.perf_counter()
        vb = volume_bars(month_trades, th)
        dt = time.perf_counter() - t0
        _perf(
            f"volume_bars on {month_trades.height:,} trades: {dt:.2f}s, {vb.height:,} bars"
        )
        assert vb["quote_volume"].sum() == pytest.approx(month_trades["qty_usd"].sum(), rel=1e-9)
        # every closed bar reaches the threshold
        assert (vb["quote_volume"].head(vb.height - 1) >= th).all()
        assert dt < 60


class TestATREdges:
    @staticmethod
    def _const_bars(n: int) -> pl.DataFrame:
        ts = T0 + np.arange(n) * NS_PER_MIN
        return time_bars(trade_frame(ts, np.full(n, 100.0), np.ones(n), np.ones(n)), "1m")

    def test_constant_price_gives_zero_atr_no_nan(self):
        bars = with_atr(self._const_bars(200), period=14)
        assert bars["tr"].null_count() == 0 and bars["atr"].null_count() == 0
        assert not bars["atr"].is_nan().any()
        assert (bars["atr"] == 0.0).all() and (bars["tr"] == 0.0).all()

    def test_single_bar_series(self):
        bars = with_atr(self._const_bars(1), period=14)
        assert bars.height == 1
        assert bars["tr"][0] == 0.0 and bars["atr"][0] == 0.0

    def test_period_longer_than_series(self):
        n = 5
        ts = T0 + np.arange(n) * NS_PER_MIN
        prices = np.array([100.0, 101.0, 99.5, 102.0, 100.5])
        bars = with_atr(time_bars(trade_frame(ts, prices, np.ones(n), np.ones(n)), "1m"), 500)
        assert bars["atr"].null_count() == 0 and not bars["atr"].is_nan().any()
        assert (bars["atr"] >= 0.0).all()
        assert bars["atr"].max() <= bars["tr"].max() + 1e-12


class TestVolZSigmaZero:
    @staticmethod
    def _flat_frame(n: int, spike_at: int | None = None) -> pl.DataFrame:
        ts = T0 + np.arange(n) * NS_PER_MIN
        qty = np.full(n, 2.0)
        if spike_at is not None:
            qty[spike_at] = 5.0
        return trade_frame(ts, np.full(n, 100.0), qty, np.ones(n))

    def test_identical_volumes_yield_null_not_nan(self):
        f = tf_features(self._flat_frame(40), None, "1m", zscore_window=8)
        z = f["vol_z"]
        assert z.null_count() == f.height  # sigma=0 -> z undefined, all null
        assert not z.is_nan().any() and not z.is_infinite().any()

    def test_spike_over_flat_baseline_never_inf(self):
        # regression: used to produce +inf ((qv - mean) / 0)
        f = tf_features(self._flat_frame(40, spike_at=30), None, "1m", zscore_window=8)
        z = f["vol_z"]
        assert not z.is_infinite().any() and not z.is_nan().any()
        assert z[30] is None  # sigma=0 baseline -> null by contract
        # degenerate impulse contract: any qv above a flat baseline fires
        assert f["impulse"][30] is True

    def test_varying_baseline_still_produces_finite_z(self):
        n = 60
        ts = T0 + np.arange(n) * NS_PER_MIN
        rng = np.random.default_rng(4)
        qty = 2.0 + 0.1 * rng.standard_normal(n)
        qty[50] = 50.0
        f = tf_features(
            trade_frame(ts, np.full(n, 100.0), qty, np.ones(n)), None, "1m", zscore_window=16
        )
        z = f["vol_z"].drop_nulls()
        assert not z.is_infinite().any() and not z.is_nan().any()
        assert f["vol_z"][50] > 10  # the planted spike is detected


class TestHugeMagnitudes:
    def test_cvd_at_1e12_volumes(self):
        n = _scaled(4_000, lo=500)
        rng = np.random.default_rng(5)
        ts = T0 + np.arange(n) * NS_PER_MIN
        qty = 1e8 * (1.0 + rng.random(n))
        side = np.where(rng.random(n) < 0.55, 1, -1)
        price = np.full(n, 10_000.0)  # qty_usd ~ 1e12 per trade
        t = trade_frame(ts, price, qty, side)
        bars = with_cvd(time_bars(t, "1m"))
        assert not bars["cvd"].is_infinite().any() and not bars["cvd_usd"].is_infinite().any()
        ref = math.fsum(float(q) * int(s) for q, s in zip(qty, side, strict=True))
        ref_usd = math.fsum(float(p * q) * int(s) for p, q, s in zip(price, qty, side, strict=True))
        assert bars["cvd"][-1] == pytest.approx(ref, rel=1e-9)
        assert bars["cvd_usd"][-1] == pytest.approx(ref_usd, rel=1e-9)

    def test_vwap_at_1e12_volumes(self):
        n = _scaled(2_000, lo=500)
        rng = np.random.default_rng(6)
        ts = T0 + np.arange(n) * NS_PER_MIN
        price = 10_000.0 * (1.0 + 0.01 * rng.standard_normal(n))
        qty = 1e8 * (1.0 + rng.random(n))
        bars = with_vwap(time_bars(trade_frame(ts, price, qty, np.ones(n)), "1m"), session="1d")
        # single-trade bars: (p*q)/q may differ from p by an ulp — allow it
        assert (
            bars.select(
                (
                    (pl.col("vwap_bar") >= pl.col("low") * (1 - 1e-12))
                    & (pl.col("vwap_bar") <= pl.col("high") * (1 + 1e-12))
                ).all()
            ).item()
        )
        assert not bars["vwap_session"].is_infinite().any()
        # first session: cumulative vwap of the last bar vs an fsum reference
        day0 = bars.filter(pl.col("ts_open") < ((T0 // NS_PER_DAY) + 1) * NS_PER_DAY)
        joined = day0.join(
            time_bars(trade_frame(ts, price, qty, np.ones(n)), "1m"), on="ts_open", how="inner"
        )
        ref = math.fsum(joined["quote_volume"].to_list()) / math.fsum(joined["volume"].to_list())
        assert day0["vwap_session"][-1] == pytest.approx(ref, rel=1e-9)


class TestAsofJoinContracts:
    @staticmethod
    def _bars(n_min: int = 5) -> pl.DataFrame:
        ts = T0 + np.arange(n_min) * NS_PER_MIN
        return time_bars(trade_frame(ts, np.full(n_min, 100.0), np.ones(n_min), np.ones(n_min)),
                         "1m")

    def test_duplicate_right_ts_last_wins(self):
        bars = self._bars()
        point = T0 + NS_PER_MIN + 5
        oi = oi_frame([point, point, point], [1.0, 2.0, 3.0])
        j = asof_join_backward(bars, oi, ["open_interest"])
        # contract: with equal ts_event the LAST row of the duplicates wins
        assert j["open_interest"].to_list() == [None, 3.0, 3.0, 3.0, 3.0]

    def test_empty_right_gives_nulls_not_error(self):
        bars = self._bars()
        empty = pl.DataFrame(schema=POLARS_SCHEMAS["open_interest"])
        j = join_open_interest(bars, empty)
        assert j.height == bars.height
        for c in ("open_interest", "open_interest_usd", "d_oi", "oi_speed_usd_per_min"):
            assert j[c].null_count() == bars.height

    def test_right_starting_after_left(self):
        bars = self._bars(6)
        # the point lands inside bar 4 (ts_close = T0+5m): visible from bar 4 on
        oi = oi_frame([T0 + 4 * NS_PER_MIN + 1], [7.0])
        j = join_open_interest(bars, oi)
        assert j["open_interest"].to_list() == [None] * 4 + [7.0, 7.0]
        speed = j["oi_speed_usd_per_min"].drop_nulls()
        assert not speed.is_infinite().any()

    def test_strictly_backward_at_equal_ts_boundary(self):
        bars = self._bars(6)
        edge = int(bars["ts_close"][2])
        # one point exactly at a bar close, one just before the next close
        oi = oi_frame([edge, edge + NS_PER_MIN - 1], [10.0, 20.0])
        j = asof_join_backward(bars, oi, ["open_interest"])
        got = j["open_interest"].to_list()
        assert got[2] is None  # ts == ts_close belongs to the NEXT bar
        assert got[3] == 20.0  # ts == ts_close-1 is visible to this bar
        assert got[4] == 20.0

    def test_duplicates_at_boundary_stay_causal(self):
        bars = self._bars(6)
        edge = int(bars["ts_close"][2])
        oi = oi_frame([edge - 1, edge, edge], [5.0, 6.0, 7.0])
        j = asof_join_backward(bars, oi, ["open_interest"])
        got = j["open_interest"].to_list()
        assert got[2] == 5.0  # only the pre-boundary point is visible
        assert got[3] == 7.0  # both boundary duplicates arrive next bar; last wins


class TestMultiTFCausality:
    def test_higher_tf_without_a_single_closed_bar(self):
        # 2 hours of trades: the 1d bar never closes inside the base range
        n = 120
        ts = T0 + np.arange(n) * NS_PER_MIN
        t = trade_frame(ts, np.full(n, 100.0), np.ones(n), np.ones(n))
        base = time_bars(t, "1m")
        mtf = build_multitf(t, None, ["1d"], zscore_window=8)
        joined = join_context(base, mtf, ["1d"])
        assert joined.height == base.height
        assert joined["1d_quote_volume"].null_count() == base.height

    def test_exactly_one_closed_higher_bar(self):
        # one trade per hour over 2 days; day-2 base bars see day-1 stats only
        day_start = ((T0 // NS_PER_DAY) + 1) * NS_PER_DAY
        ts = day_start + np.arange(48) * (NS_PER_S * 3_600)
        qty = np.where(np.arange(48) < 24, 1.0, 3.0)
        t = trade_frame(ts, np.full(48, 100.0), qty, np.ones(48))
        base = time_bars(t, "1m")
        mtf = build_multitf(t, None, ["1d"], zscore_window=8)
        joined = join_context(base, mtf, ["1d"])
        day1_qv = mtf.filter(pl.col("ts_open") == day_start)["quote_volume"][0]
        in_day1 = joined.filter(pl.col("ts_close") <= day_start + NS_PER_DAY)
        in_day2 = joined.filter(pl.col("ts_close") > day_start + NS_PER_DAY)
        assert in_day1["1d_quote_volume"].null_count() == in_day1.height
        assert in_day2.height > 0
        assert in_day2["1d_quote_volume"].to_list() == [day1_qv] * in_day2.height

    def test_boundary_base_bar_sees_tf_bar_closing_at_same_instant(self):
        # base bar closing exactly at a 5m boundary gets that 5m bar (both
        # horizons end at the boundary: same information, no lookahead)
        n = 10
        ts = T0 + np.arange(n) * NS_PER_MIN
        t = trade_frame(ts, np.full(n, 100.0), np.ones(n), np.ones(n))
        base = time_bars(t, "1m")
        mtf = build_multitf(t, None, ["5m"], zscore_window=8)
        joined = join_context(base, mtf, ["5m"])
        qv5 = mtf.sort("ts_open")["quote_volume"].to_list()  # two 5m bars
        got = joined["5m_quote_volume"].to_list()
        assert got[:4] == [None] * 4  # first 5m bar not closed yet
        assert got[4] == qv5[0]  # closes exactly with the 5th minute
        assert got[5:9] == [qv5[0]] * 4
        assert got[9] == qv5[1]

    def test_full_stack_30_days(self, month_trades):
        tfs = ["1m", "5m", "15m", "1h", "4h", "1d"]
        n = month_trades.height
        t0 = time.perf_counter()
        base = time_bars(month_trades, "1m")
        mtf = build_multitf(month_trades, None, tfs, zscore_window=96)
        joined = join_context(base, mtf, tfs[1:])
        dt = time.perf_counter() - t0
        _perf(
            f"multitf 1m->1d stack on {n:,} trades / {base.height:,} base bars: "
            f"{dt:.2f}s ({n / dt:,.0f} trades/s)"
        )
        assert joined.height == base.height
        for tf in tfs[1:]:
            assert f"{tf}_quote_volume" in joined.columns
            assert f"{tf}_vol_z" in joined.columns
        # z-scores from the stack must never be inf (sigma=0 guard)
        for tf in tfs[1:]:
            col = joined[f"{tf}_vol_z"].drop_nulls()
            assert not col.is_infinite().any() and not col.is_nan().any()
        # late bars have full higher-TF context
        assert joined.tail(10)["1d_quote_volume"].null_count() == 0
        # sampled causality check against the raw 5m features
        five = mtf.filter(pl.col("tf") == "5m").sort("ts_close")
        sample = joined.gather_every(max(1, joined.height // 20))
        for row in sample.iter_rows(named=True):
            closed = five.filter(pl.col("ts_close") <= row["ts_close"])
            if closed.height:
                assert row["5m_quote_volume"] == closed["quote_volume"][-1]
            else:
                assert row["5m_quote_volume"] is None
        assert dt < 75
        assert TIMEFRAME_NS["1d"] // TIMEFRAME_NS["1m"] == 1_440  # stack really spans 1m->1d
