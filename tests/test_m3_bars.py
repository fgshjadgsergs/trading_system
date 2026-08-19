"""M3: time/volume bars, CVD — manual fixtures and kline cross-check."""

from __future__ import annotations

import polars as pl
import pytest

from trading_system.core.schema import Side, Trade, records_to_frame
from trading_system.core.synth import synth_trades
from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S
from trading_system.features.bars import compare_klines, time_bars, volume_bars, with_cvd

T0 = 1_755_600_000 * NS_PER_S  # aligned to a minute boundary


def _trade(ts: int, price: float, qty: float, side: Side, tid: int) -> Trade:
    return Trade(
        exchange="binance_usdm",
        symbol="BTCUSDT",
        ts_event=ts,
        ts_recv=ts,
        price=price,
        qty=qty,
        qty_usd=price * qty,
        side=side,
        trade_id=tid,
    )


@pytest.fixture()
def manual_trades() -> pl.DataFrame:
    rows = [
        _trade(T0 + 1 * NS_PER_S, 100.0, 1.0, Side.BUY, 1),
        _trade(T0 + 20 * NS_PER_S, 105.0, 2.0, Side.SELL, 2),
        _trade(T0 + 59 * NS_PER_S, 95.0, 1.0, Side.BUY, 3),
        # second minute
        _trade(T0 + 61 * NS_PER_S, 96.0, 0.5, Side.SELL, 4),
        _trade(T0 + 119 * NS_PER_S, 97.0, 0.5, Side.BUY, 5),
    ]
    return records_to_frame(rows, "trade")


def test_time_bars_manual(manual_trades):
    bars = time_bars(manual_trades, "1m")
    assert bars.height == 2
    b0 = bars.row(0, named=True)
    assert b0["ts_open"] == T0 and b0["ts_close"] == T0 + NS_PER_MIN
    assert (b0["open"], b0["high"], b0["low"], b0["close"]) == (100.0, 105.0, 95.0, 95.0)
    assert b0["volume"] == pytest.approx(4.0)
    assert b0["taker_buy_volume"] == pytest.approx(2.0)
    assert b0["n_trades"] == 3
    # delta = +1 - 2 + 1
    assert b0["delta"] == pytest.approx(0.0)
    b1 = bars.row(1, named=True)
    assert (b1["open"], b1["close"], b1["n_trades"]) == (96.0, 97.0, 2)
    assert b1["delta"] == pytest.approx(0.0)
    assert b1["delta_usd"] == pytest.approx(-96.0 * 0.5 + 97.0 * 0.5)


def test_cvd_accumulates(manual_trades):
    bars = with_cvd(time_bars(manual_trades, "1m"))
    assert bars["cvd"].to_list() == pytest.approx([0.0, 0.0])
    assert bars["cvd_usd"].to_list()[1] == pytest.approx(
        (100.0 - 210.0 + 95.0) + (-48.0 + 48.5)
    )


def test_empty_minutes_skipped(manual_trades):
    sparse = manual_trades.with_columns(
        pl.when(pl.col("trade_id") > 3)
        .then(pl.col("ts_event") + 10 * NS_PER_MIN)
        .otherwise(pl.col("ts_event"))
        .alias("ts_event")
    )
    bars = time_bars(sparse, "1m")
    assert bars.height == 2  # no synthetic flat bars in between
    assert bars["ts_open"][1] - bars["ts_open"][0] == 11 * NS_PER_MIN


def test_volume_bars_threshold():
    trades = records_to_frame(synth_trades(n=5_000, seed=7), "trade")
    threshold = 200_000.0
    vb = volume_bars(trades, threshold)
    qv = vb["quote_volume"]
    # every bar except possibly the last reaches the threshold
    assert (qv.head(vb.height - 1) >= threshold).all()
    # overshoot only shrinks the count; nothing is lost
    total = trades["qty_usd"].sum()
    assert 1 <= vb.height - 1 <= total // threshold
    assert qv.sum() == pytest.approx(total)


def test_own_bars_match_exchange_klines():
    """Cross-check against an independently built kline (simulated exchange)."""
    trades = records_to_frame(synth_trades(n=20_000, seed=11), "trade")
    # independent naive implementation: plain python loop
    buckets: dict[int, list] = {}
    for row in trades.sort("ts_event", "trade_id").iter_rows(named=True):
        key = row["ts_event"] - row["ts_event"] % NS_PER_MIN
        buckets.setdefault(key, []).append(row)
    exchange_rows = []
    for ts_open in sorted(buckets):
        rs = buckets[ts_open]
        exchange_rows.append(
            {
                "exchange": "binance_usdm",
                "symbol": "BTCUSDT",
                "ts_open": ts_open,
                "open": rs[0]["price"],
                "high": max(r["price"] for r in rs),
                "low": min(r["price"] for r in rs),
                "close": rs[-1]["price"],
                "volume": sum(r["qty"] for r in rs),
                "taker_buy_volume": sum(r["qty"] for r in rs if r["side"] == 1),
            }
        )
    exchange = pl.DataFrame(exchange_rows)
    own = time_bars(trades, "1m")
    cmp = compare_klines(own, exchange)
    assert cmp.height == own.height  # same bar boundaries — no timezone drift
    assert cmp["ok"].all()


def test_day_boundary_utc():
    day_ns = 86_400 * NS_PER_S
    t_midnight = (T0 // day_ns + 1) * day_ns
    rows = [
        _trade(t_midnight - 1, 100.0, 1.0, Side.BUY, 1),
        _trade(t_midnight, 101.0, 1.0, Side.BUY, 2),
    ]
    bars = time_bars(records_to_frame(rows, "trade"), "1d")
    assert bars.height == 2
    assert bars["ts_open"].to_list() == [t_midnight - day_ns, t_midnight]
