"""Own recording vs Vision reconciliation (1.2)."""

from __future__ import annotations

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_system.collectors.vision import (
    reconcile,
    reconcile_klines,
    reconcile_trades,
)
from trading_system.core.schema import POLARS_SCHEMAS, records_to_frame
from trading_system.core.synth import synth_trades
from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S

T0 = 1_704_067_200 * NS_PER_S


def trades_frame(minute_qtys: list[list[float]]) -> pl.DataFrame:
    rows = []
    tid = 0
    for m, qtys in enumerate(minute_qtys):
        for i, q in enumerate(qtys):
            ts = T0 + m * NS_PER_MIN + (i % 60) * NS_PER_S
            tid += 1
            rows.append(
                {
                    "exchange": "binance_usdm",
                    "symbol": "BTCUSDT",
                    "ts_event": ts,
                    "ts_recv": ts,
                    "price": 100.0,
                    "qty": q,
                    "qty_usd": 100.0 * q,
                    "side": 1,
                    "trade_id": tid,
                }
            )
    return pl.DataFrame(rows, schema=POLARS_SCHEMAS["trade"])


def klines_frame(closes: list[float]) -> pl.DataFrame:
    rows = []
    for i, c in enumerate(closes):
        rows.append(
            {
                "exchange": "binance_usdm",
                "symbol": "BTCUSDT",
                "ts_open": T0 + i * NS_PER_MIN,
                "ts_close": T0 + (i + 1) * NS_PER_MIN - 1_000_000,
                "open": c - 1,
                "high": c + 2,
                "low": c - 2,
                "close": c,
                "volume": 10.0 + i,
                "quote_volume": (10.0 + i) * c,
                "taker_buy_volume": 5.0,
                "taker_buy_quote_volume": 5.0 * c,
                "n_trades": 100,
                "closed": True,
            }
        )
    return pl.DataFrame(rows, schema=POLARS_SCHEMAS["kline"])


def test_identical_trades_pass():
    own = trades_frame([[0.1, 0.2], [0.3], [0.4, 0.5, 0.6]])
    res = reconcile_trades(own, own)
    assert res.height == 3
    assert res.get_column("ok").all()
    assert res.get_column("count_rel_diff").max() == 0.0
    assert res.get_column("volume_rel_diff").max() == 0.0


def test_synth_trades_identical_pass():
    own = records_to_frame(synth_trades(n=2_000, seed=7), "trade")
    res = reconcile_trades(own, own)
    assert res.get_column("ok").all()


def test_perturbed_volume_flagged_in_the_right_minute():
    own = trades_frame([[0.1, 0.2], [0.3], [0.4]])
    vision = own.with_columns(
        pl.when(
            (pl.col("ts_event") >= T0 + NS_PER_MIN)
            & (pl.col("ts_event") < T0 + 2 * NS_PER_MIN)
        )
        .then(pl.col("qty") * 1.05)
        .otherwise(pl.col("qty"))
        .alias("qty")
    )
    res = reconcile_trades(own, vision)
    bad = res.filter(~pl.col("ok"))
    assert bad.height == 1
    assert bad.get_column("minute")[0] == T0 + NS_PER_MIN
    assert bad.get_column("volume_rel_diff")[0] == pytest.approx(0.05 / 1.05)


def test_missing_minute_flagged():
    own = trades_frame([[0.1], [0.2], [0.3]])
    vision = own.filter(pl.col("ts_event") >= T0 + NS_PER_MIN)  # first minute lost
    res = reconcile_trades(own, vision)
    bad = res.filter(~pl.col("ok"))
    assert bad.get_column("minute").to_list() == [T0]
    assert bad.get_column("count_rel_diff")[0] == pytest.approx(1.0)


def test_kline_identical_and_perturbed():
    own = klines_frame([100.0, 101.0, 102.0])
    assert reconcile_klines(own, own).get_column("ok").all()
    vision = own.with_columns(
        pl.when(pl.col("ts_open") == T0)
        .then(pl.col("close") * 1.001)
        .otherwise(pl.col("close"))
        .alias("close")
    )
    strict = reconcile_klines(own, vision)
    assert not strict.get_column("ok").all()
    assert strict.filter(~pl.col("ok")).height == 1
    loose = reconcile_klines(own, vision, rel_tol=1e-2)
    assert loose.get_column("ok").all()


def test_kline_missing_bar_flagged_even_within_tolerance():
    own = klines_frame([100.0, 101.0])
    vision = own.filter(pl.col("ts_open") != T0)
    res = reconcile_klines(own, vision, rel_tol=1.0)
    assert not res.filter(pl.col("ts_open") == T0).get_column("ok")[0]


def test_reconcile_verdict():
    own_t = trades_frame([[0.1, 0.2], [0.3]])
    own_k = klines_frame([100.0, 101.0])
    good = reconcile(own_t, own_t, own_k, own_k)
    assert good.passed
    perturbed = own_t.with_columns((pl.col("qty") * 1.01).alias("qty"))
    bad = reconcile(own_t, perturbed, own_k, own_k)
    assert not bad.passed
    assert not bad.trades.get_column("ok").all()
    assert bad.klines.get_column("ok").all()


@settings(max_examples=25, deadline=None, derandomize=True)
@given(
    qtys=st.lists(
        st.lists(
            st.floats(0.001, 100.0, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=5,
        ),
        min_size=1,
        max_size=5,
    ),
    eps=st.floats(0.01, 2.0, allow_nan=False, allow_infinity=False),
)
def test_property_identity_passes_and_perturbation_is_flagged(qtys, eps):
    own = trades_frame(qtys)
    assert reconcile_trades(own, own, volume_tol=1e-6).get_column("ok").all()
    perturbed = own.with_columns(
        pl.when(pl.col("ts_event") < T0 + NS_PER_MIN)
        .then(pl.col("qty") * (1.0 + eps))
        .otherwise(pl.col("qty"))
        .alias("qty")
    )
    res = reconcile_trades(own, perturbed, volume_tol=1e-6)
    assert not res.filter(pl.col("minute") == T0).get_column("ok")[0]
