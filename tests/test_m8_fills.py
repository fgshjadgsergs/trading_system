"""M8 fill-model tests: strict-cross limits, latency gating, taker pricing, book walk."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trading_system.backtest.engine import (
    BacktestConfig,
    Bar,
    Context,
    Order,
    OrderType,
    TradePrint,
    run_backtest,
)
from trading_system.backtest.fills import (
    LatencyModel,
    impact_bps,
    limit_crossed,
    market_fill_price,
    walk_book,
)
from trading_system.core.schema import POLARS_SCHEMAS, MarkPrice, Side, records_to_frame
from trading_system.core.timeutils import NS_PER_S


def mk_trades(rows: list[tuple[float, float, float, int]]) -> pl.DataFrame:
    """Rows of (ts_seconds, price, qty, side) -> unified trade frame."""
    dicts = [
        {
            "exchange": "binance_usdm",
            "symbol": "BTCUSDT",
            "ts_event": int(ts * NS_PER_S),
            "ts_recv": int(ts * NS_PER_S),
            "price": float(p),
            "qty": float(q),
            "qty_usd": float(p * q),
            "side": int(s),
            "trade_id": i + 1,
        }
        for i, (ts, p, q, s) in enumerate(rows)
    ]
    return pl.DataFrame(dicts, schema=POLARS_SCHEMAS["trade"])


class PlaceOnce:
    """Places a fixed order list at the close of the first bar."""

    def __init__(self, orders: list[Order]) -> None:
        self.orders = orders
        self.placed = False

    def on_bar(self, bar: Bar, ctx: Context) -> list[Order]:
        if not self.placed:
            self.placed = True
            return self.orders
        return []


def frictionless(**overrides) -> BacktestConfig:
    kwargs = dict(
        latency_ms_min=0.0,
        latency_ms_max=0.0,
        half_spread_bps=0.0,
        impact_coef_bps=0.0,
        taker_fee=5e-4,
        maker_fee=2e-4,
        bar_ns=NS_PER_S,
        seed=42,
    )
    kwargs.update(overrides)
    return BacktestConfig(**kwargs)


# -- pure fill functions ----------------------------------------------------


def test_impact_bps_grows_with_notional_and_caps():
    assert impact_bps(0.0, 1e6, 10.0, 25.0) == 0.0
    small = impact_bps(1e4, 1e6, 10.0, 25.0)
    big = impact_bps(1e5, 1e6, 10.0, 25.0)
    assert 0 < small < big
    assert big == pytest.approx(10.0 * 1e5 / 1e6)
    assert impact_bps(1e9, 1e6, 10.0, 25.0) == 25.0  # capped
    assert impact_bps(1e4, 0.0, 10.0, 25.0) == 25.0  # empty window -> cap


def test_market_fill_price_directions():
    assert market_fill_price(Side.BUY, 100.0, 10.0, 5.0) == pytest.approx(100.0 * 1.0015)
    assert market_fill_price(Side.SELL, 100.0, 10.0, 5.0) == pytest.approx(100.0 * 0.9985)


def test_walk_book_vwap_and_overflow():
    levels = [(105.0, 0.5), (106.0, 0.5), (107.0, 10.0)]
    assert walk_book(levels, 1.5) == pytest.approx((105.0 + 106.0 + 107.0) * 0.5 / 1.5)
    # remainder beyond depth pays the worst provided price
    assert walk_book([(105.0, 1.0)], 2.0) == pytest.approx((105.0 + 105.0) / 2.0)
    with pytest.raises(ValueError):
        walk_book([], 1.0)


def test_limit_crossed_is_strict():
    assert not limit_crossed(Side.BUY, 100.0, 100.0)  # touch is not enough
    assert limit_crossed(Side.BUY, 100.0, 99.99)
    assert not limit_crossed(Side.SELL, 100.0, 100.0)
    assert limit_crossed(Side.SELL, 100.0, 100.01)


def test_latency_model_bounds_and_determinism():
    lm1 = LatencyModel(200, 500, np.random.default_rng(1))
    lm2 = LatencyModel(200, 500, np.random.default_rng(1))
    d1 = [lm1.draw_ns() for _ in range(100)]
    d2 = [lm2.draw_ns() for _ in range(100)]
    assert d1 == d2
    assert all(200_000_000 <= d <= 500_000_000 for d in d1)
    assert LatencyModel(0, 0, np.random.default_rng(1)).draw_ns() == 0
    with pytest.raises(ValueError):
        LatencyModel(500, 200, np.random.default_rng(1))


# -- engine micro-scenarios -------------------------------------------------


def test_limit_fills_only_on_strict_cross_with_pro_rata_partials():
    frame = mk_trades(
        [
            (0.5, 100.0, 1.0, 1),
            (1.5, 100.0, 1.0, 1),  # closes bar 0 -> BUY limit 100 placed; no cross
            (2.2, 100.0, 5.0, -1),  # touch only -> no fill
            (2.7, 99.9, 2.0, -1),  # strict cross -> partial fill 2
            (3.4, 99.8, 10.0, -1),  # fills remaining 3
        ]
    )
    strat = PlaceOnce([Order(side=Side.BUY, qty=5.0, order_type=OrderType.LIMIT, limit_price=100.0)])
    res = run_backtest(frame, strat, frictionless())
    assert [f.qty for f in res.fills] == [2.0, 3.0]
    assert all(f.price == 100.0 and f.maker for f in res.fills)
    assert res.fills[0].ts == int(2.7 * NS_PER_S)
    assert res.fills[0].fee_usd == pytest.approx(2.0 * 100.0 * 2e-4)


def test_limit_cannot_fill_before_activation():
    frame = mk_trades(
        [
            (0.5, 100.0, 1.0, 1),
            (1.5, 100.0, 1.0, 1),  # placement at t=1.5s, latency 1s -> active 2.5s
            (2.0, 99.5, 10.0, -1),  # crosses but pre-activation -> must NOT fill
            (3.0, 99.5, 10.0, -1),  # first crossing print after activation
        ]
    )
    strat = PlaceOnce([Order(side=Side.BUY, qty=5.0, order_type=OrderType.LIMIT, limit_price=100.0)])
    res = run_backtest(frame, strat, frictionless(latency_ms_min=1000, latency_ms_max=1000))
    assert len(res.fills) == 1
    assert res.fills[0].ts == int(3.0 * NS_PER_S)


def test_market_fill_uses_pre_print_mid_after_latency():
    frame = mk_trades(
        [
            (0.5, 101.0, 1.0, 1),
            (1.5, 101.0, 1.0, 1),  # market buy placed; active at 2.5s
            (2.0, 105.0, 1.0, 1),
            (3.0, 110.0, 1.0, 1),  # execution print; pre-print mid is 105
        ]
    )
    strat = PlaceOnce([Order(side=Side.BUY, qty=1.0)])
    cfg = frictionless(latency_ms_min=1000, latency_ms_max=1000, half_spread_bps=10.0)
    res = run_backtest(frame, strat, cfg)
    assert len(res.fills) == 1
    f = res.fills[0]
    assert f.ts == int(3.0 * NS_PER_S)
    assert f.ref_mid == 105.0  # never the print it executes against
    assert f.price == pytest.approx(105.0 * 1.001)
    assert f.slippage_usd == pytest.approx(105.0 * 0.001)
    assert not f.maker


def test_market_impact_from_recent_volume():
    # execution happens on the 1.5s print PRE-print, so the rolling window
    # holds only the 0.5s print: recent volume = 100 USD
    frame = mk_trades(
        [
            (0.5, 100.0, 1.0, 1),
            (1.5, 100.0, 1.0, 1),  # bar 0 closes -> market buy qty 2, zero latency
            (2.0, 100.0, 1.0, 1),
        ]
    )
    strat = PlaceOnce([Order(side=Side.BUY, qty=2.0)])
    cfg = frictionless(impact_coef_bps=100.0, impact_cap_bps=1000.0, impact_window_s=60.0)
    res = run_backtest(frame, strat, cfg)
    assert len(res.fills) == 1
    f = res.fills[0]
    exp_imp = 100.0 * (2.0 * 100.0) / 100.0  # coef * order notional / recent volume
    assert f.price == pytest.approx(100.0 * (1.0 + exp_imp * 1e-4))


def test_book_provider_walk():
    frame = mk_trades([(0.5, 100.0, 1.0, 1), (1.5, 100.0, 1.0, 1), (2.0, 100.0, 1.0, 1)])
    strat = PlaceOnce([Order(side=Side.BUY, qty=1.5)])

    def provider(ts: int, side: Side) -> list[tuple[float, float]]:
        assert side is Side.BUY
        return [(105.0, 0.5), (106.0, 0.5), (107.0, 10.0)]

    res = run_backtest(frame, strat, frictionless(), book_provider=provider)
    assert len(res.fills) == 1
    assert res.fills[0].price == pytest.approx(106.0)


def test_funding_sign_long_pays_short_receives():
    prints = [(0.1 + 0.5 * i, 100.0, 1.0, 1) for i in range(20)]  # 10 seconds of prints
    mark = records_to_frame(
        [
            MarkPrice(
                exchange="binance_usdm",
                symbol="BTCUSDT",
                ts_event=0,
                ts_recv=0,
                mark_price=100.0,
                index_price=100.0,
                funding_rate=1e-3,
                next_funding_ts=5 * NS_PER_S,
            )
        ],
        "mark_price",
    )
    for side, expected in ((Side.BUY, 0.1), (Side.SELL, -0.1)):
        strat = PlaceOnce([Order(side=side, qty=1.0)])
        res = run_backtest(mk_trades(prints), strat, frictionless(), mark_prices=mark)
        assert res.funding_usd == pytest.approx(expected)  # rate * pos * mark
        assert res.net_pnl == pytest.approx(
            res.gross_pnl - res.fees_usd - res.slippage_usd - res.funding_usd
        )


def test_on_trade_orders_first_act_on_next_print():
    class BuyOnFirstTrade:
        def __init__(self) -> None:
            self.done = False

        def on_trade(self, trade: TradePrint, ctx: Context) -> list[Order]:
            if not self.done:
                self.done = True
                return [Order(side=Side.BUY, qty=1.0)]
            return []

    frame = mk_trades([(0.5, 100.0, 1.0, 1), (0.9, 120.0, 1.0, 1), (1.4, 130.0, 1.0, 1)])
    res = run_backtest(frame, BuyOnFirstTrade(), frictionless())
    assert len(res.fills) == 1
    f = res.fills[0]
    assert f.ts == int(0.9 * NS_PER_S)
    assert f.ref_mid == 100.0  # priced on the print BEFORE the one it reacts to


def test_invalid_orders_rejected():
    frame = mk_trades([(0.5, 100.0, 1.0, 1), (1.5, 100.0, 1.0, 1)])
    with pytest.raises(ValueError, match="qty"):
        run_backtest(frame, PlaceOnce([Order(side=Side.BUY, qty=0.0)]), frictionless())
    bad_limit = Order(side=Side.BUY, qty=1.0, order_type=OrderType.LIMIT, limit_price=None)
    with pytest.raises(ValueError, match="limit_price"):
        run_backtest(frame, PlaceOnce([bad_limit]), frictionless())
