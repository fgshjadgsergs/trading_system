"""M8 regression tests for review findings: aggregate limit capacity, impact
window eviction at fill time, funding row selection, end-of-data flush latency."""

from __future__ import annotations

import polars as pl
import pytest

from tests.test_m8_fills import PlaceOnce, frictionless, mk_trades
from trading_system.backtest.engine import (
    Bar,
    Context,
    Order,
    OrderType,
    funding_events,
    run_backtest,
)
from trading_system.core.schema import POLARS_SCHEMAS, Side
from trading_system.core.timeutils import NS_PER_S


def test_one_print_cannot_fill_more_than_its_size():
    """Three resting 5.0 buys at 100; a single 5.0 print at 99.9 provides at
    most 5.0 across ALL of them — no manufactured maker liquidity."""
    trades = mk_trades(
        [(0.2, 101.0, 1.0, 1), (1.2, 101.0, 1.0, 1), (2.5, 99.9, 5.0, -1), (3.5, 101.0, 1.0, 1)]
    )
    orders = [
        Order(side=Side.BUY, qty=5.0, order_type=OrderType.LIMIT, limit_price=100.0)
        for _ in range(3)
    ]
    result = run_backtest(trades, PlaceOnce(orders), frictionless())
    maker_qty = sum(f.qty for f in result.fills if f.maker)
    assert maker_qty == pytest.approx(5.0)


def test_impact_window_evicted_at_fill_time():
    """A market order filling after a long quiet gap must see an empty volume
    window and be charged the impact cap, not stale pre-gap volume."""
    trades = mk_trades(
        [(0.5, 100.0, 1.0, 1), (0.9, 100.0, 1.0, 1), (600.0, 100.0, 1.0, 1), (601.0, 100.0, 1.0, 1)]
    )

    class BuyBeforeGap:
        def __init__(self) -> None:
            self.placed = False

        def on_trade(self, trade, ctx):
            # placed at the 0.9s print; first eligible print is 600s
            if not self.placed and trade.ts >= int(0.9 * NS_PER_S):
                self.placed = True
                return [Order(side=Side.BUY, qty=2.0, order_type=OrderType.MARKET)]
            return None

        def on_bar(self, bar: Bar, ctx: Context):
            return None

    cfg = frictionless(impact_coef_bps=100.0, impact_cap_bps=1000.0, impact_window_s=60.0)
    result = run_backtest(trades, BuyBeforeGap(), cfg)
    taker = [f for f in result.fills if not f.maker]
    assert len(taker) == 1
    # empty window -> cap: fill at mid * (1 + 1000 bps) = 110
    assert taker[0].price == pytest.approx(100.0 * 1.10)


def test_funding_uses_row_strictly_before_funding_ts():
    """A mark row stamped exactly at the funding time belongs to the NEXT
    period; the settled rate is the last row strictly before."""
    h8 = 8 * 3600 * NS_PER_S
    rows = [
        {  # one second before funding: settled rate 1e-3
            "exchange": "binance_usdm",
            "symbol": "BTCUSDT",
            "ts_event": h8 - NS_PER_S,
            "ts_recv": h8 - NS_PER_S,
            "mark_price": 100.0,
            "index_price": 100.0,
            "funding_rate": 1e-3,
            "next_funding_ts": h8,
        },
        {  # exactly at funding: already the next period's estimate
            "exchange": "binance_usdm",
            "symbol": "BTCUSDT",
            "ts_event": h8,
            "ts_recv": h8,
            "mark_price": 100.0,
            "index_price": 100.0,
            "funding_rate": 1e-5,
            "next_funding_ts": 2 * h8,
        },
    ]
    marks = pl.DataFrame(rows, schema=POLARS_SCHEMAS["mark_price"])
    events = funding_events(marks, 0, 3 * h8)
    by_ts = {ts: rate for ts, rate, _ in events}
    assert by_ts[h8] == pytest.approx(1e-3)


def test_no_fill_after_end_of_data_for_last_print_orders():
    """An order emitted on the very last print (latency pending, no later
    print) must expire unfilled, not execute at the final mid."""
    trades = mk_trades([(0.5, 100.0, 1.0, 1), (0.9, 100.0, 1.0, 1), (1.4, 130.0, 1.0, 1)])

    class BuyOnLastPrint:
        def on_trade(self, trade, ctx):
            if trade.price >= 130.0:
                return [Order(side=Side.BUY, qty=1.0, order_type=OrderType.MARKET)]
            return None

        def on_bar(self, bar: Bar, ctx: Context):
            return None

    cfg = frictionless(latency_ms_min=500.0, latency_ms_max=500.0)
    result = run_backtest(trades, BuyOnLastPrint(), cfg)
    assert result.fills == []
    assert result.final_equity == pytest.approx(cfg.init_cash)
