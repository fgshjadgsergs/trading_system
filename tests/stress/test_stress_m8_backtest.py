"""M8 stress: engine/fills under scale, boundary limits, price storms, extreme moves.

Scenarios (scaled by env STRESS_SCALE, default 1):
- 1e5 prints through the engine with a strategy trading every bar: throughput
  floor and a tracemalloc memory ceiling;
- limit price boundaries: a print AT the limit never fills (strict cross), one
  tick through always fills — checked on 10k random boundaries;
- storms of identical prices: partial fills across 10k prints sum exactly,
  one print never fills one order beyond its own participation-capped size;
- activation at the data edge: orders born on the last print expire unfilled,
  activation exactly at the next print's ts interacts with exactly that print;
- impact with an empty / one-micro-trade volume window: the cap is paid, no
  division by zero;
- 1000 simultaneously active orders: correctness and bounded runtime;
- price x100 per bar against the position: metrics stay finite.

Everything is seeded, offline and wall-clock free except coarse elapsed-time
ceilings on the heavy tests.
"""

from __future__ import annotations

import math
import os
import time
import tracemalloc

import numpy as np
import polars as pl
import pytest

from trading_system.backtest.engine import (
    BacktestConfig,
    Bar,
    Context,
    Order,
    OrderType,
    run_backtest,
)
from trading_system.backtest.fills import impact_bps, limit_crossed, market_fill_price
from trading_system.backtest.metrics import max_drawdown, summary, trades_from_fills
from trading_system.core.schema import POLARS_SCHEMAS, Side
from trading_system.core.timeutils import NS_PER_S

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
SEED = 42
T0 = 1_755_600_000 * NS_PER_S


def n_scaled(base: int) -> int:
    return max(1, int(base * SCALE))


def frame_from_arrays(ts: np.ndarray, px: np.ndarray, qty: np.ndarray, side: np.ndarray) -> pl.DataFrame:
    n = len(ts)
    return pl.DataFrame(
        {
            "exchange": ["binance_usdm"] * n,
            "symbol": ["BTCUSDT"] * n,
            "ts_event": ts.astype(np.int64),
            "ts_recv": ts.astype(np.int64),
            "price": px.astype(np.float64),
            "qty": qty.astype(np.float64),
            "qty_usd": (px * qty).astype(np.float64),
            "side": side.astype(np.int64),
            "trade_id": np.arange(1, n + 1),
        },
        schema=POLARS_SCHEMAS["trade"],
    )


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


def frictionless(**overrides) -> BacktestConfig:
    kwargs = dict(
        latency_ms_min=0.0,
        latency_ms_max=0.0,
        half_spread_bps=0.0,
        impact_coef_bps=0.0,
        taker_fee=5e-4,
        maker_fee=2e-4,
        bar_ns=NS_PER_S,
        seed=SEED,
    )
    kwargs.update(overrides)
    return BacktestConfig(**kwargs)


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


class ChurnEveryBar:
    """Alternates a small market buy/sell on every bar close: constant churn."""

    def __init__(self, qty: float = 0.01) -> None:
        self.qty = qty
        self.i = 0

    def on_bar(self, bar: Bar, ctx: Context) -> list[Order]:
        self.i += 1
        side = Side.BUY if self.i % 2 else Side.SELL
        return [Order(side=side, qty=self.qty, order_type=OrderType.MARKET)]


# --------------------------------------------------------------------------
# 1) Scale: 1e5 prints, frequent trading, throughput + memory ceiling
# --------------------------------------------------------------------------


def test_scale_100k_prints_throughput_and_memory():
    n = n_scaled(100_000)
    rng = np.random.default_rng(SEED)
    ts = T0 + np.cumsum(rng.integers(1, 200, n)) * 1_000_000  # 1..200 ms gaps
    px = 50_000.0 * np.exp(np.cumsum(rng.normal(0.0, 2e-4, n)))
    qty = np.round(rng.lognormal(-3.0, 1.2, n), 4) + 0.0001
    side = np.where(rng.random(n) < 0.5, 1, -1)
    frame = frame_from_arrays(ts, px, qty, side)

    cfg = BacktestConfig(bar_ns=NS_PER_S, seed=SEED)
    tracemalloc.start()
    t_start = time.perf_counter()
    res = run_backtest(frame, ChurnEveryBar(qty=0.01), cfg)
    elapsed = time.perf_counter() - t_start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    throughput = n / elapsed
    # measured ~2.3e5 prints/s in this environment; the floor is deliberately
    # an order of magnitude lower so only a real regression trips it
    if n >= 10_000:  # below that, fixed overhead dominates the ratio
        assert throughput > 20_000, f"engine too slow: {throughput:,.0f} prints/s"
    assert peak < 400 * 1024 * 1024, f"peak traced memory {peak / 1e6:.0f} MB"
    assert elapsed < 60.0

    # the strategy really traded frequently and the books stay coherent
    assert res.n_bars > n // 500  # thousands of bars on 1..200 ms gaps
    assert len(res.fills) >= res.n_bars - 2  # one market order per closed bar
    assert res.net_pnl == pytest.approx(
        res.gross_pnl - res.fees_usd - res.slippage_usd - res.funding_usd, abs=1e-6
    )
    eq = res.equity_curve["equity"].to_numpy()
    assert np.all(np.isfinite(eq))
    assert np.all(np.diff(res.equity_curve["ts"].to_numpy()) >= 0)
    s = summary(res)
    assert all(math.isfinite(v) for k, v in s.items() if k != "hit_rate")
    assert math.isfinite(s["hit_rate"])  # plenty of round trips at this scale


# --------------------------------------------------------------------------
# 2) Limit price boundary: touch never fills, one tick through always fills
# --------------------------------------------------------------------------


def test_limit_touch_does_not_fill_one_tick_through_does():
    tick = 0.1
    limit = 100.0
    base = [(0.2, 100.5, 1.0, 1), (1.2, 100.5, 1.0, 1)]  # bar 0 closes -> order placed

    # touch only: 50 prints exactly at the limit -> zero fills
    touch = base + [(2.0 + 0.01 * i, limit, 5.0, -1) for i in range(50)]
    strat = PlaceOnce([Order(side=Side.BUY, qty=1.0, order_type=OrderType.LIMIT, limit_price=limit)])
    res = run_backtest(mk_trades(touch), strat, frictionless())
    assert res.fills == []

    # one tick through: fills at the limit price on the first crossing print
    through = base + [(2.0, limit, 5.0, -1), (2.5, limit - tick, 5.0, -1)]
    strat = PlaceOnce([Order(side=Side.BUY, qty=1.0, order_type=OrderType.LIMIT, limit_price=limit)])
    res = run_backtest(mk_trades(through), strat, frictionless())
    assert len(res.fills) == 1
    assert res.fills[0].price == limit and res.fills[0].maker
    assert res.fills[0].ts == int(2.5 * NS_PER_S)

    # same on the sell side (market below the limit, touch, then one tick up)
    sell_base = [(0.2, 99.5, 1.0, 1), (1.2, 99.5, 1.0, 1)]
    sell_rows = sell_base + [(2.0, limit, 5.0, 1), (2.5, limit + tick, 5.0, 1)]
    strat = PlaceOnce([Order(side=Side.SELL, qty=1.0, order_type=OrderType.LIMIT, limit_price=limit)])
    res = run_backtest(mk_trades(sell_rows), strat, frictionless())
    assert len(res.fills) == 1 and res.fills[0].ts == int(2.5 * NS_PER_S)


def test_limit_boundary_invariant_on_10k_random_boundaries():
    n = n_scaled(10_000)
    rng = np.random.default_rng(SEED)
    prices = rng.uniform(0.01, 1e6, n)
    ticks = 10.0 ** rng.uniform(-8, 0, n)
    for limit, tick in zip(prices, ticks, strict=True):
        through = limit - tick
        if through <= 0 or through == limit:  # tick lost to float precision
            continue
        # touching the limit never crosses, either side
        assert not limit_crossed(Side.BUY, limit, limit)
        assert not limit_crossed(Side.SELL, limit, limit)
        # strictly through fills; strictly away never does
        assert limit_crossed(Side.BUY, limit, through)
        assert not limit_crossed(Side.BUY, limit, limit + tick)
        assert limit_crossed(Side.SELL, limit, limit + tick)
        assert not limit_crossed(Side.SELL, limit, through)


# --------------------------------------------------------------------------
# 3) Storm of identical prices: partials sum, per-print capacity respected
# --------------------------------------------------------------------------


def test_same_price_storm_partials_sum_exactly():
    n = n_scaled(10_000)
    print_qty = 0.001
    k = max(1, (3 * n) // 10)  # the order takes ~30% of the storm to fill
    order_qty = k * print_qty
    rows = [(0.5, 100.0, 1.0, 1), (1.5, 100.0, 1.0, 1)]
    rows += [(2.0 + i * 1e-4, 99.9, print_qty, -1) for i in range(n)]
    strat = PlaceOnce(
        [Order(side=Side.BUY, qty=order_qty, order_type=OrderType.LIMIT, limit_price=100.0)]
    )
    res = run_backtest(mk_trades(rows), strat, frictionless())

    filled = sum(f.qty for f in res.fills)
    assert filled == pytest.approx(order_qty, abs=1e-9)
    assert len(res.fills) == k  # one partial per print, then done
    # no print filled the order beyond its own size
    per_ts: dict[int, float] = {}
    for f in res.fills:
        per_ts[f.ts] = per_ts.get(f.ts, 0.0) + f.qty
    assert max(per_ts.values()) <= print_qty + 1e-12
    assert all(f.price == 100.0 and f.maker for f in res.fills)


def test_same_price_storm_respects_participation_cap():
    n = n_scaled(10_000)
    capacity = 0.001 * 0.5  # per-print capacity at participation 0.5
    order_qty = max(1, n // 4) * capacity
    rows = [(0.5, 100.0, 1.0, 1), (1.5, 100.0, 1.0, 1)]
    rows += [(2.0 + i * 1e-4, 99.9, 0.001, -1) for i in range(n)]
    strat = PlaceOnce(
        [Order(side=Side.BUY, qty=order_qty, order_type=OrderType.LIMIT, limit_price=100.0)]
    )
    res = run_backtest(mk_trades(rows), strat, frictionless(limit_participation=0.5))
    per_ts: dict[int, float] = {}
    for f in res.fills:
        per_ts[f.ts] = per_ts.get(f.ts, 0.0) + f.qty
    assert max(per_ts.values()) <= capacity + 1e-12
    assert sum(f.qty for f in res.fills) == pytest.approx(order_qty, abs=1e-9)


def test_touch_only_storm_never_fills():
    n = n_scaled(10_000)
    rows = [(0.5, 100.0, 1.0, 1), (1.5, 100.0, 1.0, 1)]
    rows += [(2.0 + i * 1e-4, 100.0, 1.0, -1) for i in range(n)]  # 10k touches
    strat = PlaceOnce(
        [Order(side=Side.BUY, qty=5.0, order_type=OrderType.LIMIT, limit_price=100.0)]
    )
    res = run_backtest(mk_trades(rows), strat, frictionless())
    assert res.fills == []
    assert res.final_equity == pytest.approx(res.init_cash)


# --------------------------------------------------------------------------
# 4) Activation at the edge of the data
# --------------------------------------------------------------------------


class EmitAt:
    """Emits given orders from on_trade at the first print with ts >= at_s."""

    def __init__(self, at_s: float, orders: list[Order]) -> None:
        self.at_ns = int(at_s * NS_PER_S)
        self.orders = orders
        self.done = False

    def on_trade(self, trade, ctx):
        if not self.done and trade.ts >= self.at_ns:
            self.done = True
            return self.orders
        return None


def test_limit_order_born_on_last_print_expires_unfilled():
    rows = [(0.5, 100.0, 1.0, 1), (0.9, 100.0, 1.0, 1), (1.4, 100.0, 1.0, -1)]
    order = Order(side=Side.BUY, qty=1.0, order_type=OrderType.LIMIT, limit_price=101.0)
    res = run_backtest(mk_trades(rows), EmitAt(1.4, [order]), frictionless())
    assert res.fills == []  # nothing beyond the data edge to fill against
    assert res.final_equity == pytest.approx(res.init_cash)


def test_market_order_with_latency_beyond_data_expires():
    rows = [(0.5, 100.0, 1.0, 1), (0.9, 100.0, 1.0, 1), (1.4, 100.0, 1.0, 1)]
    res = run_backtest(
        mk_trades(rows),
        EmitAt(0.9, [Order(side=Side.BUY, qty=1.0)]),
        frictionless(latency_ms_min=10_000.0, latency_ms_max=10_000.0),
    )
    assert res.fills == []  # active only after the tape ends: expires
    assert res.final_equity == pytest.approx(res.init_cash)


def test_activation_exactly_at_next_print_interacts_with_that_print():
    # placement on the 1.0s print; latency 500ms -> ts_active == 1.5s exactly,
    # which is the ts of the next print: both order types act on that print.
    rows = [(0.2, 100.0, 1.0, 1), (1.0, 100.0, 1.0, 1), (1.5, 99.0, 5.0, -1), (2.0, 99.0, 1.0, -1)]
    cfg = frictionless(latency_ms_min=500.0, latency_ms_max=500.0)

    res = run_backtest(mk_trades(rows), EmitAt(1.0, [Order(side=Side.BUY, qty=1.0)]), cfg)
    assert len(res.fills) == 1
    assert res.fills[0].ts == int(1.5 * NS_PER_S)
    assert res.fills[0].ref_mid == 100.0  # priced on pre-print state

    limit = Order(side=Side.BUY, qty=1.0, order_type=OrderType.LIMIT, limit_price=100.0)
    res = run_backtest(mk_trades(rows), EmitAt(1.0, [limit]), cfg)
    assert len(res.fills) == 1
    assert res.fills[0].ts == int(1.5 * NS_PER_S)  # eligible exactly at activation
    assert res.fills[0].price == 100.0 and res.fills[0].maker


# --------------------------------------------------------------------------
# 5) Impact: empty and one-micro-trade volume windows
# --------------------------------------------------------------------------


def test_impact_empty_window_pays_cap_never_divides_by_zero():
    rng = np.random.default_rng(SEED)
    for notional in rng.uniform(1e-9, 1e9, n_scaled(1000)):
        assert impact_bps(float(notional), 0.0, 10.0, 25.0) == 25.0
        assert impact_bps(float(notional), -1.0, 10.0, 25.0) == 25.0  # nonsense volume -> cap
        got = impact_bps(float(notional), 1e-12, 10.0, 25.0)  # near-empty window
        assert math.isfinite(got) and got == 25.0


def test_engine_micro_trade_window_charges_cap_and_stays_finite():
    # the only volume in the 60s window before the fill is one 1e-8-coin print
    rows = [
        (0.5, 100.0, 1e-8, 1),
        (1.5, 100.0, 1e-8, 1),  # bar 0 closes -> market buy placed, zero latency
        (2.0, 100.0, 1.0, 1),
    ]
    cfg = frictionless(impact_coef_bps=100.0, impact_cap_bps=1000.0, impact_window_s=60.0)
    res = run_backtest(mk_trades(rows), PlaceOnce([Order(side=Side.BUY, qty=1.0)]), cfg)
    assert len(res.fills) == 1
    f = res.fills[0]
    assert math.isfinite(f.price)
    assert f.price == pytest.approx(100.0 * (1.0 + 1000.0 * 1e-4))  # cap paid


@pytest.mark.xfail(
    reason=(
        "design: market_fill_price with half_spread+impact >= 10000 bps drives a SELL "
        "fill price negative; impact_cap_bps sanity belongs to config validation"
    ),
    strict=True,
)
def test_sell_fill_price_nonnegative_under_absurd_impact_cap():
    assert market_fill_price(Side.SELL, 100.0, 0.0, 20_000.0) >= 0.0


# --------------------------------------------------------------------------
# 6) Order storm: 1000 simultaneously active orders
# --------------------------------------------------------------------------


def test_1000_resting_limit_orders_sweep_fills_each_exactly_once():
    n_orders = 1000
    limits = [99.0 - 0.005 * k for k in range(n_orders)]  # 99.0 down to 94.005
    orders = [
        Order(side=Side.BUY, qty=0.01, order_type=OrderType.LIMIT, limit_price=p) for p in limits
    ]
    rows = [(0.2, 100.0, 1.0, 1), (1.2, 100.0, 1.0, 1)]
    sweep_n = 1200
    for i in range(sweep_n):  # sweep from 99.5 strictly below the lowest limit
        rows.append((2.0 + i * 0.01, 99.5 - i * 0.005, 50.0, -1))
    t_start = time.perf_counter()
    res = run_backtest(mk_trades(rows), PlaceOnce(orders), frictionless())
    elapsed = time.perf_counter() - t_start
    assert elapsed < 30.0

    fills_per_order: dict[int, float] = {}
    for f in res.fills:
        fills_per_order[f.order_id] = fills_per_order.get(f.order_id, 0.0) + f.qty
    assert len(fills_per_order) == n_orders  # every order filled
    assert all(q == pytest.approx(0.01, abs=1e-12) for q in fills_per_order.values())
    assert sum(f.qty for f in res.fills) == pytest.approx(n_orders * 0.01)
    # each fill at its own limit price, never better than the sweep allows
    by_id = {o.order_id: o for o in res.orders}
    assert all(f.price == by_id[f.order_id].limit_price for f in res.fills)


def test_1000_simultaneous_market_orders_all_fill_once():
    n_orders = 1000
    orders = [Order(side=Side.BUY, qty=0.001) for _ in range(n_orders)]
    rows = [(0.2, 100.0, 1.0, 1), (1.2, 100.0, 10.0, 1), (2.0, 100.0, 10.0, 1)]
    cfg = frictionless(impact_coef_bps=10.0, impact_cap_bps=25.0)
    res = run_backtest(mk_trades(rows), PlaceOnce(orders), cfg)
    assert len(res.fills) == n_orders
    assert len({f.order_id for f in res.fills}) == n_orders  # no double execution
    assert all(math.isfinite(f.price) and f.price > 0 for f in res.fills)


# --------------------------------------------------------------------------
# 7) Extreme PnL: price x100 per bar against the position
# --------------------------------------------------------------------------


def test_price_x100_per_bar_against_short_keeps_metrics_finite():
    rows = [(0.5, 100.0, 1.0, 1), (1.5, 100.0, 1.0, 1)]
    p = 100.0
    for k in range(6):  # 100 -> 1e14 over six bars
        p *= 100.0
        rows.append((2.5 + k, p, 1.0, 1))
    strat = PlaceOnce([Order(side=Side.SELL, qty=1.0)])
    res = run_backtest(mk_trades(rows), strat, frictionless())
    assert res.net_pnl < -1e13  # catastrophic, but well-defined
    s = summary(res)
    for key, val in s.items():
        if key == "hit_rate":
            continue  # NaN with zero closed round trips is the documented contract
        assert math.isfinite(val), f"{key} not finite: {val}"
    eq = res.equity_curve["equity"].to_numpy()
    assert np.all(np.isfinite(eq))
    assert math.isfinite(max_drawdown(eq))


def test_price_collapse_with_round_trip_keeps_metrics_finite():
    rows = [(0.5, 1e10, 1.0, 1), (1.5, 1e10, 1.0, 1)]
    p = 1e10
    for k in range(5):  # collapse by x100 per bar
        p /= 100.0
        rows.append((2.5 + k, p, 1.0, -1))

    class BuyThenSell:
        def __init__(self) -> None:
            self.i = 0

        def on_bar(self, bar: Bar, ctx: Context) -> list[Order]:
            self.i += 1
            if self.i == 1:
                return [Order(side=Side.BUY, qty=1.0)]
            if self.i == 4:
                return [Order(side=Side.SELL, qty=1.0)]  # close the long into the crash
            return []

    res = run_backtest(mk_trades(rows), BuyThenSell(), frictionless())
    s = summary(res)
    assert s["n_trades"] >= 1.0
    assert all(math.isfinite(v) for v in s.values())  # incl. hit_rate: trades exist
    assert 0.0 <= s["max_drawdown"] and math.isfinite(s["max_drawdown"])


# --------------------------------------------------------------------------
# 8) Regression for the zero-qty fill fix in trades_from_fills
# --------------------------------------------------------------------------


def _fill(ts_s: float, side: Side, qty: float, price: float):
    from trading_system.backtest.engine import Fill

    return Fill(
        order_id=0,
        ts=int(ts_s * NS_PER_S),
        side=side,
        qty=qty,
        price=price,
        ref_mid=price,
        maker=False,
        fee_usd=0.0,
        slippage_usd=0.0,
    )


def test_zero_qty_fills_produce_no_phantom_trade_rows():
    real = [_fill(1.0, Side.BUY, 1.0, 100.0), _fill(3.0, Side.SELL, 1.0, 110.0)]
    noisy = [
        real[0],
        _fill(2.0, Side.SELL, 0.0, 105.0),  # zero-qty: must be invisible
        _fill(2.5, Side.BUY, 0.0, 90.0),
        real[1],
        _fill(4.0, Side.SELL, 0.0, 120.0),  # zero-qty while flat
    ]
    clean = trades_from_fills(real)
    dirty = trades_from_fills(noisy)
    assert dirty.equals(clean)
    assert len(dirty) == 1
    assert (dirty["qty"] > 0).all()
