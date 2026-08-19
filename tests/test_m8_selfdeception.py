"""M8 self-deception catchers: null strategy, lookahead, determinism, vectorbt.

These are the tests the checklist calls "ловят 90% самообмана": a random
strategy must earn nothing but pay full costs, deliberate lookahead must show
up as outsized profit that disappears once the feature is correctly lagged,
results must be seed-deterministic, and a simple rule must match an
independent vectorized implementation exactly.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trading_system.backtest.engine import BacktestConfig, Order, OrderType, run_backtest
from trading_system.backtest.strategies import RandomStrategy, TargetPositionStrategy
from trading_system.core.schema import POLARS_SCHEMAS, Side, records_to_frame
from trading_system.core.synth import synth_trades
from trading_system.core.timeutils import NS_PER_S


def test_null_strategy_mean_net_pnl_is_minus_costs():
    """Random seeded entries/exits: mean net PnL ~ -(total costs) across seeds."""
    n_seeds = 20
    net, costs = [], []
    for k in range(n_seeds):
        trades = synth_trades(n=6000, mean_gap_ms=100.0, seed=1000 + k)
        frame = records_to_frame(trades, "trade")
        cfg = BacktestConfig(bar_ns=5 * NS_PER_S, seed=100 + k)
        res = run_backtest(frame, RandomStrategy(seed=2000 + k, p_trade=0.3, qty=0.05), cfg)
        # accounting identity per seed
        assert res.net_pnl == pytest.approx(res.gross_pnl - res.total_costs_usd, abs=1e-6)
        net.append(res.net_pnl)
        costs.append(res.total_costs_usd)
    net_a = np.array(net)
    costs_a = np.array(costs)
    assert np.all(costs_a > 0)
    # |mean(net + costs)| = |mean gross| must be small vs seed dispersion
    resid = net_a + costs_a
    assert abs(resid.mean()) < 0.5 * resid.std(ddof=1)
    # and net is dominated by costs: strictly negative on average
    assert net_a.mean() < -0.5 * costs_a.mean()


def _bar_closes(frame: pl.DataFrame, bar_ns: int) -> np.ndarray:
    b = frame.with_columns((pl.col("ts_event") - pl.col("ts_event") % bar_ns).alias("bucket"))
    return (
        b.group_by("bucket", maintain_order=True)
        .agg(pl.col("price").last().alias("close"))["close"]
        .to_numpy()
    )


def test_lookahead_feature_is_hugely_profitable_and_lagging_kills_it():
    """Feeding NEXT-bar return as a feature must be visibly profitable in this
    harness, and the SAME strategy with the feature correctly lagged by one
    bar must lose that edge — proof the engine would expose lookahead bugs."""
    bar_ns = 5 * NS_PER_S
    trades = synth_trades(n=12_000, mean_gap_ms=100.0, vol=0.0002, seed=7)
    frame = records_to_frame(trades, "trade")
    closes = _bar_closes(frame, bar_ns)
    rets = np.diff(closes) / closes[:-1]
    n_bars = len(closes)
    cheat = np.zeros(n_bars)
    cheat[:-1] = np.sign(rets)  # cheat[i] = sign of NEXT bar's return: lookahead
    lagged = np.zeros(n_bars)
    lagged[1:] = np.sign(rets)  # same feature shifted to what is really known at i

    results = {}
    for name, targets in (("cheat", cheat), ("lagged", lagged)):
        cfg = BacktestConfig(bar_ns=bar_ns, seed=42)
        res = run_backtest(frame, TargetPositionStrategy(targets, qty=1.0), cfg)
        assert res.n_bars == n_bars  # target array aligned with engine bars
        results[name] = res

    assert results["cheat"].net_pnl > 0 > results["lagged"].net_pnl
    # the lookahead edge is large relative to costs, the lagged edge is gone
    assert results["cheat"].gross_pnl > 5 * abs(results["lagged"].gross_pnl)


class _MixedStrategy:
    """Random market orders plus periodic resting limit orders."""

    def __init__(self, seed: int) -> None:
        self.random = RandomStrategy(seed=seed, p_trade=0.4, qty=0.05)

    def on_bar(self, bar, ctx):
        orders = self.random.on_bar(bar, ctx)
        if bar.index % 7 == 3:
            orders.append(
                Order(
                    side=Side.BUY,
                    qty=0.01,
                    order_type=OrderType.LIMIT,
                    limit_price=bar.close * 0.9995,
                )
            )
        return orders


def test_determinism_same_seed_byte_identical_different_seed_diverges():
    trades = synth_trades(n=4000, mean_gap_ms=100.0, seed=21)
    frame = records_to_frame(trades, "trade")

    def run(engine_seed: int):
        cfg = BacktestConfig(bar_ns=5 * NS_PER_S, seed=engine_seed)
        return run_backtest(frame, _MixedStrategy(seed=77), cfg)

    a = run(42)
    b = run(42)
    c = run(43)
    assert a.equity_curve.write_csv() == b.equity_curve.write_csv()  # byte-identical
    assert a.fills == b.fills
    assert a.latencies_ns == b.latencies_ns
    assert len(a.latencies_ns) > 10
    assert a.latencies_ns != c.latencies_ns  # different seed -> different latency draws


def test_cross_check_ma_cross_matches_vectorbt():
    """Zero latency/spread/impact, fees only, market-on-bar-close: the engine
    must reproduce vectorbt's result on identical signals to sub-cent accuracy."""
    vbt = pytest.importorskip("vectorbt")
    import pandas as pd

    n_bars = 400
    rng = np.random.default_rng(11)
    closes = 20000.0 * np.exp(np.cumsum(rng.normal(0.0, 0.004, n_bars)))
    ts0 = 1_755_600_000 * NS_PER_S
    rows = [
        {
            "exchange": "binance_usdm",
            "symbol": "BTCUSDT",
            "ts_event": ts0 + i * 60 * NS_PER_S + 30 * NS_PER_S,
            "ts_recv": ts0 + i * 60 * NS_PER_S + 30 * NS_PER_S,
            "price": float(closes[i]),
            "qty": 1.0,
            "qty_usd": float(closes[i]),
            "side": 1,
            "trade_id": i + 1,
        }
        for i in range(n_bars)
    ]
    frame = pl.DataFrame(rows, schema=POLARS_SCHEMAS["trade"])

    def sma(x: np.ndarray, k: int) -> np.ndarray:
        c = np.cumsum(np.insert(x, 0, 0.0))
        return (c[k:] - c[:-k]) / k

    fast, slow = 5, 20
    state = np.zeros(n_bars, dtype=bool)
    state[slow - 1 :] = sma(closes, fast)[slow - fast :] > sma(closes, slow)
    prev = np.concatenate([[False], state[:-1]])
    entries = state & ~prev
    exits = ~state & prev
    assert entries.sum() >= 3  # the rule actually trades

    fee = 5e-4
    init_cash = 1_000_000.0
    cfg = BacktestConfig(
        latency_ms_min=0.0,
        latency_ms_max=0.0,
        half_spread_bps=0.0,
        impact_coef_bps=0.0,
        taker_fee=fee,
        bar_ns=60 * NS_PER_S,
        init_cash=init_cash,
        seed=42,
    )
    res = run_backtest(frame, TargetPositionStrategy(state.astype(float), qty=1.0), cfg)
    engine_final = init_cash + res.net_pnl

    pf = vbt.Portfolio.from_signals(
        pd.Series(closes),
        pd.Series(entries),
        pd.Series(exits),
        size=1.0,
        size_type="amount",
        fees=fee,
        init_cash=init_cash,
    )
    vbt_final = pf.final_value() if callable(pf.final_value) else pf.final_value
    n_vbt = pf.orders.count
    assert len(res.fills) == int(n_vbt() if callable(n_vbt) else n_vbt)  # same order count
    assert engine_final == pytest.approx(float(vbt_final), abs=0.01)  # sub-cent on $1M
