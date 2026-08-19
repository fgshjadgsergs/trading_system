"""M8 engine and metrics tests: bars, equity curve, trade list, decomposition."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_m8_fills import frictionless, mk_trades
from trading_system.backtest.engine import (
    BacktestConfig,
    Bar,
    Context,
    Fill,
    run_backtest,
)
from trading_system.backtest.metrics import (
    cost_waterfall,
    hit_rate,
    max_drawdown,
    pnl_decomposition,
    summary,
    trades_from_fills,
)
from trading_system.backtest.reports import _mark_frame
from trading_system.backtest.strategies import RandomStrategy
from trading_system.core.config import load_config
from trading_system.core.schema import Side, records_to_frame
from trading_system.core.synth import synth_trades
from trading_system.core.timeutils import NS_PER_S, TIMEFRAME_NS


class RecordBars:
    def __init__(self) -> None:
        self.bars: list[Bar] = []

    def on_bar(self, bar: Bar, ctx: Context) -> list:
        self.bars.append(bar)
        return []


def _fill(ts_s: float, side: Side, qty: float, price: float, fee: float = 0.0) -> Fill:
    return Fill(
        order_id=0,
        ts=int(ts_s * NS_PER_S),
        side=side,
        qty=qty,
        price=price,
        ref_mid=price,
        maker=False,
        fee_usd=fee,
        slippage_usd=0.0,
    )


def test_time_bars_ohlcv_from_prints():
    frame = mk_trades(
        [
            (0.2, 100.0, 1.0, 1),
            (0.4, 102.0, 2.0, -1),
            (0.8, 99.0, 1.0, 1),
            (1.3, 101.0, 1.0, 1),
            (2.6, 103.0, 5.0, 1),
        ]
    )
    strat = RecordBars()
    res = run_backtest(frame, strat, frictionless())
    assert res.n_bars == 3
    b0, b1, b2 = strat.bars
    assert (b0.open, b0.high, b0.low, b0.close) == (100.0, 102.0, 99.0, 99.0)
    assert b0.volume == 4.0
    assert b0.n_trades == 3
    assert (b0.ts_open, b0.ts_close) == (0, NS_PER_S)
    assert (b1.open, b1.close, b1.volume) == (101.0, 101.0, 1.0)
    assert b2.index == 2 and b2.ts_open == 2 * NS_PER_S  # flushed at end of data


def test_equity_curve_monotone_ts_and_final_row():
    trades = synth_trades(n=3000, mean_gap_ms=100.0, seed=3)
    frame = records_to_frame(trades, "trade")
    cfg = BacktestConfig(bar_ns=5 * NS_PER_S, seed=3)
    res = run_backtest(frame, RandomStrategy(seed=5, qty=0.05), cfg)
    eq = res.equity_curve
    ts = eq["ts"].to_numpy()
    assert len(eq) == res.n_bars + 1  # one row per closed bar + final mark
    assert np.all(np.diff(ts) >= 0)
    assert np.all(np.isfinite(eq["equity"].to_numpy()))
    assert eq["equity"][-1] == pytest.approx(res.final_equity)
    assert eq["fees_cum"][-1] == pytest.approx(res.fees_usd)


def test_accounting_identity_with_funding_and_costs():
    trades = synth_trades(n=5000, mean_gap_ms=100.0, seed=9)
    frame = records_to_frame(trades, "trade")
    mark = _mark_frame(trades, funding_every_min=2, rate=5e-4)
    cfg = BacktestConfig(bar_ns=5 * NS_PER_S, seed=9)
    res = run_backtest(frame, RandomStrategy(seed=17, qty=0.1), cfg, mark_prices=mark)
    assert res.funding_usd != 0.0
    assert res.fees_usd > 0.0
    assert res.net_pnl == pytest.approx(
        res.gross_pnl - res.fees_usd - res.slippage_usd - res.funding_usd, abs=1e-6
    )
    dec = pnl_decomposition(res)
    assert dec["net"] == pytest.approx(res.net_pnl)
    wf = cost_waterfall(res)
    steps = dict(zip(wf["step"].to_list(), wf["value"].to_list(), strict=True))
    assert steps["gross"] + steps["fees"] + steps["slippage"] + steps["funding"] == pytest.approx(
        steps["net"], abs=1e-6
    )


def test_trades_from_fills_round_trips_and_flip():
    fills = [
        _fill(1.0, Side.BUY, 1.0, 100.0),
        _fill(2.0, Side.BUY, 1.0, 110.0),  # avg entry 105
        _fill(3.0, Side.SELL, 2.0, 120.0),  # close long: pnl (120-105)*2
        _fill(4.0, Side.SELL, 1.0, 130.0),  # open short
        _fill(5.0, Side.BUY, 2.0, 125.0),  # close short pnl (130-125)*1, flip long 1 @ 125
        _fill(6.0, Side.SELL, 1.0, 128.0),  # close long pnl (128-125)*1
    ]
    trades = trades_from_fills(fills)
    assert len(trades) == 3
    assert trades["pnl_usd"].to_list() == pytest.approx([30.0, 5.0, 3.0])
    assert trades["direction"].to_list() == ["long", "short", "long"]
    assert trades["ts_entry"].to_list() == [NS_PER_S, 4 * NS_PER_S, 5 * NS_PER_S]
    assert hit_rate(trades) == 1.0
    assert trades_from_fills([]).is_empty()
    assert np.isnan(hit_rate(trades_from_fills([])))


def test_max_drawdown_known_path():
    assert max_drawdown(np.array([100.0, 120.0, 90.0, 130.0])) == pytest.approx(0.25)
    assert max_drawdown(np.array([1.0, 2.0, 3.0])) == 0.0
    assert max_drawdown(np.array([])) == 0.0


def test_summary_keys_and_config_from_yaml():
    trades = synth_trades(n=2000, mean_gap_ms=100.0, seed=4)
    frame = records_to_frame(trades, "trade")
    res = run_backtest(frame, RandomStrategy(seed=6, qty=0.05), BacktestConfig(bar_ns=5 * NS_PER_S))
    s = summary(res)
    assert {"net_pnl", "gross_pnl", "max_drawdown", "hit_rate", "n_trades"} <= set(s)

    cfg = BacktestConfig.from_config(load_config(), timeframe="1m", seed=7)
    assert cfg.latency_ms_min == 200.0
    assert cfg.latency_ms_max == 500.0
    assert cfg.taker_fee == 5e-4
    assert cfg.maker_fee == 2e-4
    assert cfg.half_spread_bps == 0.5
    assert cfg.bar_ns == TIMEFRAME_NS["1m"]
    assert cfg.seed == 7


def test_empty_trade_frame_rejected():
    import polars as pl

    from trading_system.core.schema import POLARS_SCHEMAS

    with pytest.raises(ValueError, match="empty"):
        run_backtest(
            pl.DataFrame(schema=POLARS_SCHEMAS["trade"]), RandomStrategy(seed=1), frictionless()
        )
