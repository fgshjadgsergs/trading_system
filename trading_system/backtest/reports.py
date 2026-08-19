"""M8 demo reports: equity curve, trade-PnL distribution, cost waterfall.

Everything is generated from seeded synthetic data (no network, no wall
clock): a synthetic trade stream is replayed through the event-driven engine
with an MA-cross strategy, realistic latency, fees, spread, impact and an
accelerated funding schedule so every cost component is visible.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from trading_system.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from trading_system.backtest.metrics import cost_waterfall, max_drawdown, trades_from_fills
from trading_system.backtest.strategies import MACrossStrategy
from trading_system.core.schema import MarkPrice, Trade, records_to_frame
from trading_system.core.synth import synth_trades
from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S
from trading_system.viz.style import PALETTE, apply_style, save_fig


def _mark_frame(trades: list[Trade], funding_every_min: int = 5, rate: float = 3e-4) -> pl.DataFrame:
    """1s mark-price stream with an accelerated funding schedule for demos."""
    step = funding_every_min * NS_PER_MIN
    out: list[MarkPrice] = []
    next_emit = trades[0].ts_event
    for t in trades:
        if t.ts_event < next_emit:
            continue
        boundary = t.ts_event - t.ts_event % step + step
        out.append(
            MarkPrice(
                exchange=t.exchange,
                symbol=t.symbol,
                ts_event=t.ts_event,
                ts_recv=t.ts_recv,
                mark_price=t.price,
                index_price=t.price,
                funding_rate=rate,
                next_funding_ts=boundary,
            )
        )
        next_emit = t.ts_event + NS_PER_S
    return records_to_frame(out, "mark_price")


def _demo_run(seed: int) -> BacktestResult:
    trades = synth_trades(n=24_000, mean_gap_ms=60.0, vol=0.0006, seed=seed)
    frame = records_to_frame(trades, "trade")
    cfg = BacktestConfig(bar_ns=5 * NS_PER_S, init_cash=100_000.0, seed=seed)
    strategy = MACrossStrategy(fast=8, slow=30, qty=0.5)
    return run_backtest(frame, strategy, cfg, mark_prices=_mark_frame(trades))


def _fig_equity(result: BacktestResult, out_dir: Path) -> Path:
    eq = result.equity_curve
    t0 = eq["ts"][0]
    minutes = (eq["ts"] - t0).to_numpy() / NS_PER_MIN
    equity = eq["equity"].to_numpy()
    peak = np.maximum.accumulate(equity)
    fig, (ax, ax_dd) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, height_ratios=[3, 1], constrained_layout=True
    )
    ax.plot(minutes, equity, color=PALETTE["neutral"], lw=1.3, label="equity")
    ax.plot(minutes, peak, color=PALETTE["accent"], lw=0.9, ls="--", label="running peak")
    ax.fill_between(minutes, equity, peak, color=PALETTE["short"], alpha=0.15)
    ax.set_ylabel("equity, USD")
    ax.legend(loc="upper left")
    ax.set_title(
        f"M8 backtest equity — net {result.net_pnl:+,.0f} USD, "
        f"max DD {max_drawdown(eq['equity']):.2%}"
    )
    dd = np.where(peak > 0, equity / peak - 1.0, 0.0)
    ax_dd.fill_between(minutes, dd, 0.0, color=PALETTE["short"], alpha=0.6)
    ax_dd.set_ylabel("drawdown")
    ax_dd.set_xlabel("minutes since start")
    return save_fig(fig, "m8_equity_curve", out_dir)


def _fig_trade_pnl(result: BacktestResult, out_dir: Path) -> Path:
    trades = trades_from_fills(result.fills)
    pnl = trades["pnl_usd"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    sns.histplot(x=pnl, bins=25, kde=len(pnl) >= 10, color=PALETTE["neutral"], ax=ax)
    ax.axvline(0.0, color=PALETTE["grid"], lw=1.0)
    ax.axvline(
        float(pnl.mean()), color=PALETTE["accent"], lw=1.4, ls="--",
        label=f"mean {pnl.mean():+,.1f} USD",
    )
    ax.set_xlabel("round-trip gross PnL, USD")
    ax.set_title(f"M8 trade PnL distribution — {len(pnl)} round trips")
    ax.legend()
    return save_fig(fig, "m8_trade_pnl_dist", out_dir)


def _fig_waterfall(result: BacktestResult, out_dir: Path) -> Path:
    wf = cost_waterfall(result)
    steps = wf["step"].to_list()
    values = wf["value"].to_list()
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    running = 0.0
    for i, (step, value) in enumerate(zip(steps, values, strict=True)):
        if step in ("gross", "net"):
            bottom, height = 0.0, value
            color = PALETTE["long"] if value >= 0 else PALETTE["short"]
            running = value if step == "gross" else running
        else:
            bottom, height = running + value, -value
            color = PALETTE["short"] if value < 0 else PALETTE["long"]
            running += value
        ax.bar(i, height, bottom=bottom, color=color, alpha=0.85, width=0.62)
        ax.annotate(
            f"{value:+,.0f}",
            (i, bottom + max(height, 0.0)),
            ha="center", va="bottom", fontsize=9,
        )
    ax.axhline(0.0, color=PALETTE["neutral"], lw=1.0)
    ax.set_xticks(range(len(steps)), steps)
    ax.set_ylabel("USD")
    ax.set_title("M8 cost waterfall: gross PnL → fees → slippage → funding → net")
    return save_fig(fig, "m8_cost_waterfall", out_dir)


def demo_reports(out_dir: Path, seed: int = 42) -> list[Path]:
    """Generate all M8 checklist figures from a seeded synthetic backtest."""
    apply_style()
    result = _demo_run(seed)
    return [
        _fig_equity(result, out_dir),
        _fig_trade_pnl(result, out_dir),
        _fig_waterfall(result, out_dir),
    ]
