"""Backtest metrics: trade list, PnL decomposition, drawdown, cost waterfall.

Accounting identity used throughout (and asserted in tests):

    net_pnl = gross_pnl - fees - slippage - funding

where ``gross_pnl`` values every fill at its frictionless reference mid and
``slippage`` is the signed price paid vs that mid (>= 0 for taker fills, may
be negative for maker fills that earn the spread).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import polars as pl

from trading_system.backtest.engine import BacktestResult, Fill
from trading_system.core.schema import Side

_EPS = 1e-12

TRADES_SCHEMA = {
    "ts_entry": pl.Int64,
    "ts_exit": pl.Int64,
    "direction": pl.Utf8,
    "qty": pl.Float64,
    "entry_price": pl.Float64,
    "exit_price": pl.Float64,
    "pnl_usd": pl.Float64,
}


def trades_from_fills(fills: Sequence[Fill]) -> pl.DataFrame:
    """Round-trip trades via average-entry accounting; one row per reducing fill.

    PnL is gross of fees (price based); position flips split into a close row
    plus a fresh entry at the flipping fill's price.
    """
    rows: list[tuple] = []
    pos = 0.0
    avg = 0.0
    ts_open = 0
    for f in fills:
        q = f.qty if f.side is Side.BUY else -f.qty
        if abs(pos) < _EPS or pos * q > 0:
            if abs(pos) < _EPS:
                ts_open = f.ts
                avg = f.price
                pos = q
            else:
                avg = (avg * abs(pos) + f.price * abs(q)) / (abs(pos) + abs(q))
                pos += q
            continue
        close_qty = min(abs(q), abs(pos))
        direction = "long" if pos > 0 else "short"
        pnl = (f.price - avg) * close_qty * (1.0 if pos > 0 else -1.0)
        rows.append((ts_open, f.ts, direction, close_qty, avg, f.price, pnl))
        new_pos = pos + q
        if abs(new_pos) < _EPS:
            pos = 0.0
        elif new_pos * pos < 0:  # flipped through zero
            pos = new_pos
            avg = f.price
            ts_open = f.ts
        else:
            pos = new_pos
    return pl.DataFrame(rows, schema=TRADES_SCHEMA, orient="row")


def max_drawdown(equity: np.ndarray | pl.Series) -> float:
    """Maximum peak-to-trough drawdown as a fraction of the running peak."""
    eq = equity.to_numpy() if isinstance(equity, pl.Series) else np.asarray(equity, dtype=float)
    if eq.size == 0:
        return 0.0
    peak = np.maximum.accumulate(eq)
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, 1.0 - eq / peak, 0.0)
    return float(np.max(dd))


def hit_rate(trades: pl.DataFrame) -> float:
    """Share of round trips with positive gross PnL; NaN when no trades."""
    if trades.is_empty():
        return math.nan
    return float((trades["pnl_usd"] > 0).mean())


def pnl_decomposition(result: BacktestResult) -> dict[str, float]:
    return {
        "gross": result.gross_pnl,
        "fees": result.fees_usd,
        "slippage": result.slippage_usd,
        "funding": result.funding_usd,
        "net": result.net_pnl,
    }


def cost_waterfall(result: BacktestResult) -> pl.DataFrame:
    """Waterfall steps from gross PnL down to net (costs stored as negatives)."""
    rows = [
        ("gross", result.gross_pnl),
        ("fees", -result.fees_usd),
        ("slippage", -result.slippage_usd),
        ("funding", -result.funding_usd),
        ("net", result.net_pnl),
    ]
    return pl.DataFrame(rows, schema={"step": pl.Utf8, "value": pl.Float64}, orient="row")


def summary(result: BacktestResult) -> dict[str, float]:
    trades = trades_from_fills(result.fills)
    return {
        "net_pnl": result.net_pnl,
        "gross_pnl": result.gross_pnl,
        "fees_usd": result.fees_usd,
        "slippage_usd": result.slippage_usd,
        "funding_usd": result.funding_usd,
        "final_equity": result.final_equity,
        "max_drawdown": max_drawdown(result.equity_curve["equity"]),
        "hit_rate": hit_rate(trades),
        "n_trades": float(len(trades)),
        "n_fills": float(len(result.fills)),
    }
