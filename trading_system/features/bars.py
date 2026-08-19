"""Time and volume bars built from the tick tape (unified 'trade' frames).

Exchange kline_1m is only a checksum of these bars; every other timeframe,
including volume bars, is built here from ticks.

Conventions: ts_open = bucket start (UTC ns), ts_close = bucket end EXCLUSIVE
(ts_open + step for time bars, last trade ts + 1 for volume bars). A feature
"known at bar close" may use events with ts_event < ts_close only.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from trading_system.core.timeutils import TIMEFRAME_NS

_BAR_AGGS = [
    pl.col("price").first().alias("open"),
    pl.col("price").max().alias("high"),
    pl.col("price").min().alias("low"),
    pl.col("price").last().alias("close"),
    pl.col("qty").sum().alias("volume"),
    pl.col("qty_usd").sum().alias("quote_volume"),
    pl.col("qty").filter(pl.col("side") == 1).sum().fill_null(0.0).alias("taker_buy_volume"),
    pl.col("qty_usd")
    .filter(pl.col("side") == 1)
    .sum()
    .fill_null(0.0)
    .alias("taker_buy_quote_volume"),
    pl.len().cast(pl.Int64).alias("n_trades"),
    (pl.col("qty") * pl.col("side")).sum().alias("delta"),
    (pl.col("qty_usd") * pl.col("side")).sum().alias("delta_usd"),
]

BAR_COLUMNS = [
    "exchange",
    "symbol",
    "ts_open",
    "ts_close",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "n_trades",
    "delta",
    "delta_usd",
]


def time_bars(trades: pl.DataFrame, timeframe: str) -> pl.DataFrame:
    """Aggregate a unified trade frame into time bars of `timeframe`.

    Empty buckets are skipped (no synthetic flat bars). Output carries kline
    columns plus taker delta in coins and USD; sorted by (symbol, ts_open).
    """
    step = TIMEFRAME_NS[timeframe]
    out = (
        trades.sort("ts_event", "trade_id")
        .with_columns((pl.col("ts_event") - pl.col("ts_event") % step).alias("ts_open"))
        .group_by("exchange", "symbol", "ts_open", maintain_order=True)
        .agg(_BAR_AGGS)
        .with_columns((pl.col("ts_open") + step).alias("ts_close"))
        .select(BAR_COLUMNS)
        .sort("symbol", "ts_open")
    )
    return out


def _volume_bar_ids(qty_usd: np.ndarray, threshold: float) -> np.ndarray:
    """Accumulate-and-reset bar ids: a bar closes on the trade that fills it."""
    ids = np.empty(len(qty_usd), dtype=np.int64)
    bar = 0
    acc = 0.0
    for i in range(len(qty_usd)):
        ids[i] = bar
        acc += qty_usd[i]
        if acc >= threshold:
            bar += 1
            acc = 0.0
    return ids


try:  # optional acceleration; semantics identical
    from numba import njit

    _volume_bar_ids = njit(cache=True)(_volume_bar_ids)
except ImportError:  # pragma: no cover
    pass


def volume_bars(trades: pl.DataFrame, usd_per_bar: float) -> pl.DataFrame:
    """Bars closing once accumulated traded USD reaches `usd_per_bar`.

    Classic accumulate-and-reset volume bars: a trade belongs entirely to the
    bar open when it arrives (no splitting), the filling trade closes the bar,
    so every closed bar's quote volume is >= the threshold.
    """
    if usd_per_bar <= 0:
        raise ValueError("usd_per_bar must be positive")
    parts = []
    for part in trades.sort("ts_event", "trade_id").partition_by(
        "exchange", "symbol", maintain_order=True
    ):
        ids = _volume_bar_ids(part["qty_usd"].to_numpy(), usd_per_bar)
        parts.append(part.with_columns(pl.Series("_bar_id", ids)))
    out = (
        pl.concat(parts)
        .group_by("exchange", "symbol", "_bar_id", maintain_order=True)
        .agg(
            pl.col("ts_event").first().alias("ts_open"),
            (pl.col("ts_event").last() + 1).alias("ts_close"),
            *_BAR_AGGS,
        )
        .drop("_bar_id")
        .select(BAR_COLUMNS)
        .sort("symbol", "ts_open")
    )
    return out


def with_cvd(bars: pl.DataFrame) -> pl.DataFrame:
    """Append cumulative volume delta (coins and USD) per symbol."""
    return bars.with_columns(
        pl.col("delta").cum_sum().over("exchange", "symbol").alias("cvd"),
        pl.col("delta_usd").cum_sum().over("exchange", "symbol").alias("cvd_usd"),
    )


def compare_klines(
    own: pl.DataFrame,
    exchange_klines: pl.DataFrame,
    price_tol: float = 1e-9,
    volume_tol: float = 1e-6,
) -> pl.DataFrame:
    """Checksum own bars against exchange klines, joined on (symbol, ts_open).

    Exchange close time (open + step - 1ms) is ignored; only ts_open aligns the
    join, which also catches timezone / bar-boundary bugs. Returns per-bar
    comparison with an `ok` flag; caller asserts `ok.all()`.
    """
    joined = own.join(
        exchange_klines,
        on=["exchange", "symbol", "ts_open"],
        how="inner",
        suffix="_ex",
    )
    return joined.with_columns(
        (
            pl.col("open").is_not_null()
            & ((pl.col("open") - pl.col("open_ex")).abs() <= price_tol)
            & ((pl.col("high") - pl.col("high_ex")).abs() <= price_tol)
            & ((pl.col("low") - pl.col("low_ex")).abs() <= price_tol)
            & ((pl.col("close") - pl.col("close_ex")).abs() <= price_tol)
            & ((pl.col("volume") - pl.col("volume_ex")).abs() <= volume_tol)
            & ((pl.col("taker_buy_volume") - pl.col("taker_buy_volume_ex")).abs() <= volume_tol)
        ).alias("ok")
    )
