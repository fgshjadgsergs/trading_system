"""VWAP and ATR on bar frames."""

from __future__ import annotations

import polars as pl

from trading_system.core.timeutils import TIMEFRAME_NS


def with_vwap(bars: pl.DataFrame, session: str = "1d") -> pl.DataFrame:
    """Bar VWAP and session-anchored VWAP (anchored at UTC session boundaries)."""
    step = TIMEFRAME_NS[session]
    return (
        bars.sort("symbol", "ts_open")
        .with_columns(
            (pl.col("quote_volume") / pl.col("volume")).alias("vwap_bar"),
            (pl.col("ts_open") - pl.col("ts_open") % step).alias("_session"),
        )
        .with_columns(
            (
                pl.col("quote_volume").cum_sum().over("exchange", "symbol", "_session")
                / pl.col("volume").cum_sum().over("exchange", "symbol", "_session")
            ).alias("vwap_session")
        )
        .drop("_session")
    )


def with_atr(bars: pl.DataFrame, period: int = 14) -> pl.DataFrame:
    """Wilder ATR: TR smoothed with EWM(alpha=1/period, adjust=False)."""
    prev_close = pl.col("close").shift(1).over("exchange", "symbol")
    tr = pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low") - prev_close).abs(),
    )
    return bars.sort("symbol", "ts_open").with_columns(
        tr.alias("tr"),
        tr.ewm_mean(alpha=1.0 / period, adjust=False)
        .over("exchange", "symbol")
        .alias("atr"),
    )
