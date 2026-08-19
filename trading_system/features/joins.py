"""As-of joins of slow series (open interest, ratios) onto bars — no lookahead.

A bar may only see points with ts_event < ts_close (bar end is exclusive), so
the join key is ts_close - 1 with a backward asof strategy.
"""

from __future__ import annotations

import polars as pl

from trading_system.core.timeutils import NS_PER_MIN


def asof_join_backward(
    bars: pl.DataFrame,
    series: pl.DataFrame,
    value_cols: list[str],
    suffix: str = "",
) -> pl.DataFrame:
    """Attach last-known values of `series[value_cols]` to each bar.

    Strictly backward: a point at exactly ts_close belongs to the NEXT bar.
    """
    right = series.select(
        "exchange", "symbol", "ts_event", *[pl.col(c).alias(c + suffix) for c in value_cols]
    ).sort("ts_event")
    out = (
        bars.sort("ts_open")
        .with_columns((pl.col("ts_close") - 1).alias("_asof_ts"))
        .join_asof(
            right,
            left_on="_asof_ts",
            right_on="ts_event",
            by=["exchange", "symbol"],
            strategy="backward",
        )
        .drop("_asof_ts", "ts_event")
        .sort("symbol", "ts_open")
    )
    return out


def join_open_interest(bars: pl.DataFrame, oi: pl.DataFrame) -> pl.DataFrame:
    """Join OI, then per-bar ΔOI (coins/USD) and ΔOI speed (USD per minute)."""
    out = asof_join_backward(bars, oi, ["open_interest", "open_interest_usd"])
    return out.with_columns(
        pl.col("open_interest").diff().over("exchange", "symbol").alias("d_oi"),
        pl.col("open_interest_usd").diff().over("exchange", "symbol").alias("d_oi_usd"),
    ).with_columns(
        (
            pl.col("d_oi_usd") / ((pl.col("ts_close") - pl.col("ts_open")) / NS_PER_MIN)
        ).alias("oi_speed_usd_per_min")
    )
