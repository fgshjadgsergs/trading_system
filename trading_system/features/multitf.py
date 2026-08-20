"""Multi-timeframe feature block: volume z-scores, taker share, impulse bars,
ΔOI speed on 1m…1d — the context inputs for dynamic liq-map weights (stage 3).

Higher-TF features attached to a base bar at time t come ONLY from higher-TF
bars already closed by t (tf_ts_close <= t), never from the bar still forming.
"""

from __future__ import annotations

import polars as pl

from trading_system.features.bars import time_bars
from trading_system.features.joins import join_open_interest

FEATURE_COLS = [
    "quote_volume",
    "vol_z",
    "taker_buy_share",
    "impulse",
    "d_oi_usd",
    "oi_speed_usd_per_min",
]


def tf_features(
    trades: pl.DataFrame,
    oi: pl.DataFrame | None,
    timeframe: str,
    zscore_window: int = 96,
    impulse_k: float = 3.0,
) -> pl.DataFrame:
    """Per-bar features of one timeframe.

    The z-score baseline is the PREVIOUS `zscore_window` bars (shifted by one)
    so a bar never normalizes against itself. A zero-variance baseline yields
    a null vol_z (never inf).
    """
    bars = time_bars(trades, timeframe)
    if oi is not None and oi.height > 0:
        bars = join_open_interest(bars, oi)
    else:
        bars = bars.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("d_oi_usd"),
            pl.lit(None, dtype=pl.Float64).alias("oi_speed_usd_per_min"),
        )
    qv = pl.col("quote_volume")
    base_mean = (
        qv.shift(1).rolling_mean(zscore_window, min_samples=8).over("exchange", "symbol")
    )
    base_std = qv.shift(1).rolling_std(zscore_window, min_samples=8).over("exchange", "symbol")
    return (
        bars.with_columns(
            # sigma=0 baseline (identical volumes) must not divide: z undefined
            pl.when(base_std > 0)
            .then((qv - base_mean) / base_std)
            .otherwise(None)
            .alias("vol_z"),
            (pl.col("taker_buy_quote_volume") / qv).alias("taker_buy_share"),
            (qv > base_mean + impulse_k * base_std).fill_null(False).alias("impulse"),
            pl.lit(timeframe).alias("tf"),
        )
        .select(
            "exchange", "symbol", "tf", "ts_open", "ts_close", *FEATURE_COLS
        )
        .sort("symbol", "ts_open")
    )


def build_multitf(
    trades: pl.DataFrame,
    oi: pl.DataFrame | None,
    timeframes: list[str],
    zscore_window: int = 96,
    impulse_k: float = 3.0,
) -> pl.DataFrame:
    """Long frame of features for every timeframe, keyed (symbol, tf, ts_close)."""
    parts = [
        tf_features(trades, oi, tf, zscore_window=zscore_window, impulse_k=impulse_k)
        for tf in timeframes
    ]
    return pl.concat(parts)


def join_context(
    base_bars: pl.DataFrame, mtf: pl.DataFrame, timeframes: list[str]
) -> pl.DataFrame:
    """Attach each timeframe's latest CLOSED-bar features to base bars.

    For a base bar closing at t the joined tf row satisfies tf_ts_close <= t-1+1
    i.e. the higher-TF bar has fully closed strictly before the base bar's end.
    """
    out = base_bars.sort("ts_open").with_columns((pl.col("ts_close") - 1).alias("_asof_ts"))
    for tf in timeframes:
        right = (
            mtf.filter(pl.col("tf") == tf)
            .select(
                "exchange",
                "symbol",
                (pl.col("ts_close") - 1).alias("_tf_key"),
                *[pl.col(c).alias(f"{tf}_{c}") for c in FEATURE_COLS],
            )
            .sort("_tf_key")
        )
        out = out.join_asof(
            right,
            left_on="_asof_ts",
            right_on="_tf_key",
            by=["exchange", "symbol"],
            strategy="backward",
        ).drop("_tf_key", strict=False)
    return out.drop("_asof_ts").sort("symbol", "ts_open")
