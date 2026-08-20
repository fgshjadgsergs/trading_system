"""Long/short side shares for allocate() from the exchange ratio streams.

v1 split the ΔOI inflow 50/50. This module derives a time-varying long share
from the unified 'ratio' stream (globalLongShortAccountRatio,
topLongShortPositionRatio, takerlongshortRatio) and attaches it to bars
strictly backward — a bar only sees ratio points published before its close.
"""

from __future__ import annotations

import polars as pl

DEFAULT_BLEND: dict[str, float] = {
    "global_ls_account": 0.4,
    "top_ls_position": 0.3,
    "taker_ls": 0.3,
}


def long_share_series(
    ratios: pl.DataFrame,
    blend: dict[str, float] | None = None,
) -> pl.DataFrame:
    """Blend per-metric long shares into one series (ts_event, long_share).

    For every timestamp where any metric publishes, each metric contributes its
    latest value at-or-before that moment; blend weights renormalize over the
    metrics that have reported at least once by then.
    """
    blend = blend or DEFAULT_BLEND
    present = [m for m in blend if ratios.filter(pl.col("metric") == m).height > 0]
    if not present:
        return pl.DataFrame(schema={"ts_event": pl.Int64, "long_share": pl.Float64})
    grid = ratios.filter(pl.col("metric").is_in(present)).select("ts_event").unique().sort("ts_event")
    acc_num = pl.lit(0.0)
    acc_den = pl.lit(0.0)
    out = grid
    for metric in present:
        series = (
            ratios.filter(pl.col("metric") == metric)
            .select("ts_event", pl.col("long_share").alias(f"_{metric}"))
            .sort("ts_event")
        )
        out = out.join_asof(series, on="ts_event", strategy="backward")
        col = pl.col(f"_{metric}")
        w = blend[metric]
        acc_num = acc_num + pl.when(col.is_not_null()).then(col * w).otherwise(0.0)
        acc_den = acc_den + pl.when(col.is_not_null()).then(w).otherwise(0.0)
    return (
        out.with_columns((acc_num / acc_den).alias("long_share"))
        .select("ts_event", "long_share")
        .sort("ts_event")
    )


def join_long_share(
    bars: pl.DataFrame,
    ratios: pl.DataFrame,
    blend: dict[str, float] | None = None,
    clip: tuple[float, float] = (0.1, 0.9),
    default: float = 0.5,
) -> pl.DataFrame:
    """Attach a causal `long_share` column to bars (last value before close).

    Shares are clipped away from 0/1 so one side never vanishes entirely on a
    noisy ratio print; bars before the first ratio point get `default`.
    """
    series = long_share_series(ratios, blend)
    if series.is_empty():
        return bars.with_columns(pl.lit(default).alias("long_share"))
    joined = (
        bars.sort("ts_open")
        .with_columns((pl.col("ts_close") - 1).alias("_asof_ts"))
        .join_asof(
            series.rename({"ts_event": "_ratio_ts"}).sort("_ratio_ts"),
            left_on="_asof_ts",
            right_on="_ratio_ts",
            strategy="backward",
        )
        .drop("_asof_ts", "_ratio_ts", strict=False)
    )
    lo, hi = clip
    return joined.with_columns(
        pl.col("long_share").clip(lo, hi).fill_null(default).alias("long_share")
    ).sort("symbol", "ts_open")
