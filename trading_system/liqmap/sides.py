"""Long/short side shares for allocate() from the exchange ratio streams.

v1 split the ΔOI inflow 50/50. This module derives a time-varying long share
from the unified 'ratio' stream (globalLongShortAccountRatio,
topLongShortPositionRatio, takerlongshortRatio) and attaches it to bars
strictly backward — a bar only sees ratio points published before its close.

Robustness/model options are opt-in; defaults reproduce the v1 behaviour
bit-for-bit (the only unconditional change is dropping non-finite shares,
which used to poison every later bar and crash allocate()).
"""

from __future__ import annotations

import polars as pl

DEFAULT_BLEND: dict[str, float] = {
    "global_ls_account": 0.4,
    "top_ls_position": 0.3,
    "taker_ls": 0.3,
}

_S_TO_NS = 1_000_000_000


def _clean_ratios(
    ratios: pl.DataFrame,
    *,
    availability: str = "ts_event",
    dedup: bool = False,
    validate_range: bool = False,
) -> pl.DataFrame:
    """Input hygiene shared by every consumer of the ratio stream.

    Non-finite shares are always dropped: NaN survives clip() and fill_null()
    in polars, so one bad print used to paint every later bar and then make
    LiqMap.allocate raise, while the calibrator path silently repaired it with
    nan_to_num — the two paths built different maps from the same lake.
    """
    out = ratios.filter(pl.col("long_share").is_finite().fill_null(False))
    if validate_range:
        out = out.filter(pl.col("long_share").is_between(0.0, 1.0))
    if availability == "ts_recv" and "ts_recv" in out.columns:
        # a point is usable only once it arrived: publication lag plus poller
        # phase can put ts_recv minutes after the exchange's grid label
        out = out.with_columns(
            pl.max_horizontal("ts_event", "ts_recv").alias("ts_event")
        )
    elif availability not in ("ts_event", "ts_recv"):
        raise ValueError("availability must be 'ts_event' or 'ts_recv'")
    if dedup:
        # retries republish the same grid point: the winner is the last one
        # ACCEPTED, not whichever row happened to land in the later part-file
        sort_keys = ["metric", "ts_event"]
        if "ts_recv" in out.columns:
            sort_keys.append("ts_recv")
        sort_keys.append("long_share")
        out = out.sort(sort_keys).unique(
            subset=["metric", "ts_event"], keep="last", maintain_order=True
        )
    return out


def long_share_series(
    ratios: pl.DataFrame,
    blend: dict[str, float] | None = None,
    *,
    max_age_s: float | None = None,
    dedup: bool = False,
    availability: str = "ts_event",
    debias_window_s: float | None = None,
    validate_range: bool = False,
) -> pl.DataFrame:
    """Blend per-metric long shares into one series (ts_event, long_share).

    For every timestamp where any metric publishes, each metric contributes its
    latest value at-or-before that moment; blend weights renormalize over the
    metrics that have reported at least once by then.

    `max_age_s` caps how old a metric's last value may be: past the cap the
    metric drops out and the existing renormalization reweights the survivors
    (without it a metric that reported once and died keeps its full weight
    forever). `debias_window_s` recenters the blend on 0.5 by subtracting its
    own causal rolling mean — the stock ratios carry a structural long tilt
    that otherwise swamps the sign of the signal.
    """
    blend = blend or DEFAULT_BLEND
    ratios = _clean_ratios(
        ratios, availability=availability, dedup=dedup, validate_range=validate_range
    )
    present = [m for m in blend if ratios.filter(pl.col("metric") == m).height > 0]
    if not present:
        return pl.DataFrame(schema={"ts_event": pl.Int64, "long_share": pl.Float64})
    grid = ratios.filter(pl.col("metric").is_in(present)).select("ts_event").unique().sort("ts_event")
    tol = int(max_age_s * _S_TO_NS) if max_age_s is not None else None
    acc_num = pl.lit(0.0)
    acc_den = pl.lit(0.0)
    out = grid
    for metric in present:
        series = (
            ratios.filter(pl.col("metric") == metric)
            .select("ts_event", pl.col("long_share").alias(f"_{metric}"))
            .sort("ts_event")
        )
        out = out.join_asof(series, on="ts_event", strategy="backward", tolerance=tol)
        col = pl.col(f"_{metric}")
        w = blend[metric]
        acc_num = acc_num + pl.when(col.is_not_null()).then(col * w).otherwise(0.0)
        acc_den = acc_den + pl.when(col.is_not_null()).then(w).otherwise(0.0)
    out = (
        out.with_columns((acc_num / acc_den).alias("long_share"))
        .select("ts_event", "long_share")
        .sort("ts_event")
    )
    if debias_window_s is not None:
        # causal: the rolling window closes on the current point, which is
        # known when the bar that reads it is allocated
        out = out.with_columns(
            pl.col("ts_event").cast(pl.Datetime("ns")).alias("_ts_dt")
        ).with_columns(
            (
                pl.col("long_share")
                - pl.col("long_share").rolling_mean_by(
                    "_ts_dt", window_size=f"{int(debias_window_s)}s", closed="right"
                )
                + 0.5
            ).alias("long_share")
        ).drop("_ts_dt")
    return out


def join_long_share(
    bars: pl.DataFrame,
    ratios: pl.DataFrame,
    blend: dict[str, float] | None = None,
    clip: tuple[float, float] = (0.1, 0.9),
    default: float = 0.5,
    *,
    max_age_s: float | None = None,
    dedup: bool = False,
    availability: str = "ts_event",
    debias_window_s: float | None = None,
    validate_range: bool = False,
    per_symbol: bool = False,
) -> pl.DataFrame:
    """Attach a causal `long_share` column to bars (last value before close).

    Shares are clipped away from 0/1 so one side never vanishes entirely on a
    noisy ratio print; bars before the first ratio point — or past `max_age_s`
    of the last one — get `default`. With `per_symbol` the series is built and
    joined per (exchange, symbol) instead of globally.
    """
    kw = dict(
        max_age_s=max_age_s,
        dedup=dedup,
        availability=availability,
        debias_window_s=debias_window_s,
        validate_range=validate_range,
    )
    keys = (
        [c for c in ("exchange", "symbol") if c in bars.columns and c in ratios.columns]
        if per_symbol
        else []
    )
    if keys:
        parts = []
        for key_vals, sub in ratios.group_by(keys, maintain_order=True):
            s = long_share_series(sub, blend, **kw)
            if s.is_empty():
                continue
            parts.append(
                s.with_columns(
                    [pl.lit(v).alias(k) for k, v in zip(keys, key_vals, strict=True)]
                )
            )
        series = pl.concat(parts) if parts else pl.DataFrame(
            schema={"ts_event": pl.Int64, "long_share": pl.Float64}
        )
    else:
        series = long_share_series(ratios, blend, **kw)
    if series.is_empty():
        return bars.with_columns(pl.lit(default).alias("long_share"))
    tol = int(max_age_s * _S_TO_NS) if max_age_s is not None else None
    right = series.rename({"ts_event": "_ratio_ts"}).sort(
        [*keys, "_ratio_ts"] if keys else "_ratio_ts"
    )
    joined = (
        bars.sort([*keys, "ts_open"] if keys else "ts_open")
        .with_columns((pl.col("ts_close") - 1).alias("_asof_ts"))
        .join_asof(
            right,
            left_on="_asof_ts",
            right_on="_ratio_ts",
            by=keys or None,
            strategy="backward",
            tolerance=tol,
        )
        .drop("_asof_ts", "_ratio_ts", strict=False)
    )
    lo, hi = clip
    return joined.with_columns(
        pl.col("long_share").clip(lo, hi).fill_null(default).alias("long_share")
    ).sort("symbol", "ts_open")
