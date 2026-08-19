"""Fractal swings, equal extremes (stop clusters), age-decayed level weights."""

from __future__ import annotations

import numpy as np
import polars as pl


def fractal_swings(bars: pl.DataFrame, n: int = 2) -> pl.DataFrame:
    """n-bar fractals: a swing high has the strictly highest high among ±n bars.

    Ties break to nothing (no swing), keeping detection unambiguous. The last
    n bars can never confirm a swing — a fractal is known only n bars later,
    at ts_confirmed = ts_close of bar i+n.
    """
    highs = bars["high"].to_numpy()
    lows = bars["low"].to_numpy()
    ts_open = bars["ts_open"].to_numpy()
    ts_conf = bars["ts_close"].to_numpy()
    rows = []
    for i in range(n, len(highs) - n):
        window_h = highs[i - n : i + n + 1]
        window_l = lows[i - n : i + n + 1]
        if highs[i] == window_h.max() and (window_h == highs[i]).sum() == 1:
            rows.append(
                {"ts_open": int(ts_open[i]), "ts_confirmed": int(ts_conf[i + n]), "kind": "high", "price": float(highs[i])}
            )
        if lows[i] == window_l.min() and (window_l == lows[i]).sum() == 1:
            rows.append(
                {"ts_open": int(ts_open[i]), "ts_confirmed": int(ts_conf[i + n]), "kind": "low", "price": float(lows[i])}
            )
    schema = {"ts_open": pl.Int64, "ts_confirmed": pl.Int64, "kind": pl.Utf8, "price": pl.Float64}
    return pl.DataFrame(rows, schema=schema).sort("ts_open")


def equal_extremes(swings: pl.DataFrame, eps: float) -> pl.DataFrame:
    """Cluster same-kind swings whose prices sit within eps of the cluster mean.

    Two or more equal highs/lows form a stop cluster — the level price is the
    volume-agnostic mean, strength is the member count.
    """
    out_rows = []
    for kind in ("high", "low"):
        pts = swings.filter(pl.col("kind") == kind).sort("price")
        cluster: list[dict] = []
        for row in pts.iter_rows(named=True):
            if cluster and abs(row["price"] - np.mean([c["price"] for c in cluster])) > eps:
                if len(cluster) >= 2:
                    out_rows.append(_cluster_row(cluster, kind))
                cluster = []
            cluster.append(row)
        if len(cluster) >= 2:
            out_rows.append(_cluster_row(cluster, kind))
    schema = {
        "kind": pl.Utf8,
        "price": pl.Float64,
        "count": pl.Int64,
        "ts_last": pl.Int64,
    }
    return pl.DataFrame(out_rows, schema=schema).sort("price")


def _cluster_row(cluster: list[dict], kind: str) -> dict:
    return {
        "kind": kind,
        "price": float(np.mean([c["price"] for c in cluster])),
        "count": len(cluster),
        "ts_last": max(c["ts_confirmed"] for c in cluster),
    }


def level_weights(
    levels: pl.DataFrame, now_ts: int, half_life_s: float = 172_800.0
) -> pl.DataFrame:
    """Weight = strength * 0.5 ** (age / half_life); age from last touch."""
    age_s = (now_ts - pl.col("ts_last")) / 1_000_000_000
    return levels.with_columns(
        (pl.col("count") * (0.5 ** (age_s / half_life_s))).alias("weight")
    )
