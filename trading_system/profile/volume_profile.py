"""Volume profile: histogram over price, POC, Value Area, HVN/LVN nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from trading_system.core.timeutils import NS_PER_S

SESSIONS_UTC: dict[str, tuple[int, int]] = {
    "asia": (0, 8),
    "eu": (8, 13),
    "us": (13, 21),
    "late": (21, 24),
}


def profile(
    trades: pl.DataFrame,
    bin_size: float,
    ts_from: int | None = None,
    ts_to: int | None = None,
) -> pl.DataFrame:
    """USD volume per price bin over an optional time window, sorted by price."""
    if bin_size <= 0:
        raise ValueError("bin_size must be positive")
    t = trades
    if ts_from is not None:
        t = t.filter(pl.col("ts_event") >= ts_from)
    if ts_to is not None:
        t = t.filter(pl.col("ts_event") < ts_to)
    return (
        t.with_columns(((pl.col("price") / bin_size).floor() * bin_size).alias("bin_lo"))
        .group_by("bin_lo")
        .agg(
            pl.col("qty_usd").sum().alias("volume_usd"),
            pl.col("qty").sum().alias("volume"),
        )
        .with_columns((pl.col("bin_lo") + bin_size / 2).alias("price"))
        .sort("bin_lo")
    )


def session_profiles(trades: pl.DataFrame, bin_size: float) -> pl.DataFrame:
    """Profile per (UTC date, named session)."""
    day_ns = 86_400 * NS_PER_S
    hour = ((pl.col("ts_event") % day_ns) / (3_600 * NS_PER_S)).floor()
    session_expr = pl.lit(None, dtype=pl.Utf8)
    for name, (lo, hi) in reversed(SESSIONS_UTC.items()):
        session_expr = (
            pl.when((hour >= lo) & (hour < hi)).then(pl.lit(name)).otherwise(session_expr)
        )
    return (
        trades.with_columns(
            (pl.col("ts_event") - pl.col("ts_event") % day_ns).alias("date_ts"),
            session_expr.alias("session"),
            ((pl.col("price") / bin_size).floor() * bin_size).alias("bin_lo"),
        )
        .group_by("date_ts", "session", "bin_lo")
        .agg(pl.col("qty_usd").sum().alias("volume_usd"))
        .with_columns((pl.col("bin_lo") + bin_size / 2).alias("price"))
        .sort("date_ts", "session", "bin_lo")
    )


@dataclass(frozen=True, slots=True)
class ValueArea:
    poc: float
    val: float  # value area low (bin center)
    vah: float  # value area high (bin center)
    share: float  # volume share actually inside the VA


def poc_price(prof: pl.DataFrame) -> float:
    return float(prof.sort("volume_usd", descending=True)["price"][0])


def value_area(prof: pl.DataFrame, pct: float = 0.70) -> ValueArea:
    """Classic two-sided expansion from POC until >= pct of total volume."""
    vols = prof["volume_usd"].to_numpy()
    prices = prof["price"].to_numpy()
    total = vols.sum()
    if total <= 0:
        raise ValueError("empty profile")
    i = int(np.argmax(vols))
    lo = hi = i
    acc = vols[i]
    while acc < pct * total and (lo > 0 or hi < len(vols) - 1):
        below = vols[lo - 1] if lo > 0 else -1.0
        above = vols[hi + 1] if hi < len(vols) - 1 else -1.0
        if below >= above:
            lo -= 1
            acc += vols[lo]
        else:
            hi += 1
            acc += vols[hi]
    return ValueArea(
        poc=float(prices[i]), val=float(prices[lo]), vah=float(prices[hi]), share=acc / total
    )


def hvn_lvn(
    prof: pl.DataFrame, neighborhood: int = 2, hvn_quantile: float = 0.75, lvn_quantile: float = 0.35
) -> pl.DataFrame:
    """Mark high/low-volume nodes: local extrema of the histogram with size gates."""
    vols = prof["volume_usd"].to_numpy()
    n = len(vols)
    hi_gate = np.quantile(vols, hvn_quantile)
    lo_gate = np.quantile(vols, lvn_quantile)
    kinds = []
    for i in range(n):
        w = vols[max(0, i - neighborhood) : min(n, i + neighborhood + 1)]
        if vols[i] == w.max() and vols[i] >= hi_gate and (w.max() > w.min()):
            kinds.append("hvn")
        elif vols[i] == w.min() and vols[i] <= lo_gate and (w.max() > w.min()):
            kinds.append("lvn")
        else:
            kinds.append(None)
    return prof.with_columns(pl.Series("node", kinds, dtype=pl.Utf8))
