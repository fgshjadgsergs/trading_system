"""Spoofing metrics over level episodes: lifetimes, cancel/fill, flicker, iceberg.

All functions consume a :class:`~trading_system.spoof.lifecycle.LevelJournal`
(or the plain frames it produces) and return polars frames, so an integrator
can join them onto any book map.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right

import polars as pl

from trading_system.core.timeutils import NS_PER_MS, NS_PER_S
from trading_system.spoof.lifecycle import LevelJournal, spoof_config


def episodes_frame(journal: LevelJournal, *, exec_min_fill: float = 0.5) -> pl.DataFrame:
    """One row per level episode with lifetime and fill accounting.

    ``outcome`` is ``alive`` while the level still stands, ``executed`` when it
    died with fill_frac >= ``exec_min_fill`` and ``canceled`` otherwise.
    """
    rows = [
        (
            ep.episode_id,
            ep.side,
            ep.price,
            ep.birth_ts,
            ep.death_ts,
            ep.last_ts,
            ep.lifetime_ns / NS_PER_MS,
            ep.max_qty,
            ep.filled_qty,
            ep.canceled_qty,
            ep.grown_qty,
            ep.n_grew,
            ep.n_reductions,
            ep.fill_frac,
            ep.was_large,
            ep.iceberg_refills,
            ep.refill_chain_id,
            ep.alive,
        )
        for ep in journal.episodes
    ]
    schema: dict[str, pl.DataType] = {
        "episode_id": pl.Int64,
        "side": pl.Utf8,
        "price": pl.Float64,
        "birth_ts": pl.Int64,
        "death_ts": pl.Int64,
        "last_ts": pl.Int64,
        "lifetime_ms": pl.Float64,
        "max_qty": pl.Float64,
        "filled_qty": pl.Float64,
        "canceled_qty": pl.Float64,
        "grown_qty": pl.Float64,
        "n_grew": pl.Int64,
        "n_reductions": pl.Int64,
        "fill_frac": pl.Float64,
        "was_large": pl.Boolean,
        "iceberg_refills": pl.Int64,
        "refill_chain_id": pl.Int64,
        "alive": pl.Boolean,
    }
    df = pl.DataFrame(rows, schema=schema, orient="row")
    return df.with_columns(
        pl.when(pl.col("alive"))
        .then(pl.lit("alive"))
        .when(pl.col("fill_frac") >= exec_min_fill)
        .then(pl.lit("executed"))
        .otherwise(pl.lit("canceled"))
        .alias("outcome")
    )


def large_level_lifetimes(episodes: pl.DataFrame) -> pl.DataFrame:
    """Dead large-level episodes split into executed vs canceled populations."""
    return episodes.filter(pl.col("was_large") & ~pl.col("alive")).select(
        "episode_id", "side", "price", "lifetime_ms", "fill_frac", "outcome"
    )


def cancel_to_fill(events: pl.DataFrame, window_ns: int = 60 * NS_PER_S) -> pl.DataFrame:
    """Canceled vs filled qty per side per time window.

    ``cancel_to_fill`` is +inf when the window saw cancels but no fills and
    NaN when it saw neither. Input is ``LevelJournal.events_frame()``.
    """
    return (
        events.with_columns(((pl.col("ts") // window_ns) * window_ns).alias("window_ts"))
        .group_by("window_ts", "side")
        .agg(
            pl.col("filled_qty").sum().alias("filled"),
            pl.col("canceled_qty").sum().alias("canceled"),
        )
        .with_columns((pl.col("canceled") / pl.col("filled")).alias("cancel_to_fill"))
        .sort("window_ts", "side")
    )


def flicker_flags(
    episodes: pl.DataFrame,
    *,
    k: int | None = None,
    window_s: float | None = None,
    price_eps: float = 0.0,
    max_fill_frac: float = 0.2,
) -> pl.DataFrame:
    """Flag flickering levels: a large level at (about) the same price that is
    born and dies unfilled >= k times within ``window_s``.

    An episode *qualifies* if it was large, died, and its fill fraction is at
    most ``max_fill_frac``. ``flicker_count`` is the number of qualifying
    births at the same (side, price bucket) within +-window of the episode's
    own birth (itself included); ``flicker`` is ``flicker_count >= k``.
    Defaults for ``k``/``window_s`` come from config ``spoof.flicker_k`` /
    ``spoof.flicker_window_s``.
    """
    cfg = spoof_config() if (k is None or window_s is None) else {}
    k = int(cfg.get("flicker_k", 3)) if k is None else int(k)
    window_s = float(cfg.get("flicker_window_s", 60)) if window_s is None else float(window_s)
    window_ns = int(window_s * NS_PER_S)

    def bucket(price: float) -> float:
        return round(price / price_eps) if price_eps > 0 else price

    qual = episodes.filter(
        pl.col("was_large") & ~pl.col("alive") & (pl.col("fill_frac") <= max_fill_frac)
    )
    births: dict[tuple[str, float], list[int]] = {}
    for side, price, b in qual.select("side", "price", "birth_ts").iter_rows():
        births.setdefault((side, bucket(price)), []).append(b)
    for v in births.values():
        v.sort()
    counts: list[int] = []
    qual_ids = set(qual["episode_id"].to_list())
    for eid, side, price, b in episodes.select("episode_id", "side", "price", "birth_ts").iter_rows():
        if eid not in qual_ids:
            counts.append(0)
            continue
        ts_list = births[(side, bucket(price))]
        counts.append(bisect_right(ts_list, b + window_ns) - bisect_left(ts_list, b - window_ns))
    return episodes.with_columns(
        pl.Series("flicker_count", counts, dtype=pl.Int64),
        (pl.Series(counts, dtype=pl.Int64) >= k).alias("flicker"),
    )


def iceberg_flags(episodes: pl.DataFrame, *, min_refills: int = 2) -> pl.DataFrame:
    """Flag iceberg behavior: repeated refills at the same price right after fills.

    Refill chains are built by the journal (``refill_chain_id``); the chain
    total is the max running refill count within the chain. ``iceberg`` is
    ``chain_refills >= min_refills``.
    """
    chained = episodes.filter(pl.col("refill_chain_id") >= 0)
    totals = (
        chained.group_by("side", "price", "refill_chain_id")
        .agg(pl.col("iceberg_refills").max().alias("chain_refills"))
        if not chained.is_empty()
        else pl.DataFrame(
            schema={
                "side": pl.Utf8,
                "price": pl.Float64,
                "refill_chain_id": pl.Int64,
                "chain_refills": pl.Int64,
            }
        )
    )
    return (
        episodes.join(totals, on=["side", "price", "refill_chain_id"], how="left")
        .with_columns(pl.col("chain_refills").fill_null(0))
        .with_columns((pl.col("chain_refills") >= min_refills).alias("iceberg"))
    )


def annotate_episodes(
    journal: LevelJournal,
    *,
    exec_min_fill: float = 0.5,
    flicker_k: int | None = None,
    flicker_window_s: float | None = None,
    price_eps: float = 0.0,
    max_flicker_fill: float = 0.2,
    iceberg_min_refills: int = 2,
) -> pl.DataFrame:
    """Episodes frame with outcome, flicker and iceberg columns in one pass."""
    df = episodes_frame(journal, exec_min_fill=exec_min_fill)
    df = flicker_flags(
        df, k=flicker_k, window_s=flicker_window_s, price_eps=price_eps,
        max_fill_frac=max_flicker_fill,
    )
    return iceberg_flags(df, min_refills=iceberg_min_refills)
