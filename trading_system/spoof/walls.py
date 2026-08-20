"""Stability-weighted wall zones for the S3 veto (track B3).

Turns scored level episodes into price zones a signal's path should respect.
Causality is explicit — at moment ts a zone may use only what is knowable:

* episodes RESOLVED before ts contribute their full stability score (the
  wall proved honest by then); their weight decays with age since death,
* episodes ALIVE at ts contribute a conservative lifetime-only prior — the
  fill/flicker/iceberg components are still unfolding, so they count as 0.

Full-life scores for live episodes (score_episodes output) are post-hoc by
construction and must not gate live decisions; see score.score_grid docs.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from trading_system.spoof.score import W_LIFE

ZONE_SCHEMA = {"lo": pl.Float64, "hi": pl.Float64, "heat_usd": pl.Float64}


def wall_zones_at(
    episodes_scored: pl.DataFrame,
    ts: int,
    band: float,
    min_score: float = 0.4,
    min_notional_usd: float = 0.0,
    dead_half_life_s: float = 3_600.0,
    live_life_weight: float = W_LIFE,
) -> pl.DataFrame:
    """Zones (lo, hi, heat_usd) active at ts from a scored episodes frame.

    heat_usd = price * max_qty * effective_score, zone = price ± band/2.
    `band` is typically the liq-map bucket size so both zone kinds merge on
    the same scale.
    """
    if episodes_scored.is_empty():
        return pl.DataFrame(schema=ZONE_SCHEMA)
    known = episodes_scored.filter(pl.col("birth_ts") <= ts)
    dead = known.filter(~pl.col("alive") & (pl.col("death_ts") <= ts))
    live = known.filter(pl.col("alive") | (pl.col("death_ts") > ts))

    rows: list[dict] = []
    age_s = (ts - dead["death_ts"].to_numpy()) / 1e9 if dead.height else np.array([])
    for i, row in enumerate(dead.iter_rows(named=True)):
        eff = row["score"] * 0.5 ** (age_s[i] / dead_half_life_s)
        notional = row["price"] * row["max_qty"]
        if eff >= min_score and notional >= min_notional_usd:
            rows.append(_zone(row["price"], band, notional * eff))

    if live.height:
        # lifetime percentile vs walls already resolved by ts (causal reference)
        ref = np.sort(dead["lifetime_ms"].to_numpy()) if dead.height else np.array([])
        for row in live.iter_rows(named=True):
            life_ms = (ts - row["birth_ts"]) / 1e6
            pct = float(np.searchsorted(ref, life_ms, side="right") / ref.size) if ref.size else 0.5
            eff = live_life_weight * min(pct, 1.0)
            notional = row["price"] * row["max_qty"]
            if eff >= min_score and notional >= min_notional_usd:
                rows.append(_zone(row["price"], band, notional * eff))
    if not rows:
        return pl.DataFrame(schema=ZONE_SCHEMA)
    return pl.DataFrame(rows, schema=ZONE_SCHEMA).sort("lo")


def _zone(price: float, band: float, heat: float) -> dict:
    return {"lo": price - band / 2, "hi": price + band / 2, "heat_usd": heat}


def merge_zones(*zone_frames: pl.DataFrame) -> pl.DataFrame:
    """Concatenate zone frames from different sources (map heat, walls)."""
    parts = [z for z in zone_frames if z is not None and z.height]
    if not parts:
        return pl.DataFrame(schema=ZONE_SCHEMA)
    return pl.concat([p.select("lo", "hi", "heat_usd") for p in parts]).sort("lo")
