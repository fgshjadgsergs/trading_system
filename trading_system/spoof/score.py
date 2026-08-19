"""Stability score in [0, 1] for level episodes; vectorized journal scoring.

The score is a documented monotonic combination of four signals:

    base = w_life * lifetime_pct + w_fill * fill_frac + w_ice * r / (r + 1)
    score = base * 0.5 ** (flicker_count / flicker_half_life)

with weights normalized to sum to 1 and ``r`` the iceberg refill count of the
episode's chain. It is non-decreasing in lifetime percentile, fill fraction
and iceberg refills (real hidden liquidity RAISES stability) and strictly
decreasing in flicker count (each ``flicker_half_life`` flickers halve the
score). ``lifetime_pct`` is the episode lifetime's percentile within a
reference population — dead large episodes of the same journal (all episodes
when none died large yet) — so persistent walls rank high without depending
on the churn of tiny levels.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from trading_system.spoof.lifecycle import LevelJournal
from trading_system.spoof.metrics import annotate_episodes

W_LIFE = 0.25
W_FILL = 0.50
W_ICE = 0.25


def stability_score(
    lifetime_pct: float,
    fill_frac: float,
    flicker_count: int,
    iceberg_refills: int,
    *,
    w_life: float = W_LIFE,
    w_fill: float = W_FILL,
    w_ice: float = W_ICE,
    flicker_half_life: float = 1.0,
) -> float:
    """Scalar stability score; see the module docstring for the formula."""
    total = w_life + w_fill + w_ice
    r = max(0, int(iceberg_refills))
    base = (
        w_life * min(max(lifetime_pct, 0.0), 1.0)
        + w_fill * min(max(fill_frac, 0.0), 1.0)
        + w_ice * (r / (r + 1.0))
    ) / total
    damp = 0.5 ** (max(0, int(flicker_count)) / flicker_half_life)
    return float(min(max(base * damp, 0.0), 1.0))


def lifetime_percentiles(episodes: pl.DataFrame) -> pl.Series:
    """Percentile of each episode's lifetime vs dead large episodes.

    Falls back to the whole population when no large episode has died yet;
    0.5 for an empty frame.
    """
    life = episodes["lifetime_ms"].to_numpy()
    ref = episodes.filter(pl.col("was_large") & ~pl.col("alive"))["lifetime_ms"].to_numpy()
    if ref.size == 0:
        ref = life
    if ref.size == 0:
        return pl.Series("lifetime_pct", np.full(len(episodes), 0.5))
    ref = np.sort(ref)
    pct = np.searchsorted(ref, life, side="right") / ref.size
    return pl.Series("lifetime_pct", np.clip(pct, 0.0, 1.0))


def score_episodes(
    annotated: pl.DataFrame,
    *,
    w_life: float = W_LIFE,
    w_fill: float = W_FILL,
    w_ice: float = W_ICE,
    flicker_half_life: float = 1.0,
) -> pl.DataFrame:
    """Vectorized stability score over an annotated episodes frame.

    Expects the columns produced by :func:`metrics.annotate_episodes`
    (``fill_frac``, ``flicker_count``, ``chain_refills``, ``lifetime_ms``,
    ``was_large``, ``alive``). Adds ``lifetime_pct`` and ``score``.
    """
    total = w_life + w_fill + w_ice
    df = annotated.with_columns(lifetime_percentiles(annotated))
    r = pl.col("chain_refills").clip(lower_bound=0)
    base = (
        w_life * pl.col("lifetime_pct").clip(0.0, 1.0)
        + w_fill * pl.col("fill_frac").clip(0.0, 1.0)
        + w_ice * (r / (r + 1))
    ) / total
    damp = pl.lit(0.5) ** (pl.col("flicker_count").clip(lower_bound=0) / flicker_half_life)
    return df.with_columns((base * damp).clip(0.0, 1.0).alias("score"))


def score_grid(journal: LevelJournal, episodes_scored: pl.DataFrame) -> pl.DataFrame:
    """Per-state level rows (ts, side, price, qty, score) for book-map weighting.

    Scores are episode-level (computed over each episode's full life), so this
    frame is for post-hoc map weighting and visualization, not for causal
    real-time decisions.
    """
    return (
        journal.grid_frame()
        .join(episodes_scored.select("episode_id", "score"), on="episode_id", how="left")
        .select("ts", "side", "price", "qty", "score")
    )


def journal_scores(
    journal: LevelJournal,
    *,
    flicker_k: int | None = None,
    flicker_window_s: float | None = None,
    price_eps: float = 0.0,
    iceberg_min_refills: int = 2,
    w_life: float = W_LIFE,
    w_fill: float = W_FILL,
    w_ice: float = W_ICE,
    flicker_half_life: float = 1.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Annotate + score a whole journal.

    Returns ``(episodes_scored, grid_scored)`` where ``grid_scored`` has
    columns (ts, side, price, qty, score).
    """
    annotated = annotate_episodes(
        journal,
        flicker_k=flicker_k,
        flicker_window_s=flicker_window_s,
        price_eps=price_eps,
        iceberg_min_refills=iceberg_min_refills,
    )
    scored = score_episodes(
        annotated,
        w_life=w_life,
        w_fill=w_fill,
        w_ice=w_ice,
        flicker_half_life=flicker_half_life,
    )
    return scored, score_grid(journal, scored)
