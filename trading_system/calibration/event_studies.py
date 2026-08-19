"""Stage 3.1 event studies: pool-touch reversal, magnet, LVN behavior.

Generic machinery: forward return/path matrices from an event index set, plus
bootstrap significance that corrects for overlapping events by resampling
whole clusters/blocks of events instead of individual ones (moving-block /
stationary bootstrap over the event set). Pure numpy, seeded rng everywhere.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Path matrices
# ---------------------------------------------------------------------------


def forward_return_paths(
    prices: np.ndarray, event_idx: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Log-return paths after each event.

    Returns (paths, kept_idx): paths[j, h] = log(P[i+h] / P[i]) for h in
    0..horizon, i = kept_idx[j]. Events with fewer than `horizon` samples of
    future are dropped so every row is complete (no ragged paths).
    """
    prices = np.asarray(prices, dtype=float)
    event_idx = np.asarray(event_idx, dtype=int)
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    kept = event_idx[(event_idx >= 0) & (event_idx + horizon < len(prices))]
    offsets = np.arange(horizon + 1)
    logp = np.log(prices)
    paths = logp[kept[:, None] + offsets[None, :]] - logp[kept, None]
    return paths, kept


# ---------------------------------------------------------------------------
# Overlap correction + bootstrap
# ---------------------------------------------------------------------------


def cluster_ids(event_idx: np.ndarray, min_gap: int) -> np.ndarray:
    """Chain events closer than `min_gap` samples into one cluster (greedy)."""
    event_idx = np.asarray(event_idx, dtype=int)
    if len(event_idx) == 0:
        return np.array([], dtype=int)
    order = np.argsort(event_idx, kind="stable")
    ids = np.empty(len(event_idx), dtype=int)
    cur = 0
    prev = event_idx[order[0]]
    ids[order[0]] = 0
    for j in order[1:]:
        if event_idx[j] - prev >= min_gap:
            cur += 1
        ids[j] = cur
        prev = event_idx[j]
    return ids


def block_ids(event_idx: np.ndarray, block_len: int) -> np.ndarray:
    """Fixed non-overlapping time blocks: events in one block resample together.

    Unlike `cluster_ids` this never chains a long run of dense events into a
    single giant cluster, so it stays usable when event windows overlap densely
    (magnet study samples every bar).
    """
    if block_len < 1:
        raise ValueError("block_len must be >= 1")
    return np.asarray(event_idx, dtype=int) // block_len


@dataclass(frozen=True)
class BootstrapResult:
    """Effect size with cluster-bootstrap CI and two-sided p-value."""

    effect: float
    event_rate: float
    control_rate: float
    ci_low: float
    ci_high: float
    p_value: float
    n_events: int
    n_clusters: int


def _resample_clusters(
    values: np.ndarray, clusters: np.ndarray, rng: np.random.Generator, n_boot: int
) -> np.ndarray:
    """Bootstrap distribution of the mean, resampling whole clusters."""
    uniq = np.unique(clusters)
    groups = [values[clusters == c] for c in uniq]
    sums = np.array([g.sum() for g in groups])
    sizes = np.array([len(g) for g in groups], dtype=float)
    k = len(uniq)
    pick = rng.integers(0, k, size=(n_boot, k))
    return sums[pick].sum(axis=1) / np.maximum(sizes[pick].sum(axis=1), 1.0)


def stationary_bootstrap_indices(
    n: int, mean_block: float, n_boot: int, rng: np.random.Generator
) -> np.ndarray:
    """Politis-Romano stationary bootstrap index matrix (n_boot, n).

    Geometric block lengths with mean `mean_block`, wrap-around continuation.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    p = min(1.0, 1.0 / max(mean_block, 1.0))
    restart = rng.random((n_boot, n)) < p
    restart[:, 0] = True
    starts = rng.integers(0, n, size=(n_boot, n))
    idx = np.empty((n_boot, n), dtype=int)
    for b in range(n_boot):
        cur = 0
        for t in range(n):
            cur = starts[b, t] if restart[b, t] else (cur + 1) % n
            idx[b, t] = cur
    return idx


def bootstrap_effect(
    event_values: np.ndarray,
    control_values: np.ndarray,
    event_clusters: np.ndarray | None = None,
    control_clusters: np.ndarray | None = None,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> BootstrapResult:
    """Difference of means (event - control) with cluster-bootstrap inference.

    Overlapping events must share a cluster id (see `block_ids`); clusters are
    the resampling unit, which corrects the CI/p-value for event overlap. The
    p-value is the two-sided bootstrap-inversion p for effect == 0.
    """
    ev = np.asarray(event_values, dtype=float)
    ct = np.asarray(control_values, dtype=float)
    if len(ev) == 0 or len(ct) == 0:
        raise ValueError("both event and control sets must be non-empty")
    ev_cl = np.zeros(len(ev), dtype=int) if event_clusters is None else np.asarray(event_clusters)
    ct_cl = (
        np.arange(len(ct), dtype=int) if control_clusters is None else np.asarray(control_clusters)
    )
    rng = np.random.default_rng(seed)
    boot_ev = _resample_clusters(ev, ev_cl, rng, n_boot)
    boot_ct = _resample_clusters(ct, ct_cl, rng, n_boot)
    boot = boot_ev - boot_ct
    effect = float(ev.mean() - ct.mean())
    alpha = 1.0 - ci
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    p_lo = (1 + np.sum(boot <= 0.0)) / (n_boot + 1)
    p_hi = (1 + np.sum(boot >= 0.0)) / (n_boot + 1)
    p = min(1.0, 2.0 * min(p_lo, p_hi))
    return BootstrapResult(
        effect=effect,
        event_rate=float(ev.mean()),
        control_rate=float(ct.mean()),
        ci_low=float(lo),
        ci_high=float(hi),
        p_value=float(p),
        n_events=len(ev),
        n_clusters=int(len(np.unique(ev_cl))),
    )


def mean_path_ci(
    paths: np.ndarray,
    clusters: np.ndarray | None = None,
    n_boot: int = 500,
    seed: int = 42,
    ci: float = 0.95,
    stat_fn: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean forward path with a cluster-bootstrap CI band per horizon step.

    Returns (mean, lo, hi), each of shape (horizon+1,). `stat_fn` maps a path
    matrix to a per-horizon statistic (default: column mean).
    """
    paths = np.asarray(paths, dtype=float)
    n = len(paths)
    if n == 0:
        raise ValueError("empty path matrix")
    cl = np.arange(n) if clusters is None else np.asarray(clusters)
    stat = stat_fn or (lambda m: m.mean(axis=0))
    rng = np.random.default_rng(seed)
    uniq = np.unique(cl)
    groups = [np.flatnonzero(cl == c) for c in uniq]
    boot = np.empty((n_boot, paths.shape[1]))
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows = np.concatenate([groups[j] for j in pick])
        boot[b] = stat(paths[rows])
    alpha = 1.0 - ci
    lo = np.quantile(boot, alpha / 2, axis=0)
    hi = np.quantile(boot, 1 - alpha / 2, axis=0)
    return stat(paths), lo, hi


# ---------------------------------------------------------------------------
# Heat-map helpers shared by the studies
# ---------------------------------------------------------------------------


def top_decile_mask(heat_row: np.ndarray, top_decile: float = 0.1) -> np.ndarray:
    """Boolean mask of the hottest `top_decile` share of cells (positive only)."""
    row = np.asarray(heat_row, dtype=float)
    k = max(1, int(np.floor(len(row) * top_decile)))
    order = np.argsort(-row, kind="stable")[:k]
    mask = np.zeros(len(row), dtype=bool)
    sel = order[row[order] > 0.0]
    mask[sel] = True
    return mask


def price_bucket(prices: np.ndarray, bucket_edges: np.ndarray) -> np.ndarray:
    """Bucket index per price; -1 when outside [edges[0], edges[-1])."""
    prices = np.asarray(prices, dtype=float)
    edges = np.asarray(bucket_edges, dtype=float)
    idx = np.searchsorted(edges, prices, side="right") - 1
    idx[(prices < edges[0]) | (prices >= edges[-1])] = -1
    return idx


def top_decile_touch_events(
    prices: np.ndarray,
    heat: np.ndarray,
    bucket_edges: np.ndarray,
    top_decile: float = 0.1,
) -> np.ndarray:
    """Indices where price enters a top-decile-heat bucket of the prior snapshot.

    heat[i] must be the snapshot known at sample i; the touch test at i uses
    heat[i-1] so no same-bar information leaks into the event definition.
    """
    prices = np.asarray(prices, dtype=float)
    if heat.shape[0] != len(prices):
        raise ValueError("heat rows must align with prices")
    buckets = price_bucket(prices, bucket_edges)
    in_pool = np.zeros(len(prices), dtype=bool)
    for i in range(1, len(prices)):
        b = buckets[i]
        if b >= 0:
            in_pool[i] = top_decile_mask(heat[i - 1], top_decile)[b]
    entering = in_pool & ~np.concatenate([[False], in_pool[:-1]])
    entering[0] = False
    return np.flatnonzero(entering)


# ---------------------------------------------------------------------------
# Study (a): reversal after touching a top-decile pool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReversalStudyResult:
    stats: BootstrapResult
    horizon: int
    k_atr: float
    event_paths: np.ndarray  # signed against approach direction: >0 == reversal move
    control_paths: np.ndarray
    event_clusters: np.ndarray
    control_clusters: np.ndarray


def _reversal_outcomes(
    prices: np.ndarray,
    atr: np.ndarray,
    idx: np.ndarray,
    k_atr: float,
    horizon: int,
    direction_lookback: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(binary reversal outcome, signed reversal-move paths, kept idx)."""
    keep = idx[(idx >= direction_lookback) & (idx + horizon < len(prices))]
    direction = np.sign(prices[keep] - prices[keep - direction_lookback])
    direction[direction == 0] = 1.0
    paths, kept = forward_return_paths(prices, keep, horizon)
    # move against the approach direction, in price units, sign-flipped so that
    # positive == reversal progress
    signed = -direction[:, None] * (np.exp(paths) - 1.0) * prices[kept, None]
    thresh = k_atr * atr[kept]
    outcome = (signed.max(axis=1) >= thresh).astype(float)
    return outcome, signed / np.maximum(atr[kept, None], 1e-12), kept


def reversal_study(
    prices: np.ndarray,
    atr: np.ndarray,
    events: np.ndarray,
    k_atr: float = 1.0,
    horizon: int = 30,
    direction_lookback: int = 5,
    n_controls_per_event: int = 4,
    n_boot: int = 1000,
    seed: int = 42,
    block_len: int | None = None,
) -> ReversalStudyResult:
    """P(reversal >= k*ATR within `horizon`) after events vs matched base rate.

    Controls are uniformly sampled non-event indices at the same horizon
    (matched horizons); significance via block bootstrap over the event set.
    """
    prices = np.asarray(prices, dtype=float)
    atr = np.asarray(atr, dtype=float)
    events = np.asarray(events, dtype=int)
    rng = np.random.default_rng(seed)
    lo, hi = direction_lookback, len(prices) - horizon - 1
    if hi <= lo:
        raise ValueError("series too short for the requested horizon")
    forbidden = set(events.tolist())
    candidates = np.array([i for i in range(lo, hi) if i not in forbidden], dtype=int)
    n_ctl = min(len(candidates), max(1, n_controls_per_event) * max(1, len(events)))
    controls = np.sort(rng.choice(candidates, size=n_ctl, replace=False))

    bl = block_len or horizon
    ev_out, ev_paths, ev_kept = _reversal_outcomes(
        prices, atr, events, k_atr, horizon, direction_lookback
    )
    ct_out, ct_paths, ct_kept = _reversal_outcomes(
        prices, atr, controls, k_atr, horizon, direction_lookback
    )
    if len(ev_out) == 0:
        raise ValueError("no usable events after horizon trimming")
    ev_cl = block_ids(ev_kept, bl)
    ct_cl = block_ids(ct_kept, bl)
    stats = bootstrap_effect(ev_out, ct_out, ev_cl, ct_cl, n_boot=n_boot, seed=seed)
    log.debug("reversal_study", effect=stats.effect, p=stats.p_value, n=stats.n_events)
    return ReversalStudyResult(
        stats=stats,
        horizon=horizon,
        k_atr=k_atr,
        event_paths=ev_paths,
        control_paths=ct_paths,
        event_clusters=ev_cl,
        control_clusters=ct_cl,
    )


# ---------------------------------------------------------------------------
# Study (b): magnet — P(reach pool at distance d within T)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MagnetResult:
    bin_edges_atr: np.ndarray  # (n_bins + 1,)
    p_reach: np.ndarray  # (n_bins,)
    ci_low: np.ndarray
    ci_high: np.ndarray
    n_samples: np.ndarray  # (n_bins,) int
    horizon: int


def magnet_study(
    prices: np.ndarray,
    atr: np.ndarray,
    heat: np.ndarray,
    bucket_edges: np.ndarray,
    horizon: int = 30,
    top_decile: float = 0.1,
    bin_edges_atr: np.ndarray | None = None,
    stride: int = 1,
    n_boot: int = 500,
    seed: int = 42,
) -> MagnetResult:
    """P(price reaches a top-decile pool at distance d within `horizon`) vs d.

    For every strided sample the nearest top-decile pool above and below the
    current price (from the concurrent snapshot) become candidates; reach means
    trading through the pool's bucket within the horizon. Per-bin CIs use a
    block bootstrap (blocks of `horizon` samples) over candidate outcomes.
    """
    prices = np.asarray(prices, dtype=float)
    atr = np.asarray(atr, dtype=float)
    edges = np.asarray(bucket_edges, dtype=float)
    if bin_edges_atr is None:
        bin_edges_atr = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0])
    centers = (edges[:-1] + edges[1:]) / 2.0
    dist_atr: list[float] = []
    reached: list[float] = []
    t_of: list[int] = []
    n = len(prices)
    for t in range(0, n - horizon - 1, stride):
        mask = top_decile_mask(heat[t], top_decile)
        if not mask.any():
            continue
        p = prices[t]
        a = max(atr[t], 1e-12)
        hot = np.flatnonzero(mask)
        fmax = prices[t + 1 : t + horizon + 1].max()
        fmin = prices[t + 1 : t + horizon + 1].min()
        above = hot[centers[hot] > p]
        below = hot[centers[hot] < p]
        for cand, is_above in ((above, True), (below, False)):
            if len(cand) == 0:
                continue
            b = cand[np.argmin(centers[cand])] if is_above else cand[np.argmax(centers[cand])]
            d = abs(centers[b] - p) / a
            hit = fmax >= edges[b] if is_above else fmin <= edges[b + 1]
            dist_atr.append(d)
            reached.append(float(hit))
            t_of.append(t)
    dist = np.asarray(dist_atr)
    hit_arr = np.asarray(reached)
    t_arr = np.asarray(t_of, dtype=int)
    n_bins = len(bin_edges_atr) - 1
    p_reach = np.full(n_bins, np.nan)
    ci_lo = np.full(n_bins, np.nan)
    ci_hi = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    rng = np.random.default_rng(seed)
    for j in range(n_bins):
        sel = (dist >= bin_edges_atr[j]) & (dist < bin_edges_atr[j + 1])
        counts[j] = int(sel.sum())
        if counts[j] == 0:
            continue
        vals = hit_arr[sel]
        cl = block_ids(t_arr[sel], horizon)
        boot = _resample_clusters(vals, cl, rng, n_boot)
        p_reach[j] = vals.mean()
        ci_lo[j], ci_hi[j] = np.quantile(boot, [0.025, 0.975])
    return MagnetResult(
        bin_edges_atr=np.asarray(bin_edges_atr, dtype=float),
        p_reach=p_reach,
        ci_low=ci_lo,
        ci_high=ci_hi,
        n_samples=counts,
        horizon=horizon,
    )


# ---------------------------------------------------------------------------
# Study (c): LVN behavior
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LvnStudyResult:
    stats: BootstrapResult  # effect on mean |log move| at the horizon
    horizon: int
    event_abs_paths: np.ndarray  # |log-return| paths after LVN entry
    control_abs_paths: np.ndarray
    event_clusters: np.ndarray
    control_clusters: np.ndarray


def zone_entry_events(prices: np.ndarray, zones: np.ndarray) -> np.ndarray:
    """Indices where price enters any [low, high] zone from outside."""
    prices = np.asarray(prices, dtype=float)
    zones = np.atleast_2d(np.asarray(zones, dtype=float))
    inside = np.zeros(len(prices), dtype=bool)
    for lo, hi in zones:
        inside |= (prices >= lo) & (prices <= hi)
    entering = inside & ~np.concatenate([[False], inside[:-1]])
    entering[0] = False
    return np.flatnonzero(entering)


def lvn_study(
    prices: np.ndarray,
    lvn_zones: np.ndarray,
    horizon: int = 20,
    n_controls_per_event: int = 4,
    n_boot: int = 1000,
    seed: int = 42,
    block_len: int | None = None,
) -> LvnStudyResult:
    """Forward path stats when price enters a low-volume node vs elsewhere.

    LVN hypothesis: price traverses low-volume areas faster, so the tested
    effect is the difference of mean |log return| at the horizon between LVN
    entries and matched non-LVN samples.
    """
    prices = np.asarray(prices, dtype=float)
    zones = np.atleast_2d(np.asarray(lvn_zones, dtype=float))
    events = zone_entry_events(prices, zones)
    events = events[events + horizon < len(prices)]
    if len(events) == 0:
        raise ValueError("no LVN entry events in the series")
    inside = np.zeros(len(prices), dtype=bool)
    for lo, hi in zones:
        inside |= (prices >= lo) & (prices <= hi)
    rng = np.random.default_rng(seed)
    candidates = np.flatnonzero(~inside[: len(prices) - horizon - 1])
    candidates = np.setdiff1d(candidates, events)
    n_ctl = min(len(candidates), max(1, n_controls_per_event) * len(events))
    controls = np.sort(rng.choice(candidates, size=n_ctl, replace=False))

    ev_paths, ev_kept = forward_return_paths(prices, events, horizon)
    ct_paths, ct_kept = forward_return_paths(prices, controls, horizon)
    bl = block_len or horizon
    ev_cl = block_ids(ev_kept, bl)
    ct_cl = block_ids(ct_kept, bl)
    stats = bootstrap_effect(
        np.abs(ev_paths[:, -1]),
        np.abs(ct_paths[:, -1]),
        ev_cl,
        ct_cl,
        n_boot=n_boot,
        seed=seed,
    )
    return LvnStudyResult(
        stats=stats,
        horizon=horizon,
        event_abs_paths=np.abs(ev_paths),
        control_abs_paths=np.abs(ct_paths),
        event_clusters=ev_cl,
        control_clusters=ct_cl,
    )
