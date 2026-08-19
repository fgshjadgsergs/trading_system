"""Stage 3.1 event-study machinery: paths, overlap-corrected bootstrap, studies."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_system.calibration.event_studies import (
    block_ids,
    bootstrap_effect,
    cluster_ids,
    forward_return_paths,
    lvn_study,
    magnet_study,
    mean_path_ci,
    price_bucket,
    reversal_study,
    stationary_bootstrap_indices,
    top_decile_mask,
    top_decile_touch_events,
)
from trading_system.calibration.synthetic import make_heat_builder, make_world

SEED = 42


# ---------------------------------------------------------------------------
# forward paths
# ---------------------------------------------------------------------------


def test_forward_return_paths_exact():
    prices = np.array([100.0, 110.0, 121.0, 133.1])
    paths, kept = forward_return_paths(prices, np.array([0, 1, 3]), horizon=2)
    # event 3 has no 2-bar future and must be dropped
    assert kept.tolist() == [0, 1]
    assert paths.shape == (2, 3)
    np.testing.assert_allclose(paths[:, 0], 0.0)
    np.testing.assert_allclose(paths[0], [0.0, np.log(1.1), np.log(1.21)], rtol=1e-12)
    np.testing.assert_allclose(paths[1], [0.0, np.log(1.1), np.log(1.21)], rtol=1e-12)


@settings(max_examples=50, deadline=None)
@given(
    seed=st.integers(0, 10_000),
    n=st.integers(10, 300),
    horizon=st.integers(1, 20),
    n_events=st.integers(1, 30),
)
def test_forward_return_paths_invariants(seed, n, horizon, n_events):
    rng = np.random.default_rng(seed)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    events = rng.integers(0, n, n_events)
    paths, kept = forward_return_paths(prices, events, horizon)
    assert paths.shape == (len(kept), horizon + 1)
    assert np.all(kept + horizon < n)
    if len(paths):
        np.testing.assert_allclose(paths[:, 0], 0.0)
        assert np.isfinite(paths).all()


# ---------------------------------------------------------------------------
# clustering / bootstrap
# ---------------------------------------------------------------------------


def test_cluster_and_block_ids():
    idx = np.array([0, 5, 100, 104, 300])
    assert cluster_ids(idx, 10).tolist() == [0, 0, 1, 1, 2]
    assert block_ids(idx, 50).tolist() == [0, 0, 2, 2, 6]
    with pytest.raises(ValueError):
        block_ids(idx, 0)


def test_bootstrap_effect_planted_vs_null():
    rng = np.random.default_rng(SEED)
    ctrl = rng.normal(0.0, 1.0, 400)
    planted = rng.normal(0.8, 1.0, 120)
    clusters = block_ids(np.arange(120) * 3, 30)  # overlapping events share blocks
    res = bootstrap_effect(planted, ctrl, clusters, n_boot=800, seed=SEED)
    assert res.p_value < 0.05
    assert res.ci_low > 0.0
    assert abs(res.effect - 0.8) < 0.3
    assert res.n_clusters == len(np.unique(clusters))

    null = rng.normal(0.0, 1.0, 120)
    res0 = bootstrap_effect(null, ctrl, clusters, n_boot=800, seed=SEED)
    assert res0.p_value > 0.05
    assert res0.ci_low < 0.0 < res0.ci_high


def test_bootstrap_effect_deterministic():
    rng = np.random.default_rng(3)
    a, b = rng.normal(0.2, 1, 50), rng.normal(0, 1, 100)
    r1 = bootstrap_effect(a, b, n_boot=300, seed=7)
    r2 = bootstrap_effect(a, b, n_boot=300, seed=7)
    assert r1 == r2


def test_stationary_bootstrap_indices():
    rng = np.random.default_rng(SEED)
    idx = stationary_bootstrap_indices(50, mean_block=5.0, n_boot=20, rng=rng)
    assert idx.shape == (20, 50)
    assert idx.min() >= 0 and idx.max() < 50
    idx2 = stationary_bootstrap_indices(50, 5.0, 20, np.random.default_rng(SEED))
    np.testing.assert_array_equal(idx, idx2)


def test_mean_path_ci_brackets_mean():
    rng = np.random.default_rng(SEED)
    paths = rng.normal(0.5, 0.2, size=(60, 11))
    mean, lo, hi = mean_path_ci(paths, n_boot=300, seed=SEED)
    assert mean.shape == lo.shape == hi.shape == (11,)
    assert np.all(lo <= mean) and np.all(mean <= hi)


# ---------------------------------------------------------------------------
# heat helpers
# ---------------------------------------------------------------------------


def test_top_decile_mask_ranks_positive_cells():
    row = np.array([0.0, 5.0, 1.0, 0.0, 3.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    mask = top_decile_mask(row, top_decile=0.2)  # budget = 2 cells
    assert mask.sum() == 2
    assert mask[1] and mask[4]
    # all-zero row -> empty mask, never "everything is hot"
    assert top_decile_mask(np.zeros(10), 0.2).sum() == 0


def test_price_bucket_bounds():
    edges = np.array([10.0, 20.0, 30.0])
    assert price_bucket(np.array([10.0, 19.9, 25.0, 30.0, 5.0]), edges).tolist() == [
        0, 0, 1, -1, -1,
    ]


def test_top_decile_touch_events_uses_prior_snapshot():
    edges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    n = 6
    heat = np.zeros((n, 10))
    heat[:, 7] = 1.0  # bucket 7 hot in every snapshot
    prices = np.array([1.5, 1.5, 7.5, 7.5, 1.5, 7.5])
    ev = top_decile_touch_events(prices, heat, edges, top_decile=0.1)
    assert ev.tolist() == [2, 5]  # entries into the hot bucket only


# ---------------------------------------------------------------------------
# study (a): reversal
# ---------------------------------------------------------------------------


def _dip_series(n=1500, period=100, seed=SEED):
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 8.0, n)
    events = []
    for c in range(period, n - 40, period):
        steps[c - 5 : c] -= 30.0
        steps[c : c + 20] += 12.0
        events.append(c)
    return 50_000.0 + np.cumsum(steps), np.full(n, 60.0), np.asarray(events)


def test_reversal_study_detects_planted_effect():
    prices, atr, events = _dip_series()
    res = reversal_study(prices, atr, events, k_atr=1.0, horizon=30, seed=SEED)
    assert res.stats.event_rate > 0.9
    assert res.stats.effect > 0.3
    assert res.stats.p_value < 0.05
    assert res.stats.ci_low > 0.0
    assert res.event_paths.shape[1] == 31


def test_reversal_study_null_is_insignificant():
    rng = np.random.default_rng(SEED)
    prices = 50_000.0 + np.cumsum(rng.normal(0.0, 8.0, 1500))
    events = np.sort(rng.choice(np.arange(100, 1400), 14, replace=False))
    res = reversal_study(prices, np.full(1500, 60.0), events, k_atr=1.0, horizon=30, seed=SEED)
    assert res.stats.p_value > 0.05
    assert abs(res.stats.effect) < 0.25


def test_reversal_study_deterministic():
    prices, atr, events = _dip_series()
    a = reversal_study(prices, atr, events, seed=SEED).stats
    b = reversal_study(prices, atr, events, seed=SEED).stats
    assert a == b


# ---------------------------------------------------------------------------
# study (b): magnet
# ---------------------------------------------------------------------------


def test_magnet_probability_decays_with_distance():
    world = make_world(n_bars=1000, seed=SEED, static_weights=(0.60, 0.10, 0.30))
    heat = make_heat_builder(world, decay_half_life_bars=200.0)(world.true_weights[0])
    atr = np.full(len(world.prices), world.atr * 5)
    res = magnet_study(
        world.prices, atr, heat, world.bucket_edges, horizon=30, stride=2, seed=SEED
    )
    valid = np.flatnonzero(~np.isnan(res.p_reach))
    assert len(valid) >= 3
    assert res.p_reach[valid[0]] > res.p_reach[valid[-1]]
    assert res.p_reach[valid[0]] > 0.5  # near pools get reached
    for j in valid:
        assert res.ci_low[j] <= res.p_reach[j] <= res.ci_high[j]
        assert res.n_samples[j] > 0


# ---------------------------------------------------------------------------
# study (c): LVN
# ---------------------------------------------------------------------------


def _lvn_series(n=3000, zone=(49_800.0, 50_200.0), accel=3.0, seed=SEED):
    # mean-reverting walk that keeps revisiting the zone; steps widen inside it
    rng = np.random.default_rng(seed)
    p = np.empty(n)
    p[0] = 50_600.0
    for t in range(1, n):
        scale = 30.0 * (accel if zone[0] <= p[t - 1] <= zone[1] else 1.0)
        p[t] = p[t - 1] + 0.01 * (50_000.0 - p[t - 1]) + rng.normal(0.0, scale)
    return p


def test_lvn_study_detects_acceleration():
    prices = _lvn_series()
    res = lvn_study(prices, np.array([[49_800.0, 50_200.0]]), horizon=20, seed=SEED)
    assert res.stats.effect > 0.0
    assert res.stats.p_value < 0.05


def test_lvn_study_null():
    prices = _lvn_series(accel=1.0, seed=SEED)
    res = lvn_study(prices, np.array([[49_800.0, 50_200.0]]), horizon=20, seed=SEED)
    assert res.stats.p_value > 0.05
