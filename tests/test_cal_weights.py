"""Stage 3.2 calibration ladder: capture rate, recovery of the true mixture,
rolling walk-forward, contextual weights, ladder selection with rollback."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from trading_system.calibration.synthetic import (
    kl_divergence,
    make_heat_builder,
    make_world,
    realized_mixture,
)
from trading_system.calibration.weights import (
    ContextualWeights,
    RollingCalibrator,
    StaticWeightCalibrator,
    calibration_curve,
    capture_details,
    capture_rate,
    compare_ladder,
    flow_divergence,
    naive_baseline_heat,
    select_rung,
)
from trading_system.core.schema import POLARS_SCHEMAS

SEED = 42
BAR = 60 * 1_000_000_000
TRUE_W = (0.60, 0.10, 0.30)


@pytest.fixture(scope="module")
def world_static():
    return make_world(seed=SEED, static_weights=TRUE_W)


@pytest.fixture(scope="module")
def build_static(world_static):
    return make_heat_builder(world_static, decay_half_life_bars=200.0)


@pytest.fixture(scope="module")
def world_regime():
    return make_world(
        seed=SEED,
        regime_period=150,
        regime_weights=((0.02, 0.08, 0.90), (0.90, 0.08, 0.02)),
        bucket_width_frac=0.002,
    )


def _ranges(world, train_frac=0.60, test_frac=0.65):
    n = len(world.ts)
    train = (int(world.ts[0]), int(world.ts[int(n * train_frac)]))
    test = (int(world.ts[int(n * test_frac)]), int(world.ts[-1]) + 1)
    return train, test


def _calibrator(lag_bars=10):
    return StaticWeightCalibrator(
        n_weights=3,
        seed=7,
        lag_ns=lag_bars * BAR,
        flow_weight=4.0,
        n_candidates=24,
        refine_sweeps=2,
    )


# ---------------------------------------------------------------------------
# capture rate
# ---------------------------------------------------------------------------


def _liq_frame(rows):
    """Rows of (ts, price, usd[, side]); side defaults to -1 (SELL = лонг-принт)."""
    return pl.DataFrame(
        [
            {
                "exchange": "binance_usdm",
                "symbol": "BTCUSDT",
                "ts_event": r[0],
                "ts_recv": r[0],
                "price": r[1],
                "qty": r[2] / r[1],
                "qty_usd": r[2],
                "side": r[3] if len(r) > 3 else -1,
            }
            for r in rows
        ],
        schema=POLARS_SCHEMAS["liquidation"],
        orient="row",
    )


def test_capture_rate_exact_hand_case():
    edges = np.arange(0.0, 11.0)  # 10 buckets [0,1)...[9,10)
    heat = np.zeros((2, 10))
    heat[0, 7] = 5.0  # snapshot 0: only bucket 7 hot
    heat[1, 2] = 5.0  # snapshot 1: only bucket 2 hot
    heat_ts = np.array([100, 200])
    liqs = _liq_frame(
        [
            (150, 7.5, 100.0),  # after snap 0, in its hot bucket -> captured
            (150, 0.5, 50.0),  # after snap 0, cold bucket -> missed
            (250, 2.5, 100.0),  # after snap 1, hot bucket -> captured
            (50, 7.5, 999.0),  # before the first snapshot -> excluded entirely
            (260, 55.0, 50.0),  # outside bucket range -> counted, not captured
        ]
    )
    cap = capture_rate(heat, heat_ts, edges, liqs, top_decile=0.1)
    assert cap == pytest.approx(200.0 / 300.0)
    # ts_range filter keeps only the last two events
    cap2 = capture_rate(heat, heat_ts, edges, liqs, top_decile=0.1, ts_range=(200, 300))
    assert cap2 == pytest.approx(100.0 / 150.0)
    # lag re-matches events to the snapshot 50ns earlier: the ts=150 events
    # stay on snap 0, ts=250/260 fall back to snap 1, ts=50 stays excluded
    cap3 = capture_rate(heat, heat_ts, edges, liqs, top_decile=0.1, lag_ns=50)
    assert cap3 == pytest.approx(200.0 / 300.0)


def test_side_aware_capture_scores_prints_against_own_half():
    """M8 (баг-фикс судьи): лонг-принт в ячейке, горячей ТОЛЬКО в
    шорт-полуматрице, side-aware не захвачен; склеенный расчёт захватывает."""
    edges = np.arange(0.0, 11.0)  # 10 бакетов
    heat3 = np.zeros((1, 2, 10))
    heat3[0, 1, 7] = 5.0  # горячо только в ШОРТ-половине (ось 1)
    heat3[0, 0, 2] = 3.0  # лонг-тепло в бакете 2
    heat_ts = np.array([100])
    long_print_at_7 = _liq_frame([(150, 7.5, 100.0, -1)])
    glued = capture_rate(heat3.sum(axis=1), heat_ts, edges, long_print_at_7)
    assert glued > 0.0  # склеенная ветка «захватывает» чужую сторону
    assert capture_rate(heat3, heat_ts, edges, long_print_at_7) == 0.0
    cap, tot, per_side = capture_details(heat3, heat_ts, edges, long_print_at_7)
    assert (cap, tot) == (0.0, 100.0)
    assert per_side["long"] == (0.0, 100.0)
    assert per_side["short"] == (0.0, 0.0)
    # принты, попадающие в тепло СВОЕЙ стороны, захвачены; итог — USD-взвешенный
    mixed = _liq_frame([(150, 2.5, 100.0, -1), (150, 7.5, 300.0, 1)])
    cap, tot, per_side = capture_details(heat3, heat_ts, edges, mixed)
    assert (cap, tot) == (400.0, 400.0)
    assert per_side["long"] == (100.0, 100.0)
    assert per_side["short"] == (300.0, 300.0)
    assert capture_rate(heat3, heat_ts, edges, mixed) == pytest.approx(1.0)
    # side-split тепло без колонки side — честная ошибка
    with pytest.raises(ValueError):
        capture_rate(heat3, heat_ts, edges, mixed.drop("side"))


def test_capture_tolerance_dilates_hot_mask():
    """R3: допуск m=2 — принт в 2 бакетах от горячей ячейки засчитан, в 3 — нет."""
    edges = np.arange(0.0, 21.0)  # 20 бакетов; top-децили: 2 ячейки, positive-only
    heat = np.zeros((1, 20))
    heat[0, 10] = 5.0
    heat_ts = np.array([100])
    print_at_8 = _liq_frame([(150, 8.5, 100.0)])  # 2 бакета от горячей
    print_at_7 = _liq_frame([(150, 7.5, 100.0)])  # 3 бакета
    assert capture_rate(heat, heat_ts, edges, print_at_8) == 0.0
    assert capture_rate(heat, heat_ts, edges, print_at_8, tolerance_buckets=2) == 1.0
    assert capture_rate(heat, heat_ts, edges, print_at_7, tolerance_buckets=2) == 0.0
    assert capture_rate(heat, heat_ts, edges, print_at_7, tolerance_buckets=3) == 1.0
    # симметрия: принт по другую сторону от горячей ячейки
    print_at_12 = _liq_frame([(150, 12.5, 100.0)])
    assert capture_rate(heat, heat_ts, edges, print_at_12, tolerance_buckets=2) == 1.0


def test_flow_divergence_prefers_true_weights(world_static, build_static):
    w = world_static
    args = (w.ts, w.bucket_edges, w.prices, w.liquidations)
    f_true = flow_divergence(build_static(np.asarray(TRUE_W)), *args)
    f_hug = flow_divergence(build_static(np.array([0.02, 0.08, 0.90])), *args)
    f_far = flow_divergence(build_static(np.array([0.90, 0.08, 0.02])), *args)
    assert f_true < f_hug
    assert f_true < f_far
    with pytest.raises(ValueError):
        flow_divergence(
            build_static(np.asarray(TRUE_W)), w.ts, w.bucket_edges, w.prices[:-1], w.liquidations
        )


def test_calibration_curve_shares(world_static, build_static):
    w = world_static
    heat = build_static(np.asarray(TRUE_W))
    pred, real = calibration_curve(heat, w.ts, w.bucket_edges, w.liquidations, n_bins=10)
    assert pred.shape == real.shape == (10,)
    assert pred.sum() == pytest.approx(1.0)
    assert real.sum() == pytest.approx(1.0)
    # hottest decile attracts more realized liquidation USD than the coldest
    assert real[-1] > real[0]


# ---------------------------------------------------------------------------
# naive baseline
# ---------------------------------------------------------------------------


def test_naive_baseline_round_numbers_and_causality(world_static):
    w = world_static
    heat = naive_baseline_heat(w.prices, w.bucket_edges)
    assert heat.shape == (len(w.prices), len(w.bucket_edges) - 1)
    # a bucket containing a multiple of 1000 carries base heat at every t
    step = 1000.0
    m = np.ceil(w.bucket_edges[0] / step) * step
    b = int(np.searchsorted(w.bucket_edges, m, side="right") - 1)
    assert np.all(heat[:, b] > 0)
    # causality: perturbing the future must not change past rows
    prices2 = w.prices.copy()
    prices2[-100:] *= 1.02
    heat2 = naive_baseline_heat(prices2, w.bucket_edges)
    np.testing.assert_array_equal(heat[: len(w.prices) - 100], heat2[: len(w.prices) - 100])


# ---------------------------------------------------------------------------
# static calibrator: recovery + beats naive OOS
# ---------------------------------------------------------------------------


def test_static_calibrator_recovers_truth_and_beats_naive(world_static, build_static):
    w = world_static
    train, test = _ranges(w)
    cal = _calibrator(lag_bars=10)
    fit = cal.fit(build_static, w.ts, w.bucket_edges, w.liquidations, w.prices, ts_range=train)

    true_w = np.asarray(TRUE_W)
    kl_fit = kl_divergence(true_w, fit.weights)
    kl_uniform = kl_divergence(true_w, np.ones(3) / 3)
    assert kl_fit < 0.10
    assert kl_fit < 0.5 * kl_uniform
    # the simulated liquidations really do carry the planted mixture
    assert kl_divergence(true_w, realized_mixture(w)) < 0.05

    heat = build_static(fit.weights)
    naive = naive_baseline_heat(w.prices, w.bucket_edges)
    oos_args = dict(ts_range=test, lag_ns=10 * BAR)
    cap_fit = capture_rate(heat, w.ts, w.bucket_edges, w.liquidations, **oos_args)
    cap_naive = capture_rate(naive, w.ts, w.bucket_edges, w.liquidations, **oos_args)
    assert cap_fit > cap_naive + 0.05  # Gate A on synthetic ground truth


def test_static_calibrator_deterministic(world_static, build_static):
    w = world_static
    train, _ = _ranges(w)
    cal1 = _calibrator()
    cal2 = _calibrator()
    f1 = cal1.fit(build_static, w.ts, w.bucket_edges, w.liquidations, w.prices, ts_range=train)
    f2 = cal2.fit(build_static, w.ts, w.bucket_edges, w.liquidations, w.prices, ts_range=train)
    np.testing.assert_array_equal(f1.weights, f2.weights)
    assert f1.objective == f2.objective


# ---------------------------------------------------------------------------
# rolling calibrator: walk-forward hygiene
# ---------------------------------------------------------------------------


class _SpyCalibrator(StaticWeightCalibrator):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen_ranges: list[tuple[int, int]] = []

    def fit(self, build_heat, heat_ts, bucket_edges, liquidations, prices, ts_range=None):
        self.seen_ranges.append(ts_range)
        return super().fit(build_heat, heat_ts, bucket_edges, liquidations, prices, ts_range)


def test_rolling_calibrator_is_walk_forward(world_static, build_static):
    w = world_static
    _, test = _ranges(w)
    spy = _SpyCalibrator(
        n_weights=3, seed=7, lag_ns=10 * BAR, flow_weight=4.0, n_candidates=8, refine_sweeps=1
    )
    train_window = 600 * BAR
    roll = RollingCalibrator(spy, train_window_ns=train_window, refit_every_ns=200 * BAR)
    cap, segments = roll.oos_capture(
        build_static, w.ts, w.bucket_edges, w.liquidations, w.prices, test, lag_ns=10 * BAR
    )
    assert len(segments) >= 2
    for seg, seen in zip(segments, spy.seen_ranges, strict=True):
        # each segment trains on the trailing window that ends where apply starts
        assert seen == (seg.apply_start_ts - train_window, seg.apply_start_ts)
        assert seg.apply_start_ts < seg.apply_end_ts
    # segments tile the apply range without overlap
    for a, b in zip(segments, segments[1:], strict=False):
        assert a.apply_end_ts == b.apply_start_ts
    assert 0.0 < cap <= 1.0


# ---------------------------------------------------------------------------
# contextual weights
# ---------------------------------------------------------------------------


def test_contextual_weights_soft_labels():
    rng = np.random.default_rng(SEED)
    n = 1200
    regime = (np.arange(n) // 100) % 2
    x = np.hstack([regime[:, None].astype(float), rng.normal(0, 1, (n, 2))])
    wa, wb = np.array([0.8, 0.1, 0.1]), np.array([0.1, 0.1, 0.8])
    y = np.where(regime[:, None] == 0, wa, wb)
    model = ContextualWeights(n_features=3, n_classes=3).fit(x, y)
    pred = model.predict_proba(x)
    assert np.max(np.abs(pred[regime == 0] - wa)) < 0.06
    assert np.max(np.abs(pred[regime == 1] - wb)) < 0.06
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_contextual_weights_hard_labels():
    rng = np.random.default_rng(SEED)
    x = rng.normal(0, 1, (800, 2))
    y = (x[:, 0] + 0.1 * rng.normal(0, 1, 800) > 0).astype(int)
    model = ContextualWeights(n_features=2, n_classes=2).fit(x, y)
    acc = float((model.predict_proba(x).argmax(axis=1) == y).mean())
    assert acc > 0.9
    with pytest.raises(ValueError):
        model.fit(x[:, :1], y)


# ---------------------------------------------------------------------------
# ladder selection
# ---------------------------------------------------------------------------


def test_select_rung_rollback_rule():
    caps = {"naive": 0.1, "static": 0.30, "rolling": 0.31, "contextual": 0.32}
    assert select_rung(caps, tolerance=0.05) == "static"
    assert select_rung(caps, tolerance=0.005) == "contextual"
    assert select_rung({"static": 0.3}, tolerance=0.05) == "static"
    with pytest.raises(ValueError):
        select_rung({"naive": 0.1}, 0.05)


def test_ladder_picks_contextual_on_regime_world(world_regime):
    w = world_regime
    build = make_heat_builder(w, decay_half_life_bars=25.0)
    train, test = _ranges(w)
    cal = StaticWeightCalibrator(
        n_weights=3, seed=7, lag_ns=0, flow_weight=4.0, n_candidates=24, refine_sweeps=2
    )
    roll = RollingCalibrator(cal, train_window_ns=600 * BAR, refit_every_ns=150 * BAR)
    res = compare_ladder(
        build,
        w.ts,
        w.bucket_edges,
        w.liquidations,
        w.prices,
        train,
        test,
        n_weights=3,
        context=w.context,
        context_labels=w.true_weights,
        calibrator=cal,
        rolling=roll,
        tolerance=0.05,
        lag_ns=0,
    )
    assert res.selected == "contextual"
    assert res.capture["contextual"] > res.capture["static"] + 0.05
    assert res.beats_naive
    assert res.contextual_model is not None


def test_ladder_rolls_back_to_static_on_context_free_world(world_static, build_static):
    w = world_static
    train, test = _ranges(w)
    cal = _calibrator(lag_bars=10)
    roll = RollingCalibrator(cal, train_window_ns=600 * BAR, refit_every_ns=150 * BAR)
    res = compare_ladder(
        build_static,
        w.ts,
        w.bucket_edges,
        w.liquidations,
        w.prices,
        train,
        test,
        n_weights=3,
        context=w.context,  # pure noise features: no exploitable context
        context_labels=w.true_weights,
        calibrator=cal,
        rolling=roll,
        tolerance=0.05,
        lag_ns=10 * BAR,
    )
    assert res.selected == "static"  # complexity without OOS gain is rolled back
    assert res.beats_naive
    assert set(res.capture) == {"naive", "static", "rolling", "contextual"}


def test_ladder_rejects_overlapping_ranges(world_static, build_static):
    w = world_static
    with pytest.raises(ValueError):
        compare_ladder(
            build_static,
            w.ts,
            w.bucket_edges,
            w.liquidations,
            w.prices,
            (0, 100),
            (50, 200),
            n_weights=3,
        )
