"""Стресс-тесты этапа 3 (калибровка): метрики, лестница весов, walk-forward,
event studies и boosted-веса на вырожденных входах и масштабе.

Сценарии: capture_rate без ликвидаций/без тепла/всё тепло в одной ячейке/все
ликвидации мимо карты; калибраторы на 0-1 событиях, нулевых таргетах и NaN в
фичах; ContextualWeights на неразделимом контексте и 50k сэмплах; boosted на
5 сэмплах (skip без lightgbm); walk-forward с embargo больше данных и данными
меньше одного фолда; бутстрап event studies на 0/1 событии и окнах за краем.

Масштаб управляется env STRESS_SCALE (по умолчанию 1.0).
"""

from __future__ import annotations

import os
import time

import numpy as np
import polars as pl
import pytest

from trading_system.calibration.event_studies import (
    bootstrap_effect,
    forward_return_paths,
    lvn_study,
    magnet_study,
    mean_path_ci,
    reversal_study,
    stationary_bootstrap_indices,
    top_decile_mask,
)
from trading_system.calibration.walkforward import (
    SymbolData,
    WalkForwardSplitter,
    run_walkforward,
    summarize_walkforward,
    tag_regimes,
)
from trading_system.calibration.weights import (
    ContextualWeights,
    RollingCalibrator,
    StaticWeightCalibrator,
    calibration_curve,
    capture_rate,
    flow_divergence,
)
from trading_system.core.schema import POLARS_SCHEMAS

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
SEED = 42
BAR = 60 * 1_000_000_000


def _n(base: int) -> int:
    return max(1, int(base * SCALE))


def _liq_frame(rows: list[tuple[int, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "exchange": "binance_usdm",
                "symbol": "BTCUSDT",
                "ts_event": ts,
                "ts_recv": ts,
                "price": price,
                "qty": usd / price,
                "qty_usd": usd,
                "side": -1,
            }
            for ts, price, usd in rows
        ],
        schema=POLARS_SCHEMAS["liquidation"],
        orient="row",
    )


EMPTY_LIQS = _liq_frame([])


def _mini_world(n: int = 60, nb: int = 20, k: int = 3):
    """Крошечный мир: класс j кладёт тепло в бакет 3j+2, цены в бакете 0..1."""
    heat_ts = (np.arange(n, dtype=np.int64) + 1) * BAR
    edges = np.arange(nb + 1, dtype=float)
    prices = np.full(n, 0.5)

    def build(w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        wm = np.tile(w, (n, 1)) if w.ndim == 1 else w
        heat = np.zeros((n, nb))
        for j in range(k):
            heat[:, 3 * j + 2] += wm[:, j]
        return heat

    return build, heat_ts, edges, prices


# ---------------------------------------------------------------------------
# 1) capture_rate: вырожденные входы без NaN-сюрпризов
# ---------------------------------------------------------------------------


def test_capture_rate_degenerate_inputs_defined():
    edges = np.arange(0.0, 11.0)
    heat = np.zeros((3, 10))
    heat[:, 4] = 7.0
    heat_ts = np.array([100, 200, 300])
    # нет ликвидаций -> 0.0, не NaN и не деление на ноль
    assert capture_rate(heat, heat_ts, edges, EMPTY_LIQS) == 0.0
    # нет тепла (пустая карта) -> маска пуста, ничего не поймано
    zero_heat = np.zeros((3, 10))
    liqs = _liq_frame([(150, 4.5, 100.0), (250, 2.5, 50.0)])
    assert capture_rate(zero_heat, heat_ts, edges, liqs) == 0.0
    # всё тепло в одной ячейке -> ловится ровно доля событий в ней
    one_cell = np.zeros((3, 10))
    one_cell[:, 4] = 1.0
    assert capture_rate(one_cell, heat_ts, edges, liqs) == pytest.approx(100.0 / 150.0)
    # все ликвидации мимо карты (вне диапазона бакетов) -> 0.0, не NaN
    off_map = _liq_frame([(150, 55.0, 100.0), (250, -3.0, 50.0)])
    assert capture_rate(heat, heat_ts, edges, off_map) == 0.0
    # все ликвидации до первого снапшота -> исключены -> 0.0
    early = _liq_frame([(50, 4.5, 100.0)])
    assert capture_rate(heat, heat_ts, edges, early) == 0.0
    # ts_range, не пересекающийся с событиями -> 0.0
    assert capture_rate(heat, heat_ts, edges, liqs, ts_range=(1_000, 2_000)) == 0.0


def test_flow_divergence_and_curve_degenerate_defined():
    build, heat_ts, edges, prices = _mini_world()
    heat = build(np.array([0.5, 0.3, 0.2]))
    # нет ликвидаций -> inf (объявленное "хуже некуда"), не NaN
    assert flow_divergence(heat, heat_ts, edges, prices, EMPTY_LIQS) == float("inf")
    # нулевое тепло -> inf
    liqs = _liq_frame([(int(heat_ts[5]), 2.5, 100.0)])
    assert flow_divergence(np.zeros_like(heat), heat_ts, edges, prices, liqs) == float("inf")
    # calibration_curve без событий -> нулевые профили правильной формы
    pred, real = calibration_curve(heat, heat_ts, edges, EMPTY_LIQS, n_bins=10)
    assert pred.shape == real.shape == (10,)
    assert not np.isnan(pred).any() and not np.isnan(real).any()
    assert real.sum() == 0.0


def test_top_decile_mask_edge_shapes():
    assert top_decile_mask(np.zeros(10)).sum() == 0  # всё нули -> ничего не "горячо"
    one = top_decile_mask(np.array([5.0]), 0.1)  # бюджет max(1, ...) = 1 ячейка
    assert one.tolist() == [True]


# ---------------------------------------------------------------------------
# 2) StaticWeightCalibrator / RollingCalibrator: 0-1 событий, NaN
# ---------------------------------------------------------------------------


def _fast_cal(**kw) -> StaticWeightCalibrator:
    return StaticWeightCalibrator(
        n_weights=3, seed=7, n_candidates=8, refine_sweeps=1, **kw
    )


def test_static_calibrator_zero_events_clean_fallback():
    build, heat_ts, edges, prices = _mini_world()
    fit = _fast_cal().fit(build, heat_ts, edges, EMPTY_LIQS, prices)
    assert np.isfinite(fit.weights).all()
    assert fit.weights.sum() == pytest.approx(1.0)
    assert (fit.weights > 0).all()
    assert fit.capture == 0.0
    assert fit.flow_kl == float("inf")  # объектив честно говорит "нечего фитить"


def test_static_calibrator_single_event_defined():
    build, heat_ts, edges, prices = _mini_world()
    liqs = _liq_frame([(int(heat_ts[10]), 2.5, 500.0)])  # в бакете класса 0
    fit = _fast_cal().fit(build, heat_ts, edges, liqs, prices)
    assert np.isfinite(fit.weights).all()
    assert fit.weights.sum() == pytest.approx(1.0)
    assert 0.0 <= fit.capture <= 1.0


def test_rolling_calibrator_window_bigger_than_data():
    build, heat_ts, edges, prices = _mini_world()
    liqs = _liq_frame([(int(heat_ts[10]), 2.5, 500.0)])
    roll = RollingCalibrator(
        _fast_cal(), train_window_ns=10_000 * BAR, refit_every_ns=30 * BAR
    )
    apply_range = (int(heat_ts[30]), int(heat_ts[-1]) + 1)
    cap, segments = roll.oos_capture(build, heat_ts, edges, liqs, prices, apply_range)
    assert len(segments) >= 1
    assert 0.0 <= cap <= 1.0
    for seg in segments:
        assert np.isfinite(seg.weights).all()
        assert seg.weights.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        RollingCalibrator(_fast_cal(), train_window_ns=0)
    with pytest.raises(ValueError):
        roll.applied_heat(build, heat_ts, [])


def test_contextual_weights_bad_inputs_contract():
    m = ContextualWeights(n_features=2, n_classes=2, n_iter=10)
    ok_x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError):
        m.fit(np.array([[1.0, np.nan], [2.0, 3.0]]), np.array([0, 1]))  # NaN в фичах
    with pytest.raises(ValueError):
        m.fit(np.array([[np.inf, 1.0], [2.0, 3.0]]), np.array([0, 1]))
    with pytest.raises(ValueError):
        m.fit(ok_x, np.array([0.0, np.nan]))  # NaN в таргетах
    with pytest.raises(ValueError):
        m.fit(np.zeros((0, 2)), np.zeros((0,)))  # 0 событий
    with pytest.raises(ValueError):
        m.fit(ok_x[:, :1], np.array([0, 1]))  # неверная размерность
    # после отбитых вызовов модель не отравлена и остаётся рабочей
    m.fit(ok_x, np.array([0, 1]))
    assert np.isfinite(m.predict_proba(ok_x)).all()


def test_contextual_weights_degenerate_targets_and_single_sample():
    # один сэмпл: sigma-guard (std=0 -> 1) не даёт NaN
    m1 = ContextualWeights(n_features=2, n_classes=2, n_iter=50)
    m1.fit(np.array([[1.0, 2.0]]), np.array([1]))
    p1 = m1.predict_proba(np.array([[1.0, 2.0], [9.0, -9.0]]))
    assert np.isfinite(p1).all()
    assert np.allclose(p1.sum(axis=1), 1.0)
    # все таргеты нулевые (не распределение): выход всё равно чистый симплекс
    m0 = ContextualWeights(n_features=2, n_classes=3, n_iter=50)
    m0.fit(np.array([[0.0, 1.0], [1.0, 0.0]]), np.zeros((2, 3)))
    p0 = m0.predict_proba(np.array([[0.5, 0.5]]))
    assert np.isfinite(p0).all()
    assert np.allclose(p0.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# 3) ContextualWeights: неразделимый контекст + 50k сэмплов
# ---------------------------------------------------------------------------


def test_contextual_unseparable_context_stays_calibrated():
    rng = np.random.default_rng(SEED)
    n = _n(2_000)
    x = rng.normal(0.0, 1.0, (n, 3))
    y = rng.integers(0, 3, n)  # случайные метки: контекст неразделим
    m = ContextualWeights(n_features=3, n_classes=3, n_iter=300).fit(x, y)
    p = m.predict_proba(x)
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)  # веса суммируются в 1
    prior = np.bincount(y, minlength=3) / n
    assert np.abs(p.mean(axis=0) - prior).max() < 0.05  # сходится к приору, не разваливается
    assert m.loss_history_[-1] <= m.loss_history_[0] + 1e-9


def test_contextual_50k_samples_throughput():
    rng = np.random.default_rng(SEED)
    n = _n(50_000)
    x = rng.normal(0.0, 1.0, (n, 4))
    y = rng.integers(0, 3, n)
    t0 = time.perf_counter()
    m = ContextualWeights(n_features=4, n_classes=3).fit(x, y)  # дефолтные 800 итераций
    elapsed = time.perf_counter() - t0
    assert elapsed < 60.0  # ~4s в прототипе; порядок сверху — регресс
    p = m.predict_proba(x[:500])
    assert np.isfinite(p).all()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 4) boosted: мало сэмплов; skip без lightgbm
# ---------------------------------------------------------------------------


def test_boosted_five_samples_clean_fallback():
    pytest.importorskip("lightgbm")
    from trading_system.calibration.boosted import BoostedWeights

    rng = np.random.default_rng(SEED)
    x = rng.normal(0.0, 1.0, (5, 2))  # ровно min_child_samples
    y = np.array([0, 1, 0, 1, 0])
    model = BoostedWeights(n_classes=2, seed=SEED).fit(x, y)
    p = model.predict_proba(x)
    assert p.shape == (5, 2)
    assert np.isfinite(p).all() and (p >= 0.0).all()
    assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)  # корректный фолбэк-симплекс
    # 2 сэмпла — минимум sklearn-валидации, всё ещё чистый выход
    p2 = BoostedWeights(n_classes=2, seed=SEED).fit(x[:2], y[:2]).predict_proba(x)
    assert np.allclose(p2.sum(axis=1), 1.0, atol=1e-9)
    # 1 сэмпл — честный ValueError, не тихий мусор
    with pytest.raises(ValueError):
        BoostedWeights(n_classes=2, seed=SEED).fit(x[:1], y[:1])


def test_boosted_predict_before_fit_contract():
    pytest.importorskip("lightgbm")
    from trading_system.calibration.boosted import BoostedWeights

    with pytest.raises(RuntimeError):
        BoostedWeights(n_classes=2).predict_proba(np.zeros((3, 2)))


# ---------------------------------------------------------------------------
# 5) walk-forward: embargo больше данных, данных меньше фолда, один режим
# ---------------------------------------------------------------------------


def test_splitter_embargo_bigger_than_data_returns_empty():
    ts = np.arange(100, dtype=np.int64) * 10  # span 990
    sp = WalkForwardSplitter(train_ns=100, test_ns=100, embargo_ns=10_000)
    assert sp.split(ts) == []  # честно пусто, не мусор


def test_splitter_data_smaller_than_one_fold():
    ts = np.arange(5, dtype=np.int64) * 10  # span 40 < train
    sp = WalkForwardSplitter(train_ns=1_000, test_ns=100, embargo_ns=0)
    assert sp.split(ts) == []
    assert sp.split(np.array([7], dtype=np.int64)) == []  # одна точка
    # ровно на грани: train покрывает всё, тесту не достаётся точек -> пусто
    sp2 = WalkForwardSplitter(train_ns=41, test_ns=100, embargo_ns=0)
    assert sp2.split(ts) == []


def test_run_walkforward_single_regime_honest_labels():
    rng = np.random.default_rng(SEED)
    n = 600
    ts = np.arange(n, dtype=np.int64) * 100
    # медленный монотонный дрейф: почти нулевая вола, стабильный режим
    prices = 50_000.0 * np.exp(np.arange(n) * 1e-5 + rng.normal(0, 1e-7, n))
    data = {"BTCUSDT": SymbolData(ts=ts, prices=prices)}
    sp = WalkForwardSplitter(train_ns=20_000, test_ns=10_000, embargo_ns=1_000)
    res = run_walkforward(data, sp, lambda s, sd, split: 1.0, regime_window=50)
    assert res.height == len(sp.split(ts))
    assert res["regime"].n_unique() == 1  # все фолды в одном режиме — и это честно
    summary = summarize_walkforward(res)
    assert summary["by_regime"].height == 1
    assert summary["overall"]["n"][0] == res.height


def test_tag_regimes_short_series_contract():
    with pytest.raises(ValueError):
        tag_regimes(np.array([100.0, 101.0]))  # < 3 точек
    tags = tag_regimes(np.full(10, 100.0))  # константная цена: не падает, без NaN
    assert not np.isnan(tags.dir_ratio).any()
    assert not np.isnan(tags.realized_vol).any()
    assert set(tags.label) == {"chop/low_vol"}


# ---------------------------------------------------------------------------
# 6) event studies: 0/1 событие, окна за краем, вырожденный бутстрап
# ---------------------------------------------------------------------------


def test_bootstrap_effect_empty_and_single_event():
    rng = np.random.default_rng(SEED)
    ctrl = rng.normal(0.0, 1.0, 100)
    with pytest.raises(ValueError):
        bootstrap_effect(np.array([]), ctrl)  # 0 событий — честная ошибка
    with pytest.raises(ValueError):
        bootstrap_effect(ctrl, np.array([]))
    # 1 событие: CI определён (вырожден по событию, но конечен), p в (0, 1]
    res = bootstrap_effect(np.array([0.7]), ctrl, n_boot=200, seed=SEED)
    assert np.isfinite(res.effect)
    assert np.isfinite(res.ci_low) and np.isfinite(res.ci_high)
    assert res.ci_low <= res.ci_high
    assert 0.0 < res.p_value <= 1.0
    assert res.n_events == 1 and res.n_clusters == 1


def test_events_windows_past_series_edge():
    rng = np.random.default_rng(SEED)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 200)))
    # все события в конце: окно вылезает за край -> все выброшены, форма честная
    tail_events = np.array([195, 197, 199])
    paths, kept = forward_return_paths(prices, tail_events, horizon=30)
    assert len(kept) == 0
    assert paths.shape == (0, 31)
    with pytest.raises(ValueError):
        mean_path_ci(paths)  # пустая матрица -> честная ошибка, не NaN-CI
    with pytest.raises(ValueError):
        reversal_study(prices, np.full(200, 1.0), tail_events, horizon=30, seed=SEED)
    with pytest.raises(ValueError):
        # серия короче горизонта целиком
        reversal_study(prices[:10], np.full(10, 1.0), np.array([2]), horizon=30, seed=SEED)


def test_magnet_study_zero_heat_explicitly_empty():
    rng = np.random.default_rng(SEED)
    n = 150
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    heat = np.zeros((n, 12))  # пустая карта: кандидатов нет
    edges = np.linspace(prices.min() * 0.9, prices.max() * 1.1, 13)
    res = magnet_study(prices, np.full(n, 1.0), heat, edges, horizon=10, seed=SEED)
    assert np.isnan(res.p_reach).all()  # явно пусто, а не ложные вероятности
    assert (res.n_samples == 0).all()
    assert np.isnan(res.ci_low).all() and np.isnan(res.ci_high).all()


def test_lvn_study_no_entries_contract():
    prices = np.linspace(100.0, 200.0, 300)  # монотонно мимо зоны
    with pytest.raises(ValueError):
        lvn_study(prices, np.array([[10.0, 20.0]]), horizon=10, seed=SEED)


def test_mean_path_ci_single_path_and_bootstrap_edges():
    paths = np.array([[0.0, 0.1, 0.2]])
    mean, lo, hi = mean_path_ci(paths, n_boot=50, seed=SEED)
    np.testing.assert_allclose(mean, paths[0])
    np.testing.assert_allclose(lo, paths[0])  # один кластер: CI вырожден, но определён
    np.testing.assert_allclose(hi, paths[0])
    idx = stationary_bootstrap_indices(1, mean_block=5.0, n_boot=10, rng=np.random.default_rng(SEED))
    assert idx.shape == (10, 1) and (idx == 0).all()
    with pytest.raises(ValueError):
        stationary_bootstrap_indices(0, 5.0, 10, np.random.default_rng(SEED))
