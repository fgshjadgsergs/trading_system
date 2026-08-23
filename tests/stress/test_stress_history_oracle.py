"""Оракул HeatHistory: каузальность срезов, доступ по времени, защита от
взрыва плотной матрицы и стоимость хранения кадров."""

from __future__ import annotations

import os
import tracemalloc

import numpy as np
import pytest

from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.history import HeatHistory
from trading_system.liqmap.map import LiqMap, StaticWeights

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
MIN = 60_000_000_000


def make_map(bucket_size: float = 10.0) -> LiqMap:
    return LiqMap(
        leverage_grid=[5, 10, 25, 50, 100],
        buckets=PriceBuckets(bucket_size),
        weight_fn=StaticWeights(np.array([1.0, 2.0, 3.0, 2.0, 1.0])),
        decay_half_life_s=86_400.0,
    )


def test_frames_are_immutable_snapshots():
    """Кадр — снимок, а не вид на живую карту: последующие allocate/consume
    не могут задним числом изменить уже записанную историю."""
    lm = make_map()
    hist = HeatHistory(lm)
    lm.allocate(1_000.0, 50_000.0)
    hist.record(1 * MIN)
    t0 = hist.total_at(0)
    lm.allocate(9_000.0, 50_000.0)
    lm.consume(49_000.0, 51_000.0)
    lm.decay(3_600.0)
    hist.record(2 * MIN)
    assert hist.total_at(0) == pytest.approx(t0)  # прошлое не переписано
    assert hist.total_at(1) != pytest.approx(t0)


def test_index_at_is_strict_or_inclusive_by_request():
    lm = make_map()
    hist = HeatHistory(lm)
    for k in range(1, 6):
        lm.allocate(1_000.0, 50_000.0)
        hist.record(k * MIN)
    assert hist.index_at(0) is None  # до первого кадра
    assert hist.index_at(1 * MIN) is None  # строго ДО: свой же кадр не виден
    assert hist.index_at(1 * MIN, inclusive=True) == 0
    assert hist.index_at(3 * MIN + 1) == 2
    assert hist.index_at(3 * MIN) == 1
    assert hist.index_at(3 * MIN, inclusive=True) == 2
    assert hist.index_at(99 * MIN) == 4  # позже всех — последний кадр
    # оракул перебором: strict == последний кадр с ts_frame < ts
    rng = np.random.default_rng(4)
    for _ in range(int(2000 * SCALE)):
        t = int(rng.integers(0, 7 * MIN))
        want = [i for i, ts in enumerate(hist.ts) if ts < t]
        assert hist.index_at(t) == (want[-1] if want else None)
        want_inc = [i for i, ts in enumerate(hist.ts) if ts <= t]
        assert hist.index_at(t, inclusive=True) == (want_inc[-1] if want_inc else None)


def test_at_ts_accessors_match_index_accessors():
    lm = make_map()
    hist = HeatHistory(lm)
    for k in range(1, 10):
        lm.allocate(1_000.0 * k, 50_000.0 + 50 * k)
        hist.record(k * MIN)
    assert hist.pools_at_ts(5 * MIN + 1) == hist.pools_at(hist.index_at(5 * MIN + 1))
    assert hist.zones_at_ts(5 * MIN + 1) == hist.zones_at(hist.index_at(5 * MIN + 1))
    assert hist.pools_at_ts(0) == []
    assert hist.zones_at_ts(0) == ([], [], [])


def test_out_of_order_record_rejected():
    lm = make_map()
    hist = HeatHistory(lm)
    lm.allocate(1_000.0, 50_000.0)
    hist.record(5 * MIN)
    hist.record(5 * MIN)  # равные ts допустимы (дубль бара)
    with pytest.raises(ValueError):
        hist.record(4 * MIN)


def test_total_at_order_independent_and_exact():
    """total_at не зависит от порядка обхода бакетов (fsum)."""
    lm = make_map(bucket_size=1.0)
    hist = HeatHistory(lm)
    rng = np.random.default_rng(7)
    for _ in range(int(200 * SCALE)):
        lm.allocate(float(rng.uniform(1e-3, 1e9)), float(rng.uniform(1_000, 90_000)))
    hist.record(MIN)
    frame = hist._frames[0]
    import math
    vals = list(frame.values())
    rng.shuffle(vals)
    assert hist.total_at(0) == math.fsum(sorted(frame.values()))
    assert hist.total_at(0) == pytest.approx(math.fsum(vals), rel=1e-15)
    assert hist.total_at(0) == pytest.approx(lm.total_heat(), rel=1e-12)


def test_matrix_matches_frames_and_caps_exploded_range():
    lm = make_map(bucket_size=1.0)
    hist = HeatHistory(lm)
    lm.allocate(1e6, 100.0)
    hist.record(MIN)
    lm.allocate(1e6, 100_000.0)  # взрыв диапазона индексов
    hist.record(2 * MIN)
    ts, prices, H = hist.matrix()
    assert H.shape[1] == 2 and len(prices) == H.shape[0]
    assert H.sum() == pytest.approx(hist.total_at(0) + hist.total_at(1), rel=1e-9)
    assert H.shape[0] > 90_000  # плотная матрица по всему диапазону
    ts_c, prices_c, Hc = hist.matrix(max_rows=500)
    assert Hc.shape == (500, 2)
    assert len(prices_c) == 500
    # окно уехало к самой тяжёлой области, а не обрезано слепо с краю
    heaviest = max(hist._frames[1], key=lambda i: hist._frames[1][i])
    center_price = (heaviest + 0.5) * lm.buckets.bucket_size
    assert prices_c[0] <= center_price <= prices_c[-1]
    assert Hc.sum() > 0


def test_frame_memory_cost_per_bar():
    """Стоимость хранения кадра линейна и предсказуема (для планирования
    live-прогонов: 7 дней 5m ~ 2016 кадров)."""
    lm = make_map(bucket_size=25.0)
    hist = HeatHistory(lm)
    n = int(300 * SCALE)
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    price = 50_000.0
    rng = np.random.default_rng(1)
    for k in range(n):
        price *= float(np.exp(rng.normal(0, 0.002)))
        lm.step(price * 0.999, price * 1.001, price, 1e6, dt_s=300.0)
        hist.record((k + 1) * MIN)
    used = tracemalloc.get_traced_memory()[0] - base
    tracemalloc.stop()
    per_bar = used / n
    assert 0 < per_bar < 200_000, per_bar  # sanity: не мегабайты на бар
