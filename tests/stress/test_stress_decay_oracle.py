"""Оракул LiqMap.decay: дрейф последовательного полураспада против точной
рекурсии (Fraction), композиция decay(a)+decay(b) ≈ decay(a+b), и точный
учёт пыли при удалении бакетов (раньше остаток ≤1e-15 испарялся из
инварианта молча — по чуть-чуть на каждое удаление)."""

from __future__ import annotations

import math
import os
from fractions import Fraction

import numpy as np
import pytest

from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.map import LiqMap, StaticWeights

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))


def make_map(half_life_s: float = 86_400.0) -> LiqMap:
    return LiqMap(
        leverage_grid=[5, 10, 25, 50, 100],
        buckets=PriceBuckets(10.0),
        weight_fn=StaticWeights(np.array([1.0, 2.0, 3.0, 2.0, 1.0])),
        decay_half_life_s=half_life_s,
    )


def test_sequential_decay_matches_exact_recurrence():
    """k шагов float-рекурсии h -= h*(1-keep) против ТОЧНОГО h0*keep^k в
    рациональной арифметике того же float-keep: дрейф — единицы ulp на шаг."""
    lm = make_map()
    side = list(lm.heat)[0]
    h0 = 123_456.789
    lm.heat[side][100] = h0
    lm.contributed = h0
    dt = 300.0
    k = int(2000 * SCALE)
    for _ in range(k):
        lm.decay(dt)
    keep = 0.5 ** (dt / lm.decay_half_life_s)  # тот же float, что в decay
    exact = Fraction(h0) * Fraction(keep) ** k
    got = lm.heat[side][100]
    rel = abs(Fraction(got) - exact) / exact
    assert rel < Fraction(1, 10**9), float(rel)
    assert lm.mass_balance_error() / h0 < 1e-12


def test_decay_composition():
    """decay(a); decay(b) эквивалентно decay(a+b) с точностью до округления
    самой степени (не накопления): rel < 1e-12."""
    rng = np.random.default_rng(9)
    for _ in range(int(300 * SCALE)):
        a = float(rng.uniform(0, 5_000))
        b = float(rng.uniform(0, 5_000))
        m1, m2 = make_map(), make_map()
        side = list(m1.heat)[0]
        for m in (m1, m2):
            m.heat[side][7] = 1e9
            m.contributed = 1e9
        m1.decay(a)
        m1.decay(b)
        m2.decay(a + b)
        h1, h2 = m1.heat[side].get(7, 0.0), m2.heat[side].get(7, 0.0)
        assert h1 == pytest.approx(h2, rel=1e-12)


def test_dust_deletion_accounted_exactly():
    """5000 циклов «маленький allocate → полное затухание в пыль»: раньше
    каждый удалённый бакет тёк в инвариант остатком ≤1e-15 (итого ~1e-8 rel
    на малой массе), теперь пыль зачисляется в decayed — инвариант 1e-12."""
    lm = make_map()
    n = int(5000 * SCALE)
    for _ in range(n):
        lm.allocate(1e-6, 50_000.0)
        lm.decay(1e12)  # keep -> 0: всё в пыль и удаляется
    assert lm.total_heat() == 0.0
    scale = max(lm.contributed, 1e-12)
    assert lm.mass_balance_error() / scale < 1e-12
    # то же для пути removal: частичные снятия загоняют бакеты в пыль
    lm2 = make_map()
    for _ in range(int(500 * SCALE)):
        lm2.allocate(1e-6, 50_000.0)
        for _ in range(60):  # серия снятий по 30% до порога пыли
            lm2.allocate(-lm2.total_heat() * 0.3, 50_000.0)
            if lm2.total_heat() == 0.0:
                break
    assert lm2.mass_balance_error() / max(lm2.contributed, 1e-12) < 1e-12


def test_decay_bounds_and_infinite_dt():
    lm = make_map()
    lm.allocate(1_000.0, 50_000.0)
    before = {s: dict(h) for s, h in lm.heat.items()}
    lm.decay(0.0)  # keep == 1: ничего не меняется
    assert {s: dict(h) for s, h in lm.heat.items()} == before
    lm.decay(float("inf"))  # keep == 0: всё затухло, учтено точно
    assert lm.total_heat() == 0.0
    assert lm.mass_balance_error() < 1e-9
    assert math.isfinite(lm.decayed) and lm.decayed == pytest.approx(1_000.0)
    # монотонность: за любой dt >= 0 ни один бакет не растёт
    lm3 = make_map()
    lm3.allocate(1e6, 50_000.0)
    snap = {s: dict(h) for s, h in lm3.heat.items()}
    lm3.decay(12_345.0)
    for s, h in lm3.heat.items():
        for idx, v in h.items():
            assert v <= snap[s][idx]
