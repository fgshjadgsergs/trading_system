"""Оракул LiqMap.consume: геометрия снятия против точной арифметики.

Контракт: бакет [lo, hi) снимается ⇔ он пересекает замкнутый путь
[path_lo, path_hi]; на нижней границе строгость (бакет, кончающийся ровно
на path_lo, не задет), на верхней — включительность (бакет, начинающийся
ровно на path_hi, задет: цена path_hi лежит в его полуинтервале).
Fractional-режим: краевые бакеты теряют долю тепла, равную доле
пройденного интервала (равномерность тепла внутри бакета).
"""

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


def make_map(bucket_size: float = 10.0, **kw) -> LiqMap:
    return LiqMap(
        leverage_grid=[5, 10, 25, 50, 100],
        buckets=PriceBuckets(bucket_size),
        weight_fn=StaticWeights(np.array([1.0, 2.0, 3.0, 2.0, 1.0])),
        decay_half_life_s=86_400.0,
        **kw,
    )


def test_consumed_iff_interval_touches_path():
    """Точная арифметика (Fraction на float-границах): снятые бакеты касаются
    пути, выжившие — либо на положительном расстоянии, либо кончаются ровно
    на path_lo (полуоткрытость)."""
    rng = np.random.default_rng(5)
    for _ in range(int(2000 * SCALE)):
        size = float(10 ** rng.uniform(-2, 2))
        lm = make_map(bucket_size=size)
        price = float(10 ** rng.uniform(1, 4))
        for _ in range(rng.integers(1, 4)):
            lm.allocate(float(rng.uniform(1e3, 1e6)), price * float(rng.uniform(0.9, 1.1)))
        before = {s: dict(h) for s, h in lm.heat.items()}
        span = price * float(rng.uniform(0.001, 0.2))
        plo = price - span * float(rng.random())
        phi = plo + span
        lm.consume(plo, phi)
        flo, fhi = Fraction(plo), Fraction(phi)
        for s, h_before in before.items():
            for idx, h in h_before.items():
                blo, bhi = Fraction(lm.buckets.lo(idx)), Fraction(lm.buckets.hi(idx))
                survived = idx in lm.heat[s]
                touches = bhi > flo and blo <= fhi  # [lo,hi) ∩ [plo,phi] ≠ ∅
                if touches:
                    assert not survived, (size, plo, phi, idx, h)
                else:
                    assert survived and lm.heat[s][idx] == h, (size, plo, phi, idx)


def test_zero_width_path_consumes_containing_bucket_only():
    lm = make_map(bucket_size=10.0)
    lm.heat[list(lm.heat)[0]].update({4: 100.0, 5: 200.0, 6: 300.0})
    lm.contributed = 600.0
    taken = lm.consume(55.0, 55.0)  # точка внутри бакета [50, 60)
    assert taken == 200.0
    assert lm.total_heat() == 400.0
    # точка ровно на границе 60: полуоткрытость — задет только [60, 70)
    lm2 = make_map(bucket_size=10.0)
    lm2.heat[list(lm2.heat)[0]].update({5: 200.0, 6: 300.0})
    lm2.contributed = 500.0
    assert lm2.consume(60.0, 60.0) == 300.0


def test_fractional_edges_hand_case():
    """Путь [45, 65] по бакетам {[40,50): 100, [50,60): 100, [60,70): 100}:
    края теряют ровно пройденную долю (50 и 50), середина — всё."""
    lm = make_map(bucket_size=10.0, fractional_edge_consume=True)
    side = list(lm.heat)[0]
    lm.heat[side].update({4: 100.0, 5: 100.0, 6: 100.0})
    lm.contributed = 300.0
    taken = lm.consume(45.0, 65.0)
    assert taken == pytest.approx(200.0)
    assert lm.heat[side][4] == pytest.approx(50.0)
    assert 5 not in lm.heat[side]
    assert lm.heat[side][6] == pytest.approx(50.0)
    assert lm.mass_balance_error() < 1e-9
    # путь внутри одного бакета: съедена только его доля
    lm2 = make_map(bucket_size=10.0, fractional_edge_consume=True)
    lm2.heat[side][4] = 100.0
    lm2.contributed = 100.0
    assert lm2.consume(43.0, 47.0) == pytest.approx(40.0)
    assert lm2.heat[side][4] == pytest.approx(60.0)


def test_fractional_probe_and_scan_branches_agree():
    """Обе ветви consume (range-probe и полный скан) с fractional дают
    идентичный heat и taken."""
    rng = np.random.default_rng(6)
    for _ in range(int(1000 * SCALE)):
        size = float(10 ** rng.uniform(-1, 1))
        heat = {int(i): float(h) for i, h in
                zip(rng.integers(0, 3000, 40), 10 ** rng.uniform(2, 6, 40), strict=True)}
        plo = float(rng.uniform(0, 3000)) * size
        phi = plo + float(rng.uniform(0, 30)) * size
        results = []
        for pad in (0, 5000):  # 5000 фиктивных бакетов → узкий путь → probe
            lm = make_map(bucket_size=size, fractional_edge_consume=True)
            side = list(lm.heat)[0]
            lm.heat[side].update(heat)
            for j in range(pad):
                lm.heat[side][10_000_000 + j] = 1.0
            taken = lm.consume(plo, phi)
            core = {i: h for i, h in lm.heat[side].items() if i < 10_000_000}
            results.append((taken, core))
        assert results[0][0] == results[1][0]
        assert results[0][1] == results[1][1]


def test_mass_invariant_with_fractional_consume():
    rng = np.random.default_rng(7)
    lm = make_map(bucket_size=50.0, fractional_edge_consume=True)
    price = 50_000.0
    for _ in range(int(20_000 * SCALE)):
        price = min(max(price * float(np.exp(rng.normal(0, 0.002))), 45_000.0), 55_000.0)
        k = rng.random()
        if k < 0.5:
            lm.allocate(float(rng.uniform(0, 1e6)), price)
        elif k < 0.7:
            lm.allocate(-float(rng.uniform(0, 1e5)), price)
        else:
            lm.consume(price * 0.998, price * 1.002)
    scale = max(1.0, lm.contributed)
    assert lm.mass_balance_error() / scale < 1e-12
    assert all(h >= 0.0 for sh in lm.heat.values() for h in sh.values())
    assert math.isfinite(lm.total_heat())
