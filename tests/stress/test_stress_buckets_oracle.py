"""Оракул PriceBuckets: контракт содержания в float-арифметике, Decimal-сверка
вдали от границ, монотонность, стыковка тайлов и консервация rebucket."""

from __future__ import annotations

import math
import os
from decimal import Decimal, getcontext

import numpy as np
import pytest

from trading_system.liqmap.buckets import PriceBuckets, rebucket

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
getcontext().prec = 60


def test_containment_holds_at_ulp_boundaries():
    """lo(i) <= P < hi(i) в той же арифметике, что consume() — даже в ±3 ulp
    от float-границы (до фикса index(): ~24k нарушений на 1.4M точек)."""
    rng = np.random.default_rng(0)
    for _ in range(int(30_000 * SCALE)):
        size = float(10 ** rng.uniform(-6, 4))
        b = PriceBuckets(size)
        p = int(rng.integers(1, 10_000_000)) * size
        for _ in range(3):
            p = math.nextafter(p, -math.inf)
        for _ in range(7):
            if p > 0:
                i = b.index(p)
                assert b.lo(i) <= p < b.hi(i), (size, p, i)
            p = math.nextafter(p, math.inf)


def test_matches_decimal_floor_away_from_boundaries():
    rng = np.random.default_rng(1)
    for _ in range(int(20_000 * SCALE)):
        size = float(10 ** rng.uniform(-6, 4))
        price = float(10 ** rng.uniform(-4, 6))
        b = PriceBuckets(size)
        exact = math.floor(Decimal(price) / Decimal(size))
        # у самой границы float-тайл может законно отличаться на 1
        frac = (Decimal(price) / Decimal(size)) % 1
        near = min(frac, 1 - frac) < Decimal("1e-12")
        assert abs(b.index(price) - int(exact)) <= (1 if near else 0), (size, price)


def test_index_monotone_and_tiles_adjacent():
    rng = np.random.default_rng(2)
    b = PriceBuckets(float(10 ** rng.uniform(-3, 2)))
    prices = np.sort(10 ** rng.uniform(-2, 5, size=int(100_000 * SCALE)))
    idx = [b.index(float(p)) for p in prices]
    assert all(a <= c for a, c in zip(idx, idx[1:], strict=False))
    for i in rng.integers(-(10**9), 10**9, size=1000):
        assert b.hi(int(i)) == b.lo(int(i) + 1)
        assert b.lo(int(i)) < b.center(int(i)) < b.hi(int(i))


def test_rebucket_conserves_mass_exactly():
    rng = np.random.default_rng(3)
    for _ in range(int(200 * SCALE)):
        old = PriceBuckets(float(10 ** rng.uniform(-2, 1)))
        new = PriceBuckets(float(10 ** rng.uniform(-2, 1)))
        heat = {int(i): float(h) for i, h in
                zip(rng.integers(0, 100_000, 300), 10 ** rng.uniform(0, 8, 300), strict=True)}
        out = rebucket(heat, old, new)
        assert math.isclose(math.fsum(heat.values()), math.fsum(out.values()),
                            rel_tol=1e-12)
    # тождественная сетка — тождественное отображение
    b = PriceBuckets(0.5)
    heat = {3: 1.0, 10: 2.5}
    assert rebucket(heat, b, b) == heat


def test_from_atr_clip_and_guards():
    assert PriceBuckets.from_atr(1e-15).bucket_size == 1e-9
    assert PriceBuckets.from_atr(100.0, fraction=0.1).bucket_size == pytest.approx(10.0)
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            PriceBuckets.from_atr(bad)
        if bad == bad:  # NaN уже отбит выше
            with pytest.raises(ValueError):
                PriceBuckets(bad)
