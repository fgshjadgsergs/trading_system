"""Слоистая карта (смесь экспонент): эквивалентность одной карте при равных
полураспадах, инвариант массы, убывающая интенсивность затухания."""

from __future__ import annotations

import math

import numpy as np
import pytest

from trading_system.core.schema import Side
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.map import LiqMap, StaticWeights
from trading_system.liqmap.mixture import MixtureLiqMap

GRID = [5, 10, 25, 50, 100]
W = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
HOUR = 3_600.0


def kw(**over):
    base = dict(
        leverage_grid=GRID,
        buckets=PriceBuckets(10.0),
        weight_fn=StaticWeights(W),
    )
    base.update(over)
    return base


def storm(m, n: int = 400, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    price = 50_000.0
    for _ in range(n):
        price = float(np.clip(price * np.exp(rng.normal(0, 0.004)), 40_000.0, 60_000.0))
        m.step(price * 0.997, price * 1.003, price,
               float(rng.normal(5e5, 3e5)), dt_s=900.0)


def agg(m) -> dict:
    return {int(i): float(h) for side in (Side.BUY, Side.SELL)
            for i, h in m.heat[side].items()}


def test_single_component_is_bit_identical_to_plain_map():
    plain = LiqMap(decay_half_life_s=6 * HOUR, **kw())
    mix = MixtureLiqMap([(1.0, 6 * HOUR)], **kw())
    storm(plain)
    storm(mix)
    a, b = agg(plain), agg(mix)
    assert a.keys() == b.keys()
    for i in a:
        assert a[i] == b[i]  # бит в бит: слой — это тот же LiqMap
    assert plain.total_heat() == pytest.approx(mix.total_heat(), rel=1e-15)


def test_equal_half_lives_reproduce_one_map():
    """Три слоя с одинаковым T½ — это одна карта: разложение массы по слоям
    не меняет ни геометрию, ни баланс (расхождение — только порядок сложения)."""
    plain = LiqMap(decay_half_life_s=12 * HOUR, close_out_fraction=0.6, **kw())
    mix = MixtureLiqMap([(0.5, 12 * HOUR), (0.3, 12 * HOUR), (0.2, 12 * HOUR)],
                        close_out_fraction=0.6, **kw())
    storm(plain)
    storm(mix)
    a, b = agg(plain), agg(mix)
    assert a.keys() == b.keys()
    for i in a:
        assert b[i] == pytest.approx(a[i], rel=1e-12)
    assert mix.total_heat() == pytest.approx(plain.total_heat(), rel=1e-12)


def test_by_leverage_split_preserves_placement():
    """Тиры по плечу с равными T½ размещают тепло ровно как одна карта:
    маска перенормируется внутри группы, а слой получает долю группы."""
    plain = LiqMap(decay_half_life_s=8 * HOUR, **kw())
    mix = MixtureLiqMap.by_leverage(
        [(10.0, 8 * HOUR), (50.0, 8 * HOUR), (1e9, 8 * HOUR)],
        leverage_grid=GRID, weight_fn=StaticWeights(W), buckets=PriceBuckets(10.0),
    )
    plain.allocate(1e6, 50_000.0)
    mix.allocate(1e6, 50_000.0)
    a, b = agg(plain), agg(mix)
    assert a.keys() == b.keys()
    for i in a:
        assert b[i] == pytest.approx(a[i], rel=1e-12)
    assert mix.total_heat() == pytest.approx(1e6, rel=1e-12)


def test_mass_invariant_and_nonnegative_under_storm():
    mix = MixtureLiqMap([(0.7, 2 * HOUR), (0.3, 168 * HOUR)],
                        close_out_fraction=0.5, **kw())
    rng = np.random.default_rng(11)
    price = 50_000.0
    for _ in range(2_000):
        price = float(np.clip(price * np.exp(rng.normal(0, 0.003)), 45_000.0, 55_000.0))
        k = rng.random()
        if k < 0.55:
            mix.allocate(float(rng.uniform(0, 1e6)), price)
        elif k < 0.75:
            mix.allocate(-float(rng.uniform(0, 3e5)), price)
        elif k < 0.9:
            mix.consume(price * 0.998, price * 1.002)
        else:
            mix.decay(float(rng.uniform(0, 4 * HOUR)))
    assert mix.mass_balance_error() / max(mix.contributed, 1.0) < 1e-12
    assert all(h >= 0.0 for side in mix.heat.values() for h in side.values())
    assert math.isfinite(mix.total_heat())


def test_hazard_decreases_with_age():
    """Ключевое свойство смеси: эффективный полураспад РАСТЁТ с возрастом
    тепла (у одной экспоненты он константа)."""
    mix = MixtureLiqMap([(0.75, 4 * HOUR), (0.25, 336 * HOUR)], **kw())
    ages = [0.0, 12 * HOUR, 48 * HOUR, 240 * HOUR]
    effs = [mix.effective_half_life(a) for a in ages]
    assert effs == sorted(effs)
    assert effs[0] < 24 * HOUR < effs[-1]
    one = MixtureLiqMap([(1.0, 24 * HOUR)], **kw())
    flat = [one.effective_half_life(a) for a in ages]
    assert all(f == pytest.approx(24 * HOUR, rel=1e-6) for f in flat)


def test_old_heat_outlives_a_single_exponential():
    """Практическое следствие: при одинаковой массе через сутки смесь
    сохраняет заметно больше тепла через неделю, чем одна экспонента."""
    mix = MixtureLiqMap([(0.75, 4 * HOUR), (0.25, 336 * HOUR)], **kw())
    exp = LiqMap(decay_half_life_s=24 * HOUR, **kw())
    for m in (mix, exp):
        m.allocate(1e6, 50_000.0)
        m.decay(24 * HOUR)
    day1_mix, day1_exp = mix.total_heat(), exp.total_heat()
    for m in (mix, exp):
        m.decay(6 * 24 * HOUR)
    assert mix.total_heat() / day1_mix > 5 * (exp.total_heat() / day1_exp)


def test_consume_is_local_across_layers():
    mix = MixtureLiqMap([(0.5, HOUR), (0.5, 100 * HOUR)], **kw())
    mix.allocate(1e6, 50_000.0)
    before = agg(mix)
    mix.consume(49_000.0, 50_500.0)
    after = agg(mix)
    for i, h in before.items():
        lo, hi = mix.buckets.lo(i), mix.buckets.hi(i)
        if hi > 49_000.0 and lo <= 50_500.0:
            assert i not in after
        else:
            assert after[i] == h


def test_validation():
    with pytest.raises(ValueError):
        MixtureLiqMap([], **kw())
    with pytest.raises(ValueError):
        MixtureLiqMap([(1.0, -5.0)], **kw())
    with pytest.raises(ValueError):
        MixtureLiqMap([(0.0, HOUR), (0.0, 2 * HOUR)], **kw())
    with pytest.raises(ValueError):
        MixtureLiqMap([(1.0, float("nan"))], **kw())
    with pytest.raises(ValueError):  # тиры не по возрастанию
        MixtureLiqMap.by_leverage([(50.0, HOUR), (10.0, HOUR)],
                                  leverage_grid=GRID, weight_fn=StaticWeights(W),
                                  buckets=PriceBuckets(10.0))
    with pytest.raises(ValueError):  # тир без единого плеча в сетке
        MixtureLiqMap.by_leverage([(1.0, HOUR), (1e9, HOUR)],
                                  leverage_grid=GRID, weight_fn=StaticWeights(W),
                                  buckets=PriceBuckets(10.0))


def test_history_and_overlay_accept_mixture(tmp_path):
    import polars as pl

    from trading_system.liqmap.history import HeatHistory
    from trading_system.liqmap.terminal import terminal_heat_overlay

    mix = MixtureLiqMap([(0.7, 4 * HOUR), (0.3, 168 * HOUR)], **kw())
    hist = HeatHistory(mix)
    rows = []
    price = 50_000.0
    rng = np.random.default_rng(5)
    for i in range(60):
        price = float(price * np.exp(rng.normal(0, 0.004)))
        lo, hi = price * 0.996, price * 1.004
        mix.step(lo, hi, price, float(rng.normal(4e5, 2e5)), dt_s=3_600.0)
        hist.record((i + 1) * 3_600_000_000_000)
        rows.append({"ts_open": i * 3_600_000_000_000,
                     "ts_close": (i + 1) * 3_600_000_000_000,
                     "open": price, "high": hi, "low": lo, "close": price})
    bars = pl.DataFrame(rows)
    assert len(hist) == 60
    assert hist.total_at(59) == pytest.approx(mix.total_heat(), rel=1e-12)
    assert terminal_heat_overlay(bars, hist, name="mix", out_dir=tmp_path).exists()
