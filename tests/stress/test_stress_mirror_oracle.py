"""Оракул зеркала real_data: где быстрый билдер обязан совпадать с точным
LiqMap-реплеем БИТ-В-БИТ, а где расходится и насколько.

Существующая сверка допускает медианную ошибку тоталов 5% — этого хватает,
чтобы пропустить регрессию. При плоской ставке маржи и общей сетке
эквивалентность на самом деле машинная, и именно это здесь закрепляется —
поячеечно, а не по суммам.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest

from trading_system.calibration.real_data import (
    bars_to_arrays,
    bucket_grid,
    make_real_heat_builder,
)
from trading_system.collectors.brackets import bracket_liq_price_fn
from trading_system.core.liquidation import DEFAULT_BRACKETS
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.history import HeatHistory
from trading_system.liqmap.map import LiqMap, StaticWeights

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
MIN = 60_000_000_000


def synth_bars(n: int, seed: int = 0, vol: float = 0.004) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    price, ts, rows = 3_000.0, 0, []
    for _ in range(n):
        price *= float(np.exp(rng.normal(0, vol)))
        hi = price * (1 + abs(rng.normal(0, vol / 2)))
        lo = price * (1 - abs(rng.normal(0, vol / 2)))
        ts += MIN
        rows.append(
            dict(
                symbol="X", ts_open=ts - MIN, ts_close=ts, open=price, high=hi, low=lo,
                close=price, volume=1.0, quote_volume=price,
                d_oi_usd=float(rng.uniform(-2e5, 1e6)), atr=price * vol,
                long_share=float(rng.uniform(0.3, 0.7)),
            )
        )
    return pl.DataFrame(rows)


def replay(bars: pl.DataFrame, grid: np.ndarray, w: np.ndarray, edges: np.ndarray,
           **map_kw) -> tuple[np.ndarray, list[float]]:
    """Точный LiqMap-реплей, спроецированный на ту же сетку бакетов."""
    lm = LiqMap(
        leverage_grid=list(grid),
        buckets=PriceBuckets(float(edges[1] - edges[0])),
        weight_fn=StaticWeights(w),
        decay_half_life_s=86_400.0,
        **map_kw,
    )
    hist = HeatHistory(lm)
    for r in bars.iter_rows(named=True):
        lm.step(r["low"], r["high"], r["close"], r["d_oi_usd"], dt_s=60.0,
                long_share=r["long_share"])
        hist.record(r["ts_close"])
    nb = len(edges) - 1
    rows = np.zeros((len(hist), nb))
    base = int(np.searchsorted(edges, edges[0], side="right") - 1)
    for i in range(len(hist)):
        for idx, h in ((k, v) for k, v in hist._frames[i].items()):
            j = idx - int(round(edges[0] / (edges[1] - edges[0]))) - base
            if 0 <= j < nb:
                rows[i, j] += h
    return rows, [hist.total_at(i) for i in range(len(hist))]


def test_mirror_is_cellwise_exact_at_flat_mmr():
    """Плоская ставка маржи, общая сетка: расхождение — машинный ноль,
    и не только в суммах, но в каждой ячейке."""
    bars = synth_bars(int(150 * SCALE), seed=1)
    arr = bars_to_arrays(bars)
    edges = bucket_grid(arr, atr_fraction=0.5)
    grid = np.array([10.0, 25.0, 50.0])
    w = np.array([0.5, 0.3, 0.2])
    fast = make_real_heat_builder(arr, grid, edges, bar_s=60.0)(w)
    exact_rows, exact_tot = replay(bars, grid, w, edges)
    tot = np.array(exact_tot)
    m = tot > 0
    rel = np.abs(fast.sum(axis=1)[m] - tot[m]) / tot[m]
    assert rel.max() < 1e-12, rel.max()
    scale = max(tot.max(), 1.0)
    assert np.abs(fast - exact_rows).max() / scale < 1e-12


def test_typical_account_makes_bracket_path_exact_too():
    """С брекет-таблицами зеркало приближает размер счёта долей слайса, и
    расхождение реально; с typical_account_usd (M1) qty перестаёт зависеть
    от весов и эквивалентность снова машинная."""
    bars = synth_bars(int(120 * SCALE), seed=2)
    arr = bars_to_arrays(bars)
    edges = bucket_grid(arr, atr_fraction=0.5)
    grid = np.array([10.0, 25.0, 50.0])
    w = np.array([0.2, 0.3, 0.5])
    liq_fn = bracket_liq_price_fn({"X": DEFAULT_BRACKETS}, "X")

    approx = make_real_heat_builder(arr, grid, edges, bar_s=60.0, liq_fn=liq_fn)(w)
    rows_a, tot_a = replay(bars, grid, w, edges, liq_price_fn=liq_fn)
    tot = np.array(tot_a)
    m = tot > 0
    err_approx = float(np.abs(approx.sum(axis=1)[m] - tot[m]).max() / max(tot.max(), 1.0))

    exact = make_real_heat_builder(
        arr, grid, edges, bar_s=60.0, liq_fn=liq_fn, typical_account_usd=20_000.0
    )(w)
    rows_e, tot_e = replay(bars, grid, w, edges, liq_price_fn=liq_fn,
                           typical_account_usd=20_000.0)
    tot2 = np.array(tot_e)
    m2 = tot2 > 0
    err_exact = float(np.abs(exact.sum(axis=1)[m2] - tot2[m2]).max() / max(tot2.max(), 1.0))
    assert err_exact < 1e-12, (err_exact, err_approx)
    assert err_exact <= err_approx  # M1 не ухудшает согласованность путей


def test_builder_rows_are_causal():
    """Строка t зависит только от баров до t: обрезание хвоста не меняет
    ни одной прошлой строки."""
    bars = synth_bars(int(120 * SCALE), seed=3)
    arr = bars_to_arrays(bars)
    edges = bucket_grid(arr, atr_fraction=0.5)
    grid = np.array([10.0, 25.0])
    w = np.array([0.6, 0.4])
    full = make_real_heat_builder(arr, grid, edges, bar_s=60.0)(w)
    k = len(bars) // 2
    head_arr = bars_to_arrays(bars.head(k))
    head = make_real_heat_builder(head_arr, grid, edges, bar_s=60.0)(w)
    assert np.array_equal(full[:k], head)


def test_builder_is_scale_invariant_and_deterministic():
    """Веса нормируются внутри (умножение на константу ничего не меняет),
    а повторный вызов даёт бит-в-бит тот же результат."""
    bars = synth_bars(int(80 * SCALE), seed=4)
    arr = bars_to_arrays(bars)
    edges = bucket_grid(arr, atr_fraction=0.5)
    grid = np.array([10.0, 25.0, 50.0])
    build = make_real_heat_builder(arr, grid, edges, bar_s=60.0)
    w = np.array([0.2, 0.5, 0.3])
    a = build(w)
    assert np.array_equal(a, build(w))
    assert np.allclose(a, build(w * 7.0), rtol=1e-12, atol=0)
    assert (a >= 0).all() and np.isfinite(a).all()


def test_split_sides_halves_sum_to_glued():
    bars = synth_bars(int(80 * SCALE), seed=6)
    arr = bars_to_arrays(bars)
    edges = bucket_grid(arr, atr_fraction=0.5)
    grid = np.array([10.0, 50.0])
    w = np.array([0.5, 0.5])
    glued = make_real_heat_builder(arr, grid, edges, bar_s=60.0)(w)
    split = make_real_heat_builder(arr, grid, edges, bar_s=60.0, split_sides=True)(w)
    assert np.allclose(split[:, 0, :] + split[:, 1, :], glued, rtol=1e-12, atol=0)
