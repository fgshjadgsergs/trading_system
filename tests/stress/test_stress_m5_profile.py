"""Стресс-тесты M5: профиль объёма и свинги на вырожденных входах и масштабе.

Сценарии: весь объём в одной цене (POC/VA определены), равномерный профиль
(детерминированный тайбрейк VA, покрытие >= 70%), монотонная серия без свингов
и зигзаг где каждый бар — экстремум, кластеры equal_extremes при eps=0 и
eps=huge, пропускная на 10k свингов/баров, случайные профили (инварианты VA).

Масштаб управляется env STRESS_SCALE (по умолчанию 1.0).
"""

from __future__ import annotations

import os
import time

import numpy as np
import polars as pl
import pytest

from trading_system.profile.swings import equal_extremes, fractal_swings, level_weights
from trading_system.profile.volume_profile import hvn_lvn, poc_price, profile, value_area

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
SEED = 42
T0 = 1_755_600_000_000_000_000
MIN_NS = 60_000_000_000


def _n(base: int) -> int:
    return max(1, int(base * SCALE))


def _trades(prices: list[float], qtys: list[float]) -> pl.DataFrame:
    n = len(prices)
    return pl.DataFrame(
        {
            "ts_event": [T0 + i for i in range(n)],
            "price": prices,
            "qty": qtys,
            "qty_usd": [p * q for p, q in zip(prices, qtys, strict=True)],
        }
    )


def _prof(vols: list[float], bin_size: float = 1.0, p0: float = 100.0) -> pl.DataFrame:
    prices = [p0 + i * bin_size for i in range(len(vols))]
    trades = pl.DataFrame(
        {
            "ts_event": [T0 + i for i in range(len(vols))],
            "price": prices,
            "qty": [v / p for v, p in zip(vols, prices, strict=True)],
            "qty_usd": vols,  # ровно v, без float-шума p*(v/p)
        }
    )
    return profile(trades, bin_size=bin_size)


def _bars(highs: list[float], lows: list[float]) -> pl.DataFrame:
    n = len(highs)
    return pl.DataFrame(
        {
            "ts_open": [T0 + i * MIN_NS for i in range(n)],
            "ts_close": [T0 + (i + 1) * MIN_NS for i in range(n)],
            "open": [h - 0.25 for h in highs],
            "high": highs,
            "low": lows,
            "close": [h - 0.25 for h in highs],
        }
    )


# ---------------------------------------------------------------------------
# 1) весь объём в одной цене
# ---------------------------------------------------------------------------


def test_all_volume_single_price():
    trades = _trades([100.0] * 500, [2.0] * 500)
    prof = profile(trades, bin_size=5.0)
    assert prof.height == 1
    center = 100.0 - (100.0 % 5.0) + 2.5  # 102.5: центр бина [100, 105)
    assert poc_price(prof) == center
    va = value_area(prof, pct=0.70)
    assert va.poc == va.val == va.vah == center  # VA не пустая: ровно один бин
    assert va.share == pytest.approx(1.0)  # покрытие — весь объём
    nodes = hvn_lvn(prof)
    assert nodes.height == 1  # не падает на профиле из одного бина


def test_value_area_empty_profile_contract():
    empty = _trades([], []).pipe(profile, bin_size=1.0)
    assert empty.height == 0
    with pytest.raises(ValueError):
        value_area(empty)
    with pytest.raises(ValueError):
        profile(_trades([100.0], [1.0]), bin_size=0.0)
    with pytest.raises(ValueError):
        profile(_trades([100.0], [1.0]), bin_size=-1.0)


# ---------------------------------------------------------------------------
# 2) равномерный профиль
# ---------------------------------------------------------------------------


def test_uniform_profile_deterministic_tiebreak():
    n_bins = 20
    prof = _prof([1_000.0] * n_bins)
    va1 = value_area(prof, pct=0.70)
    va2 = value_area(prof, pct=0.70)
    assert va1 == va2  # детерминизм
    prices = prof["price"].to_numpy()
    # argmax равных -> первый бин; расширение при равных (below >= above) -> вниз,
    # но лоу уже на краю, значит растёт только vah: ровно ceil(0.7*20)=14 бинов
    assert va1.poc == prices[0]
    assert va1.val == prices[0]
    assert va1.vah == prices[13]
    assert va1.share == pytest.approx(14 / 20)
    assert va1.share >= 0.70
    # на плоском профиле нет ни HVN ни LVN (гейт w.max() > w.min())
    nodes = hvn_lvn(prof)
    assert nodes["node"].null_count() == n_bins


def test_random_profiles_va_invariants():
    """VA-инварианты держатся на сотнях случайных профилей."""
    rng = np.random.default_rng(SEED)
    for _ in range(_n(300)):
        n_bins = int(rng.integers(1, 60))
        vols = rng.uniform(0.1, 1_000.0, n_bins).tolist()
        prof = _prof(vols)
        va = value_area(prof, pct=0.70)
        assert va.val <= va.poc <= va.vah
        assert va.share >= 0.70 - 1e-12 or prof.height == 1
        vols_arr = prof["volume_usd"].to_numpy()
        largest = vols_arr.max() / vols_arr.sum()
        assert va.share <= 0.70 + max(largest, 0.05) + 1e-9
        assert poc_price(prof) == va.poc


# ---------------------------------------------------------------------------
# 3) монотонная серия / каждый бар — экстремум
# ---------------------------------------------------------------------------


def test_monotonic_series_yields_no_swings():
    n = 300
    highs = [100.0 + i for i in range(n)]
    lows = [99.0 + i for i in range(n)]
    sw = fractal_swings(_bars(highs, lows), n=2)
    assert sw.height == 0  # пусто, не падает
    # вниз по цепочке тоже не падает
    clusters = equal_extremes(sw, eps=1.0)
    assert clusters.height == 0
    weighted = level_weights(clusters, now_ts=T0)
    assert weighted.height == 0 and "weight" in weighted.columns


def test_zigzag_every_bar_extremum():
    """Зигзаг: каждый внутренний бар — строгий экстремум своего окна n=1."""
    n = 101
    highs = [110.0 if i % 2 == 0 else 100.0 for i in range(n)]
    lows = [h - 1.0 for h in highs]
    sw = fractal_swings(_bars(highs, lows), n=1)
    # чётные внутренние бары — свинг-хаи, нечётные — свинг-лоу
    expected_highs = len([i for i in range(1, n - 1) if i % 2 == 0])
    expected_lows = len([i for i in range(1, n - 1) if i % 2 == 1])
    assert sw.filter(pl.col("kind") == "high").height == expected_highs
    assert sw.filter(pl.col("kind") == "low").height == expected_lows
    assert set(sw.filter(pl.col("kind") == "high")["price"].to_list()) == {110.0}
    # подтверждение всегда на n баров позже
    assert (
        sw.select(((pl.col("ts_confirmed") - pl.col("ts_open")) >= MIN_NS).all()).item()
    )


# ---------------------------------------------------------------------------
# 4) кластеры: eps=0, eps=huge, пропускная на 10k
# ---------------------------------------------------------------------------


def _swings_frame(prices: np.ndarray, kinds: list[str]) -> pl.DataFrame:
    n = len(prices)
    return pl.DataFrame(
        {
            "ts_open": np.arange(n, dtype=np.int64) * MIN_NS + T0,
            "ts_confirmed": (np.arange(n, dtype=np.int64) + 1) * MIN_NS + T0,
            "kind": kinds,
            "price": prices.astype(float),
        }
    )


def test_equal_extremes_eps_zero_exact_matches_only():
    prices = np.array([100.0, 100.0, 100.0 + 1e-9, 200.0, 200.0, 300.0])
    sw = _swings_frame(prices, ["high"] * 6)
    clusters = equal_extremes(sw, eps=0.0)
    # 1e-9 от бегущего среднего уже не "равен" при eps=0 -> кластеры только из точных
    by_price = {round(r["price"], 6): r["count"] for r in clusters.iter_rows(named=True)}
    assert by_price.get(100.0) == 2
    assert by_price.get(200.0) == 2
    assert 300.0 not in by_price  # одиночка не кластер


def test_equal_extremes_huge_eps_single_cluster_per_kind():
    rng = np.random.default_rng(SEED)
    n = 400
    prices = rng.uniform(1.0, 1e6, n)
    kinds = ["high" if i < n // 2 else "low" for i in range(n)]
    clusters = equal_extremes(_swings_frame(prices, kinds), eps=1e12)
    assert clusters.height == 2  # всё в один кластер на каждый kind
    assert set(clusters["kind"].to_list()) == {"high", "low"}
    assert clusters["count"].sum() == n


def test_equal_extremes_throughput_10k():
    n = _n(10_000)
    rng = np.random.default_rng(SEED)
    prices = rng.uniform(40_000.0, 60_000.0, n)
    kinds = ["high" if i % 2 == 0 else "low" for i in range(n)]
    sw = _swings_frame(prices, kinds)
    t0 = time.perf_counter()
    moderate = equal_extremes(sw, eps=5.0)
    t_moderate = time.perf_counter() - t0
    t0 = time.perf_counter()
    worst = equal_extremes(sw, eps=1e9)  # один гигантский кластер: худший случай O(n^2)
    t_worst = time.perf_counter() - t0
    assert moderate.height > 0
    assert worst.height == 2
    # ~0.1s / ~1.5s на 10k в прототипе; порядок сверху — регресс
    assert t_moderate < 15.0
    assert t_worst < 45.0
    # консистентность: сумма участников не превышает входа
    assert moderate["count"].sum() <= n
    assert (moderate["count"] >= 2).all()


def test_fractal_swings_throughput_10k_bars():
    n = _n(10_000)
    rng = np.random.default_rng(SEED)
    base = 50_000 + np.cumsum(rng.normal(0, 100.0, n))
    highs = base + np.abs(rng.normal(0, 50.0, n))
    lows = base - np.abs(rng.normal(0, 50.0, n))
    bars = _bars(highs.tolist(), lows.tolist())
    t0 = time.perf_counter()
    sw = fractal_swings(bars, n=2)
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0  # ~0.05s в прототипе
    assert 0 < sw.height < n
    # каузальность держится на масштабе
    assert (
        sw.select(((pl.col("ts_confirmed") - pl.col("ts_open")) >= 2 * MIN_NS).all()).item()
    )
