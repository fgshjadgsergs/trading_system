"""Стресс сигнального слоя M7 (signals/detectors): масштаб, границы, вырождения.

1) Префиксная согласованность на 50k баров: s1/s2 детерминированы и
   edge-triggered — префикс данных даёт префикс событий (при дефолтных
   параметрах; для s2 при return_bars >= 4 это НЕ так — задокументировано).
2) Границы: пул ровно на k*ATR (включительно — сигнал), H(P*) ровно θ*ΣH
   (включительно — сигнал), прокол s2 ровно на уровне (строго '>' — НЕ прокол),
   зона s3 ровно на квантили q (включительно — блокирует), касание края пути
   (включительно — блокирует).
3) Вырождения: пустая карта пулов, все пулы тронуты, зоны покрывают всё
   (вето 100%) и ничего, ATR=0 (вырожденный «магнит в себя» — документируем).
4) Шторм: 1000 пулов / 1000 зон — тайминги уходят в reports/stress-m6.
"""

from __future__ import annotations

import os
import time
from functools import cache

import numpy as np
import polars as pl
import pytest

from trading_system.core.timeutils import NS_PER_MIN, NS_PER_S
from trading_system.signals.detectors import s1_magnet, s2_sweep_reversal, s3_filter

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
T0 = 1_755_600_000 * NS_PER_S
N_BARS = max(2_000, int(50_000 * SCALE))


def make_bars(ohlc: list[tuple[float, float, float, float]], atr=10.0) -> pl.DataFrame:
    n = len(ohlc)
    atrs = atr if isinstance(atr, list) else [atr] * n
    return pl.DataFrame(
        {
            "ts_open": [T0 + i * NS_PER_MIN for i in range(n)],
            "ts_close": [T0 + (i + 1) * NS_PER_MIN for i in range(n)],
            "open": [r[0] for r in ohlc],
            "high": [r[1] for r in ohlc],
            "low": [r[2] for r in ohlc],
            "close": [r[3] for r in ohlc],
            "atr": pl.Series(atrs, dtype=pl.Float64),
        }
    )


def pools_frame(rows: list[tuple[float, float, int | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "price": [r[0] for r in rows],
            "heat_usd": [r[1] for r in rows],
            "touched_ts": pl.Series([r[2] for r in rows], dtype=pl.Int64),
        }
    )


@cache
def walk_bars() -> pl.DataFrame:
    """Сидированный случайный блуждающий день из N_BARS минутных баров."""
    rng = np.random.default_rng(1234)
    steps = rng.normal(0.0, 5.0, N_BARS).cumsum()
    closes = 1_000.0 + steps - steps.min() + 100.0
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + rng.uniform(0.5, 5.0, N_BARS)
    lows = np.minimum(opens, closes) - rng.uniform(0.5, 5.0, N_BARS)
    idx = np.arange(N_BARS)
    return pl.DataFrame(
        {
            "ts_open": T0 + idx * NS_PER_MIN,
            "ts_close": T0 + (idx + 1) * NS_PER_MIN,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "atr": np.full(N_BARS, 10.0),
        }
    )


def _cut_ts(cut: int) -> int:
    return T0 + cut * NS_PER_MIN


@cache
def storm_numbers() -> dict:
    """Тайминги масштабных прогонов; их печатает отчёт reports/stress-m6."""
    bars = walk_bars()
    closes = bars["close"].to_numpy()
    lo, hi = float(closes.min()), float(closes.max())
    out: dict = {"n_bars": N_BARS}

    pools = pools_frame(
        [(float(p), float(1e6 * (1 + k)), None) for k, p in enumerate(np.linspace(lo, hi, 7))]
    )
    t0 = time.perf_counter()
    ev1 = s1_magnet(bars, pools, k_atr=3.0, min_heat_share=0.05)
    out["t_s1_day_s"] = time.perf_counter() - t0
    out["s1_day_events"] = ev1.height

    levels = pl.DataFrame(
        {
            "kind": ["high", "low"],
            "price": [float(np.percentile(closes, 90)), float(np.percentile(closes, 10))],
            "count": [3, 3],
        }
    )
    t0 = time.perf_counter()
    ev2 = s2_sweep_reversal(bars, levels, return_bars=3)
    out["t_s2_day_s"] = time.perf_counter() - t0
    out["s2_day_events"] = ev2.height

    n_storm_bars = max(200, int(1_000 * SCALE))
    rng = np.random.default_rng(7)
    storm_pools = pools_frame(
        [(float(p), float(rng.uniform(1e5, 1e7)), None) for p in np.linspace(lo, hi, 1_000)]
    )
    sb = bars.head(n_storm_bars)
    t0 = time.perf_counter()
    ev_storm = s1_magnet(sb, storm_pools, k_atr=3.0, min_heat_share=0.0)
    out["t_s1_storm_s"] = time.perf_counter() - t0
    out["s1_storm_bars"] = n_storm_bars
    out["s1_storm_events"] = ev_storm.height

    n_events = 1_000
    prices = rng.uniform(lo, hi, n_events)
    targets = prices + rng.uniform(-50.0, 50.0, n_events)
    events = pl.DataFrame(
        {
            "ts": T0 + np.arange(n_events) * NS_PER_MIN,
            "signal": ["s1"] * n_events,
            "side": np.where(targets > prices, 1, -1).astype(np.int8),
            "price": prices,
            "target": targets,
            "meta": np.zeros(n_events),
        }
    )
    grid = np.linspace(lo, hi, 1_000)
    storm_zones = pl.DataFrame(
        {"lo": grid, "hi": grid + 0.5, "heat_usd": rng.uniform(1e5, 1e7, 1_000)}
    )
    t0 = time.perf_counter()
    blocked = s3_filter(events, storm_zones, dense_quantile=0.99)
    out["t_s3_storm_s"] = time.perf_counter() - t0
    out["s3_storm_blocked"] = int(blocked["blocked"].sum())
    out["s3_storm_events"] = n_events
    return out


# -- 1) масштаб: префиксная согласованность -----------------------------------


def test_s1_prefix_consistency_full_day():
    bars = walk_bars()
    closes = bars["close"].to_numpy()
    pools = pools_frame(
        [
            (float(p), float(1e6 * (1 + k)), None)
            for k, p in enumerate(np.linspace(float(closes.min()), float(closes.max()), 7))
        ]
    )
    full = s1_magnet(bars, pools, k_atr=3.0, min_heat_share=0.05)
    # edge-trigger: каждый пул срабатывает не более одного раза
    assert full["target"].n_unique() == full.height
    assert full["ts"].is_sorted()
    for cut in [1, 137, N_BARS // 10, N_BARS // 2, int(N_BARS * 0.8), N_BARS]:
        pref = s1_magnet(bars.head(cut), pools, k_atr=3.0, min_heat_share=0.05)
        assert pref.equals(full.filter(pl.col("ts") <= _cut_ts(cut))), f"cut={cut}"


def test_s2_prefix_consistency_full_day():
    bars = walk_bars()
    closes = bars["close"].to_numpy()
    levels = pl.DataFrame(
        {
            "kind": ["high", "low"],
            "price": [float(np.percentile(closes, 90)), float(np.percentile(closes, 10))],
            "count": [3, 3],
        }
    )
    full = s2_sweep_reversal(bars, levels, return_bars=3)
    assert full.height > 0  # день реально даёт свипы
    assert full["ts"].is_sorted()
    for cut in [N_BARS // 50, N_BARS // 5, N_BARS // 2, int(N_BARS * 0.9), N_BARS]:
        pref = s2_sweep_reversal(bars.head(cut), levels, return_bars=3)
        assert pref.equals(full.filter(pl.col("ts") <= _cut_ts(cut))), f"cut={cut}"


def test_s2_prefix_inconsistency_at_return_bars_4():
    """ДИЗАЙН-СЛАБОСТЬ (документируем, не чиним): при return_bars >= 4 два
    вложенных прокола делают s2 префиксно-НЕсогласованным: на полном дне бар
    отдаётся ПЕРВОМУ проколу (эпизод 2 подавляется прыжком i=fired_at+1), а на
    префиксе, обрезавшем окно первого прокола, срабатывает ВТОРОЙ прокол —
    другой ts и другой target. При дефолтном return_bars=3 окна не пересекаются
    так глубоко и согласованность выполняется (тест выше)."""
    lvl = pl.DataFrame({"kind": ["high"], "price": [110.0], "count": [2]})
    ohlc = [
        (100.0, 105.0, 95.0, 104.0),
        (104.0, 112.0, 90.0, 108.0),  # прокол 1 (глубокий low 90)
        (108.0, 109.0, 101.0, 105.0),  # возврат под уровень
        (105.0, 111.0, 100.0, 107.0),  # прокол 2 (low 100)
        (107.0, 108.0, 98.0, 99.0),  # реверс только для прокола 2 (99 < 100, но >= 90)
        (99.0, 100.0, 85.0, 88.0),  # реверс для прокола 1 (88 < 90)
    ]
    bars = make_bars(ohlc)
    full = s2_sweep_reversal(bars, lvl, return_bars=4)
    pref = s2_sweep_reversal(bars.head(5), lvl, return_bars=4)
    # фактическое поведение: полный день — одно событие прокола 1 на баре 5
    assert full.height == 1
    assert full["ts"][0] == T0 + 6 * NS_PER_MIN and full["target"][0] == 90.0
    # префикс из 5 баров — событие прокола 2 на баре 4, которого нет в полном
    assert pref.height == 1
    assert pref["ts"][0] == T0 + 5 * NS_PER_MIN and pref["target"][0] == 100.0
    assert not pref.equals(full.filter(pl.col("ts") <= T0 + 5 * NS_PER_MIN))


# -- 2) границы ---------------------------------------------------------------


def test_s1_pool_exactly_at_k_atr_fires():
    bars = make_bars([(100.0, 101.0, 99.0, 100.0)], atr=10.0)
    # |price - close| == k*ATR: граница включительна — сигнал есть
    assert s1_magnet(bars, pools_frame([(130.0, 1e6, None)]), k_atr=3.0).height == 1
    # на волосок дальше — сигнала нет
    assert s1_magnet(bars, pools_frame([(130.0 + 1e-9, 1e6, None)]), k_atr=3.0).height == 0


def test_s1_heat_share_exactly_at_theta_fires():
    bars = make_bars([(100.0, 101.0, 99.0, 100.0)], atr=10.0)
    # пул в зоне держит ровно θ=0.25 суммарного heat: граница включительна
    pools = pools_frame([(105.0, 2.5e6, None), (400.0, 7.5e6, None)])
    assert s1_magnet(bars, pools, k_atr=3.0, min_heat_share=0.25).height == 1
    # чуть тяжелее дальний пул — доля < θ, сигнала нет
    pools2 = pools_frame([(105.0, 2.5e6, None), (400.0, 7.5e6 + 1.0, None)])
    assert s1_magnet(bars, pools2, k_atr=3.0, min_heat_share=0.25).height == 0


def test_s2_pierce_exactly_at_level_is_not_a_pierce():
    lvl = pl.DataFrame({"kind": ["high"], "price": [110.0], "count": [2]})
    base = [(100.0, 105.0, 98.0, 104.0)]
    tail = [(108.0, 109.0, 95.0, 96.0)]  # заведомый реверс, если был прокол
    # high == уровню: строгое '>' -> прокола нет
    touch = base + [(104.0, 110.0, 103.0, 108.0)] + tail
    assert s2_sweep_reversal(make_bars(touch), lvl, return_bars=3).height == 0
    # на волосок выше — прокол и сигнал
    pierce = base + [(104.0, 110.0 + 1e-9, 103.0, 108.0)] + tail
    assert s2_sweep_reversal(make_bars(pierce), lvl, return_bars=3).height == 1


def _one_event(price: float, target: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts": [T0],
            "signal": ["s1"],
            "side": [1 if target > price else -1],
            "price": [price],
            "target": [target],
            "meta": [0.0],
        },
        schema_overrides={"side": pl.Int8},
    )


def test_s3_zone_exactly_at_quantile_blocks():
    # порог = квантиль всех heat; зона с heat РОВНО на пороге плотная ('>=')
    zones = pl.DataFrame({"lo": [104.0, 200.0], "hi": [106.0, 202.0], "heat_usd": [1e6, 2e6]})
    ev = s3_filter(_one_event(100.0, 110.0), zones, dense_quantile=1.0)
    assert not ev["blocked"][0]  # порог 2e6, зона на пути держит 1e6 < порога
    zones_eq = zones.with_columns(pl.Series("heat_usd", [2e6, 2e6]))
    ev2 = s3_filter(_one_event(100.0, 110.0), zones_eq, dense_quantile=1.0)
    assert ev2["blocked"][0]  # heat == порогу: блокирует


def test_s3_zone_edge_touching_path_blocks():
    # zone.lo == верхнему краю пути: пересечение включительно — блокирует
    edge = pl.DataFrame({"lo": [110.0], "hi": [112.0], "heat_usd": [1e6]})
    assert bool(s3_filter(_one_event(100.0, 110.0), edge, dense_quantile=0.0)["blocked"][0])
    beyond = pl.DataFrame({"lo": [110.0 + 1e-9], "hi": [112.0], "heat_usd": [1e6]})
    assert not bool(s3_filter(_one_event(100.0, 110.0), beyond, dense_quantile=0.0)["blocked"][0])


# -- 3) вырождения ------------------------------------------------------------


def test_s1_degenerate_pools():
    bars = make_bars([(100.0, 101.0, 99.0, 100.0), (100.0, 102.0, 98.0, 101.0)], atr=10.0)
    # карта пуста
    assert s1_magnet(bars, pools_frame([]), k_atr=3.0).height == 0
    # все пулы тронуты до начала
    assert s1_magnet(bars, pools_frame([(105.0, 1e6, T0)]), k_atr=3.0).height == 0
    # тронут РОВНО на закрытии бара: touched_ts <= ts -> этот бар уже не сигнал
    assert s1_magnet(bars, pools_frame([(105.0, 1e6, T0 + NS_PER_MIN)]), k_atr=3.0).height == 0
    # тронут после первого закрытия: первый бар успевает выстрелить
    ev = s1_magnet(bars, pools_frame([(105.0, 1e6, T0 + NS_PER_MIN + 1)]), k_atr=3.0)
    assert ev.height == 1 and ev["ts"][0] == T0 + NS_PER_MIN


def test_s1_zero_atr_documented_self_magnet():
    """ДИЗАЙН-ОСОБЕННОСТЬ (документируем): ATR=0 не отфильтровывается — радиус
    поиска k*ATR вырождается в 0, и пул РОВНО на цене закрытия даёт событие с
    target == price (магнит «в себя», side=-1, нулевая дистанция до цели)."""
    bars = make_bars([(100.0, 101.0, 99.0, 100.0)], atr=0.0)
    ev = s1_magnet(bars, pools_frame([(100.0, 1e6, None)]), k_atr=3.0, min_heat_share=0.0)
    assert ev.height == 1
    assert ev["target"][0] == ev["price"][0] == 100.0
    assert ev["side"][0] == -1
    # пул хоть чуть в стороне при ATR=0 недостижим
    ev2 = s1_magnet(bars, pools_frame([(100.5, 1e6, None)]), k_atr=3.0, min_heat_share=0.0)
    assert ev2.height == 0


def test_s1_null_atr_bar_skipped():
    bars = make_bars([(100.0, 101.0, 99.0, 100.0), (100.0, 101.0, 99.0, 100.0)],
                     atr=[None, 10.0])
    ev = s1_magnet(bars, pools_frame([(105.0, 1e6, None)]), k_atr=3.0)
    assert ev.height == 1  # первый бар (atr null) пропущен, второй стреляет
    assert ev["ts"][0] == T0 + 2 * NS_PER_MIN


def test_s3_veto_all_and_nothing():
    n = 50
    rng = np.random.default_rng(3)
    prices = rng.uniform(100.0, 200.0, n)
    targets = prices + rng.uniform(-20.0, 20.0, n)
    events = pl.DataFrame(
        {
            "ts": T0 + np.arange(n) * NS_PER_MIN,
            "signal": ["s1"] * n,
            "side": np.where(targets > prices, 1, -1).astype(np.int8),
            "price": prices,
            "target": targets,
            "meta": np.zeros(n),
        }
    )
    # одна зона покрывает весь диапазон: вето 100%
    all_zone = pl.DataFrame({"lo": [0.0], "hi": [1e6], "heat_usd": [1e9]})
    out = s3_filter(events, all_zone, dense_quantile=0.9)
    assert int(out["blocked"].sum()) == n
    # зон нет вовсе: порог inf, вето 0%
    empty = pl.DataFrame(schema={"lo": pl.Float64, "hi": pl.Float64, "heat_usd": pl.Float64})
    out2 = s3_filter(events, empty, dense_quantile=0.9)
    assert int(out2["blocked"].sum()) == 0
    # зоны есть, но все вне пути: вето 0%
    off_path = pl.DataFrame({"lo": [1e5], "hi": [1e5 + 1], "heat_usd": [1e9]})
    out3 = s3_filter(events, off_path, dense_quantile=0.0)
    assert int(out3["blocked"].sum()) == 0
    # пустые события с зонами: колонки на месте
    out4 = s3_filter(events.head(0), all_zone)
    assert out4.height == 0 and "blocked" in out4.columns


# -- 4) шторм -----------------------------------------------------------------


def test_storm_budgets():
    res = storm_numbers()
    budget = 30.0 * max(1.0, SCALE)
    assert res["t_s1_day_s"] < budget  # 50k баров x 7 пулов
    assert res["t_s2_day_s"] < budget  # 50k баров x 2 уровня
    assert res["t_s1_storm_s"] < budget  # 1000 пулов
    assert res["t_s3_storm_s"] < budget  # 1000 зон x 1000 событий
    assert res["s1_day_events"] >= 1
    assert res["s2_day_events"] >= 1
    assert 0 < res["s3_storm_blocked"] < res["s3_storm_events"]
