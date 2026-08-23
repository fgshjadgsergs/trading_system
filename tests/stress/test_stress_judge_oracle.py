"""Оракул судьи Gate A: каузальность бейзлайна, честность side-aware
сравнения, отсутствие фиктивного захвата и корректность лага.

Судья решает, работает ли система вообще, поэтому его свойства проверяются
жёстче самой карты: слабый или подглядывающий бейзлайн = самообман.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest

from trading_system.calibration.weights import (
    capture_details,
    capture_rate,
    naive_baseline_heat,
)
from trading_system.core.schema import Side

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
MIN = 60_000_000_000


def liq_frame(rows: list[tuple[int, float, float, Side]]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_event": [r[0] for r in rows],
            "price": [r[1] for r in rows],
            "qty_usd": [r[2] for r in rows],
            "side": [r[3] for r in rows],
        }
    )


def test_baseline_is_causal_row_by_row():
    """Строка t бейзлайна зависит только от prices[:t+1]: продолжение ряда
    (в том числе скачок на порядок) не меняет ни одной прошлой строки."""
    rng = np.random.default_rng(3)
    n = int(300 * SCALE)
    prices = 500.0 * np.exp(np.cumsum(rng.normal(0, 0.003, n)))
    edges = np.arange(100.0, 20_000.0, 5.0)
    head = naive_baseline_heat(prices, edges)
    for tail in (np.full(50, 5_000.0), np.full(50, 120.0), prices[:50]):
        full = naive_baseline_heat(np.concatenate([prices, tail]), edges)
        assert np.array_equal(head, full[:n])


def test_side_split_baseline_conserves_mass_and_can_beat_or_lose_to_glued():
    """Half-матрицы бейзлайна складываются в исходную (масса не потеряна), а
    какая форма сильнее — зависит от данных: под лагом геометрия «ниже цены»
    режется по цене СРЕЗА, а не события. Поэтому судья обязан брать
    сильнейший вариант как оппонента (см. stage3_report), иначе Gate A
    проходится за счёт ослабления противника."""
    rng = np.random.default_rng(5)
    n = int(200 * SCALE)
    prices = 3_000.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    edges = np.arange(2_000.0, 4_500.0, 2.0)
    ts = np.arange(n, dtype=np.int64) * MIN
    glued = naive_baseline_heat(prices, edges)
    split = naive_baseline_heat(prices, edges, split_sides=True)
    assert np.allclose(split[:, 0, :] + split[:, 1, :], glued)
    # лонг-тепло строго ниже цены бара, шорт — строго выше
    centers = 0.5 * (edges[:-1] + edges[1:])
    for t in (0, n // 2, n - 1):
        assert not split[t, 0, centers >= prices[t]].any()
        assert not split[t, 1, centers < prices[t]].any()
    rows = []
    for i in range(50, n, 3):
        p = float(prices[i]) * (0.995 if i % 2 else 1.005)
        side = Side.SELL if i % 2 else Side.BUY  # SELL = ликвидирован лонг
        rows.append((int(ts[i]) + MIN // 2, p, 1_000.0, side))
    liqs = liq_frame(rows)
    c_glued = capture_rate(glued, ts, edges, liqs, 0.1, None, MIN)
    c_split = capture_rate(split, ts, edges, liqs, 0.1, None, MIN)
    strongest = max(c_glued, c_split)
    assert strongest >= c_glued and strongest >= c_split
    assert 0.0 <= c_split <= 1.0 and 0.0 <= c_glued <= 1.0


def test_empty_map_captures_nothing():
    """Пустая карта не должна получать фиктивный захват: топ-дециль нулевой
    строки пуст, а не «все ячейки одинаково горячие»."""
    n, nb = 20, 100
    edges = np.arange(1_000.0, 1_000.0 + nb + 1, 1.0)
    ts = np.arange(n, dtype=np.int64) * MIN
    liqs = liq_frame([(int(ts[10]) + 1, 1_050.0, 5_000.0, Side.SELL)])
    assert capture_rate(np.zeros((n, nb)), ts, edges, liqs, 0.1) == 0.0
    # и в side-aware форме тоже
    assert capture_rate(np.zeros((n, 2, nb)), ts, edges, liqs, 0.1) == 0.0
    # карта, равномерно горячая везде, захватывает не больше доли top_decile
    flat = np.ones((n, nb))
    rng = np.random.default_rng(2)
    many = liq_frame([
        (int(ts[10]) + 1, float(rng.uniform(1_000.0, 1_100.0)), 100.0, Side.SELL)
        for _ in range(int(2000 * SCALE))
    ])
    got = capture_rate(flat, ts, edges, many, 0.1)
    assert got <= 0.16, got  # ~10% ячеек, попадание случайно


def test_lag_excludes_the_snapshot_of_the_event_itself():
    """С lag=1 бар событие судится по срезу, снятому баром раньше: карта,
    «нарисованная» ровно под уже случившееся событие, не получает очков."""
    n, nb = 30, 60
    edges = np.arange(500.0, 500.0 + nb + 1, 1.0)
    ts = np.arange(n, dtype=np.int64) * MIN
    hot_cell = 40
    heat = np.zeros((n, nb))
    heat[20:, hot_cell] = 100.0  # тепло появилось ровно на баре 20
    price = float(edges[hot_cell]) + 0.5
    liqs = liq_frame([(int(ts[20]) + MIN // 2, price, 1_000.0, Side.SELL)])
    assert capture_rate(heat, ts, edges, liqs, 0.1, None, 0) == pytest.approx(1.0)
    assert capture_rate(heat, ts, edges, liqs, 0.1, None, MIN) == 0.0
    # событие до первого среза не считается вовсе (не портит знаменатель)
    early = liq_frame([(int(ts[0]) - 1, price, 1_000.0, Side.SELL)])
    cap, tot = capture_details(heat, ts, edges, early, 0.1)
    assert tot == 0.0 and cap == 0.0


def test_side_aware_scores_each_print_against_its_own_half():
    """Лонг-принт в ячейке, горячей только в шорт-половине, не засчитывается
    (баг склеенного режима, найденный панелью)."""
    n, nb = 10, 50
    edges = np.arange(100.0, 100.0 + nb + 1, 1.0)
    ts = np.arange(n, dtype=np.int64) * MIN
    split = np.zeros((n, 2, nb))
    split[:, 1, 30] = 500.0  # горячо только в шорт-половине
    glued = split[:, 0, :] + split[:, 1, :]
    price = float(edges[30]) + 0.5
    long_print = liq_frame([(int(ts[5]) + 1, price, 1_000.0, Side.SELL)])
    assert capture_rate(glued, ts, edges, long_print, 0.1) == pytest.approx(1.0)
    assert capture_rate(split, ts, edges, long_print, 0.1) == 0.0
    short_print = liq_frame([(int(ts[5]) + 1, price, 1_000.0, Side.BUY)])
    assert capture_rate(split, ts, edges, short_print, 0.1) == pytest.approx(1.0)
    # пер-сторонняя разбивка согласована с общим числом
    both = pl.concat([long_print, short_print])
    cap, tot, per = capture_details(split, ts, edges, both, 0.1)
    assert per["long"] == (0.0, 1_000.0) and per["short"] == (1_000.0, 1_000.0)
    assert cap == 1_000.0 and tot == 2_000.0


def test_tolerance_dilation_is_symmetric_and_bounded():
    n, nb = 8, 41
    edges = np.arange(0.0, nb + 1, 1.0)
    ts = np.arange(n, dtype=np.int64) * MIN
    heat = np.zeros((n, nb))
    heat[:, 20] = 10.0
    for dist, tol, want in ((2, 2, 1.0), (3, 2, 0.0), (3, 3, 1.0), (-2, 2, 1.0)):
        liqs = liq_frame([(int(ts[4]) + 1, 20.5 + dist, 1_000.0, Side.SELL)])
        got = capture_rate(heat, ts, edges, liqs, 0.05, None, 0, tol)
        assert got == pytest.approx(want), (dist, tol)


def test_gate_a_canary_does_not_pass_on_noise():
    """КАНАРЕЙКА: на мире, где принты НЕ связаны с картой (ликвидации сыплются
    у текущей цены независимо от ΔOI), вердикт Gate A не должен проходить.

    Без выравнивания площади тревоги старое правило `static > naive` проходило
    здесь почти всегда: карта помечает горячими ~10% сетки, разреженный
    бейзлайн — около процента, и метрика мерила площадь, а не точность.
    """
    import polars as pl

    from scripts.stage3_report import analyze
    from trading_system.calibration.synthetic import make_world
    from trading_system.core.timeutils import NS_PER_MIN

    world = make_world(n_bars=1_500, seed=11)
    closes = world.prices
    lows = np.minimum(np.concatenate([[closes[0]], closes[:-1]]), closes)
    highs = np.maximum(np.concatenate([[closes[0]], closes[:-1]]), closes)
    bars = pl.DataFrame(
        {
            "ts_open": world.ts - NS_PER_MIN,
            "ts_close": world.ts,
            "close": closes,
            "low": lows,
            "high": highs,
            "open": np.concatenate([[closes[0]], closes[:-1]]),
            "quote_volume": np.full(len(closes), 1.0),
            "d_oi_usd": world.entry_notional,
            "atr": np.full(len(closes), world.atr),
        }
    )
    # шумовые принты: у текущей цены, без всякой связи с картой
    rng = np.random.default_rng(11)
    idx = rng.integers(len(closes) // 3, len(closes), size=400)
    noise = pl.DataFrame(
        {
            "ts_event": (world.ts[idx] + NS_PER_MIN // 2).astype(np.int64),
            "price": closes[idx] * (1.0 + rng.normal(0, 0.0005, len(idx))),
            "qty_usd": np.abs(rng.lognormal(9.0, 1.0, len(idx))),
            "side": [Side.SELL if v else Side.BUY for v in rng.integers(0, 2, len(idx))],
        }
    )
    cfg = {
        "project": {"seed": 11},
        "liqmap": {
            "leverage_grid": [10, 25, 50, 100],
            "bucket_atr_fraction": 0.1,
            "decay_half_life_s": 86_400,
            "long_share_default": 0.5,
            "maint_margin_rate_flat": 0.005,
        },
    }
    res = analyze(
        bars, noise, cfg, tmp_dir(), world.symbol, timeframe="1m",
        test_frac=0.3, embargo_days=0.05, n_candidates=8, seed=11,
    )
    area = res["alert_area"]
    assert area["naive"] >= 0.5 * area["static"], area  # площади выровнены
    assert res["gate_a"] is not True, (res["capture"], area, res["lift"])


def tmp_dir():
    import tempfile
    from pathlib import Path

    return Path(tempfile.mkdtemp())
