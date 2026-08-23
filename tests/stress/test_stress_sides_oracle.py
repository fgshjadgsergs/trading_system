"""Оракул долей сторон: находки панели по sides.py, каждая — тестом.

Панель (3 линзы) нашла: NaN переживает clip/fill_null и роняет allocate,
пока калибраторный путь его тихо чинит; метрика, отчитавшаяся один раз и
умершая, весит в бленде вечно; ратио 47-часовой давности используется молча;
победитель дубля — «кто позже лёг в part-файл»; join без by=symbol отдаёт
барам BTC долю ETH; бленд стоковых ратио сдвинут структурно, из-за чего знак
отклонения от 0.5 — монета.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest

from trading_system.liqmap.sides import (
    DEFAULT_BLEND,
    join_long_share,
    long_share_series,
)

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
MIN = 60_000_000_000
NAN = float("nan")


def ratio_frame(rows: list[tuple[str, int, float]], *, ts_recv: list[int] | None = None,
                symbol: str = "BTCUSDT") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "exchange": ["binance_usdm"] * len(rows),
            "symbol": [symbol] * len(rows),
            "ts_event": [r[1] for r in rows],
            "ts_recv": ts_recv if ts_recv is not None else [r[1] for r in rows],
            "metric": [r[0] for r in rows],
            "long_share": [r[2] for r in rows],
            "short_share": [1.0 - r[2] for r in rows],
            "ratio": [r[2] / max(1.0 - r[2], 1e-9) for r in rows],
        }
    )


def bar_frame(ts_closes: list[int], symbol: str = "BTCUSDT") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": [symbol] * len(ts_closes),
            "exchange": ["binance_usdm"] * len(ts_closes),
            "ts_open": [t - 5 * MIN for t in ts_closes],
            "ts_close": ts_closes,
        }
    )


def test_blend_matches_hand_computation():
    """Дефолтный путь = ручной расчёт по формуле ренормировки."""
    r = ratio_frame([
        ("global_ls_account", 10 * MIN, 0.70),
        ("top_ls_position", 10 * MIN, 0.50),
        ("taker_ls", 10 * MIN, 0.60),
    ])
    got = long_share_series(r)["long_share"][0]
    want = (0.4 * 0.70 + 0.3 * 0.50 + 0.3 * 0.60) / 1.0
    assert got == pytest.approx(want, rel=1e-15)
    # только одна метрика -> чистое её значение (ренормировка)
    r1 = ratio_frame([("global_ls_account", 10 * MIN, 0.70)])
    assert long_share_series(r1)["long_share"][0] == pytest.approx(0.70, rel=1e-12)


def test_nan_point_neither_poisons_nor_crashes():
    """NaN в одной метрике: сосед не заражён, серия конечна, бары валидны."""
    r = ratio_frame([
        ("global_ls_account", 10 * MIN, NAN),
        ("top_ls_position", 10 * MIN, 0.60),
        ("global_ls_account", 20 * MIN, 0.70),
    ])
    s = long_share_series(r)
    assert np.isfinite(s["long_share"].to_numpy()).all()
    assert s["long_share"][0] == pytest.approx(0.60, rel=1e-12)  # только top
    bars = bar_frame([15 * MIN, 25 * MIN])
    ls = join_long_share(bars, r)["long_share"].to_numpy()
    assert np.isfinite(ls).all()
    # inf и отрицательные тоже не проходят
    r2 = ratio_frame([
        ("global_ls_account", 10 * MIN, float("inf")),
        ("top_ls_position", 10 * MIN, 0.55),
    ])
    assert long_share_series(r2)["long_share"][0] == pytest.approx(0.55, rel=1e-12)
    r3 = ratio_frame([
        ("global_ls_account", 10 * MIN, -1.0),
        ("top_ls_position", 10 * MIN, 0.55),
    ])
    assert long_share_series(r3, validate_range=True)["long_share"][0] == pytest.approx(0.55)


def test_dead_metric_loses_weight_under_staleness_cap():
    """Метрика отчиталась раз и умерла: без кэпа весит вечно, с кэпом уходит."""
    rows = [("taker_ls", 0, 0.95)]
    rows += [("global_ls_account", k * 5 * MIN, 0.50) for k in range(1, 25)]
    r = ratio_frame(rows)
    last_no_cap = long_share_series(r)["long_share"][-1]
    assert last_no_cap == pytest.approx((0.4 * 0.5 + 0.3 * 0.95) / 0.7, rel=1e-12)
    last_cap = long_share_series(r, max_age_s=900.0)["long_share"][-1]
    assert last_cap == pytest.approx(0.5, rel=1e-12)  # остался только global


def test_stale_series_gives_default_to_far_bars():
    """Бар дальше max_age_s от последней точки получает default, не мертвечину."""
    r = ratio_frame([("global_ls_account", 0, 0.85), ("top_ls_position", 0, 0.85)])
    bars = bar_frame([10 * MIN, 60 * MIN, 48 * 60 * MIN])
    no_cap = join_long_share(bars, r)["long_share"].to_numpy()
    assert np.allclose(no_cap, 0.85)  # текущее поведение: возраст не ограничен
    capped = join_long_share(bars, r, max_age_s=1800.0)["long_share"].to_numpy()
    assert capped[0] == pytest.approx(0.85)
    assert capped[1] == pytest.approx(0.5) and capped[2] == pytest.approx(0.5)


def test_dedup_is_permutation_invariant():
    """Ретраи поллера: без дедупа побеждает порядок строк, с дедупом — ts_recv."""
    base = [("global_ls_account", 10 * MIN, 0.20), ("global_ls_account", 10 * MIN, 0.80)]
    recv = [10 * MIN + 1_000, 10 * MIN + 5_000]  # второй принят позже
    rng = np.random.default_rng(3)
    plain, deduped = set(), set()
    for _ in range(int(20 * SCALE)):
        order = rng.permutation(2)
        rows = [base[i] for i in order]
        rec = [recv[i] for i in order]
        r = ratio_frame(rows, ts_recv=rec)
        plain.add(round(float(long_share_series(r)["long_share"][0]), 12))
        deduped.add(round(float(long_share_series(r, dedup=True)["long_share"][0]), 12))
    assert len(plain) == 2  # зависит от порядка строк — исходная проблема
    assert deduped == {0.80}  # победил принятый позже, независимо от порядка


def test_per_symbol_join_does_not_leak_across_symbols():
    """Бары BTC не должны получать долю ETH (join без by=symbol — футган)."""
    r = pl.concat([
        ratio_frame([("global_ls_account", 10 * MIN, 0.90)], symbol="BTCUSDT"),
        ratio_frame([("global_ls_account", 10 * MIN, 0.10)], symbol="ETHUSDT"),
    ])
    bars = bar_frame([15 * MIN], symbol="BTCUSDT")
    leaked = join_long_share(bars, r)["long_share"][0]
    assert leaked == pytest.approx(0.10)  # текущее поведение: победил ETH
    fixed = join_long_share(bars, r, per_symbol=True)["long_share"][0]
    assert fixed == pytest.approx(0.90)
    # односимвольный вход: per_symbol ничего не меняет
    r_btc = ratio_frame([("global_ls_account", 10 * MIN, 0.65)])
    a = join_long_share(bars, r_btc)["long_share"][0]
    b = join_long_share(bars, r_btc, per_symbol=True)["long_share"][0]
    assert a == pytest.approx(b)


def test_availability_ts_recv_hides_late_points():
    """Точка с меткой грида в прошлом, но принятая позже закрытия бара,
    не должна быть видна этому бару."""
    r = ratio_frame(
        [("global_ls_account", 10 * MIN, 0.80)], ts_recv=[14 * MIN]
    )
    bars = bar_frame([12 * MIN, 20 * MIN])
    by_event = join_long_share(bars, r)["long_share"].to_numpy()
    assert by_event[0] == pytest.approx(0.80)  # lookahead: точка ещё не пришла
    by_recv = join_long_share(bars, r, availability="ts_recv")["long_share"].to_numpy()
    assert by_recv[0] == pytest.approx(0.5)  # честно: бар её знать не мог
    assert by_recv[1] == pytest.approx(0.80)


def test_debias_recovers_sign_and_stays_causal():
    """Структурный сдвиг стоковых ратио съедает знак; де-биас его возвращает."""
    n = int(400 * SCALE)
    rng = np.random.default_rng(11)
    wobble = rng.normal(0.0, 0.04, n)
    rows = [("global_ls_account", k * 5 * MIN, float(np.clip(0.65 + wobble[k], 0.01, 0.99)))
            for k in range(n)]
    r = ratio_frame(rows)
    raw = long_share_series(r)["long_share"].to_numpy()
    deb = long_share_series(r, debias_window_s=6 * 3600.0)["long_share"].to_numpy()
    assert (raw > 0.5).mean() > 0.99  # знак сырого бленда — константа
    tail = slice(n // 4, None)
    signs_ok = ((deb[tail] > 0.5) == (wobble[tail] > 0)).mean()
    assert signs_ok > 0.8  # де-биас вернул знак колебания
    assert abs(float(np.mean(deb[tail])) - 0.5) < 0.02
    # каузальность: добавление будущих точек не меняет прошлые значения
    r_more = ratio_frame(rows + [("global_ls_account", (n + k) * 5 * MIN, 0.99)
                                 for k in range(20)])
    deb2 = long_share_series(r_more, debias_window_s=6 * 3600.0)["long_share"].to_numpy()
    assert np.allclose(deb, deb2[:n], rtol=0, atol=0)


def test_defaults_are_unchanged_on_clean_streams():
    """Все новые опции выключены по умолчанию: результат на чистом потоке
    совпадает с ручной формулой, а флаги-гигиены на нём — no-op."""
    rng = np.random.default_rng(5)
    rows = []
    for k in range(int(200 * SCALE)):
        for m in DEFAULT_BLEND:
            rows.append((m, k * 5 * MIN, float(rng.uniform(0.3, 0.8))))
    r = ratio_frame(rows)
    base = long_share_series(r)["long_share"].to_numpy()
    for kw in ({"dedup": True}, {"validate_range": True}, {"max_age_s": 3600.0},
               {"availability": "ts_recv"}):
        got = long_share_series(r, **kw)["long_share"].to_numpy()
        assert np.allclose(base, got, rtol=0, atol=0), kw
