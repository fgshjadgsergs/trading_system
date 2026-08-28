"""LakeBarFeed на реальном формате лейка рекордера.

Тестовый лейк пишется теми же writer'ами, что и record_live.py
(core.io.write_batch поверх records_to_frame), с теми же записями:
kline с промежуточными незакрытыми снимками (websocket шлёт бар много раз
до закрытия), open_interest реже баров и с лагом (REST-поллер), ratio трёх
метрик с битыми точками. Проверяемые контракты:

1. эквивалентность инкрементального опроса одному poll() по полному лейку;
2. каузальная придержка бара до прихода OI, значение ΔOI от неё не зависит;
3. незакрытый kline не выдаётся; после закрытия — ровно один раз;
4. битые ratio-точки отфильтрованы, бленд каузален и лежит в [0.1, 0.9];
5. рестарт фида поверх живого LiveMapState: дубли отбрасываются, состояние
   сходится с непрерывным прогоном;
6. дыра в OI в начале: бары без известного ΔOI пропускаются, не мусорятся;
7. пустой/частичный лейк не роняет poll();
8. регрессия: open_interest_usd=NaN (record_live.py зовёт parse_open_interest
   без цены) не отравляет ΔOI и не служит водяным знаком придержки.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from trading_system.core.io import write_batch
from trading_system.core.schema import (
    STREAM_OF_TYPE,
    Kline,
    OpenInterest,
    RatioPoint,
    records_to_frame,
)
from trading_system.platform.feed import LakeBarFeed
from trading_system.platform.state import Bar, LiveMapState

EXCHANGE = "binance_usdm"
SYMBOL = "BTCUSDT"
S = 1_000_000_000  # нс в секунде
MIN_NS = 60 * S
T0 = 1_767_571_200 * S  # 2026-01-05 00:00:00 UTC — база партиций date=/hour=
BAR_SPAN = MIN_NS - 1_000_000  # binance: close_time = open_time + 59_999 мс

LakeRecord = Kline | OpenInterest | RatioPoint


def bar_close(i: int) -> int:
    return T0 + i * MIN_NS + BAR_SPAN


def mk_kline(i: int, *, closed: bool = True, close_px: float | None = None) -> Kline:
    ts_open = T0 + i * MIN_NS
    o = 65_000.0 + 10.0 * i
    c = o + 5.0 if close_px is None else close_px
    return Kline(
        exchange=EXCHANGE,
        symbol=SYMBOL,
        ts_open=ts_open,
        ts_close=ts_open + BAR_SPAN,
        open=o,
        high=max(o, c) + 3.0,
        low=min(o, c) - 3.0,
        close=c,
        volume=12.5,
        quote_volume=12.5 * o,
        taker_buy_volume=6.0,
        taker_buy_quote_volume=6.0 * o,
        n_trades=100 + i,
        closed=closed,
    )


def mk_oi(ts_s: float, usd: float) -> OpenInterest:
    ts = T0 + int(ts_s * S)
    return OpenInterest(
        exchange=EXCHANGE,
        symbol=SYMBOL,
        ts_event=ts,
        ts_recv=ts + 200_000_000,  # лаг REST-ответа
        open_interest=usd / 65_000.0 if math.isfinite(usd) else 1_500.0,
        open_interest_usd=usd,
    )


def mk_ratio(ts_s: float, metric: str, ls: float) -> RatioPoint:
    ts = T0 + int(ts_s * S)
    short = 1.0 - ls
    return RatioPoint(
        exchange=EXCHANGE,
        symbol=SYMBOL,
        ts_event=ts,
        ts_recv=ts + 300_000_000,
        metric=metric,
        long_share=ls,
        short_share=short,
        ratio=ls / short if math.isfinite(ls) and short > 0 else 1.0,
    )


def write(lake: Path, recs: list[LakeRecord]) -> None:
    """Дозапись в лейк штатным writer'ом рекордера (append-only батчи)."""
    by_stream: dict[str, list[LakeRecord]] = {}
    for rec in recs:
        by_stream.setdefault(STREAM_OF_TYPE[type(rec)], []).append(rec)
    for stream, rows in by_stream.items():
        write_batch(lake, stream, records_to_frame(rows, stream))


def timeline() -> list[LakeRecord]:
    """Полный «живой» лейк в порядке поступления записей.

    Бары 0..9 закрыты, бар 10 — незакрытый хвост; промежуточные незакрытые
    снимки баров 3 и 7 (реальный kline-стрим); OI каждые ~130 с (реже баров,
    с лагом), первый замер на 130 с — позже закрытий баров 0 и 1 (дыра);
    ratio трёх метрик, три точки битые (вне [0,1] и NaN), одна точка позже
    закрытия последнего бара (проверка каузальности).
    """
    events: list[tuple[float, LakeRecord]] = []
    for i in range(10):
        events.append((i * 60 + 59.999, mk_kline(i)))
    events.append((200.0, mk_kline(3, closed=False, close_px=65_017.0)))
    events.append((440.0, mk_kline(7, closed=False, close_px=65_055.0)))
    events.append((620.0, mk_kline(10, closed=False, close_px=65_099.0)))
    for ts_s, usd in [(130, 100e6), (260, 108e6), (390, 103e6), (520, 111e6), (650, 118e6)]:
        events.append((float(ts_s), mk_oi(ts_s, usd)))
    for ts_s, metric, ls in [
        (41, "global_ls_account", 0.55),
        (43, "taker_ls", 1.5),  # битая: вне [0, 1]
        (47, "top_ls_position", 0.58),
        (227, "top_ls_position", float("nan")),  # битая: NaN
        (233, "taker_ls", 0.52),
        (341, "global_ls_account", 0.60),
        (347, "top_ls_position", 0.62),
        (449, "taker_ls", -0.2),  # битая: вне [0, 1]
        (533, "taker_ls", 0.48),
        (641, "global_ls_account", 0.99),  # после закрытия последнего бара
    ]:
        events.append((float(ts_s), mk_ratio(ts_s, metric, ls)))
    events.sort(key=lambda e: e[0])  # стабильная сортировка, тай-брейков нет
    return [rec for _, rec in events]


# Ожидание по timeline(): бары 0-1 старше первого OI-замера и пропускаются,
# выдаются бары 2..9.
EXPECTED_TS = [bar_close(i) for i in range(2, 10)]


def assert_bars_equal(got: list[Bar], ref: list[Bar]) -> None:
    assert [(b.ts_open, b.ts_close) for b in got] == [(b.ts_open, b.ts_close) for b in ref]
    for g, r in zip(got, ref, strict=True):
        assert (g.open, g.high, g.low, g.close) == (r.open, r.high, r.low, r.close)
        assert g.d_oi_usd == pytest.approx(r.d_oi_usd, abs=1e-6)
        if r.long_share is None:
            assert g.long_share is None
        else:
            assert g.long_share == pytest.approx(r.long_share, abs=1e-12)


# --------------------------------------------------------------------------- #
# 1. эквивалентность: случайные границы опроса == один poll по полному лейку
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [11, 23, 37])
def test_random_poll_boundaries_match_single_poll(tmp_path, seed):
    recs = timeline()
    ref_lake = tmp_path / "ref"
    write(ref_lake, recs)
    ref = LakeBarFeed(ref_lake, SYMBOL).poll()
    assert [b.ts_close for b in ref] == EXPECTED_TS

    rng = np.random.default_rng(seed)
    n_cuts = int(rng.integers(3, 7))
    cuts = sorted(rng.choice(np.arange(1, len(recs)), size=n_cuts, replace=False).tolist())
    lake = tmp_path / f"inc-{seed}"
    feed = LakeBarFeed(lake, SYMBOL)
    got: list[Bar] = []
    prev = 0
    for cut in [*cuts, len(recs)]:
        write(lake, recs[prev:cut])
        prev = cut
        got.extend(feed.poll())
        if rng.random() < 0.5:  # повторный опрос без новых данных не дублирует
            got.extend(feed.poll())
    assert_bars_equal(got, ref)


# --------------------------------------------------------------------------- #
# 2. каузальная придержка: бар ждёт OI, значение ΔOI от придержки не зависит
# --------------------------------------------------------------------------- #
def test_holdback_until_oi_arrives_value_unchanged(tmp_path):
    bars = [mk_kline(i) for i in range(5)]
    oi_early = [mk_oi(35, 100e6), mk_oi(95, 104e6), mk_oi(155, 101e6), mk_oi(215, 107e6)]
    oi_late = mk_oi(275, 110e6)

    lake = tmp_path / "hold"
    write(lake, bars + oi_early)
    feed = LakeBarFeed(lake, SYMBOL)
    first = feed.poll()
    # свежайший OI (215 с) старше закрытия бара 3 (239.999 с) — бары 3, 4 придержаны
    assert [b.ts_close for b in first] == [bar_close(i) for i in range(3)]
    assert feed.poll() == []  # без новых данных придержанное не выдаётся

    write(lake, [oi_late])
    second = feed.poll()  # OI доехал до 275 с: выходит ровно бар 3, бар 4 ещё ждёт
    assert [b.ts_close for b in second] == [bar_close(3)]
    assert second[0].d_oi_usd == pytest.approx(107e6 - 101e6)

    # тот же лейк целиком, без придержки: ΔOI бара 3 совпадает побайтно
    ref_lake = tmp_path / "hold-ref"
    write(ref_lake, bars + oi_early + [oi_late])
    ref = {b.ts_close: b.d_oi_usd for b in LakeBarFeed(ref_lake, SYMBOL).poll()}
    assert ref[bar_close(3)] == second[0].d_oi_usd


# --------------------------------------------------------------------------- #
# 3. незакрытый kline не выдаётся; после закрытия — ровно один раз
# --------------------------------------------------------------------------- #
def test_unclosed_kline_emitted_exactly_once_after_close(tmp_path):
    lake = tmp_path / "unclosed"
    oi_rows = [mk_oi(35, 100e6), mk_oi(95, 104e6), mk_oi(155, 101e6),
               mk_oi(215, 107e6), mk_oi(275, 110e6)]
    unclosed = mk_kline(3, closed=False, close_px=64_990.0)
    write(lake, [mk_kline(i) for i in range(3)] + [unclosed] + oi_rows)

    feed = LakeBarFeed(lake, SYMBOL)
    got = feed.poll()
    # OI покрывает закрытие бара 3 — не выдан он только потому, что не закрыт
    assert [b.ts_close for b in got] == [bar_close(i) for i in range(3)]

    # рекордер дописал закрытую версию (дважды — реплей после реконнекта)
    write(lake, [mk_kline(3), mk_kline(3)])
    got2 = feed.poll()
    assert [b.ts_close for b in got2] == [bar_close(3)]
    assert got2[0].close == mk_kline(3).close  # закрытая версия, не промежуточная
    assert feed.poll() == []  # дубли закрытой строки не дают второго бара


# --------------------------------------------------------------------------- #
# 4. битые ratio-точки отфильтрованы, бленд каузален и лежит в [0.1, 0.9]
# --------------------------------------------------------------------------- #
def test_broken_ratio_points_filtered_blend_causal(tmp_path):
    lake = tmp_path / "ratio"
    write(lake, timeline())
    bars = {b.ts_close: b for b in LakeBarFeed(lake, SYMBOL).poll()}
    assert sorted(bars) == EXPECTED_TS
    for b in bars.values():
        assert b.long_share is not None
        assert 0.1 <= b.long_share <= 0.9

    # у taker_ls к 179.999 с только битая точка (1.5) — метрика выпадает,
    # веса 0.4/0.3 перенормируются
    assert bars[bar_close(2)].long_share == pytest.approx((0.4 * 0.55 + 0.3 * 0.58) / 0.7)
    # NaN top_ls_position@227 отфильтрован: asof берёт последнюю валидную (0.58)
    assert bars[bar_close(3)].long_share == pytest.approx(
        0.4 * 0.55 + 0.3 * 0.58 + 0.3 * 0.52)
    # -0.2 taker_ls@449 отфильтрован; точка 0.99@641 (после закрытия бара 9)
    # лежит в лейке во время poll, но не влияет — каузальность asof
    assert bars[bar_close(9)].long_share == pytest.approx(
        0.4 * 0.60 + 0.3 * 0.62 + 0.3 * 0.48)


# --------------------------------------------------------------------------- #
# 5. рестарт: свежий фид + уже видевший бары LiveMapState == непрерывный прогон
# --------------------------------------------------------------------------- #
def test_restart_dedup_by_state_converges(tmp_path):
    recs = timeline()
    split = 13  # после записи k0..k4 и OI@130,260: фаза 1 выдаёт бары 2 и 3
    n_phase1 = 2  # бар 4 (close 299.999 c) придержан: OI дошёл лишь до 260 c

    # непрерывный прогон: один фид через обе фазы
    lake_a = tmp_path / "cont"
    state_a = LiveMapState(SYMBOL, 50.0)
    feed_a = LakeBarFeed(lake_a, SYMBOL)
    write(lake_a, recs[:split])
    phase1 = feed_a.poll()
    assert len(phase1) == n_phase1
    for b in phase1:
        assert state_a.apply_bar(b)
    write(lake_a, recs[split:])
    for b in feed_a.poll():
        assert state_a.apply_bar(b)

    # рестарт: после фазы 1 фид пересоздан и перечитывает лейк с нуля
    lake_b = tmp_path / "restart"
    state_b = LiveMapState(SYMBOL, 50.0)
    feed_b = LakeBarFeed(lake_b, SYMBOL)
    write(lake_b, recs[:split])
    for b in feed_b.poll():
        assert state_b.apply_bar(b)
    write(lake_b, recs[split:])
    replay = LakeBarFeed(lake_b, SYMBOL).poll()  # свежий фид: с самого начала
    applied = [state_b.apply_bar(b) for b in replay]
    assert applied == [False] * n_phase1 + [True] * (len(EXPECTED_TS) - n_phase1)
    assert state_b.meta()["dropped_old_bars"] == n_phase1

    assert state_b.snapshot()["frames"] == state_a.snapshot()["frames"]
    assert state_b.snapshot()["last_ts"] == state_a.snapshot()["last_ts"]


# --------------------------------------------------------------------------- #
# 6. дыра в OI в начале: бары без известного ΔOI пропускаются, не мусорятся
# --------------------------------------------------------------------------- #
def test_oi_gap_at_start_skips_bars_without_delta(tmp_path):
    lake = tmp_path / "gap"
    write(lake, [mk_kline(i) for i in range(4)])
    feed = LakeBarFeed(lake, SYMBOL)
    assert feed.poll() == []  # OI ещё нет вовсе — всё придержано

    write(lake, [mk_oi(130, 100e6), mk_oi(260, 108e6)])
    got = feed.poll()
    # бары 0, 1 закрылись до первого замера: ΔOI неизвестен — их нет и не будет
    assert [b.ts_close for b in got] == [bar_close(2), bar_close(3)]
    assert got[0].d_oi_usd == 0.0  # первая известная точка — база, не мусор
    assert all(math.isfinite(b.d_oi_usd) for b in got)

    write(lake, [mk_oi(400, 111e6)])
    assert feed.poll() == []  # пропущенные бары не возвращаются задним числом


# --------------------------------------------------------------------------- #
# 7. пустой/частичный лейк не роняет poll()
# --------------------------------------------------------------------------- #
def test_empty_and_partial_lakes(tmp_path):
    assert LakeBarFeed(tmp_path / "missing", SYMBOL).poll() == []  # каталога нет

    empty = tmp_path / "empty"
    empty.mkdir()
    assert LakeBarFeed(empty, SYMBOL).poll() == []

    only_klines = tmp_path / "only-klines"  # нет open_interest
    write(only_klines, [mk_kline(i) for i in range(3)])
    assert LakeBarFeed(only_klines, SYMBOL).poll() == []

    no_ratio = tmp_path / "no-ratio"  # нет ratio: бары идут, long_share=None
    write(no_ratio, [mk_kline(i) for i in range(3)]
          + [mk_oi(35, 100e6), mk_oi(95, 104e6), mk_oi(155, 101e6), mk_oi(215, 107e6)])
    got = LakeBarFeed(no_ratio, SYMBOL).poll()
    assert [b.ts_close for b in got] == [bar_close(i) for i in range(3)]
    assert all(b.long_share is None for b in got)


# --------------------------------------------------------------------------- #
# 8. регрессия: open_interest_usd=NaN из реального рекордера
# --------------------------------------------------------------------------- #
def _oi(ts_s: float, coins: float, usd: float) -> OpenInterest:
    ts = T0 + int(ts_s * S)
    return OpenInterest(EXCHANGE, SYMBOL, ts, ts + 200_000_000, coins, usd)


def test_nan_open_interest_usd_ignored(tmp_path):
    """Рекордер мог писать open_interest_usd=NaN (parse без цены), но замер в
    МОНЕТАХ там настоящий. Контракт: NaN-USD не отравляет ΔOI и не черствит
    его молча — если USD-замер на границе старее замера монет, ΔOI бара
    считается фолбэком Δмонеты × close бара; бары выпускаются по водяному
    знаку МОНЕТ (NaN-строка тоже двигает придержку)."""
    nan = float("nan")
    # монеты — единый истинный ряд; usd = монеты*65000 там, где цена была
    recs = [
        _oi(35, 1_000.0, 65e6),
        _oi(95, 1_010.0, nan),      # дыра в USD
        _oi(155, 1_025.0, 66.625e6),
        _oi(215, 1_040.0, nan),     # дыра в USD на хвосте
    ]
    lake = tmp_path / "lake"
    write(lake, [mk_kline(i) for i in range(4)] + recs)
    bars = LakeBarFeed(lake, SYMBOL).poll()
    # водяной знак монет = 215с -> выпущены бары 0..2 (close 59.999/119.999/179.999)
    assert [b.ts_close for b in bars] == [bar_close(i) for i in range(3)]
    assert all(math.isfinite(b.d_oi_usd) for b in bars)
    assert bars[0].d_oi_usd == 0.0                                   # базовая точка
    assert bars[1].d_oi_usd == pytest.approx((1_010 - 1_000) * bars[1].close)  # фолбэк
    assert bars[2].d_oi_usd == pytest.approx((1_025 - 1_010) * bars[2].close)  # USD чёрствый после дыры

    # USD нет вовсе: весь ΔOI из монет, мусора нет
    all_nan = tmp_path / "all-nan"
    write(all_nan, [mk_kline(i) for i in range(3)]
          + [_oi(35, 1_000.0, nan), _oi(95, 1_004.0, nan), _oi(175, 1_010.0, nan)])
    got = LakeBarFeed(all_nan, SYMBOL).poll()
    assert [b.ts_close for b in got] == [bar_close(0), bar_close(1)]
    assert got[0].d_oi_usd == 0.0
    assert got[1].d_oi_usd == pytest.approx((1_004 - 1_000) * got[1].close)


def test_usd_series_used_when_fresh(tmp_path):
    """Целый USD-ряд (замер к каждой границе свежий) идёт диффом по USD —
    фолбэк не подменяет его пересчётом монет."""
    lake = tmp_path / "lake"
    recs = [_oi(35, 1_000.0, 65.0e6), _oi(95, 1_010.0, 65.8e6),
            _oi(155, 1_025.0, 66.6e6)]
    write(lake, [mk_kline(i) for i in range(3)] + recs)
    bars = LakeBarFeed(lake, SYMBOL).poll()
    assert [b.ts_close for b in bars] == [bar_close(0), bar_close(1)]
    assert bars[0].d_oi_usd == 0.0
    assert bars[1].d_oi_usd == pytest.approx(65.8e6 - 65.0e6)
