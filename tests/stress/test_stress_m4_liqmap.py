"""Стресс-тесты M4: карта ликвидаций под вырожденными и масштабными нагрузками.

Сценарии: вырожденный ATR (NaN/0/отрицательный), взрыв ценового диапазона,
инвариант массы на 1e5 операций, рост памяти HeatHistory на 10k баров,
экстремумы decay, полуоткрытость бакетов на 10k случайных границ, брекеты
(пустые/однотировые/за последним тиром, инвариант стороны на 10k параметров),
клип ратио-серий и строгая каузальность sides, адверсариальная сверка
быстрого билдера real_data с точным LiqMap-реплеем.

Масштаб управляется env STRESS_SCALE (по умолчанию 1.0).
"""

from __future__ import annotations

import math
import os
import time
import tracemalloc

import numpy as np
import polars as pl
import pytest

from trading_system.calibration.real_data import (
    BarArrays,
    bars_to_arrays,
    bucket_grid,
    make_real_heat_builder,
)
from trading_system.collectors.brackets import load_brackets, parse_leverage_brackets
from trading_system.core.liquidation import (
    DEFAULT_BRACKETS,
    BinanceUsdmLiquidation,
    MarginBracket,
    bracket_for,
    liq_price,
)
from trading_system.core.schema import POLARS_SCHEMAS, Side
from trading_system.liqmap.buckets import PriceBuckets, rebucket
from trading_system.liqmap.history import HeatHistory
from trading_system.liqmap.map import LiqMap, StaticWeights
from trading_system.liqmap.sides import join_long_share, long_share_series

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
SEED = 42
NAN = float("nan")


def _n(base: int) -> int:
    return max(1, int(base * SCALE))


def make_map(bucket_size: float = 10.0, half_life_s: float = 3_600.0) -> LiqMap:
    return LiqMap(
        leverage_grid=[5, 10, 25, 50, 100],
        buckets=PriceBuckets(bucket_size),
        weight_fn=StaticWeights(np.array([1.0, 2.0, 3.0, 2.0, 1.0])),
        decay_half_life_s=half_life_s,
    )


# ---------------------------------------------------------------------------
# 1) вырожденный ATR -> ширина бакета
# ---------------------------------------------------------------------------


def test_degenerate_atr_bucket_contract():
    """ATR<=0/NaN/inf и мусорный bucket_size обязаны падать ValueError сразу,
    а не давать NaN-сетку/инвертированные интервалы много позже."""
    for bad_atr in (0.0, -5.0, NAN, float("inf")):
        with pytest.raises(ValueError):
            PriceBuckets.from_atr(bad_atr)
    with pytest.raises(ValueError):
        PriceBuckets.from_atr(10.0, fraction=0.0)
    with pytest.raises(ValueError):
        PriceBuckets.from_atr(10.0, fraction=-1.0)
    # прямое конструирование с мусором тоже отбито (guard в __post_init__)
    for bad_size in (0.0, -2.0, NAN, float("inf")):
        with pytest.raises(ValueError):
            PriceBuckets(bad_size)
    # ATR -> 0: клип min_size, индексы конечны и согласованы
    b = PriceBuckets.from_atr(1e-300, fraction=0.1, min_size=1e-9)
    assert b.bucket_size == 1e-9
    idx = b.index(50_000.0)
    assert isinstance(idx, int)
    assert b.lo(idx) <= 50_000.0 < b.hi(idx) + 1e-3  # float-погрешность на 5e13 индексе
    assert b.lo(idx) < b.hi(idx)


def test_liqmap_rejects_nan_inputs():
    """NaN ΔOI/цена/веса/decay dt не должны тихо отравлять карту."""
    lm = make_map()
    lm.allocate(1_000.0, 50_000.0)
    with pytest.raises(ValueError):
        lm.allocate(NAN, 50_000.0)
    with pytest.raises(ValueError):
        lm.allocate(float("inf"), 50_000.0)
    with pytest.raises(ValueError):
        lm.allocate(100.0, NAN)
    with pytest.raises(ValueError):
        lm.allocate(100.0, -5.0)
    with pytest.raises(ValueError):
        lm.decay(NAN)
    with pytest.raises(ValueError):
        lm.consume(NAN, 50_000.0)
    with pytest.raises(ValueError):
        lm.consume(50_000.0, NAN)
    with pytest.raises(ValueError):
        StaticWeights(np.array([1.0, NAN]))
    bad_w = LiqMap([5.0], PriceBuckets(10.0), lambda ctx: np.array([NAN]))
    with pytest.raises(ValueError):
        bad_w.allocate(100.0, 50_000.0)
    # после всех отбитых вызовов карта не отравлена
    assert math.isfinite(lm.total_heat())
    assert lm.total_heat() == pytest.approx(1_000.0)
    assert lm.mass_balance_error() < 1e-9


# ---------------------------------------------------------------------------
# 2) взрыв диапазона: 100 -> 100000 при мелком бакете
# ---------------------------------------------------------------------------


def test_range_explosion_storage_is_sparse():
    """Хранилище heat — sparse dict: скачок цены 100 -> 100000 при бакете 0.01
    занимает O(грид), а не O(диапазон)."""
    lm = make_map(bucket_size=0.01)
    lm.allocate(1e6, 100.0)
    lm.allocate(1e6, 100_000.0)
    occupied = sum(len(h) for h in lm.heat.values())
    assert occupied <= 4 * len(lm.leverage_grid)  # 2 цены x 2 стороны x грид максимум
    assert lm.mass_balance_error() < 1e-6
    idxs = sorted(set(lm.heat[Side.BUY]) | set(lm.heat[Side.SELL]))
    span = idxs[-1] - idxs[0] + 1
    assert span > 10_000_000  # диапазон индексов и правда взорван


@pytest.mark.xfail(
    reason="дизайн: snapshot()/HeatHistory.matrix() строят плотные массивы по всему "
    "занятому диапазону индексов (при бакете 0.01 и скачке 100->100000 это ~11.9M "
    "ячеек = ~286MB на три массива при 20 занятых бакетах); нужен sparse-вид "
    "или клип диапазона — менять формат снапшота без владельца API нельзя",
    strict=True,
)
def test_snapshot_memory_proportional_to_occupied():
    lm = make_map(bucket_size=1.0)
    lm.allocate(1e6, 100.0)
    lm.allocate(1e6, 100_000.0)
    occupied = sum(len(h) for h in lm.heat.values())
    snap = lm.snapshot()
    dense_bytes = snap["prices"].nbytes + snap["long"].nbytes + snap["short"].nbytes
    # желаемое свойство: память вида пропорциональна занятым бакетам
    assert dense_bytes <= occupied * 8 * 3 * 10


# ---------------------------------------------------------------------------
# 3) инвариант массы на 1e5 операций
# ---------------------------------------------------------------------------


def test_mass_invariant_100k_random_ops():
    """|ΣH - (C - X - R - D)| остаётся на уровне float-шума после 1e5 операций."""
    n = _n(100_000)
    rng = np.random.default_rng(SEED)
    lm = make_map(bucket_size=50.0)
    kinds = rng.random(n)
    amounts = rng.uniform(0.0, 1e6, n)
    steps = rng.normal(0.0, 0.002, n)
    dts = rng.uniform(0.0, 900.0, n)
    price = 50_000.0
    t0 = time.perf_counter()
    for i in range(n):
        price = min(max(price * float(np.exp(steps[i])), 45_000.0), 55_000.0)
        k = kinds[i]
        if k < 0.45:
            lm.allocate(amounts[i], price)
        elif k < 0.65:
            lm.allocate(-amounts[i], price)
        elif k < 0.85:
            lm.consume(price * 0.995, price * 1.005)
        else:
            lm.decay(dts[i])
    elapsed = time.perf_counter() - t0
    ops_per_s = n / max(elapsed, 1e-9)
    assert ops_per_s > 1_000  # деградация на порядок от ~15k ops/s — регресс
    assert all(h >= 0.0 for sh in lm.heat.values() for h in sh.values())
    scale = max(1.0, lm.contributed)
    assert lm.mass_balance_error() / scale < 1e-9
    total = lm.total_heat()
    assert math.isfinite(total) and total >= 0.0


# ---------------------------------------------------------------------------
# 4) HeatHistory: рост памяти на 10k баров
# ---------------------------------------------------------------------------


def test_heathistory_memory_growth_bounded_10k_bars():
    """Память истории растёт линейно по барам; байт/бар ограничен сверху."""
    n = _n(10_000)
    rng = np.random.default_rng(SEED)
    lm = make_map(bucket_size=50.0)
    hist = HeatHistory(lm)
    steps = rng.normal(0.0, 0.002, n)
    d_ois = rng.uniform(-2e5, 6e5, n)
    price = 50_000.0
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    for i in range(n):
        price = min(max(price * float(np.exp(steps[i])), 45_000.0), 55_000.0)
        lm.step(price * 0.999, price * 1.001, price, float(d_ois[i]), dt_s=300.0)
        hist.record(i)
    cur, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    bytes_per_bar = (cur - base) / n
    assert len(hist) == n
    # ~23KB/бар при ~360 занятых бакетах на кадр; 100KB/бар — явный регресс
    assert bytes_per_bar < 100_000
    # каузальные представления по-прежнему согласованы после 10k кадров
    i_last = n - 1
    assert hist.total_at(i_last) == pytest.approx(lm.total_heat(), rel=1e-9)
    lo, hi, heat = hist.zones_at(i_last)
    assert len(lo) == len(hi) == len(heat)
    assert sum(heat) == pytest.approx(hist.total_at(i_last), rel=1e-9)
    pools = hist.pools_at(i_last, k=8)
    assert all(h >= 0 for _, h in pools)


def test_heathistory_rejects_grid_change_mid_history():
    lm = make_map(bucket_size=10.0)
    hist = HeatHistory(lm)
    lm.allocate(1_000.0, 50_000.0)
    hist.record(0)
    lm.rebucket_to(PriceBuckets(25.0))
    with pytest.raises(ValueError):
        hist.record(1)


# ---------------------------------------------------------------------------
# 5) decay: dt=0, dt=1e9, экстремумы T1/2
# ---------------------------------------------------------------------------


def test_decay_extremes():
    lm = make_map()
    lm.allocate(100_000.0, 50_000.0)
    before = {s: dict(h) for s, h in lm.heat.items()}
    assert lm.decay(0.0) == 0.0
    assert {s: dict(h) for s, h in lm.heat.items()} == before  # dt=0: H не меняется
    lost = lm.decay(1e9)  # ~278k полупериодов: H -> 0 без NaN
    assert math.isfinite(lost)
    assert lost == pytest.approx(100_000.0)
    assert lm.total_heat() == 0.0
    assert lm.mass_balance_error() < 1e-6
    # dt=inf допустим и означает полный распад
    lm2 = make_map()
    lm2.allocate(1_000.0, 50_000.0)
    lm2.decay(float("inf"))
    assert lm2.total_heat() == 0.0
    # T1/2 = inf — распада нет вообще
    lm3 = make_map(half_life_s=float("inf"))
    lm3.allocate(1_000.0, 50_000.0)
    assert lm3.decay(1e12) == 0.0
    assert lm3.total_heat() == pytest.approx(1_000.0)
    # T1/2 <= 0 / NaN — отбой на конструкторе, а не ZeroDivisionError в decay
    for bad_hl in (0.0, -1.0, NAN):
        with pytest.raises(ValueError):
            make_map(half_life_s=bad_hl)
    # крошечный T1/2 конечен и не даёт NaN
    lm4 = make_map(half_life_s=1e-12)
    lm4.allocate(1_000.0, 50_000.0)
    assert math.isfinite(lm4.decay(1.0))
    assert lm4.total_heat() == 0.0


# ---------------------------------------------------------------------------
# 6) consume: битый вход и полуоткрытость на 10k случайных границ
# ---------------------------------------------------------------------------


def test_consume_broken_bar_contract():
    lm = make_map()
    lm.allocate(1_000.0, 50_000.0)
    with pytest.raises(ValueError):
        lm.consume(50_100.0, 50_000.0)  # low > high
    with pytest.raises(ValueError):
        lm.step(50_100.0, 50_000.0, 50_050.0, 100.0, dt_s=60.0)
    assert lm.total_heat() == pytest.approx(1_000.0)  # неуспешный вызов ничего не тронул


def test_consume_half_open_boundary_10k_random():
    """Бакет [lo, hi), кончающийся ровно на path_lo, не задет — всегда."""
    n = _n(10_000)
    rng = np.random.default_rng(SEED)
    sizes = rng.uniform(1e-3, 100.0, n)
    idxs = rng.integers(-1_000, 1_000, n)
    lens = rng.uniform(0.1, 5.0, n)
    for i in range(n):
        size = float(sizes[i])
        idx = int(idxs[i])
        lm = LiqMap([10.0], PriceBuckets(size), StaticWeights(np.array([1.0])))
        lm.heat[Side.BUY][idx] = 100.0
        lm.contributed = 100.0
        boundary = lm.buckets.hi(idx)  # путь начинается ровно на границе бакета
        taken = lm.consume(boundary, boundary + size * float(lens[i]))
        assert taken == 0.0
        assert lm.total_heat() == 100.0
        # путь, реально входящий в бакет, забирает его целиком
        assert lm.consume(lm.buckets.lo(idx), boundary) == pytest.approx(100.0)
        assert lm.total_heat() == 0.0


def test_rebucket_conserves_mass_extreme_ratio():
    lm = make_map(bucket_size=0.5)
    lm.allocate(777_777.0, 50_000.0)
    lm.rebucket_to(PriceBuckets(5_000.0))  # укрупнение x10000
    assert lm.total_heat() == pytest.approx(777_777.0)
    assert lm.mass_balance_error() < 1e-6
    assert rebucket({}, PriceBuckets(1.0), PriceBuckets(3.0)) == {}


# ---------------------------------------------------------------------------
# 7) liquidation.py + brackets.py
# ---------------------------------------------------------------------------


def test_liq_price_side_invariant_10k_flat():
    """При mmr < 1/L цена ликвидации лонга всегда < входа, шорта — > входа."""
    n = _n(10_000)
    rng = np.random.default_rng(SEED)
    entries = np.exp(rng.uniform(np.log(0.01), np.log(200_000.0), n))
    levs = rng.uniform(1.0, 125.0, n)
    for i in range(n):
        entry, lev = float(entries[i]), float(levs[i])
        mmr = float(rng.uniform(0.0, 0.999 / lev))
        lp_long = liq_price(entry, lev, Side.BUY, mmr)
        lp_short = liq_price(entry, lev, Side.SELL, mmr)
        assert 0.0 <= lp_long < entry
        assert lp_short > entry
        assert math.isfinite(lp_short)


def test_bracket_engine_selfconsistent_10k_and_bounded_time():
    """Самосогласованный подбор тира: конечный, детерминированный, укладывается
    в бюджет (нет зацикливания), сторона нарушается только там, где применённый
    тировый mmr >= 1/L (зона мгновенной ликвидации)."""
    n = _n(10_000)
    rng = np.random.default_rng(SEED)
    f = BinanceUsdmLiquidation(brackets={"BTCUSDT": DEFAULT_BRACKETS})
    entries = np.exp(rng.uniform(np.log(0.01), np.log(200_000.0), n))
    levs = rng.uniform(1.0, 125.0, n)
    qtys = np.exp(rng.uniform(np.log(1e-6), np.log(1e5), n))
    t0 = time.perf_counter()
    for i in range(n):
        entry, lev, qty = float(entries[i]), float(levs[i]), float(qtys[i])
        for side in (Side.BUY, Side.SELL):
            lp = f.liq_price(entry, lev, side, symbol="BTCUSDT", qty=qty)
            assert math.isfinite(lp) and lp >= 0.0
            mmr_cap = max(
                bracket_for(lp * qty, DEFAULT_BRACKETS).mmr,
                bracket_for(entry * qty, DEFAULT_BRACKETS).mmr,
            )
            if side is Side.BUY:
                assert lp < entry or mmr_cap >= 1.0 / lev
            else:
                assert lp > entry or mmr_cap >= 1.0 / lev
        assert (time.perf_counter() - t0) < 60.0  # таймаут-guard от зацикливания


def test_bracket_notional_beyond_last_tier_and_boundaries():
    # далеко за последним конечным капом: последний тир (inf) применяется
    b = bracket_for(1e12, DEFAULT_BRACKETS)
    assert b is DEFAULT_BRACKETS[-1]
    f = BinanceUsdmLiquidation(brackets={"BTCUSDT": DEFAULT_BRACKETS})
    # 50B USD нотионала на плече 2 (1/L=0.5 > mmr последнего тира 0.25)
    lp = f.liq_price(50_000.0, 2, Side.SELL, symbol="BTCUSDT", qty=1e6)
    assert math.isfinite(lp) and lp > 50_000.0
    # ровно на границах тиров: нотионал == cap принадлежит нижнему тиру
    for cap_bracket in DEFAULT_BRACKETS[:-1]:
        assert bracket_for(cap_bracket.max_notional_usd, DEFAULT_BRACKETS) is cap_bracket
        eps = cap_bracket.max_notional_usd * 1e-12
        above = bracket_for(cap_bracket.max_notional_usd + max(eps, 1e-6), DEFAULT_BRACKETS)
        assert above.mmr >= cap_bracket.mmr
    # L=1 лонг клампится в 0, L=125 в первом тире совпадает с плоской формулой
    assert f.liq_price(100.0, 1, Side.BUY, symbol="BTCUSDT", qty=1.0) == 0.0
    lp125 = f.liq_price(100.0, 125, Side.BUY, symbol="BTCUSDT", qty=1.0)
    assert lp125 == pytest.approx(liq_price(100.0, 125, Side.BUY, 0.004), rel=1e-12)


@pytest.mark.xfail(
    reason="дизайн: таблицы брекетов не несут initialLeverage-капы биржи, поэтому "
    "гигантский слайс на высоком плече попадает в тир с mmr >= 1/L (зона мгновенной "
    "ликвидации) и его lp оказывается по НЕПРАВИЛЬНУЮ сторону от входа — шорт-тепло "
    "ложится ниже цены. На реальной бирже такая позиция неоткрываема; корректный фикс "
    "требует initialLeverage в таблицах, а не правки формулы",
    strict=True,
)
def test_bracket_map_side_invariant_for_giant_slices():
    from trading_system.collectors.brackets import bracket_liq_price_fn

    fn = bracket_liq_price_fn({"BTCUSDT": DEFAULT_BRACKETS}, "BTCUSDT")
    lm = LiqMap([100.0], PriceBuckets(10.0), StaticWeights(np.array([1.0])), liq_price_fn=fn)
    lm.allocate(1e9, 50_000.0)  # слайс ~500M USD -> тир mmr=0.25 >= 1/100
    snap = lm.snapshot()
    # желаемое свойство карты: шорт-тепло строго выше входа, лонг — строго ниже
    assert snap["prices"][snap["short"] > 0].min() > 50_000.0
    assert snap["prices"][snap["long"] > 0].max() < 50_000.0


def test_empty_and_single_tier_brackets_contract(tmp_path):
    with pytest.raises(ValueError):
        bracket_for(100.0, ())
    f = BinanceUsdmLiquidation(brackets={"X": ()})
    with pytest.raises(ValueError):
        f.liq_price(100.0, 10, Side.BUY, symbol="X", qty=1.0)
    # однотировая таблица == плоская формула с её mmr/cum
    one = (MarginBracket(float("inf"), 0.01, 25.0),)
    f1 = BinanceUsdmLiquidation(brackets={"Y": one})
    got = f1.liq_price(1_000.0, 10, Side.BUY, symbol="Y", qty=2.0)
    assert got == pytest.approx(liq_price(1_000.0, 10, Side.BUY, 0.01, cum=25.0, qty=2.0))
    # parse и load оба выбрасывают пустые таблицы (не отдают их дальше)
    payload = [
        {"symbol": "EMPTY", "brackets": []},
        {
            "symbol": "OK",
            "brackets": [{"notionalCap": 1e6, "maintMarginRatio": 0.01, "cum": 0.0}],
        },
    ]
    assert set(parse_leverage_brackets(payload)) == {"OK"}
    cache = tmp_path / "brackets.json"
    cache.write_text(
        '{"EMPTY": [], "OK": [{"notionalCap": 1e6, "maintMarginRatio": 0.01, "cum": 0.0}]}',
        encoding="utf-8",
    )
    assert set(load_brackets(cache)) == {"OK"}


def test_liq_price_rejects_nan_garbage():
    """NaN-вход не должен тихо давать lp=0 (кламп max(0, NaN))."""
    for bad in (NAN, float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError):
            liq_price(bad, 10, Side.BUY, 0.005)
        with pytest.raises(ValueError):
            liq_price(100.0, bad, Side.BUY, 0.005)
        with pytest.raises(ValueError):
            liq_price(100.0, 10, Side.BUY, 0.005, qty=bad)
    with pytest.raises(ValueError):
        liq_price(100.0, 10, Side.BUY, NAN)
    with pytest.raises(ValueError):
        liq_price(100.0, 10, Side.BUY, 0.005, cum=NAN)
    with pytest.raises(ValueError):
        liq_price(100.0, 10, Side.BUY, 0.005, cum=-1.0)


# ---------------------------------------------------------------------------
# 8) sides.py: экстремальные ратио, пустые, каузальность на границе
# ---------------------------------------------------------------------------

T0 = 1_755_600_000_000_000_000
MIN_NS = 60_000_000_000


def _ratio_rows(points: list[tuple[int, str, float]]) -> pl.DataFrame:
    rows = [
        {
            "exchange": "binance_usdm",
            "symbol": "BTCUSDT",
            "ts_event": ts,
            "ts_recv": ts,
            "metric": metric,
            "long_share": share,
            "short_share": 1.0 - share,
            "ratio": 999.0 if share >= 1.0 else share / (1.0 - share),
        }
        for ts, metric, share in points
    ]
    return pl.DataFrame(rows, schema=POLARS_SCHEMAS["ratio"])


def _bars(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "symbol": ["BTCUSDT"] * n,
            "ts_open": [T0 + i * MIN_NS for i in range(n)],
            "ts_close": [T0 + (i + 1) * MIN_NS for i in range(n)],
        }
    )


def test_sides_extreme_ratios_always_clipped():
    n = _n(2_000)
    rng = np.random.default_rng(SEED)
    metrics = ["global_ls_account", "top_ls_position", "taker_ls"]
    # экстремумы: все лонги (1.0), все шорты (0.0) и случайный мусор около краёв
    shares = rng.choice([0.0, 1.0, 0.001, 0.999], size=n)
    pts = [
        (T0 + int(i * MIN_NS / 4), metrics[i % 3], float(shares[i])) for i in range(n)
    ]
    bars = _bars(max(2, n // 4))
    joined = join_long_share(bars, _ratio_rows(pts), clip=(0.1, 0.9))
    ls = joined["long_share"].to_numpy()
    assert np.all(ls >= 0.1) and np.all(ls <= 0.9)  # клип держится всегда
    assert not np.isnan(ls).any()
    # чисто все-лонги -> ровно верхний клип
    all_long = join_long_share(_bars(4), _ratio_rows([(T0 - 1, "taker_ls", 1.0)]))
    assert (all_long["long_share"] == 0.9).all()
    all_short = join_long_share(_bars(4), _ratio_rows([(T0 - 1, "taker_ls", 0.0)]))
    assert (all_short["long_share"] == 0.1).all()


def test_sides_empty_unknown_and_future_ratios():
    bars = _bars(5)
    # пустой поток -> default всюду
    empty = _ratio_rows([])
    assert (join_long_share(bars, empty)["long_share"] == 0.5).all()
    assert long_share_series(empty).is_empty()
    # только неизвестные метрики -> как пустой
    unknown = _ratio_rows([(T0, "mystery_metric", 0.99)])
    assert (join_long_share(bars, unknown)["long_share"] == 0.5).all()
    # все точки позже последнего бара -> default всюду (ничего из будущего)
    future = _ratio_rows([(T0 + 100 * MIN_NS, "taker_ls", 0.9)])
    assert (join_long_share(bars, future)["long_share"] == 0.5).all()


def test_sides_strict_causality_at_close_boundary():
    """Точка ровно в ts_close бара видна только следующему бару."""
    bars = _bars(4)
    close_1 = int(bars["ts_close"][1])
    at_close = join_long_share(bars, _ratio_rows([(close_1, "taker_ls", 0.8)]))
    assert at_close["long_share"].to_list() == pytest.approx([0.5, 0.5, 0.8, 0.8])
    just_before = join_long_share(bars, _ratio_rows([(close_1 - 1, "taker_ls", 0.8)]))
    assert just_before["long_share"].to_list() == pytest.approx([0.5, 0.8, 0.8, 0.8])


# ---------------------------------------------------------------------------
# 9) real_data.py: вырожденные бары и адверсариальная сверка с LiqMap
# ---------------------------------------------------------------------------


def _adversarial_bars(n: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    steps = rng.normal(0.0, 0.05, n)  # 5%/бар — экстремальная волатильность
    steps[rng.random(n) < 0.05] *= 4.0  # шоковые бары до ~20%
    close = 50_000.0 * np.exp(np.cumsum(steps))
    opn = np.concatenate([[close[0]], close[:-1]])
    low = np.minimum(opn, close) * (1.0 - np.abs(rng.normal(0.0, 0.01, n)))
    high = np.maximum(opn, close) * (1.0 + np.abs(rng.normal(0.0, 0.01, n)))
    flat = rng.random(n) < 0.10  # 10% баров с нулевым диапазоном (low == high)
    low[flat] = close[flat]
    high[flat] = close[flat]
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.abs(high - prev_close))
    atr = pl.Series(tr).ewm_mean(alpha=1 / 14).to_numpy()
    ts = (np.arange(n, dtype=np.int64) + 1) * MIN_NS + T0
    ts[n // 2 :] += 37 * MIN_NS  # гэп в данных посреди серии
    return pl.DataFrame(
        {
            "ts_open": ts - MIN_NS,
            "ts_close": ts,
            "open": opn,
            "close": close,
            "low": low,
            "high": high,
            "quote_volume": np.abs(rng.normal(1e6, 3e5, n)),
            "d_oi_usd": rng.uniform(-3e5, 8e5, n),
            "atr": atr,
        }
    )


def test_real_builder_survives_degenerate_bars():
    bars = _adversarial_bars(60, seed=SEED)
    arr = bars_to_arrays(bars)
    edges = bucket_grid(arr, atr_fraction=0.5)
    heat = make_real_heat_builder(arr, np.array([10.0, 25.0]), edges, bar_s=60.0)(
        np.array([0.5, 0.5])
    )
    assert heat.shape == (bars.height, len(edges) - 1)
    assert np.isfinite(heat).all() and (heat >= 0.0).all()
    # один бар — не падает
    one = bars.head(1)
    a1 = bars_to_arrays(one)
    e1 = bucket_grid(a1, atr_fraction=0.5)
    h1 = make_real_heat_builder(a1, np.array([10.0]), e1, bar_s=60.0)(np.array([1.0]))
    assert h1.shape[0] == 1 and np.isfinite(h1).all()
    # ATR весь NaN -> честный ValueError, а не NaN-сетка
    all_nan = one.with_columns(pl.lit(NAN).alias("atr"))
    with pytest.raises(ValueError):
        bucket_grid(bars_to_arrays(all_nan))


def test_real_builder_adversarial_volatility_matches_liqmap():
    """Экстремальная волатильность: медианная отн. ошибка тоталов < 5%."""
    n = _n(800)
    bars = _adversarial_bars(n, seed=SEED)
    arr = bars_to_arrays(bars)
    edges = bucket_grid(arr, atr_fraction=0.5)
    grid = np.array([10.0, 25.0])
    w = np.array([0.6, 0.4])
    heat = make_real_heat_builder(arr, grid, edges, bar_s=60.0)(w)
    lm = LiqMap(
        leverage_grid=list(grid),
        buckets=PriceBuckets(float(edges[1] - edges[0])),
        weight_fn=StaticWeights(w),
        decay_half_life_s=86_400.0,
    )
    hist = HeatHistory(lm)
    for row in bars.iter_rows(named=True):
        lm.step(row["low"], row["high"], row["close"], row["d_oi_usd"], dt_s=60.0)
        hist.record(row["ts_close"])
    fast = heat.sum(axis=1)
    exact = np.array([hist.total_at(i) for i in range(len(hist))])
    mask = exact > 0
    assert mask.sum() > n // 2
    rel = np.abs(fast[mask] - exact[mask]) / exact[mask]
    assert float(np.median(rel)) < 0.05
    # каузальность: тотал строки не превышает накопленный приток
    inflow = np.cumsum(np.maximum(arr.d_oi_usd, 0.0))
    assert np.all(fast <= inflow + 1e-6)


def test_real_builder_drops_never_liquidating_slices():
    """Плечо 1 (лонг не ликвидируется, lp=0) не должно копить массу в нижнем
    крайнем бакете — LiqMap такие слайсы пропускает."""
    n = 5
    arr = BarArrays(
        ts=(np.arange(n, dtype=np.int64) + 1) * MIN_NS,
        close=np.full(n, 100.0),
        low=np.full(n, 99.0),
        high=np.full(n, 101.0),
        d_oi_usd=np.full(n, 1_000.0),
        long_share=np.full(n, 1.0),  # только лонги
        atr=np.full(n, 1.0),
    )
    edges = np.arange(50.0, 151.0, 1.0)
    heat = make_real_heat_builder(arr, np.array([1.0]), edges, bar_s=60.0)(np.array([1.0]))
    assert heat.sum() == 0.0  # 1x-лонги никогда не ликвидируются — массы нет
    # а шорты 1x на месте
    arr_short = BarArrays(
        ts=arr.ts, close=arr.close, low=arr.low, high=arr.high,
        d_oi_usd=arr.d_oi_usd, long_share=np.zeros(n), atr=arr.atr,
    )
    heat_s = make_real_heat_builder(arr_short, np.array([1.0]), edges, bar_s=60.0)(
        np.array([1.0])
    )
    assert heat_s.sum() > 0.0
