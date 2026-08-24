"""M4: liquidation formula vs exchange calculator, mass conservation, consume/decay, regression hash."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_system.core.liquidation import (
    DEFAULT_BRACKETS,
    BinanceUsdmLiquidation,
    bracket_for,
    liq_price,
)
from trading_system.core.schema import Side
from trading_system.liqmap.buckets import PriceBuckets, rebucket
from trading_system.liqmap.map import Context, LiqMap, StaticWeights

# -- liq_price vs exchange calculator -----------------------------------------
# Expected values from Binance's own formula LP = (WB + cum -/+ q*E) / (q*(MMR -/+ 1))
# with WB = q*E/L (isolated), computed independently below.

FLAT_CASES = [
    # entry, leverage, side, mmr, expected
    (100.0, 10, Side.BUY, 0.005, 90.45226130653266),
    (100.0, 10, Side.SELL, 0.005, 109.45273631840796),
    (50_000.0, 20, Side.BUY, 0.005, 47_738.69346733668),
    (50_000.0, 20, Side.SELL, 0.005, 52_238.80597014925),
    (50_000.0, 125, Side.BUY, 0.004, 49_799.19678714859),
    (50_000.0, 125, Side.SELL, 0.004, 50_199.20318725099),
    (3.0, 5, Side.BUY, 0.01, 2.4242424242424243),
    (3.0, 5, Side.SELL, 0.01, 3.5643564356435644),
    (0.25, 50, Side.BUY, 0.005, 0.24623115577889445),
    (0.25, 50, Side.SELL, 0.005, 0.25373134328358204),
    (100.0, 3, Side.BUY, 0.025, 68.37606837606837),
    (100.0, 3, Side.SELL, 0.025, 130.0813008130081),
    (100.0, 1, Side.BUY, 0.005, 0.0),  # 1x long with flat mmr never liquidates above 0
]


@pytest.mark.parametrize("entry,lev,side,mmr,expected", FLAT_CASES)
def test_liq_price_matches_exchange_calculator(entry, lev, side, mmr, expected):
    assert liq_price(entry, lev, side, mmr) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("entry,lev,side,mmr,expected", FLAT_CASES)
def test_liq_price_matches_raw_binance_form(entry, lev, side, mmr, expected):
    """Cross-check against the raw (WB + cum -/+ qE) / (q(MMR -/+ 1)) form, q=1."""
    wb = entry / lev
    if side is Side.BUY:
        raw = (wb - entry) / (mmr - 1)
        raw = max(0.0, raw)
    else:
        raw = (wb + entry) / (mmr + 1)
    assert liq_price(entry, lev, side, mmr) == pytest.approx(raw, rel=1e-12)


def test_liq_price_with_bracket_cum():
    # 10 BTC at 50k -> notional 500k -> tier mmr=1%, cum=1300
    f = BinanceUsdmLiquidation(brackets={"BTCUSDT": DEFAULT_BRACKETS})
    got = f.liq_price(50_000.0, 10, Side.BUY, symbol="BTCUSDT", qty=10.0)
    assert got == pytest.approx(45_323.23232323232, rel=1e-12)
    # raw exchange form: (WB + cum - q*E) / (q*(MMR - 1))
    raw = (50_000.0 + 1_300.0 - 10 * 50_000.0) / (10 * (0.01 - 1))
    assert got == pytest.approx(raw, rel=1e-12)
    # long liq below entry, short above
    short = f.liq_price(50_000.0, 10, Side.SELL, symbol="BTCUSDT", qty=10.0)
    assert short > 50_000.0 > got


def test_liq_price_bracket_tier_self_consistent():
    """The applied tier must contain the notional AT the liquidation price,
    not at entry (Binance computes maintenance from mark-price notional)."""
    f = BinanceUsdmLiquidation(brackets={"BTCUSDT": DEFAULT_BRACKETS})
    # 6 BTC long @50k 2x: entry notional 300k (tier 3), but at LP the notional
    # is ~150k (tier 2) -> tier-2 solution is the self-consistent one
    lp = f.liq_price(50_000.0, 2, Side.BUY, symbol="BTCUSDT", qty=6.0)
    assert lp == pytest.approx((50_000.0 * 0.5 - 50.0 / 6.0) / (1 - 0.005), rel=1e-12)
    assert 50_000 < lp * 6.0 <= 250_000  # lands in the tier that produced it
    # entry-tier (wrong) answer would have been lower: heat placed too deep
    wrong = (50_000.0 * 0.5 - 1_300.0 / 6.0) / (1 - 0.01)
    assert lp > wrong
    # short crossing UP a tier: 6 @40k 2x, entry 240k (tier 2) -> LP in tier 3
    lps = f.liq_price(40_000.0, 2, Side.SELL, symbol="BTCUSDT", qty=6.0)
    assert lps == pytest.approx((40_000.0 * 1.5 + 1_300.0 / 6.0) / (1 + 0.01), rel=1e-12)
    assert 250_000 < lps * 6.0 <= 1_000_000
    # 1x long still clamps to 0 (never liquidates above zero)
    assert f.liq_price(100.0, 1, Side.BUY, symbol="BTCUSDT", qty=1.0) == 0.0


def test_bracket_lookup_monotonic():
    mmrs = [bracket_for(n, DEFAULT_BRACKETS).mmr for n in (1e4, 1e5, 5e5, 5e6, 1.5e7, 3e7, 7e7, 1.5e8, 5e8)]
    assert mmrs == sorted(mmrs)
    assert bracket_for(1e4, DEFAULT_BRACKETS).mmr == 0.004


def test_liq_price_validation():
    with pytest.raises(ValueError):
        liq_price(0.0, 10, Side.BUY, 0.005)
    with pytest.raises(ValueError):
        liq_price(100.0, 0, Side.BUY, 0.005)
    with pytest.raises(ValueError):
        liq_price(100.0, 10, Side.BUY, 1.5)


# -- map mechanics ------------------------------------------------------------


def make_map(long_share: float = 0.5) -> LiqMap:
    return LiqMap(
        leverage_grid=[5, 10, 25, 50, 100],
        buckets=PriceBuckets(bucket_size=10.0),
        weight_fn=StaticWeights(np.array([1.0, 2.0, 3.0, 2.0, 1.0])),
        long_share=long_share,
        decay_half_life_s=3_600.0,
    )


def test_allocate_places_heat_on_both_sides():
    lm = make_map()
    lm.allocate(1_000_000.0, 50_000.0)
    snap = lm.snapshot()
    assert lm.total_heat() == pytest.approx(1_000_000.0)
    assert snap["long"].sum() == pytest.approx(500_000.0)
    assert snap["short"].sum() == pytest.approx(500_000.0)
    # long pools strictly below entry, short strictly above
    assert snap["prices"][snap["long"] > 0].max() < 50_000.0
    assert snap["prices"][snap["short"] > 0].min() > 50_000.0


def test_consume_zeroes_traversed_zones_only():
    lm = make_map()
    lm.allocate(1_000_000.0, 50_000.0)
    before = {s: dict(h) for s, h in lm.heat.items()}
    taken = lm.consume(49_000.0, 50_500.0)
    assert taken > 0
    for side, side_heat in lm.heat.items():
        for idx, h in side_heat.items():
            lo, hi = lm.buckets.lo(idx), lm.buckets.hi(idx)
            assert hi < 49_000.0 or lo > 50_500.0  # nothing left inside the path
            assert h == before[side][idx]  # untouched pools unchanged
    assert lm.mass_balance_error() < 1e-6


def test_consume_respects_half_open_bucket_boundary():
    """A pool in [49990, 50000) survives a path that starts exactly at 50000."""
    lm = LiqMap([10], PriceBuckets(10.0), StaticWeights(np.array([1.0])))
    lm.heat[Side.BUY][lm.buckets.index(49_995.0)] = 100.0
    lm.contributed = 100.0
    taken = lm.consume(50_000.0, 50_050.0)
    assert taken == 0.0
    assert lm.total_heat() == pytest.approx(100.0)
    # but a path that actually enters the bucket takes it
    assert lm.consume(49_999.9, 50_050.0) == pytest.approx(100.0)


def test_consume_returns_all_heat_on_full_sweep():
    lm = make_map()
    lm.allocate(500_000.0, 50_000.0)
    taken = lm.consume(0.0, 10_000_000.0)
    assert taken == pytest.approx(500_000.0)
    assert lm.total_heat() == pytest.approx(0.0)


def test_decay_half_life():
    lm = make_map()
    lm.allocate(100_000.0, 50_000.0)
    lost = lm.decay(3_600.0)  # exactly one half-life
    assert lost == pytest.approx(50_000.0, rel=1e-9)
    assert lm.total_heat() == pytest.approx(50_000.0, rel=1e-9)
    assert lm.mass_balance_error() < 1e-6


def test_negative_doi_removes_proportionally():
    lm = make_map(long_share=0.7)
    lm.allocate(1_000_000.0, 50_000.0)
    lm.allocate(-400_000.0, 50_100.0)
    assert lm.total_heat() == pytest.approx(600_000.0)
    snap = lm.snapshot()
    assert snap["long"].sum() == pytest.approx(0.6 * 700_000.0)
    assert snap["short"].sum() == pytest.approx(0.6 * 300_000.0)
    assert lm.removed == pytest.approx(400_000.0)
    assert lm.mass_balance_error() < 1e-6


def test_weight_context_signature():
    seen: list[Context | None] = []

    def w(ctx: Context | None) -> np.ndarray:
        seen.append(ctx)
        return np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    lm = LiqMap([5, 10, 25, 50, 100], PriceBuckets(10.0), w)
    ctx = Context(ts=123, features={"vol_z_5m": 2.5})
    lm.allocate(1000.0, 50_000.0, ctx)
    assert seen == [ctx]


def test_rebucket_conserves_mass():
    lm = make_map()
    lm.allocate(750_000.0, 50_000.0)
    lm.rebucket_to(PriceBuckets(bucket_size=25.0))
    assert lm.total_heat() == pytest.approx(750_000.0)
    assert lm.mass_balance_error() < 1e-6
    assert rebucket({}, PriceBuckets(1.0), PriceBuckets(2.0)) == {}


@settings(max_examples=60, deadline=None)
@given(
    ops=st.lists(
        st.tuples(
            st.sampled_from(["alloc", "close", "consume", "decay"]),
            st.floats(min_value=0.0, max_value=1e6, allow_nan=False),
            st.floats(min_value=1_000.0, max_value=100_000.0, allow_nan=False),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_mass_invariant_over_arbitrary_op_sequences(ops):
    lm = make_map()
    for kind, amount, price in ops:
        if kind == "alloc":
            lm.allocate(amount, price)
        elif kind == "close":
            lm.allocate(-amount, price)
        elif kind == "consume":
            lm.consume(price * 0.99, price * 1.01)
        else:
            lm.decay(amount % 7_200.0)
    total = lm.total_heat()
    assert total >= -1e-9
    assert all(h >= 0 for sh in lm.heat.values() for h in sh.values())
    scale = max(1.0, lm.contributed)
    assert lm.mass_balance_error() / scale < 1e-9


def test_dropped_counter_accounts_for_never_liquidating_slices():
    """M5: contributed + dropped == Σ положительных ΔOI (доли и веса в сумме 1)."""
    lm = LiqMap(
        leverage_grid=[1, 10, 25],  # 1x лонг никогда не ликвидируется (lp=0)
        buckets=PriceBuckets(10.0),
        weight_fn=StaticWeights(np.array([1.0, 2.0, 1.0])),
        long_share=0.6,
    )
    rng = np.random.default_rng(3)
    d_ois = rng.uniform(1.0, 1e6, 50)
    price = 50_000.0
    for d in d_ois:  # только allocate: без consume/decay/remove
        price *= float(np.exp(rng.normal(0, 0.001)))
        lm.allocate(float(d), price)
    total_in = float(d_ois.sum())
    assert lm.dropped > 0.0  # 1x-лонг-слайсы действительно отброшены
    assert abs((lm.contributed + lm.dropped) - total_in) / total_in < 1e-12
    assert lm.mass_balance_error() / total_in < 1e-12
    # dropped не входит в баланс массы: ΣH == contributed
    assert abs(lm.total_heat() - lm.contributed) / total_in < 1e-12


def make_bracket_map(typical_account_usd: float | None) -> LiqMap:
    from trading_system.collectors.brackets import bracket_liq_price_fn

    return LiqMap(
        leverage_grid=[5, 10, 25, 50, 100],
        buckets=PriceBuckets(bucket_size=10.0),
        weight_fn=StaticWeights(np.array([1.0, 2.0, 3.0, 2.0, 1.0])),
        liq_price_fn=bracket_liq_price_fn({"BTCUSDT": DEFAULT_BRACKETS}, "BTCUSDT"),
        long_share=0.5,
        decay_half_life_s=3_600.0,
        typical_account_usd=typical_account_usd,
    )


def test_typical_account_validation():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            make_bracket_map(bad)


def test_typical_account_tier1_lp_is_qty_independent():
    """M1: пока нотионал счёта в тире 1 (cum=0), lp не зависит от typical —
    карты для typical=10k и typical=40k совпадают бит-в-бит."""
    maps = [make_bracket_map(10_000.0), make_bracket_map(40_000.0)]
    rng_states = [np.random.default_rng(7), np.random.default_rng(7)]
    for lm, rng in zip(maps, rng_states, strict=True):
        price = 50_000.0
        for _ in range(50):
            price *= float(np.exp(rng.normal(0, 0.002)))
            lm.step(price * 0.999, price * 1.001, price, float(rng.uniform(0.0, 5e5)), dt_s=60.0)
    assert maps[0].heat == maps[1].heat  # dict-и с float-ами: точное равенство
    assert maps[0].contributed == maps[1].contributed


def make_blur_map(sigma0: float = 5.0, sigma1: float = 0.0) -> LiqMap:
    return LiqMap(
        leverage_grid=[5, 10, 25, 50, 100],
        buckets=PriceBuckets(bucket_size=10.0),
        weight_fn=StaticWeights(np.array([1.0, 2.0, 3.0, 2.0, 1.0])),
        long_share=0.5,
        decay_half_life_s=3_600.0,
        blur_sigma0_bps=sigma0,
        blur_sigma1=sigma1,
    )


def test_blur_param_validation():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            make_blur_map(sigma0=bad)
        with pytest.raises(ValueError):
            make_blur_map(sigma1=bad)


def test_blur_conserves_mass_exactly():
    """R1: раздача amount по ядру консервативна — contributed набирает ровно
    amount, инвариант массы держится на 1e-12."""
    lm = make_blur_map()
    lm.allocate(1_000_000.0, 50_000.0)
    assert lm.mass_balance_error() / 1e6 < 1e-12
    assert abs(lm.total_heat() - 1_000_000.0) / 1e6 < 1e-12
    # и под смесью операций
    rng = np.random.default_rng(5)
    price = 50_000.0
    for _ in range(500):
        price *= float(np.exp(rng.normal(0, 0.002)))
        lm.step(price * 0.999, price * 1.001, price, float(rng.uniform(-2e5, 6e5)), dt_s=60.0)
    scale = max(1.0, lm.contributed)
    assert lm.mass_balance_error() / scale < 1e-12


def test_blur_side_trim_no_heat_past_price():
    """R1: гауссов хвост за текущей ценой обрезан — лонг-тепла на/выше цены
    раскладки нет, шорт-тепла на/ниже нет (и при дистанционном sigma1)."""
    for sigma0, sigma1 in ((5.0, 0.0), (20.0, 0.5), (0.0, 1.0)):
        lm = make_blur_map(sigma0=sigma0, sigma1=sigma1)
        price = 50_000.0
        lm.allocate(2_000_000.0, price)
        snap = lm.snapshot()
        longs = snap["prices"][snap["long"] > 0]
        shorts = snap["prices"][snap["short"] > 0]
        assert longs.size and shorts.size
        assert longs.max() < price  # центры лонг-клеток строго ниже цены
        assert shorts.min() > price
        assert lm.mass_balance_error() / 2e6 < 1e-12


def test_regression_hash_blur_config():
    """R1 pin: бит-стабильность активированной конфигурации sigma0=5.0 bps."""
    lm = make_blur_map()
    rng = np.random.default_rng(42)
    price = 50_000.0
    for _ in range(200):
        price *= float(np.exp(rng.normal(0, 0.002)))
        lo, hi = price * 0.999, price * 1.001
        lm.step(lo, hi, price, d_oi_usd=float(rng.uniform(-2e5, 6e5)), dt_s=60.0)
    snap = lm.snapshot()
    payload = np.round(
        np.concatenate([snap["prices"], snap["long"], snap["short"]]), 6
    ).tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == "f60c4d1721ba13a13880b40017435569269f375951f46c14f04cfa8a95103a17"


def test_neumaier_accumulators_survive_many_small_ops():
    """N3: инвариант массы держится на 1e-12 после многих мелких операций."""
    lm = make_map()
    rng = np.random.default_rng(11)
    price = 50_000.0
    for _ in range(3_000):
        price *= float(np.exp(rng.normal(0, 0.002)))
        lm.allocate(float(rng.uniform(0.0, 1e6)), price)
        lm.consume(price * 0.999, price * 1.001)
        lm.decay(60.0)
        lm.allocate(-float(rng.uniform(0.0, 1e5)), price)
    scale = max(1.0, lm.contributed)
    assert lm.mass_balance_error() / scale < 1e-12


# -- regression hash ----------------------------------------------------------


def test_regression_hash_of_heat_matrix():
    """Fixed input -> byte-stable H matrix (guards accidental model drift)."""
    lm = make_map()
    rng = np.random.default_rng(42)
    price = 50_000.0
    for _ in range(200):
        price *= float(np.exp(rng.normal(0, 0.002)))
        lo, hi = price * 0.999, price * 1.001
        lm.step(lo, hi, price, d_oi_usd=float(rng.uniform(-2e5, 6e5)), dt_s=60.0)
    snap = lm.snapshot()
    payload = np.round(
        np.concatenate([snap["prices"], snap["long"], snap["short"]]), 6
    ).tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == "d3da19d2e9de12334f1e3512fd66de4d7a24d1c262cf56dac9027169dc8596ea"


def test_regression_hash_typical_account_bracket_config():
    """M1 pin: бит-стабильность активированной конфигурации typical=20k + брекеты."""
    lm = make_bracket_map(20_000.0)
    rng = np.random.default_rng(42)
    price = 50_000.0
    for _ in range(200):
        price *= float(np.exp(rng.normal(0, 0.002)))
        lo, hi = price * 0.999, price * 1.001
        lm.step(lo, hi, price, d_oi_usd=float(rng.uniform(-2e5, 6e5)), dt_s=60.0)
    snap = lm.snapshot()
    payload = np.round(
        np.concatenate([snap["prices"], snap["long"], snap["short"]]), 6
    ).tobytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert digest == "404a8d3db288741bad51bdb8111dcb29c614b02c59d61948d1f3d299a218a19b"


def test_demo_reports(tmp_path):
    from trading_system.liqmap.reports import demo_reports

    paths = demo_reports(tmp_path, seed=42)
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 5_000


def test_terminal_overlay_renders(tmp_path):
    from trading_system.liqmap.reports import build_demo_map
    from trading_system.liqmap.terminal import terminal_heat_overlay

    bars, lm, hist = build_demo_map(seed=42)
    p = terminal_heat_overlay(bars, hist, name="terminal_test", out_dir=tmp_path)
    assert p.exists() and p.stat().st_size > 20_000


def test_terminal_overlay_empty_map_renders(tmp_path):
    """Пустая карта (нулевой приток OI) раньше роняла matrix() по IndexError
    на occupied[0]; теперь отдаёт пустую сетку, а оверлей рисует пустой
    экран с невырожденным диапазоном цены."""
    import polars as pl

    from trading_system.liqmap.history import HeatHistory
    from trading_system.liqmap.terminal import terminal_heat_overlay

    lm = LiqMap(
        leverage_grid=[10],
        buckets=PriceBuckets(1.0),
        weight_fn=StaticWeights(np.array([1.0])),
        decay_half_life_s=1e9,
    )
    hist = HeatHistory(lm)
    step_ns = 60_000_000_000
    bars = pl.DataFrame({
        "ts_open": [0, step_ns], "ts_close": [step_ns, 2 * step_ns],
        "open": [100.0, 100.0], "high": [100.0, 100.0],
        "low": [100.0, 100.0], "close": [100.0, 100.0],
    })
    for t in (step_ns, 2 * step_ns):
        lm.step(100.0, 100.0, 100.0, 0.0, dt_s=60.0)
        hist.record(t)
    ts, prices, H = hist.matrix()
    assert ts.shape == (2,) and prices.size == 0 and H.shape == (0, 2)
    assert terminal_heat_overlay(bars, hist, name="empty", out_dir=tmp_path).exists()


def test_terminal_overlay_quantile_contrast_is_opt_outable(tmp_path):
    """Квантильная нормировка цвета — только представление: файл рисуется и
    с ней, и с полным диапазоном (q_lo=0, q_hi=1 — прежнее поведение)."""
    from trading_system.liqmap.reports import build_demo_map
    from trading_system.liqmap.terminal import terminal_heat_overlay

    bars, lm, hist = build_demo_map(seed=42)
    a = terminal_heat_overlay(bars, hist, name="q_default", out_dir=tmp_path)
    b = terminal_heat_overlay(bars, hist, name="q_full", out_dir=tmp_path,
                              q_lo=0.0, q_hi=1.0)
    assert a.exists() and b.exists()
    assert a.read_bytes() != b.read_bytes()  # нормировка действительно применяется
