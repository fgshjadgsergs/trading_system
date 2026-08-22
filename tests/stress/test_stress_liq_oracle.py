"""Независимый оракул liq_price: закрытая форма против численного решения
самого уравнения маржи (equity(P) == maintenance(P), кусочные тиры).

Оракул не переиспользует формулу — только определяющее уравнение + бисекция:

    long:  q*E/L + q*(P-E) = m(N)*q*P - c(N),  N = q*P
    short: q*E/L + q*(E-P) = m(N)*q*P - c(N)
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from trading_system.core.liquidation import (
    DEFAULT_BRACKETS,
    BinanceUsdmLiquidation,
    admissible_qty,
    bracket_for,
)
from trading_system.core.schema import Side

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
SYM = "BTCUSDT"
ENGINE = BinanceUsdmLiquidation(brackets={SYM: DEFAULT_BRACKETS})


def margin_gap(P: float, E: float, L: float, q: float, side: Side) -> float:
    b = bracket_for(q * P, DEFAULT_BRACKETS)
    equity = q * E / L + (q * (P - E) if side is Side.BUY else q * (E - P))
    return equity - (b.mmr * q * P - b.cum)


def bisect_root(E: float, L: float, q: float, side: Side) -> float:
    lo, hi = 0.0, max(E * 4, 1.0)
    for _ in range(200):
        if margin_gap(lo, E, L, q, side) * margin_gap(hi, E, L, q, side) <= 0:
            break
        hi *= 2
    else:
        return float("nan")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if margin_gap(lo, E, L, q, side) * margin_gap(mid, E, L, q, side) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def test_closed_form_solves_margin_equation_random_sweep():
    rng = np.random.default_rng(42)
    n = int(4000 * SCALE)
    checked = 0
    for _ in range(n):
        E = float(10 ** rng.uniform(-2, 5))
        L = float(rng.choice([1, 2, 3, 5, 10, 20, 25, 50, 75, 100, 125]))
        q = float(10 ** rng.uniform(1, 8.7)) / E  # нотионал входа 10..5e8 USD
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        lp = ENGINE.liq_price(E, L, side, symbol=SYM, qty=q)
        if side is Side.BUY and lp == 0.0:
            continue  # long, который не ликвидируется (L=1) — корректный clamp
        # движок клэмпит агрегат до максимального допустимого счёта на плече;
        # оракул решает уравнение для того же представительного счёта
        q_eff = min(q, admissible_qty(DEFAULT_BRACKETS, E, L, side))
        root = bisect_root(E, L, q_eff, side)
        if np.isnan(root):
            continue
        assert abs(lp - root) <= 1e-9 * max(root, 1.0), (E, L, q, side, lp, root)
        # закрытая красная зона: цена ликвидации всегда по верную сторону входа
        assert (lp < E) if side is Side.BUY else (lp > E), (E, L, q, side, lp)
        checked += 1
    assert checked > n * 0.9


def test_admissible_qty_puts_liq_notional_on_admissible_cap():
    """Клэмп-счёт сидит нотионалом на цене ликвидации ровно на капе
    последнего тира с mmr < 1/L; его тир никогда не «рождён ликвидируемым»."""
    rng = np.random.default_rng(11)
    for _ in range(int(2000 * SCALE)):
        E = float(10 ** rng.uniform(-1, 5))
        L = float(rng.choice([5, 10, 20, 50, 100, 125]))
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        q_max = admissible_qty(DEFAULT_BRACKETS, E, L, side)
        adm = [b for b in DEFAULT_BRACKETS if b.mmr < 1 / L][-1]
        lp = ENGINE.liq_price(E, L, side, symbol=SYM, qty=q_max)
        assert q_max * lp == pytest.approx(adm.max_notional_usd, rel=1e-9)
        # нотионал сидит на капе с точностью до ulp — проверяем тир чуть внутри
        assert bracket_for(q_max * lp * (1 - 1e-9), DEFAULT_BRACKETS).mmr < 1 / L


def test_tier_boundaries_match_equation():
    for b_prev in DEFAULT_BRACKETS[:-1]:
        cap = b_prev.max_notional_usd
        for side in (Side.BUY, Side.SELL):
            E = 100.0
            q = cap / E  # нотионал ликвидации оказывается в окрестности cap
            q_eff = min(q, admissible_qty(DEFAULT_BRACKETS, E, 10, side))
            lp = ENGINE.liq_price(E, 10, side, symbol=SYM, qty=q)
            root = bisect_root(E, 10, q_eff, side)
            assert abs(lp - root) <= 1e-9 * max(root, 1.0), (cap, side, lp, root)


def test_flat_formula_solves_constant_mmr_equation():
    flat = BinanceUsdmLiquidation(flat_mmr=0.005)
    rng = np.random.default_rng(7)
    for _ in range(int(2000 * SCALE)):
        E = float(10 ** rng.uniform(-2, 5))
        L = float(rng.choice([2, 3, 5, 10, 25, 50, 125]))
        side = Side.BUY if rng.random() < 0.5 else Side.SELL
        lp = flat.liq_price(E, L, side)
        gap = E / L + (lp - E if side is Side.BUY else E - lp) - 0.005 * lp
        assert abs(gap) <= 1e-9 * max(E, 1.0)
