"""M9 vol-target sizing: EWMA estimator math, caps, zero-vol guard, step rounding."""

from __future__ import annotations

import math

import pytest

from trading_system.risk.sizing import (
    EwmaVol,
    VolTargetSizer,
    round_qty_to_step,
    vol_target_position_usd,
)

# --------------------------------------------------------------------------
# round_qty_to_step
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("qty", "step", "expected"),
    [
        (0.37, 0.1, 0.3),  # rounds DOWN, never oversizes
        (0.30000000000000004, 0.1, 0.3),  # float dust does not lose a step
        (1.0, 0.001, 1.0),
        (0.0009, 0.001, 0.0),
        (5.0, 1.0, 5.0),
        (-1.0, 0.1, 0.0),
        (0.0, 0.1, 0.0),
    ],
)
def test_round_qty_to_step(qty, step, expected):
    assert round_qty_to_step(qty, step) == pytest.approx(expected, abs=1e-12)


def test_round_qty_bad_step_raises():
    with pytest.raises(ValueError):
        round_qty_to_step(1.0, 0.0)


# --------------------------------------------------------------------------
# vol_target_position_usd
# --------------------------------------------------------------------------


def test_vol_target_basic_math():
    usd, capped, reason = vol_target_position_usd(100_000.0, 0.01, 0.02, 1e12)
    assert usd == pytest.approx(50_000.0)  # equity * target / realized
    assert not capped and reason == "vol-target"


def test_vol_target_cap():
    usd, capped, reason = vol_target_position_usd(100_000.0, 0.01, 0.001, 100_000.0)
    assert usd == pytest.approx(100_000.0)  # raw 1e6 hits the cap
    assert capped and "capped" in reason


@pytest.mark.parametrize("bad_vol", [0.0, 1e-12, -0.01, math.nan, math.inf])
def test_vol_target_zero_vol_guard(bad_vol):
    """No vol information -> size zero, never explode toward infinity."""
    usd, capped, reason = vol_target_position_usd(100_000.0, 0.01, bad_vol, 1e12)
    assert usd == 0.0 and not capped
    assert "zero-vol guard" in reason


@pytest.mark.parametrize(("equity", "target"), [(0.0, 0.01), (-5.0, 0.01), (100.0, 0.0)])
def test_vol_target_degenerate_inputs(equity, target):
    usd, capped, _ = vol_target_position_usd(equity, target, 0.02, 1e12)
    assert usd == 0.0 and not capped


# --------------------------------------------------------------------------
# EwmaVol
# --------------------------------------------------------------------------


def test_ewma_halflife_is_exact():
    """After exactly `halflife` zero-return updates the variance halves."""
    est = EwmaVol(halflife_bars=20.0, bars_per_day=1440.0)
    r = 0.01
    est.update(r)  # seeds var = r^2
    v0 = est.daily_vol**2
    for _ in range(20):
        est.update(0.0)
    assert est.daily_vol**2 == pytest.approx(v0 / 2.0, rel=1e-9)


def test_ewma_constant_returns_converge():
    est = EwmaVol(halflife_bars=10.0, bars_per_day=1440.0)
    for _ in range(600):
        est.update(0.001)
    assert est.daily_vol == pytest.approx(0.001 * math.sqrt(1440.0), rel=1e-6)


def test_ewma_starts_at_zero_and_resets():
    est = EwmaVol(halflife_bars=10.0, bars_per_day=1440.0)
    assert est.daily_vol == 0.0
    est.update(0.01)
    assert est.daily_vol > 0.0 and est.n == 1
    est.reset()
    assert est.daily_vol == 0.0 and est.n == 0


@pytest.mark.parametrize(("hl", "bpd"), [(0.0, 1440.0), (-1.0, 1440.0), (10.0, 0.0)])
def test_ewma_bad_params_raise(hl, bpd):
    with pytest.raises(ValueError):
        EwmaVol(halflife_bars=hl, bars_per_day=bpd)


# --------------------------------------------------------------------------
# VolTargetSizer end to end
# --------------------------------------------------------------------------


def make_sizer(**kw) -> VolTargetSizer:
    defaults = dict(
        target_daily_vol=0.01,
        max_position_usd=100_000.0,
        qty_step=0.001,
        halflife_bars=10.0,
        bars_per_day=1440.0,
    )
    defaults.update(kw)
    return VolTargetSizer(**defaults)


def test_sizer_zero_before_any_update():
    s = make_sizer()
    res = s.size(equity=100_000.0, price=50_000.0)
    assert res.position_usd == 0.0 and res.qty == 0.0
    assert "zero-vol guard" in res.reason


def test_sizer_end_to_end_matches_formula():
    s = make_sizer()
    for _ in range(600):
        s.update(0.001)  # daily vol -> 0.001 * sqrt(1440) ~= 0.03795
    res = s.size(equity=100_000.0, price=50_000.0)
    expected_usd = 100_000.0 * 0.01 / (0.001 * math.sqrt(1440.0))
    assert res.position_usd == pytest.approx(expected_usd, rel=1e-6)
    assert not res.capped
    assert res.vol_used == pytest.approx(0.001 * math.sqrt(1440.0), rel=1e-6)
    # qty is the notional at the mark, rounded DOWN to the step
    assert res.qty == pytest.approx(
        round_qty_to_step(expected_usd / 50_000.0, 0.001), abs=1e-12
    )
    assert res.qty * 50_000.0 <= res.position_usd + 1e-6  # never oversized


def test_sizer_cap_applies():
    s = make_sizer(max_position_usd=10_000.0)
    for _ in range(600):
        s.update(0.0001)  # tiny vol -> huge raw size
    res = s.size(equity=100_000.0, price=50_000.0)
    assert res.position_usd == pytest.approx(10_000.0)
    assert res.capped


def test_sizer_zero_price_gives_zero_qty():
    s = make_sizer()
    s.update(0.001)
    res = s.size(equity=100_000.0, price=0.0)
    assert res.qty == 0.0
