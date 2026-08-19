"""M11 live-PnL tracker vs the backtest expectation envelope."""

from __future__ import annotations

import numpy as np
import pytest

from trading_system.monitoring.alerts import Severity
from trading_system.monitoring.pnl_tracker import NS_PER_DAY, PnlTracker

T0 = 1_755_600_000 * 1_000_000_000

MEAN, STD = 50.0, 100.0


def T(days: float) -> int:
    return T0 + int(days * NS_PER_DAY)


def make_tracker(z: float = 3.0) -> PnlTracker:
    return PnlTracker(MEAN, STD, z_threshold=z, start_ts=T0)


def test_start_tick_is_never_breached():
    tr = PnlTracker(MEAN, STD)  # start_ts from the first observation
    status = tr.observe(T(0), 0.0)
    assert status.t_days == 0.0 and status.z == 0.0 and not status.breached


def test_z_math_after_one_day():
    tr = make_tracker()
    status = tr.observe(T(1.0), MEAN + 2.0 * STD)
    assert status.expected == pytest.approx(MEAN)
    assert status.z == pytest.approx(2.0)
    assert status.band == pytest.approx(3.0 * STD)
    assert not status.breached


def test_breach_below_fires_critical_alert():
    tr = make_tracker()
    status = tr.observe(T(1.0), MEAN - 3.5 * STD)
    assert status.breached and status.z == pytest.approx(-3.5)
    alert = tr.alert_for(status)
    assert alert is not None
    assert alert.severity is Severity.CRITICAL
    assert alert.key == "pnl_divergence"
    assert "below" in alert.message
    assert alert.ts == T(1.0)


def test_breach_above_is_a_model_mismatch_too():
    tr = make_tracker()
    status = tr.observe(T(4.0), 4.0 * MEAN + 3.1 * STD * 2.0)  # sigma = STD*sqrt(4)
    assert status.breached and status.z == pytest.approx(3.1)
    alert = tr.alert_for(status)
    assert alert is not None and "above" in alert.message


def test_inside_envelope_returns_no_alert():
    tr = make_tracker()
    status = tr.observe(T(1.0), MEAN)
    assert tr.alert_for(status) is None


def test_sqrt_time_scaling():
    tr = make_tracker()
    shortfall = -250.0
    z1 = tr.observe(T(1.0), MEAN * 1.0 + shortfall).z
    tr.reset()
    tr.start_ts = T0
    z4 = tr.observe(T(4.0), MEAN * 4.0 + shortfall).z
    assert z4 == pytest.approx(z1 / 2.0)  # same shortfall, twice the sigma


def test_envelope_arrays():
    tr = make_tracker()
    tr.observe(T(0.0), 0.0)
    ts = np.array([T(0.0), T(1.0), T(4.0)], dtype=np.int64)
    expected, lower, upper = tr.envelope(ts)
    assert expected == pytest.approx([0.0, MEAN, 4.0 * MEAN])
    assert upper - expected == pytest.approx([0.0, 3.0 * STD, 6.0 * STD])
    assert expected - lower == pytest.approx(upper - expected)


def test_envelope_without_start_raises():
    tr = PnlTracker(MEAN, STD)
    with pytest.raises(ValueError):
        tr.envelope(np.array([T0]))


def test_reset_clears_start():
    tr = make_tracker()
    tr.observe(T(1.0), 0.0)
    tr.reset()
    assert tr.start_ts is None and tr.last_status is None


def test_bad_params_raise():
    with pytest.raises(ValueError):
        PnlTracker(MEAN, 0.0)
    with pytest.raises(ValueError):
        PnlTracker(MEAN, STD, z_threshold=0.0)
