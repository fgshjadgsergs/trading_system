"""M9 hard limits: DailyStop (day rollover, stickiness) and KillSwitch scenarios.

All time is injected UTC-ns timestamps — no wall clock anywhere.
"""

from __future__ import annotations

import pytest

from trading_system.risk.limits import NS_PER_DAY, DailyStop, KillSwitch, utc_day_key

NS_PER_S = 1_000_000_000
DAY0_NOON = 100 * NS_PER_DAY + NS_PER_DAY // 2


def test_utc_day_key_boundary():
    assert utc_day_key(5 * NS_PER_DAY - 1) == 4
    assert utc_day_key(5 * NS_PER_DAY) == 5


# --------------------------------------------------------------------------
# DailyStop
# --------------------------------------------------------------------------


def test_daily_stop_trips_at_threshold():
    stop = DailyStop(daily_stop_pct=0.03)
    assert not stop.update(DAY0_NOON, 100_000.0).halted  # first tick sets the baseline
    assert not stop.update(DAY0_NOON + NS_PER_S, 97_100.0).halted  # -2.9%
    state = stop.update(DAY0_NOON + 2 * NS_PER_S, 97_000.0)  # exactly -3%
    assert state.halted
    assert state.ts == DAY0_NOON + 2 * NS_PER_S
    assert "daily stop" in state.reason


def test_daily_stop_is_sticky_within_the_day():
    stop = DailyStop(daily_stop_pct=0.03)
    stop.update(DAY0_NOON, 100_000.0)
    assert stop.update(DAY0_NOON + NS_PER_S, 96_000.0).halted
    # equity fully recovers — still halted for the rest of the day
    late = DAY0_NOON + 3_600 * NS_PER_S
    assert stop.update(late, 101_000.0).halted


def test_daily_stop_day_rollover_resets_baseline_and_halt():
    stop = DailyStop(daily_stop_pct=0.03)
    stop.update(DAY0_NOON, 100_000.0)
    assert stop.update(DAY0_NOON + NS_PER_S, 96_000.0).halted
    next_day = DAY0_NOON + NS_PER_DAY
    state = stop.update(next_day, 96_000.0)
    assert not state.halted  # new day, new baseline at 96k
    assert stop.day_start_equity == pytest.approx(96_000.0)
    # -3% from the NEW baseline trips again
    assert stop.update(next_day + NS_PER_S, 96_000.0 * 0.969).halted


def test_daily_stop_gain_never_halts():
    stop = DailyStop(daily_stop_pct=0.03)
    stop.update(DAY0_NOON, 100_000.0)
    assert not stop.update(DAY0_NOON + NS_PER_S, 130_000.0).halted


def test_daily_stop_injectable_day_key():
    # 1-hour "days" via an injected key: rollover after one hour of ts
    stop = DailyStop(daily_stop_pct=0.03, day_key=lambda ts: ts // (3_600 * NS_PER_S))
    t0 = 42 * 3_600 * NS_PER_S + 1
    stop.update(t0, 100_000.0)
    assert stop.update(t0 + NS_PER_S, 90_000.0).halted
    assert not stop.update(t0 + 3_600 * NS_PER_S, 90_000.0).halted  # next fake day


def test_daily_stop_reset():
    stop = DailyStop(daily_stop_pct=0.03)
    stop.update(DAY0_NOON, 100_000.0)
    stop.update(DAY0_NOON + NS_PER_S, 90_000.0)
    assert stop.state.halted
    stop.reset()
    assert not stop.state.halted and stop.day_start_equity is None


def test_daily_stop_bad_pct_raises():
    with pytest.raises(ValueError):
        DailyStop(daily_stop_pct=0.0)


# --------------------------------------------------------------------------
# KillSwitch
# --------------------------------------------------------------------------


def make_switch(**kw) -> KillSwitch:
    defaults = dict(max_consecutive_errors=3, stale_after_s=5.0)
    defaults.update(kw)
    return KillSwitch(**defaults)


def test_kill_switch_consecutive_errors_trip():
    ks = make_switch()
    t = DAY0_NOON
    assert not ks.record_error(t).tripped
    assert not ks.record_error(t + 1).tripped
    state = ks.record_error(t + 2)
    assert state.tripped and state.flatten
    assert "consecutive errors" in state.reason


def test_kill_switch_success_breaks_the_run():
    ks = make_switch()
    t = DAY0_NOON
    ks.record_error(t)
    ks.record_error(t + 1)
    ks.record_success(t + 2)  # run broken
    assert ks.consecutive_errors == 0
    assert not ks.record_error(t + 3).tripped
    assert not ks.record_error(t + 4).tripped
    assert ks.record_error(t + 5).tripped  # three in a row again


def test_kill_switch_stale_data_trips():
    ks = make_switch(stale_after_s=5.0)
    t0 = DAY0_NOON
    ks.record_market_data(t0)
    assert not ks.check(t0 + 5 * NS_PER_S).tripped  # exactly at the limit: fine
    state = ks.check(t0 + 5 * NS_PER_S + 1)
    assert state.tripped and state.flatten
    assert "stale market data" in state.reason


def test_kill_switch_fresh_ticks_keep_it_armed():
    ks = make_switch(stale_after_s=5.0)
    t = DAY0_NOON
    for i in range(20):
        ks.record_market_data(t + i * 4 * NS_PER_S)
        assert not ks.check(t + i * 4 * NS_PER_S + NS_PER_S).tripped


def test_kill_switch_first_check_arms_baseline_not_trips():
    ks = make_switch(stale_after_s=5.0)
    t0 = DAY0_NOON
    assert not ks.check(t0).tripped  # no data ever seen: arm, do not judge
    assert ks.check(t0 + 6 * NS_PER_S).tripped  # still nothing 6s later: dead feed


def test_kill_switch_market_data_is_monotonic():
    ks = make_switch(stale_after_s=5.0)
    t0 = DAY0_NOON
    ks.record_market_data(t0 + 10 * NS_PER_S)
    ks.record_market_data(t0)  # late out-of-order tick must not rewind freshness
    assert not ks.check(t0 + 14 * NS_PER_S).tripped
    assert ks.check(t0 + 16 * NS_PER_S).tripped


def test_kill_switch_sticky_until_reset():
    ks = make_switch(max_consecutive_errors=1)
    assert ks.record_error(DAY0_NOON).tripped
    ks.record_success(DAY0_NOON + 1)  # success does NOT un-trip
    assert ks.state.tripped
    assert ks.check(DAY0_NOON + 2).tripped
    ks.reset()
    assert not ks.state.tripped and ks.consecutive_errors == 0


def test_kill_switch_bad_params_raise():
    with pytest.raises(ValueError):
        KillSwitch(max_consecutive_errors=0, stale_after_s=5.0)
    with pytest.raises(ValueError):
        KillSwitch(max_consecutive_errors=1, stale_after_s=0.0)
