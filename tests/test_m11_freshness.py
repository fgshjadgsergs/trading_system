"""M11 freshness tracking and gap alerting on a fully injected clock."""

from __future__ import annotations

import pytest

from trading_system.monitoring.alerts import Severity
from trading_system.monitoring.freshness import FreshnessTracker, GapEvent, GapMonitor

NS_PER_S = 1_000_000_000
T0 = 1_755_600_000 * NS_PER_S

LIMITS = {"depth": 5.0, "agg_trade": 10.0}


def T(seconds: float) -> int:
    return T0 + int(seconds * NS_PER_S)


# --------------------------------------------------------------------------
# FreshnessTracker
# --------------------------------------------------------------------------


def test_ages_and_stale_detection():
    tr = FreshnessTracker(LIMITS, start_ts=T0)
    tr.observe("depth", T(0))
    tr.observe("agg_trade", T(0))
    ages = tr.ages_s(T(3))
    assert ages["depth"] == pytest.approx(3.0)
    assert tr.stale(T(3)) == []
    stale = tr.stale(T(6))  # depth (limit 5) is over; agg_trade (limit 10) is not
    assert [s.stream for s in stale] == ["depth"]
    assert stale[0].age_s == pytest.approx(6.0)
    assert stale[0].limit_s == 5.0
    assert stale[0].last_ts == T(0)


def test_never_seen_stream_ages_from_start():
    tr = FreshnessTracker(LIMITS, start_ts=T0)
    stale = tr.stale(T(11))  # nothing ever observed
    assert {s.stream for s in stale} == {"depth", "agg_trade"}
    assert all(s.last_ts is None for s in stale)


def test_stale_sorted_worst_first():
    tr = FreshnessTracker(LIMITS, start_ts=T0)
    tr.observe("depth", T(14))  # age 6 at T(20) -> 1.2x limit
    tr.observe("agg_trade", T(0))  # age 20 at T(20) -> 2.0x limit
    stale = tr.stale(T(20))
    assert [s.stream for s in stale] == ["agg_trade", "depth"]


def test_injected_clock_is_used():
    now = [T(0)]
    tr = FreshnessTracker(LIMITS, clock=lambda: now[0], start_ts=T0)
    tr.observe("depth", T(0))
    tr.observe("agg_trade", T(0))
    assert tr.stale() == []
    now[0] = T(7)
    assert [s.stream for s in tr.stale()] == ["depth"]


def test_no_clock_and_no_ts_raises():
    tr = FreshnessTracker(LIMITS, start_ts=T0)
    with pytest.raises(ValueError):
        tr.stale()


def test_alert_severity_escalates_at_3x_limit():
    tr = FreshnessTracker(LIMITS, start_ts=T0)
    tr.observe("depth", T(0))
    tr.observe("agg_trade", T(0))
    (warn,) = tr.alerts(T(8))
    assert warn.severity is Severity.WARNING
    assert warn.key == "stale:depth"
    assert "depth" in warn.message and warn.ts == T(8)
    alerts = tr.alerts(T(16))  # depth age 16 > 3*5 -> CRITICAL; agg_trade 16 > 10 -> WARNING
    by_key = {a.key: a for a in alerts}
    assert by_key["stale:depth"].severity is Severity.CRITICAL
    assert by_key["stale:agg_trade"].severity is Severity.WARNING


def test_out_of_order_observe_does_not_rewind():
    tr = FreshnessTracker(LIMITS, start_ts=T0)
    tr.observe("depth", T(10))
    tr.observe("depth", T(2))  # late arrival
    assert tr.last_ts("depth") == T(10)


def test_reset_forgets_everything():
    tr = FreshnessTracker(LIMITS, start_ts=T0)
    tr.observe("depth", T(0))
    tr.reset()
    assert tr.last_ts("depth") is None


# --------------------------------------------------------------------------
# GapMonitor
# --------------------------------------------------------------------------


def gap(end_s: float, dur_s: float, stream: str = "agg_trade") -> GapEvent:
    return GapEvent(stream=stream, ts_start=T(end_s - dur_s), ts_end=T(end_s))


def test_short_gap_is_ignored():
    gm = GapMonitor(min_gap_s=2.0)
    assert gm.on_gap(gap(10.0, 1.9)) == []


def test_single_gap_warns():
    gm = GapMonitor(min_gap_s=2.0, burst_n=3, burst_window_s=60.0)
    alerts = gm.on_gap(gap(10.0, 4.0))
    assert len(alerts) == 1
    assert alerts[0].severity is Severity.WARNING
    assert alerts[0].key == "gap:agg_trade"
    assert "4.0s" in alerts[0].message


def test_burst_escalates_to_critical():
    gm = GapMonitor(min_gap_s=2.0, burst_n=3, burst_window_s=60.0)
    assert len(gm.on_gap(gap(10.0, 3.0))) == 1
    assert len(gm.on_gap(gap(20.0, 3.0))) == 1
    alerts = gm.on_gap(gap(30.0, 3.0))  # third within 60s
    assert [a.severity for a in alerts] == [Severity.WARNING, Severity.CRITICAL]
    assert alerts[1].key == "gap_burst:agg_trade"


def test_old_gaps_fall_out_of_the_burst_window():
    gm = GapMonitor(min_gap_s=2.0, burst_n=3, burst_window_s=60.0)
    gm.on_gap(gap(10.0, 3.0))
    gm.on_gap(gap(20.0, 3.0))
    alerts = gm.on_gap(gap(200.0, 3.0))  # first two aged out
    assert len(alerts) == 1 and alerts[0].severity is Severity.WARNING


def test_bursts_tracked_per_stream():
    gm = GapMonitor(min_gap_s=2.0, burst_n=2, burst_window_s=60.0)
    gm.on_gap(gap(10.0, 3.0, "depth"))
    alerts = gm.on_gap(gap(20.0, 3.0, "agg_trade"))  # different stream: no burst
    assert len(alerts) == 1
