"""M11 fire drill: break every component on a fake clock — the right alert must
fire BEFORE a human would notice, repeats are rate-limited, and the run is
fully deterministic.
"""

from __future__ import annotations

import pytest

from trading_system.monitoring.drill import (
    DEPTH_BREAK_S,
    FRESHNESS_LIMITS_S,
    GAP_TIMES_S,
    run_drill,
)


@pytest.fixture(scope="module")
def drill():
    return run_drill(seed=42)


def test_drill_passes_overall(drill):
    assert drill.passed


@pytest.mark.parametrize("component", ["depth", "gaps", "pnl"])
def test_alert_fires_before_human_notices(drill, component):
    break_e = drill.first(component, "break")
    alert_e = drill.first(component, "alert")
    human_e = drill.first(component, "human")
    assert break_e is not None and alert_e is not None and human_e is not None
    assert break_e.ts <= alert_e.ts, "alert cannot precede its own fault"
    assert alert_e.ts < human_e.ts, "alert must beat the human-noticeable threshold"


def test_stale_stream_alert_is_prompt_and_correct(drill):
    alert_e = drill.first("depth", "alert")
    # fires within the freshness limit plus one tick of the stream dying
    assert alert_e.t_s - DEPTH_BREAK_S <= FRESHNESS_LIMITS_S["depth"] + 2.0
    assert "depth" in alert_e.label
    assert any(a.source == "freshness" and "depth" in a.message for a in drill.alerts)


def test_gap_burst_alert_is_prompt_and_correct(drill):
    alert_e = drill.first("gaps", "alert")
    assert alert_e.t_s <= GAP_TIMES_S[0] + 1.0  # first gap already alerts
    gap_alerts = [a for a in drill.alerts if a.source == "gaps"]
    assert any(a.key == "gap:agg_trade" for a in gap_alerts)
    assert any(a.key == "gap_burst:agg_trade" for a in gap_alerts)  # burst escalated


def test_pnl_divergence_alert_is_correct(drill):
    pnl_alerts = [a for a in drill.alerts if a.source == "pnl"]
    assert pnl_alerts and pnl_alerts[0].key == "pnl_divergence"
    assert "below" in pnl_alerts[0].message  # the drill injects a shortfall
    break_e = drill.first("pnl", "break")
    assert pnl_alerts[0].ts > break_e.ts


def test_dedup_rate_limits_repeats(drill):
    # a dead stream re-alerts every tick; the dedup sink must eat the repeats
    assert drill.suppressed > 0
    depth_alerts = [a for a in drill.alerts if a.key == "stale:depth"]
    # delivered at most once per cooldown window across the whole drill
    assert 1 <= len(depth_alerts) <= 3


def test_drill_is_deterministic():
    a = run_drill(seed=42)
    b = run_drill(seed=42)
    assert a.events == b.events
    assert a.alerts == b.alerts
    assert a.suppressed == b.suppressed
    assert (a.pnl_cum == b.pnl_cum).all()
