"""M11 stress: freshness under clock anomalies, alert floods, PnL at 1e5 scale,
dashboard validator on garbage, drill cascades.

Scenarios (scaled by env STRESS_SCALE, default 1):
- freshness: clocks jumping backwards, data stamped in the future, `now`
  before tracker start — status always defined, never an exception; 1e5
  observations under a bounded elapsed ceiling;
- gap monitor: 10k gap events including zero/negative durations and identical
  timestamps — alerts well-formed, burst window bounded;
- alerts: floods of 10k identical alerts (including identical ts) through the
  dedup sink — exactly one delivery per cooldown window, exact suppression
  accounting, distinct keys unaffected;
- pnl tracker: 1e5 fills with zero quantities and positions flipping through
  zero — the tracked cumulative PnL matches an independent mark-to-market
  recomputation and the round-trip accounting of trades_from_fills;
- dashboard validator: empty state and 10k-panel volumes of valid and broken
  panels, plus structurally hostile values — problems reported, no crash;
- drill: cascaded runs back to back and shared components reset between
  scripted scenarios — the system returns to a clean state every time.

Seeded, offline, no sleeps; only coarse elapsed ceilings on heavy loops.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest

from trading_system.backtest.engine import Fill
from trading_system.backtest.metrics import trades_from_fills
from trading_system.core.schema import Side
from trading_system.monitoring.alerts import Alert, DedupSink, ListSink, Severity
from trading_system.monitoring.dashboard import validate_dashboard
from trading_system.monitoring.drill import run_drill
from trading_system.monitoring.freshness import FreshnessTracker, GapEvent, GapMonitor
from trading_system.monitoring.pnl_tracker import NS_PER_DAY, PnlTracker

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
SEED = 42
NS_PER_S = 1_000_000_000
T0 = 1_755_600_000 * NS_PER_S
LIMITS = {"depth": 5.0, "agg_trade": 10.0, "mark_price": 5.0}


def n_scaled(base: int) -> int:
    return max(1, int(base * SCALE))


def T(seconds: float) -> int:
    return T0 + int(seconds * NS_PER_S)


# --------------------------------------------------------------------------
# 1) Freshness: clock anomalies and 1e5-observation throughput
# --------------------------------------------------------------------------


def test_freshness_clock_jumping_backwards_keeps_status_defined():
    tr = FreshnessTracker(LIMITS, start_ts=T0)
    tr.observe("depth", T(100))
    tr.observe("depth", T(40))  # late/rewound tick must not rewind freshness
    assert tr.last_ts("depth") == T(100)
    # `now` earlier than the last observation: age clamps to zero, no negative
    ages = tr.ages_s(T(50))
    assert ages["depth"] == 0.0
    assert all(a >= 0.0 for a in ages.values())
    stale = tr.stale(T(50))  # never-seen streams age from start: well-formed
    assert {s.stream for s in stale} <= set(LIMITS)
    for s in stale:
        assert s.age_s >= 0.0 and s.last_ts is None
    # `now` before the tracker even started
    assert all(a >= 0.0 for a in tr.ages_s(T0 - 60 * NS_PER_S).values())


def test_freshness_data_from_the_future_never_goes_negative_or_raises():
    tr = FreshnessTracker(LIMITS, start_ts=T0)
    tr.observe("depth", T(10_000))  # producer clock is far ahead
    tr.observe("agg_trade", T(1))
    ages = tr.ages_s(T(100))
    assert ages["depth"] == 0.0  # future data reads as perfectly fresh
    assert ages["agg_trade"] == pytest.approx(99.0)
    stale = tr.stale(T(100))
    # depth (future-stamped) is fresh; agg_trade is 99s old; mark_price was
    # never seen and ages from tracker start
    assert {s.stream for s in stale} == {"agg_trade", "mark_price"}
    alerts = tr.alerts(T(100))
    assert {a.key for a in alerts} == {"stale:agg_trade", "stale:mark_price"}
    assert all(a.severity in (Severity.WARNING, Severity.CRITICAL) for a in alerts)


def test_freshness_1e5_observations_bounded_time():
    tr = FreshnessTracker(LIMITS, start_ts=T0)
    rng = np.random.default_rng(SEED)
    n = n_scaled(100_000)
    streams = list(LIMITS)
    picks = rng.integers(0, len(streams), n)
    jitter = rng.integers(-2 * NS_PER_S, 2 * NS_PER_S, n)  # out-of-order arrivals
    t_start = time.perf_counter()
    for i in range(n):
        tr.observe(streams[int(picks[i])], T0 + i * 1_000_000 + int(jitter[i]))
        if i % 1000 == 0:
            tr.stale(T0 + i * 1_000_000)
    elapsed = time.perf_counter() - t_start
    assert elapsed < 30.0, f"1e5 observations took {elapsed:.1f}s"
    ages = tr.ages_s(T0 + n * 1_000_000)
    assert all(0.0 <= a < 10.0 for a in ages.values())  # every stream stayed live


def test_gap_monitor_storm_10k_events_incl_degenerate():
    gm = GapMonitor(min_gap_s=1.0, burst_n=3, burst_window_s=300.0)
    rng = np.random.default_rng(SEED)
    n = n_scaled(10_000)
    n_alerts = n_criticals = 0
    t_start = time.perf_counter()
    for i in range(n):
        end = T(float(i))
        dur = float(rng.choice([-5.0, 0.0, 0.5, 2.0, 30.0]))
        alerts = gm.on_gap(GapEvent("agg_trade", end - int(dur * NS_PER_S), end))
        if dur < 1.0:
            assert alerts == []  # too short / degenerate: ignored
        else:
            assert 1 <= len(alerts) <= 2
            n_alerts += len(alerts)
            n_criticals += sum(1 for a in alerts if a.severity is Severity.CRITICAL)
        for a in alerts:
            assert a.ts == end and a.source == "gaps"
    elapsed = time.perf_counter() - t_start
    assert elapsed < 30.0
    assert n_alerts > 0 and n_criticals > 0  # bursts escalated during the storm
    # the per-stream burst deque stays bounded by the window, not the storm size
    assert len(gm._recent["agg_trade"]) <= 301


# --------------------------------------------------------------------------
# 2) Alerts: 10k-identical floods and identical-ts storms
# --------------------------------------------------------------------------


def _alert(t_s: float, key: str = "stale:depth") -> Alert:
    return Alert(
        severity=Severity.WARNING,
        source="freshness",
        message="stream depth stale",
        ts=T(t_s),
        key=key,
    )


def test_flood_10k_identical_alerts_is_rate_limited_exactly():
    inner = ListSink()
    dedup = DedupSink(inner, cooldown_s=300.0)
    n = n_scaled(10_000)
    t_start = time.perf_counter()
    delivered = sum(1 for i in range(n) if dedup.emit(_alert(float(i))))  # 1s apart
    elapsed = time.perf_counter() - t_start
    assert elapsed < 20.0
    expected = math.ceil(n / 300)  # one delivery per full cooldown window
    assert delivered == expected
    assert len(inner.alerts) == expected
    assert dedup.suppressed == n - expected


def test_flood_of_alerts_with_identical_ts_delivers_exactly_one():
    inner = ListSink()
    dedup = DedupSink(inner, cooldown_s=300.0)
    n = n_scaled(10_000)
    results = [dedup.emit(_alert(0.0)) for _ in range(n)]
    assert results[0] is True
    assert not any(results[1:])
    assert len(inner.alerts) == 1
    assert dedup.suppressed == n - 1


def test_distinct_keys_all_pass_through_the_flood():
    inner = ListSink()
    dedup = DedupSink(inner, cooldown_s=300.0)
    n = n_scaled(10_000)
    for i in range(n):
        assert dedup.emit(_alert(0.0, key=f"k{i}")) is True
    assert len(inner.alerts) == n
    assert dedup.suppressed == 0
    # and a repeat of every key at the same ts is fully suppressed
    for i in range(n):
        assert dedup.emit(_alert(0.0, key=f"k{i}")) is False
    assert dedup.suppressed == n


# --------------------------------------------------------------------------
# 3) PnL tracker: 1e5 fills, zero quantities, flips through zero
# --------------------------------------------------------------------------


def test_pnl_1e5_trades_matches_independent_recomputation():
    n = max(10_000, n_scaled(100_000))  # floor keeps flips-through-zero plentiful
    rng = np.random.default_rng(SEED)
    px = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 5e-4, n)))
    # signed deltas with zero quantities mixed in; the running position
    # repeatedly crosses zero (long -> short and back)
    dq = np.round(rng.normal(0.0, 1.0, n), 3)
    dq[rng.random(n) < 0.10] = 0.0
    pos = np.cumsum(dq)
    assert (np.sign(pos[:-1]) * np.sign(pos[1:]) < 0).sum() > 10  # real flips

    # independent recompute #1: vectorized mark-to-market
    cum = np.concatenate([[0.0], np.cumsum(pos[:-1] * np.diff(px))])

    # independent recompute #2: average-entry round trips + unrealized tail
    fills = [
        Fill(
            order_id=i,
            ts=T(float(i)),
            side=Side.BUY if dq[i] >= 0 else Side.SELL,
            qty=abs(float(dq[i])),
            price=float(px[i]),
            ref_mid=float(px[i]),
            maker=False,
            fee_usd=0.0,
            slippage_usd=0.0,
        )
        for i in range(n)
    ]
    trades = trades_from_fills(fills)
    assert (trades["qty"] > 0).all()  # no phantom zero-qty rows (fix regression)
    realized = float(trades["pnl_usd"].sum())
    final_pos = float(pos[-1])
    # reconstruct the open average entry the same way the trade builder does
    p, avg = 0.0, 0.0
    for i in range(n):
        q = float(dq[i])
        if q == 0.0:
            continue
        if abs(p) < 1e-12 or p * q > 0:
            avg = float(px[i]) if abs(p) < 1e-12 else (avg * abs(p) + px[i] * abs(q)) / (abs(p) + abs(q))
            p += q
        else:
            new_p = p + q
            if abs(new_p) < 1e-12:
                p = 0.0
            elif new_p * p < 0:
                p, avg = new_p, float(px[i])
            else:
                p = new_p
    unrealized = (float(px[-1]) - avg) * p
    assert p == pytest.approx(final_pos, abs=1e-6)
    assert realized + unrealized == pytest.approx(float(cum[-1]), abs=max(1.0, abs(cum[-1]) * 1e-6))

    # feed the mark-to-market stream through the tracker at 1e5 scale
    tracker = PnlTracker(expected_daily_mean=50.0, expected_daily_std=100.0, start_ts=T0)
    ts = T0 + (np.arange(n, dtype=np.int64) + 1) * (NS_PER_DAY // (2 * n))  # half a day
    t_start = time.perf_counter()
    statuses = [tracker.observe(int(ts[i]), float(cum[i])) for i in range(n)]
    elapsed = time.perf_counter() - t_start
    assert elapsed < 30.0, f"1e5 observations took {elapsed:.1f}s"

    for i in rng.integers(1, n, 200):  # sampled independent z recomputation
        st = statuses[int(i)]
        t_days = (int(ts[i]) - T0) / NS_PER_DAY
        sigma = 100.0 * math.sqrt(t_days)
        z = (cum[i] - 50.0 * t_days) / sigma
        assert st.t_days == pytest.approx(t_days)
        assert st.z == pytest.approx(z, rel=1e-9, abs=1e-9)
        assert st.breached == (abs(z) >= 3.0)
        assert math.isfinite(st.z) and math.isfinite(st.band)
        alert = tracker.alert_for(st)
        assert (alert is not None) == st.breached


def test_pnl_observation_before_start_is_defined_not_breached():
    tracker = PnlTracker(50.0, 100.0, start_ts=T0)
    st = tracker.observe(T0 - NS_PER_DAY, 1e9)  # clock jumped before start
    assert st.t_days == 0.0 and st.z == 0.0 and not st.breached
    assert tracker.alert_for(st) is None


# --------------------------------------------------------------------------
# 4) Dashboard validator: empty state, hostile values, 10k-panel volume
# --------------------------------------------------------------------------


def test_validator_on_completely_empty_state():
    problems = validate_dashboard({})
    assert problems  # everything required is reported missing
    assert any("title" in p for p in problems)
    assert any("panels" in p for p in problems)
    # and reporting is pure: a second call returns the same problems
    assert validate_dashboard({}) == problems


@pytest.mark.parametrize(
    "panels",
    ["oops", {"a": 1}, [1, "x", None], [[]], 42, None],
    ids=["str", "dict", "mixed-list", "nested-list", "int", "none"],
)
def test_validator_never_crashes_on_malformed_panels(panels):
    problems = validate_dashboard({"title": "t", "uid": "u", "schemaVersion": 1, "panels": panels})
    assert any("panels" in p for p in problems)


def test_validator_survives_hostile_panel_values():
    dashboard = {
        "title": None,
        "uid": 1,
        "schemaVersion": "x",
        "panels": [
            {"title": float("nan"), "id": None},  # NaN/None where strings belong
            {"title": "", "id": 0, "type": "", "targets": [], "gridPos": {}},
            {},  # nothing at all
        ],
    }
    problems = validate_dashboard(dashboard)
    assert problems and all(isinstance(p, str) for p in problems)


def test_validator_10k_panels_valid_and_broken():
    n = max(3, n_scaled(10_000))  # at least the three required panels
    required = [
        "Stream freshness age (s)",
        "Gap events per stream",
        "PnL divergence z-score",
    ]

    def panel(i: int, title: str, complete: bool = True) -> dict:
        p = {"id": i, "type": "timeseries", "title": title, "targets": [], "gridPos": {}}
        if not complete:
            del p["targets"]
        return p

    valid = {
        "title": "t",
        "uid": "u",
        "schemaVersion": 39,
        "panels": [panel(i, required[i % 3] if i < 3 else f"p{i}") for i in range(n)],
    }
    t_start = time.perf_counter()
    assert validate_dashboard(valid) == []
    broken = dict(valid)
    broken["panels"] = [panel(i, f"p{i}", complete=False) for i in range(n)]
    problems = validate_dashboard(broken)
    elapsed = time.perf_counter() - t_start
    assert elapsed < 30.0
    missing_targets = [p for p in problems if "missing targets" in p]
    assert len(missing_targets) == n  # one finding per broken panel, none dropped


# --------------------------------------------------------------------------
# 5) Drill cascade: clean state between runs
# --------------------------------------------------------------------------


def test_drill_cascade_is_reproducible_after_other_drills():
    first = run_drill(seed=42)
    for seed in (7, 8, 9):  # unrelated drills in between
        mid = run_drill(seed=seed)
        assert mid.events and mid.alerts
    again = run_drill(seed=42)
    assert again.events == first.events
    assert again.alerts == first.alerts
    assert again.suppressed == first.suppressed
    assert (again.pnl_cum == first.pnl_cum).all()
    assert again.passed and first.passed


def _scripted_scenario(
    tracker: FreshnessTracker, gaps: GapMonitor, dedup: DedupSink, pnl: PnlTracker
) -> tuple[tuple, int]:
    """One deterministic mini-drill on shared components; returns its outputs."""
    delivered = []
    for i in range(120):
        now = T(float(i))
        if i < 30:
            tracker.observe("depth", now)
        tracker.observe("agg_trade", now)
        tracker.observe("mark_price", now)
        for alert in tracker.alerts(now):
            if dedup.emit(alert):
                delivered.append(alert)
        if i in (60, 70, 80):
            for alert in gaps.on_gap(GapEvent("agg_trade", now - 4 * NS_PER_S, now)):
                if dedup.emit(alert):
                    delivered.append(alert)
        st = pnl.observe(now, -40.0 * i)
        alert = pnl.alert_for(st)
        if alert is not None and dedup.emit(alert):
            delivered.append(alert)
    return tuple(delivered), dedup.suppressed


def test_shared_components_return_to_clean_state_after_reset():
    sink = ListSink()
    dedup = DedupSink(sink, cooldown_s=300.0)
    tracker = FreshnessTracker(LIMITS, start_ts=T0)
    gaps = GapMonitor(min_gap_s=2.0, burst_n=3, burst_window_s=60.0)
    pnl = PnlTracker(50.0, 100.0, start_ts=T0)

    first, first_suppressed = _scripted_scenario(tracker, gaps, dedup, pnl)
    assert first and first_suppressed > 0  # the scenario really fired and deduped

    tracker.reset()
    gaps.reset()
    dedup.reset()
    pnl.reset()
    # clean state: nothing remembered anywhere
    assert all(tracker.last_ts(s) is None for s in LIMITS)
    assert gaps._recent == {}
    assert dedup.suppressed == 0
    assert pnl.start_ts is None and pnl.last_status is None

    pnl.start_ts = T0  # re-arm exactly as constructed
    tracker._start_ts = T0
    second, second_suppressed = _scripted_scenario(tracker, gaps, dedup, pnl)
    assert second == first  # byte-identical rerun: no residue leaked through
    assert second_suppressed == first_suppressed
    assert sink.alerts == list(first) + list(second)  # sink kept both runs in order
