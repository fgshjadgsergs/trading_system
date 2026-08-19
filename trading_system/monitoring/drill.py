"""Monitoring drill: deliberately break each component on a fake clock.

One scripted scenario breaks the depth stream (silence), injects a gap burst
on agg_trade and makes live PnL drift away from the backtest envelope. The
drill records, per component, when the fault was injected ("break"), when the
first alert fired ("alert") and when a human would plausibly notice ("human").
The monitoring exam passes when every alert precedes its human threshold.
Everything runs on synthetic time — no wall clock, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from trading_system.core.timeutils import NS_PER_S
from trading_system.monitoring.alerts import Alert, DedupSink, ListSink
from trading_system.monitoring.freshness import FreshnessTracker, GapEvent, GapMonitor
from trading_system.monitoring.pnl_tracker import NS_PER_DAY, PnlTracker

START_TS = 1_755_600_000 * NS_PER_S  # matches core.synth's synthetic day

FRESHNESS_LIMITS_S = {"depth": 5.0, "agg_trade": 10.0, "mark_price": 5.0}
DEPTH_BREAK_S = 120.0
GAP_TIMES_S = (240.0, 252.0, 264.0)  # gap end times; each gap lasts GAP_LEN_S
GAP_LEN_S = 4.0
STREAM_DRILL_S = 600.0  # freshness/gap phase length
HUMAN_NOTICE_STREAM_S = 600.0  # a dead stream is human-obvious within 10 min

PNL_MEAN_DAILY = 50.0
PNL_STD_DAILY = 100.0
PNL_BREAK_DAYS = 1.0 / 24.0  # drift starts after one clean hour
PNL_DRIFT_DAILY = -1200.0  # live shortfall vs expectation after the break
PNL_HUMAN_LOSS = 3.0 * PNL_STD_DAILY  # shortfall a human notices on the PnL page
PNL_SAMPLE_DAYS = 1.0 / 288.0  # 5-minute samples
PNL_TOTAL_DAYS = 0.5


@dataclass(frozen=True, slots=True)
class DrillEvent:
    ts: int
    t_s: float  # seconds since drill start
    component: str  # "depth" | "gaps" | "pnl"
    kind: str  # "break" | "alert" | "human"
    label: str


@dataclass
class DrillResult:
    start_ts: int
    events: list[DrillEvent] = field(default_factory=list)
    alerts: list[Alert] = field(default_factory=list)
    suppressed: int = 0
    freshness_t_s: np.ndarray = field(default_factory=lambda: np.empty(0))
    freshness_ages: dict[str, np.ndarray] = field(default_factory=dict)
    pnl_t_days: np.ndarray = field(default_factory=lambda: np.empty(0))
    pnl_cum: np.ndarray = field(default_factory=lambda: np.empty(0))
    pnl_tracker: PnlTracker | None = None

    def first(self, component: str, kind: str) -> DrillEvent | None:
        for e in self.events:
            if e.component == component and e.kind == kind:
                return e
        return None

    def alert_before_human(self, component: str) -> bool:
        alert = self.first(component, "alert")
        human = self.first(component, "human")
        return alert is not None and human is not None and alert.ts < human.ts

    @property
    def passed(self) -> bool:
        return all(self.alert_before_human(c) for c in ("depth", "gaps", "pnl"))


def run_drill(seed: int = 42, dedup_cooldown_s: float = 300.0) -> DrillResult:
    """Run the full scripted drill; deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    res = DrillResult(start_ts=START_TS)
    sink = ListSink()
    dedup = DedupSink(sink, cooldown_s=dedup_cooldown_s)

    def note(t_s: float, component: str, kind: str, label: str) -> None:
        res.events.append(
            DrillEvent(ts=START_TS + int(t_s * NS_PER_S), t_s=t_s, component=component, kind=kind, label=label)
        )

    # --- phase 1: stream freshness + gap burst, 1s ticks --------------------
    tracker = FreshnessTracker(FRESHNESS_LIMITS_S, start_ts=START_TS)
    gaps = GapMonitor(min_gap_s=2.0, burst_n=3, burst_window_s=60.0)
    note(DEPTH_BREAK_S, "depth", "break", "depth stream goes silent")
    note(DEPTH_BREAK_S + HUMAN_NOTICE_STREAM_S, "depth", "human", "operator opens the dashboard")
    note(GAP_TIMES_S[0] - GAP_LEN_S, "gaps", "break", "gap burst starts on agg_trade")
    note(GAP_TIMES_S[0] + HUMAN_NOTICE_STREAM_S, "gaps", "human", "operator notices choppy data")

    t_grid: list[float] = []
    ages: dict[str, list[float]] = {s: [] for s in FRESHNESS_LIMITS_S}
    depth_alerted = gaps_alerted = False
    n_steps = int(STREAM_DRILL_S)
    for i in range(n_steps + 1):
        t_s = float(i)
        now = START_TS + int(t_s * NS_PER_S)
        in_gap = any(g - GAP_LEN_S < t_s <= g for g in GAP_TIMES_S)
        if t_s < DEPTH_BREAK_S:
            tracker.observe("depth", now)
        if not in_gap:
            tracker.observe("agg_trade", now)
        tracker.observe("mark_price", now)
        for stream, age in tracker.ages_s(now).items():
            ages[stream].append(age)
        t_grid.append(t_s)
        for alert in tracker.alerts(now):
            if dedup.emit(alert) and not depth_alerted and "depth" in alert.message:
                depth_alerted = True
                note(t_s, "depth", "alert", alert.message)
        if t_s in GAP_TIMES_S:
            event = GapEvent(
                stream="agg_trade",
                ts_start=now - int(GAP_LEN_S * NS_PER_S),
                ts_end=now,
            )
            for alert in gaps.on_gap(event):
                if dedup.emit(alert) and not gaps_alerted:
                    gaps_alerted = True
                    note(t_s, "gaps", "alert", alert.message)
    res.freshness_t_s = np.asarray(t_grid)
    res.freshness_ages = {s: np.asarray(v) for s, v in ages.items()}

    # --- phase 2: PnL divergence, 5-minute samples over half a day ----------
    pnl = PnlTracker(PNL_MEAN_DAILY, PNL_STD_DAILY, z_threshold=3.0, start_ts=START_TS)
    note(PNL_BREAK_DAYS * 86_400.0, "pnl", "break", "live edge decays: drift vs backtest begins")
    t_days = np.arange(0.0, PNL_TOTAL_DAYS + 1e-12, PNL_SAMPLE_DAYS)
    noise = rng.normal(0.0, 0.05 * PNL_STD_DAILY * np.sqrt(PNL_SAMPLE_DAYS), len(t_days))
    cum = np.empty(len(t_days))
    level = 0.0
    pnl_alerted = human_noted = False
    for i, t in enumerate(t_days):
        if i > 0:
            drift = PNL_MEAN_DAILY + (PNL_DRIFT_DAILY if t > PNL_BREAK_DAYS else 0.0)
            level += drift * PNL_SAMPLE_DAYS + noise[i]
        cum[i] = level
        ts = START_TS + int(t * NS_PER_DAY)
        status = pnl.observe(ts, level)
        if status.breached and not pnl_alerted:
            alert = pnl.alert_for(status)
            assert alert is not None
            if dedup.emit(alert):
                pnl_alerted = True
                note(t * 86_400.0, "pnl", "alert", alert.message)
        if not human_noted and level - status.expected <= -PNL_HUMAN_LOSS:
            human_noted = True
            note(t * 86_400.0, "pnl", "human", "shortfall big enough to catch the eye")
    res.pnl_t_days = t_days
    res.pnl_cum = cum
    res.pnl_tracker = pnl

    res.alerts = sink.alerts
    res.suppressed = dedup.suppressed
    res.events.sort(key=lambda e: e.ts)
    return res
