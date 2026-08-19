"""Stream freshness tracking and gap alerting.

FreshnessTracker compares the last ts_recv per stream against per-stream age
limits (configs monitoring.freshness_limits_s); the clock is injectable so
drills and tests control time completely. GapMonitor turns collector gap
events into alerts and escalates bursts.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from trading_system.core.timeutils import NS_PER_S
from trading_system.monitoring.alerts import Alert, Severity


@dataclass(frozen=True, slots=True)
class StaleStream:
    stream: str
    age_s: float
    limit_s: float
    last_ts: int | None  # None == never seen since tracker start


class FreshnessTracker:
    """Last ts_recv per stream vs per-stream limits -> stale streams."""

    def __init__(
        self,
        limits_s: Mapping[str, float],
        clock: Callable[[], int] | None = None,
        start_ts: int | None = None,
    ) -> None:
        self.limits_s = dict(limits_s)
        self._clock = clock
        self._start_ts = start_ts
        self._last: dict[str, int] = {}

    def _now(self, now_ts: int | None) -> int:
        if now_ts is not None:
            return now_ts
        if self._clock is None:
            raise ValueError("no now_ts given and no clock injected")
        return self._clock()

    def observe(self, stream: str, ts_recv: int) -> None:
        """Note a message received on a stream at ts_recv (UTC ns)."""
        if self._start_ts is None:
            self._start_ts = ts_recv
        cur = self._last.get(stream)
        if cur is None or ts_recv > cur:
            self._last[stream] = ts_recv

    def last_ts(self, stream: str) -> int | None:
        return self._last.get(stream)

    def ages_s(self, now_ts: int | None = None) -> dict[str, float]:
        """Age in seconds per configured stream; never-seen streams age from start."""
        now = self._now(now_ts)
        if self._start_ts is None:
            self._start_ts = now
        out: dict[str, float] = {}
        for stream in self.limits_s:
            base = self._last.get(stream, self._start_ts)
            out[stream] = max(0.0, (now - base) / NS_PER_S)
        return out

    def stale(self, now_ts: int | None = None) -> list[StaleStream]:
        """Streams whose age exceeds their limit, worst first."""
        now = self._now(now_ts)
        out = [
            StaleStream(
                stream=s, age_s=age, limit_s=self.limits_s[s], last_ts=self._last.get(s)
            )
            for s, age in self.ages_s(now).items()
            if age > self.limits_s[s]
        ]
        out.sort(key=lambda x: x.age_s / x.limit_s, reverse=True)
        return out

    def alerts(self, now_ts: int | None = None) -> list[Alert]:
        """One WARNING (or CRITICAL at 3x limit) alert per currently stale stream."""
        now = self._now(now_ts)
        alerts = []
        for s in self.stale(now):
            sev = Severity.CRITICAL if s.age_s > 3 * s.limit_s else Severity.WARNING
            alerts.append(
                Alert(
                    severity=sev,
                    source="freshness",
                    message=f"stream {s.stream} stale: {s.age_s:.1f}s > {s.limit_s:.0f}s limit",
                    ts=now,
                    key=f"stale:{s.stream}",
                )
            )
        return alerts

    def reset(self) -> None:
        self._last.clear()
        self._start_ts = None


@dataclass(frozen=True, slots=True)
class GapEvent:
    """A detected data gap on a stream: [ts_start, ts_end] with nothing between."""

    stream: str
    ts_start: int
    ts_end: int

    @property
    def duration_s(self) -> float:
        return (self.ts_end - self.ts_start) / NS_PER_S


class GapMonitor:
    """Alert on gap events; escalate to CRITICAL on a burst per stream."""

    def __init__(
        self,
        min_gap_s: float = 1.0,
        burst_n: int = 3,
        burst_window_s: float = 300.0,
    ) -> None:
        self.min_gap_s = min_gap_s
        self.burst_n = burst_n
        self.burst_window_ns = int(burst_window_s * NS_PER_S)
        self._recent: dict[str, deque[int]] = {}

    def on_gap(self, event: GapEvent) -> list[Alert]:
        """Feed one gap event; returns the alerts it raises (possibly empty)."""
        if event.duration_s < self.min_gap_s:
            return []
        q = self._recent.setdefault(event.stream, deque())
        q.append(event.ts_end)
        while q and event.ts_end - q[0] > self.burst_window_ns:
            q.popleft()
        alerts = [
            Alert(
                severity=Severity.WARNING,
                source="gaps",
                message=f"gap on {event.stream}: {event.duration_s:.1f}s",
                ts=event.ts_end,
                key=f"gap:{event.stream}",
            )
        ]
        if len(q) >= self.burst_n:
            alerts.append(
                Alert(
                    severity=Severity.CRITICAL,
                    source="gaps",
                    message=(
                        f"gap burst on {event.stream}: {len(q)} gaps within "
                        f"{self.burst_window_ns / NS_PER_S:.0f}s"
                    ),
                    ts=event.ts_end,
                    key=f"gap_burst:{event.stream}",
                )
            )
        return alerts

    def reset(self) -> None:
        self._recent.clear()
