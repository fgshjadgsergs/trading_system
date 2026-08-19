"""Live PnL vs backtest expectation envelope.

The backtest promises daily PnL ~ (mean, std). Live cumulative PnL over t days
is expected at mean*t with standard deviation std*sqrt(t); the tracker alerts
when the cumulative divergence z-score leaves the +/- z_threshold band in
either direction (over-performing is a model mismatch too).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trading_system.core.timeutils import NS_PER_S
from trading_system.monitoring.alerts import Alert, Severity

NS_PER_DAY = 86_400 * NS_PER_S


@dataclass(frozen=True, slots=True)
class PnlStatus:
    ts: int
    t_days: float
    cum_pnl: float
    expected: float  # mean * t_days
    band: float  # z_threshold * std * sqrt(t_days)
    z: float
    breached: bool


class PnlTracker:
    """Cumulative live PnL vs the backtest's daily mean/std envelope."""

    def __init__(
        self,
        expected_daily_mean: float,
        expected_daily_std: float,
        z_threshold: float = 3.0,
        start_ts: int | None = None,
    ) -> None:
        if expected_daily_std <= 0:
            raise ValueError("expected_daily_std must be positive")
        if z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
        self.mean = expected_daily_mean
        self.std = expected_daily_std
        self.z_threshold = z_threshold
        self.start_ts = start_ts
        self.last_status: PnlStatus | None = None

    def observe(self, ts: int, cum_pnl: float) -> PnlStatus:
        """Feed the current cumulative live PnL (since tracker start)."""
        if self.start_ts is None:
            self.start_ts = ts
        t = max(0.0, (ts - self.start_ts) / NS_PER_DAY)
        expected = self.mean * t
        if t <= 0:
            z = 0.0
            band = 0.0
        else:
            sigma = self.std * math.sqrt(t)
            z = (cum_pnl - expected) / sigma
            band = self.z_threshold * sigma
        status = PnlStatus(
            ts=ts,
            t_days=t,
            cum_pnl=cum_pnl,
            expected=expected,
            band=band,
            z=z,
            breached=abs(z) >= self.z_threshold,
        )
        self.last_status = status
        return status

    def alert_for(self, status: PnlStatus) -> Alert | None:
        """Alert for a breached status; None inside the envelope."""
        if not status.breached:
            return None
        direction = "below" if status.z < 0 else "above"
        return Alert(
            severity=Severity.CRITICAL,
            source="pnl",
            message=(
                f"live PnL diverged {direction} backtest envelope: z={status.z:+.2f} "
                f"(cum {status.cum_pnl:.2f} vs expected {status.expected:.2f} "
                f"+/- {status.band:.2f}) after {status.t_days:.3f}d"
            ),
            ts=status.ts,
            key="pnl_divergence",
        )

    def envelope(self, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(expected, lower, upper) cumulative-PnL envelope over timestamps ts."""
        if self.start_ts is None:
            raise ValueError("tracker has no start_ts yet")
        t = np.maximum(0.0, (np.asarray(ts, dtype=np.float64) - self.start_ts) / NS_PER_DAY)
        expected = self.mean * t
        band = self.z_threshold * self.std * np.sqrt(t)
        return expected, expected - band, expected + band

    def reset(self) -> None:
        self.start_ts = None
        self.last_status = None
