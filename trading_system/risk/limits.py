"""Hard risk limits: daily stop and kill switch.

Both are pure state machines: every decision is driven by timestamps passed in
(UTC ns), never by the wall clock, and both expose an explicit reset().
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from trading_system.core.timeutils import NS_PER_S

NS_PER_DAY = 86_400 * NS_PER_S


def utc_day_key(ts: int) -> int:
    """Day index with the boundary at UTC midnight (epoch is UTC midnight)."""
    return ts // NS_PER_DAY


@dataclass(frozen=True, slots=True)
class HaltState:
    """Result of a limit check: whether trading must halt and why."""

    halted: bool
    reason: str | None = None
    ts: int | None = None


@dataclass(frozen=True, slots=True)
class KillState:
    """Kill-switch verdict: tripped implies a flatten intent plus halt."""

    tripped: bool
    flatten: bool = False
    reason: str | None = None
    ts: int | None = None


class DailyStop:
    """Halt when running day PnL <= -daily_stop_pct * day-start equity.

    The day boundary is UTC midnight derived from the event ts via an
    injectable day_key function. The halt is sticky for the rest of the day
    (equity recovering does not un-halt); a new day or reset() clears it.
    """

    def __init__(
        self,
        daily_stop_pct: float,
        day_key: Callable[[int], int] = utc_day_key,
    ) -> None:
        if daily_stop_pct <= 0:
            raise ValueError("daily_stop_pct must be positive")
        self.daily_stop_pct = daily_stop_pct
        self._day_key = day_key
        self._day: int | None = None
        self._day_start_equity: float | None = None
        self._state = HaltState(halted=False)

    @property
    def day_start_equity(self) -> float | None:
        return self._day_start_equity

    @property
    def state(self) -> HaltState:
        return self._state

    def update(self, ts: int, equity: float) -> HaltState:
        """Feed the current equity mark; returns the halt state after this tick."""
        day = self._day_key(ts)
        if self._day is None or day != self._day:
            self._day = day
            self._day_start_equity = equity
            self._state = HaltState(halted=False)
            return self._state
        if self._state.halted:
            return self._state
        assert self._day_start_equity is not None
        pnl = equity - self._day_start_equity
        if pnl <= -self.daily_stop_pct * self._day_start_equity:
            self._state = HaltState(
                halted=True,
                reason=(
                    f"daily stop: day pnl {pnl:.2f} <= "
                    f"-{self.daily_stop_pct:.4f} * {self._day_start_equity:.2f}"
                ),
                ts=ts,
            )
        return self._state

    def reset(self) -> None:
        """Forget the day baseline and any halt."""
        self._day = None
        self._day_start_equity = None
        self._state = HaltState(halted=False)


class KillSwitch:
    """Trip on n consecutive errors OR stale market data (ts gap).

    Tripping produces a flatten intent + halt. The trip is sticky until
    reset(). Staleness is measured against the last record_market_data ts; the
    first check() with no data seen arms the baseline at that check's ts, so
    an idle switch does not trip retroactively at startup.
    """

    def __init__(self, max_consecutive_errors: int, stale_after_s: float) -> None:
        if max_consecutive_errors < 1:
            raise ValueError("max_consecutive_errors must be >= 1")
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        self.max_consecutive_errors = max_consecutive_errors
        self.stale_after_ns = int(stale_after_s * NS_PER_S)
        self._consecutive_errors = 0
        self._last_md_ts: int | None = None
        self._state = KillState(tripped=False)

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    @property
    def state(self) -> KillState:
        return self._state

    def record_market_data(self, ts: int) -> None:
        """Note a market-data heartbeat (any stream tick counts)."""
        if self._last_md_ts is None or ts > self._last_md_ts:
            self._last_md_ts = ts

    def record_success(self, ts: int) -> None:
        """A successful operation breaks the consecutive-error run."""
        self._consecutive_errors = 0

    def record_error(self, ts: int) -> KillState:
        """Count one error; trips at max_consecutive_errors in a row."""
        if self._state.tripped:
            return self._state
        self._consecutive_errors += 1
        if self._consecutive_errors >= self.max_consecutive_errors:
            self._state = KillState(
                tripped=True,
                flatten=True,
                reason=f"{self._consecutive_errors} consecutive errors",
                ts=ts,
            )
        return self._state

    def check(self, now_ts: int) -> KillState:
        """Evaluate staleness at now_ts; returns the current verdict."""
        if self._state.tripped:
            return self._state
        if self._last_md_ts is None:
            self._last_md_ts = now_ts  # arm baseline; nothing to judge yet
            return self._state
        gap = now_ts - self._last_md_ts
        if gap > self.stale_after_ns:
            self._state = KillState(
                tripped=True,
                flatten=True,
                reason=f"stale market data: {gap / NS_PER_S:.1f}s without a tick",
                ts=now_ts,
            )
        return self._state

    def reset(self) -> None:
        """Explicit re-arm after human intervention."""
        self._consecutive_errors = 0
        self._last_md_ts = None
        self._state = KillState(tripped=False)
