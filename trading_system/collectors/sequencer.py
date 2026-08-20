"""Binance futures depth sequencing: snapshot + U/u/pu contiguity enforcement.

Protocol (Binance USDT-M "How to manage a local order book correctly"):
1. Buffer depth diffs while waiting for a REST snapshot.
2. Drop buffered events whose final_update_id (u) <= snapshot last_update_id.
3. The first applied event must straddle the snapshot id:
   first_update_id (U) <= last_update_id <= final_update_id (u).
4. Every subsequent event must have prev_final_update_id (pu) == previous u.

Any violation emits a GapEvent and requests a resync via callback — the
sequencer never silently continues over a hole in the stream.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import structlog

from trading_system.core.schema import BookSnapshot, DepthDiff

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class GapEvent:
    """Sequence violation: expected/got are update-ids at the break point."""

    symbol: str
    ts: int  # ts_event (UTC ns) of the violating diff
    expected: int  # last applied final_update_id (or snapshot id)
    got: int  # violating diff's pu (stream break) or U (snapshot too old)


class DepthSequencer:
    """Order-book diff gatekeeper for one symbol.

    add_diff/set_snapshot return the diffs that are safe to apply, in order.
    Actual book reconstruction lives in M2; this class only guarantees that
    what it releases is gap-free relative to the accepted snapshot.
    """

    def __init__(
        self,
        symbol: str,
        *,
        on_gap: Callable[[GapEvent], None] | None = None,
        on_resync: Callable[[], None] | None = None,
        max_buffer: int = 100_000,
    ) -> None:
        self.symbol = symbol
        self._on_gap = on_gap
        self._on_resync = on_resync
        self._max_buffer = max_buffer
        self._buffer: deque[DepthDiff] = deque()  # deque: O(1) drop-oldest on overflow
        self._synced = False
        self._awaiting_first = False
        self._last_u = -1
        self._snapshot: BookSnapshot | None = None
        self.gaps: list[GapEvent] = []

    @property
    def synced(self) -> bool:
        return self._synced

    @property
    def snapshot(self) -> BookSnapshot | None:
        """The accepted snapshot the released diff stream is relative to."""
        return self._snapshot

    @property
    def buffered(self) -> int:
        return len(self._buffer)

    def add_diff(self, diff: DepthDiff) -> list[DepthDiff]:
        """Feed one diff; returns [diff] when contiguous, [] otherwise."""
        if not self._synced:
            self._buffer_diff(diff)
            return []
        return self._process(diff)

    def set_snapshot(self, snap: BookSnapshot) -> list[DepthDiff]:
        """Accept a REST snapshot and drain the buffer through sequencing rules."""
        if snap.symbol != self.symbol:
            raise ValueError(f"snapshot for {snap.symbol}, sequencer for {self.symbol}")
        self._snapshot = snap
        self._last_u = snap.last_update_id
        self._synced = True
        self._awaiting_first = True
        pending = list(self._buffer)
        self._buffer.clear()
        ready: list[DepthDiff] = []
        for i, d in enumerate(pending):
            ready.extend(self._process(d))
            if not self._synced:
                # _process buffered the violating diff; keep the rest in order
                self._buffer.extend(pending[i + 1 :])
                break
        return ready

    # ------------------------------------------------------------------ #
    def _buffer_diff(self, diff: DepthDiff) -> None:
        if len(self._buffer) >= self._max_buffer:
            dropped = self._buffer.popleft()
            log.warning(
                "seq_buffer_overflow", symbol=self.symbol, dropped_u=dropped.final_update_id
            )
        self._buffer.append(diff)

    def _process(self, diff: DepthDiff) -> list[DepthDiff]:
        if diff.final_update_id <= self._last_u:
            return []  # entirely covered by the snapshot / already applied
        if self._awaiting_first:
            if diff.first_update_id <= self._last_u or diff.prev_final_update_id == self._last_u:
                self._awaiting_first = False
                self._last_u = diff.final_update_id
                return [diff]
            # snapshot too old: a hole exists between snapshot id and this diff
            self._gap(diff, expected=self._last_u, got=diff.first_update_id)
            return []
        if diff.prev_final_update_id == self._last_u:
            self._last_u = diff.final_update_id
            return [diff]
        self._gap(diff, expected=self._last_u, got=diff.prev_final_update_id)
        return []

    def _gap(self, diff: DepthDiff, *, expected: int, got: int) -> None:
        ev = GapEvent(symbol=self.symbol, ts=diff.ts_event, expected=expected, got=got)
        self.gaps.append(ev)
        self._synced = False
        self._awaiting_first = False
        self._buffer_diff(diff)  # still valid data once a fresh snapshot arrives
        log.warning("depth_gap", symbol=self.symbol, expected=expected, got=got)
        if self._on_gap is not None:
            self._on_gap(ev)
        if self._on_resync is not None:
            self._on_resync()
