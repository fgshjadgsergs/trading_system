"""Time x price heat history for overlay heatmaps (assumes a stable bucket grid)."""

from __future__ import annotations

import bisect
import math

import numpy as np

from trading_system.core.schema import Side
from trading_system.liqmap.map import LiqMap


class HeatHistory:
    def __init__(self, liqmap: LiqMap) -> None:
        self._map = liqmap
        self._bucket_size = liqmap.buckets.bucket_size
        self.ts: list[int] = []
        self._frames: list[dict[int, float]] = []

    def record(self, ts: int) -> None:
        if self._map.buckets.bucket_size != self._bucket_size:
            raise ValueError("bucket grid changed mid-history; start a new HeatHistory")
        if self.ts and ts < self.ts[-1]:
            # frames are a time series: an out-of-order record would silently
            # break every at-time lookup (and the overlay's x axis)
            raise ValueError(f"ts {ts} precedes the last recorded {self.ts[-1]}")
        combined: dict[int, float] = {}
        for side in (Side.BUY, Side.SELL):
            for idx, h in self._map.heat[side].items():
                combined[idx] = combined.get(idx, 0.0) + h
        self.ts.append(ts)
        self._frames.append(combined)

    def __len__(self) -> int:
        return len(self._frames)

    def total_at(self, i: int) -> float:
        """Total heat (USD) in the snapshot recorded at index i."""
        # fsum: the readout must not depend on dict iteration order
        return math.fsum(self._frames[i].values())

    def index_at(self, ts: int, inclusive: bool = False) -> int | None:
        """Index of the last snapshot at-or-before (`inclusive`) / strictly
        before `ts`, or None when nothing was recorded yet.

        The causal accessor for consumers that live in time rather than in
        frame numbers: indexing frames by bar position silently desyncs as
        soon as one bar is not recorded. Use the default (strict) form to ask
        what the map knew when an external event happened — the event must not
        see the snapshot taken at its own instant — and `inclusive=True` for a
        signal firing at a bar close, which does see that bar's own snapshot.
        """
        i = (bisect.bisect_right if inclusive else bisect.bisect_left)(self.ts, ts) - 1
        return i if i >= 0 else None

    def pools_at_ts(self, ts: int, k: int = 8, inclusive: bool = False) -> list[tuple[float, float]]:
        """Top-k pools of the snapshot `index_at(ts, inclusive)` selects."""
        i = self.index_at(ts, inclusive)
        return [] if i is None else self.pools_at(i, k)

    def zones_at_ts(
        self, ts: int, inclusive: bool = False
    ) -> tuple[list[float], list[float], list[float]]:
        """Zones of the snapshot `index_at(ts, inclusive)` selects."""
        i = self.index_at(ts, inclusive)
        return ([], [], []) if i is None else self.zones_at(i)

    def pools_at(self, i: int, k: int = 8) -> list[tuple[float, float]]:
        """Top-k pools of snapshot i as (bucket center price, heat), descending.

        Snapshot i is the map state at the close of bar i — the causal view a
        signal firing at that close is allowed to see.
        """
        frame = self._frames[i]
        top = sorted(frame.items(), key=lambda kv: -kv[1])[:k]
        return [((idx + 0.5) * self._bucket_size, heat) for idx, heat in top]

    def zones_at(self, i: int) -> tuple[list[float], list[float], list[float]]:
        """Snapshot i as parallel (lo, hi, heat) lists for zone filters."""
        frame = self._frames[i]
        lo = [idx * self._bucket_size for idx in frame]
        hi = [(idx + 1) * self._bucket_size for idx in frame]
        return lo, hi, list(frame.values())

    def matrix(self, max_rows: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (ts[n], bucket_center_prices[m], H[m, n]) for imshow overlays.

        The grid is dense over the occupied index span, so a large price jump
        at a fine bucket size can blow it up (100 -> 100000 at 0.01 is ~12M
        rows). `max_rows` caps the span around the heaviest region: rows are
        trimmed from whichever end carries less mass, so the plotted window
        keeps the pools that matter.
        """
        if not self._frames:
            return np.array([]), np.array([]), np.zeros((0, 0))
        occupied = sorted({i for f in self._frames for i in f})
        if not occupied:  # кадры есть, но карта пуста (нулевой приток OI)
            return np.asarray(self.ts), np.array([]), np.zeros((0, len(self._frames)))
        lo, hi = occupied[0], occupied[-1]
        if max_rows is not None and hi - lo + 1 > max_rows:
            mass: dict[int, float] = {}
            for f in self._frames:
                for i, h in f.items():
                    mass[i] = mass.get(i, 0.0) + h
            # widen a window around the heaviest bucket, always taking the
            # richer neighbouring side, until max_rows is used up
            center = max(mass, key=lambda i: mass[i])
            lo = hi = center
            while hi - lo + 1 < max_rows:
                left = mass.get(lo - 1, 0.0) if lo - 1 >= occupied[0] else -1.0
                right = mass.get(hi + 1, 0.0) if hi + 1 <= occupied[-1] else -1.0
                if left < 0 and right < 0:
                    break
                if right >= left:
                    hi += 1
                else:
                    lo -= 1
        idx_range = np.arange(lo, hi + 1)
        H = np.zeros((len(idx_range), len(self._frames)))
        for j, f in enumerate(self._frames):
            for i, h in f.items():
                if lo <= i <= hi:
                    H[i - lo, j] = h
        prices = (idx_range + 0.5) * self._bucket_size
        return np.asarray(self.ts), prices, H
