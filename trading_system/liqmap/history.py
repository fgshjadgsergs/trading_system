"""Time x price heat history for overlay heatmaps (assumes a stable bucket grid)."""

from __future__ import annotations

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
        return sum(self._frames[i].values())

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

    def matrix(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (ts[n], bucket_center_prices[m], H[m, n]) for imshow overlays."""
        if not self._frames:
            return np.array([]), np.array([]), np.zeros((0, 0))
        occupied = sorted({i for f in self._frames for i in f})
        lo, hi = occupied[0], occupied[-1]
        idx_range = np.arange(lo, hi + 1)
        H = np.zeros((len(idx_range), len(self._frames)))
        for j, f in enumerate(self._frames):
            for i, h in f.items():
                H[i - lo, j] = h
        prices = (idx_range + 0.5) * self._bucket_size
        return np.asarray(self.ts), prices, H
