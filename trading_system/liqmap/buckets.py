"""Price buckets sized as a fraction of ATR (never a fixed tick).

The grid is anchored at price 0 with step = atr * fraction, so bucket indices
are stable across updates until the grid is re-sized; re-bucketing maps mass
to nearest new centers and conserves it exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PriceBuckets:
    bucket_size: float

    @classmethod
    def from_atr(cls, atr: float, fraction: float = 0.1, min_size: float = 1e-9) -> PriceBuckets:
        if atr <= 0 or fraction <= 0:
            raise ValueError("atr and fraction must be positive")
        return cls(bucket_size=max(atr * fraction, min_size))

    def index(self, price: float) -> int:
        return math.floor(price / self.bucket_size)

    def center(self, index: int) -> float:
        return (index + 0.5) * self.bucket_size

    def lo(self, index: int) -> float:
        return index * self.bucket_size

    def hi(self, index: int) -> float:
        return (index + 1) * self.bucket_size


def rebucket(
    heat: dict[int, float], old: PriceBuckets, new: PriceBuckets
) -> dict[int, float]:
    """Move heat between grids by bucket centers; total mass is conserved."""
    out: dict[int, float] = {}
    for idx, h in heat.items():
        j = new.index(old.center(idx))
        out[j] = out.get(j, 0.0) + h
    return out
