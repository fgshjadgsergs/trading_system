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

    def __post_init__(self) -> None:
        # `not >` (instead of `<=`) also rejects NaN; a NaN/0/negative size
        # would otherwise produce inverted [lo, hi) intervals or blow up
        # much later inside index() with a confusing error.
        if not (self.bucket_size > 0 and math.isfinite(self.bucket_size)):
            raise ValueError("bucket_size must be positive and finite")

    @classmethod
    def from_atr(cls, atr: float, fraction: float = 0.1, min_size: float = 1e-9) -> PriceBuckets:
        if not (atr > 0 and math.isfinite(atr)) or not fraction > 0:
            raise ValueError("atr and fraction must be positive and finite")
        return cls(bucket_size=max(atr * fraction, min_size))

    def index(self, price: float) -> int:
        # floor(price/size) может ошибиться на 1 у float-границы (деление
        # округляется через границу); контракт lo(i) <= price < hi(i) обязан
        # держаться в ТОЙ ЖЕ арифметике, что lo/hi, — иначе consume() видит
        # бакет, численно не содержащий свою цену. Коррекция до содержащего
        # тайла (не больше пары шагов: ошибка деления — единицы ulp).
        i = math.floor(price / self.bucket_size)
        while price < i * self.bucket_size:
            i -= 1
        while price >= (i + 1) * self.bucket_size:
            i += 1
        return i

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
