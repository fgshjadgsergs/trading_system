"""Смесь экспонент вместо одной: слоистая карта тепла.

Зачем. Одна экспонента означает ПОСТОЯННУЮ интенсивность закрытия: шанс
позиции исчезнуть в следующий час не зависит от того, сколько она уже
прожила. У рынка это не так — импульсная froth на 100x живёт часы, а
крупная позиция на 3x переживает недели, — и наблюдается это как «на
минутках картинка понятная, а на старших ТФ всё гаснет слишком быстро»:
единственный T½ приходится выбирать между свежестью и памятью.

Смесь экспонент решает это без привязки к таймфрейму. Тепло разложено по
K слоям со своими полураспадами; каждый слой — обычный `LiqMap` (вся
проверенная арифметика, инвариант массы и оракулы работают послойно),
геометрия снятия у слоёв общая. Итоговая функция выживания

    S(t) = Σ w_k · 2^(−t/T_k)

имеет УБЫВАЮЩУЮ интенсивность: молодое тепло гаснет быстро (сохраняется
контраст на минутках), а дожившее живёт долго (уровни держатся на дневках).
Эффективный полураспад растёт с возрастом тепла — см. `effective_half_life`.

Два способа задать слои:
  * `MixtureLiqMap(components=[(доля, T½), ...])` — доли по массе;
  * `MixtureLiqMap.by_leverage([(макс. плечо, T½), ...], ...)` — слой на
    группу плеч: полураспад привязан к плечу, а плечо известно в момент
    размещения (физически защитимее доли «на глаз»).

Смесь — opt-in надстройка: сам `LiqMap` не изменён, дефолтный путь один в
один прежний, якорь-хэш регрессии не затронут.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from trading_system.core.schema import Side
from trading_system.liqmap.map import Context, LiqMap, WeightFn


class MaskedWeights:
    """Веса исходной функции, обрезанные до группы плеч и перенормированные
    внутри неё. `share(context)` — какая доля исходной массы приходится на
    группу: слой получает ровно её, поэтому суммарное размещение по всем
    слоям совпадает с размещением одной карты."""

    def __init__(self, inner: WeightFn, mask: np.ndarray) -> None:
        self._inner = inner
        self._mask = np.asarray(mask, dtype=bool)

    def __call__(self, context: Context | None = None) -> np.ndarray:
        w = np.asarray(self._inner(context), dtype=float) * self._mask
        s = float(w.sum())
        return w / s if s > 0.0 else w  # пустая группа: слой ничего не размещает

    def share(self, context: Context | None = None) -> float:
        w = np.asarray(self._inner(context), dtype=float)
        total = float(w.sum())
        return float(w[self._mask].sum() / total) if total > 0.0 else 0.0


class MixtureLiqMap:
    """K слоёв `LiqMap` с разными полураспадами под общим интерфейсом.

    Интерфейс совпадает с `LiqMap` в той части, которой пользуются
    `HeatHistory`, оверлеи и зоны: `heat`, `buckets`, `allocate`, `consume`,
    `decay`, `step`, `total_heat`, `snapshot`, счётчики массы. `heat` —
    АГРЕГАТ (свежий словарь на каждое обращение), писать в него бесполезно:
    состояние живёт в слоях (`layers`).
    """

    def __init__(
        self,
        components: Sequence[tuple[float, float]],
        *,
        close_out_fraction: float = 1.0,
        **map_kwargs: Any,
    ) -> None:
        if not components:
            raise ValueError("нужен хотя бы один слой")
        weights = [float(w) for w, _ in components]
        halves = [float(t) for _, t in components]
        if any(not (w >= 0 and math.isfinite(w)) for w in weights) or sum(weights) <= 0:
            raise ValueError("доли слоёв должны быть конечными, >= 0 и в сумме > 0")
        if any(not t > 0 for t in halves):  # `not >` отсекает и NaN; +inf = без затухания
            raise ValueError("полураспад слоя должен быть положительным")
        if not 0.0 <= close_out_fraction <= 1.0:
            raise ValueError("close_out_fraction in [0, 1]")
        map_kwargs.pop("decay_half_life_s", None)
        total_w = math.fsum(weights)
        self._fixed_shares = [w / total_w for w in weights]
        # close_out_fraction применяется ОДИН раз на уровне смеси (снятие
        # пропорционально общей массе), слои получают уже готовую сумму
        self.close_out_fraction = close_out_fraction
        self.layers = [
            LiqMap(decay_half_life_s=t, close_out_fraction=1.0, **map_kwargs) for t in halves
        ]
        self.half_lives = halves
        self._share_fns: list[Callable[[Context | None], float]] = [
            (lambda _ctx, s=s: s) for s in self._fixed_shares
        ]

    # -- конструкторы ---------------------------------------------------------
    @classmethod
    def by_leverage(
        cls,
        tiers: Sequence[tuple[float, float]],
        *,
        leverage_grid: Sequence[float],
        weight_fn: WeightFn,
        close_out_fraction: float = 1.0,
        **map_kwargs: Any,
    ) -> MixtureLiqMap:
        """Слой на группу плеч: `tiers` = [(верхняя граница плеча, T½), ...]
        по возрастанию границы; последняя группа забирает всё, что выше."""
        if not tiers:
            raise ValueError("нужен хотя бы один тир")
        bounds = [float(b) for b, _ in tiers]
        if bounds != sorted(bounds):
            raise ValueError("границы плеч должны идти по возрастанию")
        grid = np.asarray(leverage_grid, dtype=float)
        masks, lo = [], -math.inf
        for k, hi in enumerate(bounds):
            hi_eff = math.inf if k == len(bounds) - 1 else hi
            masks.append((grid > lo) & (grid <= hi_eff))
            lo = hi
        if not all(m.any() for m in masks):
            raise ValueError("есть тир без единого плеча в сетке")
        obj = cls.__new__(cls)
        obj.close_out_fraction = close_out_fraction
        obj.half_lives = [float(t) for _, t in tiers]
        obj.layers = [
            LiqMap(
                leverage_grid=grid,
                weight_fn=MaskedWeights(weight_fn, m),
                decay_half_life_s=t,
                close_out_fraction=1.0,
                **map_kwargs,
            )
            for m, t in zip(masks, obj.half_lives, strict=True)
        ]
        obj._fixed_shares = []
        obj._share_fns = [
            (lambda ctx, f=lm.weight_fn: f.share(ctx)) for lm in obj.layers
        ]
        return obj

    # -- состояние ------------------------------------------------------------
    @property
    def buckets(self):  # noqa: ANN201 - тот же тип, что у слоя
        return self.layers[0].buckets

    @property
    def leverage_grid(self) -> np.ndarray:
        return self.layers[0].leverage_grid

    @property
    def heat(self) -> dict[Side, dict[int, float]]:
        agg: dict[Side, dict[int, float]] = {Side.BUY: {}, Side.SELL: {}}
        for lm in self.layers:
            for side, side_heat in lm.heat.items():
                dst = agg[side]
                for idx, h in side_heat.items():
                    dst[idx] = dst.get(idx, 0.0) + h
        return agg

    def layer_heat(self, k: int) -> dict[Side, dict[int, float]]:
        return self.layers[k].heat

    def total_heat(self) -> float:
        return math.fsum(lm.total_heat() for lm in self.layers)

    def _sum(self, name: str) -> float:
        return math.fsum(getattr(lm, name) for lm in self.layers)

    @property
    def contributed(self) -> float:
        return self._sum("contributed")

    @property
    def consumed(self) -> float:
        return self._sum("consumed")

    @property
    def decayed(self) -> float:
        return self._sum("decayed")

    @property
    def removed(self) -> float:
        return self._sum("removed")

    @property
    def dropped(self) -> float:
        return self._sum("dropped")

    def mass_balance_error(self) -> float:
        return abs(
            self.total_heat()
            - (self.contributed - self.consumed - self.decayed - self.removed)
        )

    # -- обновление -----------------------------------------------------------
    def allocate(
        self,
        d_oi_usd: float,
        price: float,
        context: Context | None = None,
        long_share: float | None = None,
    ) -> None:
        if not math.isfinite(d_oi_usd):
            raise ValueError("d_oi_usd must be finite")
        if d_oi_usd == 0.0:
            return
        if d_oi_usd < 0.0:
            self._remove_proportional(-d_oi_usd * self.close_out_fraction)
            return
        for lm, share_fn in zip(self.layers, self._share_fns, strict=True):
            share = share_fn(context)
            if share > 0.0:
                lm.allocate(d_oi_usd * share, price, context, long_share=long_share)

    def _remove_proportional(self, amount_usd: float) -> None:
        """Закрытия снимают со слоёв ОДНУ И ТУ ЖЕ долю массы: закрывающий
        поток не знает, какого возраста позиция. При равных T½ это в точности
        воспроизводит снятие одной карты."""
        total = self.total_heat()
        if total <= 0.0:
            return
        factor = min(amount_usd / total, 1.0)
        for lm in self.layers:
            mass = lm.total_heat()
            if mass > 0.0:
                lm.remove_proportional(factor * mass)

    def consume(self, path_lo: float, path_hi: float) -> float:
        return math.fsum(lm.consume(path_lo, path_hi) for lm in self.layers)

    def decay(self, dt_s: float) -> float:
        return math.fsum(lm.decay(dt_s) for lm in self.layers)

    def step(
        self,
        bar_low: float,
        bar_high: float,
        bar_close: float,
        d_oi_usd: float,
        dt_s: float,
        context: Context | None = None,
        long_share: float | None = None,
    ) -> None:
        self.consume(bar_low, bar_high)
        self.allocate(d_oi_usd, bar_close, context, long_share=long_share)
        self.decay(dt_s)

    def rebucket_to(self, new_buckets) -> None:  # noqa: ANN001 - тот же тип, что у слоя
        for lm in self.layers:
            lm.rebucket_to(new_buckets)

    # -- виды -----------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        heat = self.heat
        idxs = sorted(set(heat[Side.BUY]) | set(heat[Side.SELL]))
        if not idxs:
            return {"prices": np.array([]), "long": np.array([]), "short": np.array([])}
        rng = np.arange(idxs[0], idxs[-1] + 1)
        return {
            "prices": np.array([self.buckets.center(int(i)) for i in rng]),
            "long": np.array([heat[Side.BUY].get(int(i), 0.0) for i in rng]),
            "short": np.array([heat[Side.SELL].get(int(i), 0.0) for i in rng]),
        }

    # -- свойства закона выживания -------------------------------------------
    def survival(self, age_s: float) -> float:
        """Доля исходного тепла, дожившая до возраста `age_s` (в отсутствие
        снятий и закрытий) — S(t) = Σ w_k · 2^(−t/T_k)."""
        shares = self._fixed_shares or self._current_shares()
        return math.fsum(
            w * 0.5 ** (age_s / t) for w, t in zip(shares, self.half_lives, strict=True)
        )

    def _current_shares(self) -> list[float]:
        """Доли по МАССЕ на данный момент (для тиров по плечу они меняются)."""
        total = self.total_heat()
        if total <= 0.0:
            return [f(None) for f in self._share_fns]
        return [lm.total_heat() / total for lm in self.layers]

    def effective_half_life(self, age_s: float = 0.0, hi_s: float = 3.15e8) -> float:
        """Полураспад ТЕПЛА ВОЗРАСТА `age_s`: h, при котором S(age+h) = S(age)/2.

        Для одной экспоненты это константа, для смеси растёт с возрастом —
        это и есть «убывающая интенсивность»: чем дольше уровень прожил, тем
        медленнее он гаснет дальше."""
        target = self.survival(age_s) * 0.5
        lo, hi = 0.0, hi_s
        if self.survival(age_s + hi) > target:  # не дожили до половины и за 10 лет
            return math.inf
        for _ in range(200):  # бисекция: S монотонно убывает
            mid = 0.5 * (lo + hi)
            if self.survival(age_s + mid) > target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
