"""Живое состояние карты одного символа и дельта-протокол для фронтенда.

Контракт «дорисовки»: карта на сервере продвигается строго вперёд — один
закрытый бар = один шаг step() = один новый кадр. Клиент хранит ts
последнего полученного кадра и запрашивает только новые (`delta`). Полная
перезагрузка нужна лишь когда сменилась `epoch` — идентичность состояния
сервера (рестарт, другой код, другие параметры): реплей из лейка
детерминирован, поэтому после обновления сервер приходит к тому же (или
осознанно новому) состоянию, а клиент это видит по эпохе и один раз
перечитывает снапшот вместо того, чтобы молча рисовать поверх чужого.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import threading
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from trading_system.core.schema import Side
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.map import LiqMap, StaticWeights

GRID = [3.0, 5.0, 10.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 75.0, 100.0, 125.0]
SEED_W = np.array([1, 2, 4, 6, 5, 5, 4, 4, 3, 2, 2, 1], dtype=float)


@dataclass(frozen=True, slots=True)
class Bar:
    """Закрытый бар, единица продвижения карты."""

    ts_open: int
    ts_close: int
    open: float
    high: float
    low: float
    close: float
    d_oi_usd: float
    long_share: float | None = None


class LiveMapState:
    """Карта + история кадров одного символа с потокобезопасным доступом.

    Кадры хранятся кольцом (`max_frames`): дельта-протокол отдаёт только
    хвост, а клиент, отставший дальше кольца, получает признак gap и
    перечитывает снапшот.
    """

    def __init__(
        self,
        symbol: str,
        bucket_size: float,
        *,
        decay_half_life_s: float = float("inf"),
        close_out_fraction: float = 0.0,
        fractional_edge_consume: bool = False,
        blur_sigma0_bps: float | None = None,
        max_frames: int = 3000,
        heat_floor_usd: float = 1.0,
    ) -> None:
        if not (bucket_size > 0 and np.isfinite(bucket_size)):
            raise ValueError("bucket_size must be positive and finite")
        self.symbol = symbol
        self.map = LiqMap(
            leverage_grid=GRID,
            buckets=PriceBuckets(bucket_size),
            weight_fn=StaticWeights(SEED_W),
            decay_half_life_s=decay_half_life_s,
            close_out_fraction=close_out_fraction,
            fractional_edge_consume=fractional_edge_consume,
            blur_sigma0_bps=blur_sigma0_bps,
        )
        self._heat_floor = heat_floor_usd
        self._frames: deque[dict[str, Any]] = deque(maxlen=max_frames)
        self._bars: deque[dict[str, Any]] = deque(maxlen=max_frames)
        self._last_ts: int | None = None
        self._dropped_old = 0  # баров отвергнуто из-за немонотонного ts
        self._lock = threading.Lock()
        # эпоха — идентичность ПАРАМЕТРОВ состояния; конкретный прогон
        # добавляет соль после первого бара (см. _seal_epoch)
        self._epoch_basis = json.dumps(
            {
                "symbol": symbol,
                "bucket_size": bucket_size,
                "half_life_s": decay_half_life_s,
                "close_out": close_out_fraction,
                "fractional_edges": fractional_edge_consume,
                "blur_bps": blur_sigma0_bps,
                "grid": GRID,
                "w": SEED_W.tolist(),
                "v": 1,  # поднимать при несовместимой смене семантики кадров
            },
            sort_keys=True,
        )
        self.epoch: str | None = None

    # -- продвижение ----------------------------------------------------------
    def apply_bar(self, bar: Bar) -> bool:
        """Один закрытый бар -> один кадр. Возвращает False для дублей/прошлого.

        Идемпотентность по ts_close: повторная подача того же бара (перезапуск
        фида, перекрытие окон опроса) не двигает карту.
        """
        with self._lock:
            if self._last_ts is not None and bar.ts_close <= self._last_ts:
                self._dropped_old += 1
                return False
            # ВСЯ валидация до первого касания карты/эпохи: map.step() делает
            # consume -> allocate, и ValueError из allocate (битый d_oi_usd или
            # long_share) оставил бы карту с уже снятым по пути бара теплом
            d_oi = float(bar.d_oi_usd or 0.0)
            if (
                not (bar.low <= bar.high)
                or not np.isfinite([bar.open, bar.high, bar.low, bar.close, d_oi]).all()
                or (bar.long_share is not None and not 0.0 <= bar.long_share <= 1.0)
            ):
                raise ValueError(f"broken bar for {self.symbol}: {bar}")
            if self.epoch is None:
                self._seal_epoch(bar.ts_close)
            dt_s = max((bar.ts_close - bar.ts_open) / 1e9, 0.0)
            self.map.step(
                bar.low, bar.high, bar.close,
                d_oi, dt_s=dt_s, long_share=bar.long_share,
            )
            self._frames.append({"ts": bar.ts_close, "cols": self._sparse_cols()})
            self._bars.append(
                {
                    "ts": bar.ts_close, "ts_open": bar.ts_open,
                    "o": bar.open, "h": bar.high, "l": bar.low, "c": bar.close,
                }
            )
            self._last_ts = bar.ts_close
            return True

    def _seal_epoch(self, first_ts: int) -> None:
        basis = f"{self._epoch_basis}|first_ts={first_ts}"
        self.epoch = hashlib.sha256(basis.encode()).hexdigest()[:16]

    def _sparse_cols(self) -> list[list[float]]:
        """Занятые бакеты агрегатом по сторонам, отсечка пыли ниже floor."""
        agg: dict[int, float] = {}
        for side in (Side.BUY, Side.SELL):
            for idx, h in self.map.heat[side].items():
                agg[idx] = agg.get(idx, 0.0) + h
        floor = self._heat_floor
        return [[int(i), round(h, 2)] for i, h in sorted(agg.items()) if h >= floor]

    # -- выдача ---------------------------------------------------------------
    def meta(self) -> dict[str, Any]:
        with self._lock:
            return {
                "symbol": self.symbol,
                "epoch": self.epoch,
                "bucket_size": self.map.buckets.bucket_size,
                "frames": len(self._frames),
                "last_ts": self._last_ts,
                "total_heat": self.map.total_heat(),
                "dropped_old_bars": self._dropped_old,
            }

    def snapshot(self, max_cols: int = 1500) -> dict[str, Any]:
        """Полное состояние для первой отрисовки: хвост кадров + бары."""
        with self._lock:
            frames = list(self._frames)[-max_cols:]
            bars = list(self._bars)[-max_cols:]
            return {
                "type": "snapshot",
                "symbol": self.symbol,
                "epoch": self.epoch,
                "bucket_size": self.map.buckets.bucket_size,
                "frames": frames,
                "bars": bars,
                "last_ts": self._last_ts,
            }

    def delta(self, since_ts: int, epoch: str | None) -> dict[str, Any]:
        """Кадры строго после since_ts. gap=True -> клиент должен перечитать
        снапшот (сменилась эпоха или запрошенное уже выпало из кольца)."""
        with self._lock:
            if epoch is not None and epoch != self.epoch:
                return {"type": "delta", "epoch": self.epoch, "gap": True,
                        "frames": [], "bars": [], "last_ts": self._last_ts}
            frames = list(itertools.takewhile(
                lambda f: f["ts"] > since_ts, reversed(self._frames)))
            # отстал дальше кольца: самый старый хранимый кадр всё ещё новее
            gap = bool(
                self._frames
                and self._frames[0]["ts"] > since_ts
                and len(frames) == len(self._frames)
                and since_ts != 0
            )
            frames.reverse()
            bars = [b for b in self._bars if b["ts"] > since_ts]
            return {
                "type": "delta", "epoch": self.epoch, "gap": gap,
                "frames": frames, "bars": bars, "last_ts": self._last_ts,
            }
