"""Источники закрытых баров для живой карты.

DemoBarFeed — сидированный ряд в реальном (или ускоренном) времени, чтобы
платформу можно было поднять и проверить без сети и без лейка.

LakeBarFeed — инкрементальное чтение живого лейка рекордера: закрытые
1m-клайны + открытый интерес (ΔOI по asof-значениям на закрытиях баров) +
долю лонгов из ратио-потока. Каждый poll() отдаёт только НОВЫЕ бары строго
после последнего отданного; сборка каузальна — бар не выдаётся, пока для
него нет значения OI не старше закрытия.
"""

from __future__ import annotations

import bisect
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
import structlog

from trading_system.platform.state import Bar

log = structlog.get_logger(__name__)

MIN_NS = 60_000_000_000


class DemoBarFeed:
    """Минутные бары сидированного случайного блуждания, по часам wall-clock.

    speed > 1 ускоряет время: при speed=60 «минутный» бар выходит каждую
    секунду. История стартует за history_bars до текущего момента, чтобы
    карта на старте не была пустой.
    """

    def __init__(
        self,
        symbol: str,
        price0: float = 65_000.0,
        daily_vol: float = 0.028,
        oi_daily_usd: float = 50e6,
        seed: int = 7,
        speed: float = 60.0,
        history_bars: int = 600,
    ) -> None:
        self.symbol = symbol
        self._rng = np.random.default_rng(seed)
        self._price = price0
        self._vol = daily_vol * math.sqrt(60.0 / 86_400.0)
        self._per_bar_oi = oi_daily_usd / 1440.0
        self._speed = speed
        self._t0 = time.time()
        self._emitted = 0
        self._history_bars = history_bars
        self._pending_history = history_bars
        self._ts0 = int(time.time() * 1e9) - history_bars * MIN_NS

    def _make_bar(self, i: int) -> Bar:
        r = float(self._rng.normal(0, self._vol))
        o = self._price
        c = max(o * math.exp(r), 1e-9)
        hi = max(o, c) * (1 + abs(float(self._rng.normal(0, 0.35))) * self._vol)
        lo = min(o, c) * (1 - abs(float(self._rng.normal(0, 0.35))) * self._vol)
        impulse = abs(r) / (self._vol + 1e-12)
        d_oi = (impulse - 0.7) * self._per_bar_oi * 1.6 + float(
            self._rng.normal(0, self._per_bar_oi * 0.5))
        ls = float(np.clip(0.5 + 0.25 * math.tanh(r / (self._vol + 1e-12)), 0.15, 0.85))
        self._price = c
        ts_open = self._ts0 + i * MIN_NS
        return Bar(ts_open, ts_open + MIN_NS, o, hi, lo, c, d_oi, ls)

    def poll(self) -> list[Bar]:
        if self._pending_history:
            n, self._pending_history = self._pending_history, 0
            out = [self._make_bar(i) for i in range(n)]
            self._emitted = n
            return out
        due = self._history_bars + int(
            (time.time() - self._t0) * self._speed / 60.0)
        out = []
        while self._emitted < due:
            out.append(self._make_bar(self._emitted))
            self._emitted += 1
        return out


class LakeBarFeed:
    """Закрытые 1m-бары из лейка рекордера, строго вперёд и каузально.

    ΔOI бара = OI(asof закрытие) − OI(asof закрытие предыдущего бара) в USD.
    Бар придерживается, пока свежайший OI-замер старше его закрытия: иначе
    ΔOI досчитался бы задним числом и кадр пришлось бы переписывать — а
    контракт платформы запрещает переписывать прошлое.
    """

    def __init__(self, lake: str | Path, symbol: str, *, ls_blend: dict[str, float] | None = None) -> None:
        self.lake = Path(lake)
        self.symbol = symbol
        self._last_ts = 0
        self._prev_oi_usd: float | None = None
        self._prev_oi_coins: float | None = None
        self._ls_blend = ls_blend or {"global_ls_account": 0.4,
                                      "top_ls_position": 0.3, "taker_ls": 0.3}

    def _read(self, stream: str) -> pl.DataFrame:
        from trading_system.core.io import read_stream
        try:
            return read_stream(self.lake, stream, symbol=self.symbol)
        except FileNotFoundError:
            return pl.DataFrame()

    @staticmethod
    def _asof(ts_sorted: list[int], values: list[float], ts: int) -> float | None:
        i = bisect.bisect_right(ts_sorted, ts) - 1
        return values[i] if i >= 0 else None

    @staticmethod
    def _asof_ts(ts_sorted: list[int], values: list[float], ts: int) -> tuple[int, float] | None:
        i = bisect.bisect_right(ts_sorted, ts) - 1
        return (ts_sorted[i], values[i]) if i >= 0 else None

    def poll(self) -> list[Bar]:
        klines = self._read("kline")
        if klines.is_empty():
            return []
        klines = (
            klines.filter(pl.col("closed") & (pl.col("ts_close") > self._last_ts))
            .sort("ts_close")
            .unique(subset=["ts_close"], keep="first", maintain_order=True)
        )
        if klines.is_empty():
            return []
        oi = self._read("open_interest")
        if oi.is_empty():
            return []
        # open_interest_usd может быть NaN (рекордер без цены на момент
        # замера): NaN не значение — он не служит водяным знаком придержки и
        # не попадает в ΔOI. Основной ряд — USD; там, где USD неизвестен,
        # ΔOI восстанавливается из МОНЕТ × цена закрытия бара (фолбэк для
        # уже записанных лейков и первых секунд до первой mark-цены).
        oi = oi.sort("ts_event")
        usd = oi.filter(pl.col("open_interest_usd").is_finite())
        coins = oi.filter(pl.col("open_interest").is_finite())
        if coins.is_empty():
            return []
        usd_ts = usd["ts_event"].to_list()
        usd_val = usd["open_interest_usd"].to_list()
        coin_ts = coins["ts_event"].to_list()
        coin_val = coins["open_interest"].to_list()
        newest_oi = coin_ts[-1]

        ratios = self._read("ratio")
        if not ratios.is_empty() and "long_share" in ratios.columns:
            r = (
                ratios.filter(pl.col("long_share").is_finite()
                              & pl.col("long_share").is_between(0.0, 1.0))
                .sort("ts_event")
            )
            # каузальный бленд: по каждой метрике последний замер <= ts,
            # смешанный весами; метрика без замера выпадает с перенормировкой
            per_metric = {
                m: (g["ts_event"].to_list(), g["long_share"].to_list())
                for m, g in r.group_by("metric", maintain_order=True)
            }
            per_metric = {
                (m[0] if isinstance(m, tuple) else m): v for m, v in per_metric.items()
            }
            def blended(ts: int) -> float | None:
                acc, wsum = 0.0, 0.0
                for m, w in self._ls_blend.items():
                    tv = per_metric.get(m)
                    if not tv:
                        continue
                    v = self._asof(tv[0], tv[1], ts)
                    if v is not None:
                        acc += w * v
                        wsum += w
                return float(np.clip(acc / wsum, 0.1, 0.9)) if wsum > 0 else None
            ls_fn = blended
        else:
            def ls_fn(ts: int) -> float | None:
                return None

        out: list[Bar] = []
        for row in klines.iter_rows(named=True):
            tsc = row["ts_close"]
            if tsc > newest_oi:
                break  # OI ещё не доехал: бар придержан до следующего poll
            usd_at = self._asof_ts(usd_ts, usd_val, tsc)
            coins_at = self._asof_ts(coin_ts, coin_val, tsc)
            if coins_at is None:
                self._last_ts = tsc  # бар старше первого замера OI: ΔOI неизвестен
                self._prev_oi_usd = None
                self._prev_oi_coins = None
                continue
            c_ts, coins_now = coins_at
            # USD-диф только если USD-замер на этой границе не СТАРЕЕ замера
            # монет: asof молча тянет чёрствый USD через дыру NaN-замеров, а
            # монеты уже ушли — тогда честнее Δмонеты × close этого бара
            usd_fresh = usd_at is not None and usd_at[0] >= c_ts
            if usd_fresh and self._prev_oi_usd is not None:
                d_oi = usd_at[1] - self._prev_oi_usd
            elif self._prev_oi_coins is not None:
                d_oi = (coins_now - self._prev_oi_coins) * row["close"]
            else:
                d_oi = 0.0  # первый бар с известным OI: базовая точка
            out.append(Bar(row["ts_open"], tsc, row["open"], row["high"],
                           row["low"], row["close"], d_oi, ls_fn(tsc)))
            self._prev_oi_usd = usd_at[1] if usd_fresh else None
            self._prev_oi_coins = coins_now
            self._last_ts = tsc
        return out
