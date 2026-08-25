"""Таймфрейм — это окно ПОКАЗА, а не пересчёт модели: карта строится один
раз на базовом разрешении, старший ТФ выбирает моменты показа."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from scripts.map_multitf import aggregate_bars
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.history import HeatHistory
from trading_system.liqmap.map import LiqMap, StaticWeights

MIN_NS = 60_000_000_000


def make_history(n: int = 120) -> tuple[LiqMap, HeatHistory, pl.DataFrame]:
    lm = LiqMap(leverage_grid=[5, 10, 25, 50], buckets=PriceBuckets(1.0),
                weight_fn=StaticWeights(np.array([1.0, 2.0, 2.0, 1.0])),
                decay_half_life_s=6 * 3_600.0)
    hist = HeatHistory(lm)
    rng = np.random.default_rng(4)
    price, rows = 1_000.0, []
    for i in range(n):
        price = float(price * np.exp(rng.normal(0, 0.002)))
        lo, hi = price * 0.998, price * 1.002
        lm.step(lo, hi, price, float(rng.normal(3e5, 1e5)), dt_s=60.0)
        hist.record((i + 1) * MIN_NS)
        rows.append({"ts_open": i * MIN_NS, "ts_close": (i + 1) * MIN_NS,
                     "open": price, "high": hi, "low": lo, "close": price,
                     "volume": 10.0 + i, "d_oi_usd": 1.0, "long_share": 0.5})
    return lm, hist, pl.DataFrame(rows)


def test_resample_picks_the_frame_at_or_before_each_instant():
    _, hist, bars = make_history()
    coarse = aggregate_bars(bars, 15)
    view = hist.resample(coarse["ts_close"])
    assert len(view) == coarse.height
    for j, ts in enumerate(coarse["ts_close"]):
        i = hist.index_at(int(ts), inclusive=True)
        assert view.zones_at(j) == hist.zones_at(i)
        assert view.total_at(j) == hist.total_at(i)


def test_resample_is_a_view_not_a_rebuild():
    """Итоговый кадр окна показа — тот же самый кадр базовой карты, до бита."""
    lm, hist, bars = make_history()
    for factor in (5, 15, 60):
        view = hist.resample(aggregate_bars(bars, factor)["ts_close"])
        assert view.total_at(len(view) - 1) == hist.total_at(len(hist) - 1)
        assert view.pools_at(len(view) - 1, k=5) == hist.pools_at(len(hist) - 1, k=5)


def test_resample_rejects_unordered_instants_and_handles_early_ones():
    _, hist, _ = make_history(10)
    with pytest.raises(ValueError):
        hist.resample([5 * MIN_NS, 3 * MIN_NS])
    assert len(hist.resample([0])) == 0  # раньше первого кадра — показывать нечего
    assert len(hist.resample([])) == 0


def test_aggregate_bars_matches_manual_ohlc():
    _, _, bars = make_history(60)
    coarse = aggregate_bars(bars, 15)
    assert coarse.height == 4
    for j in range(4):
        chunk = bars[j * 15:(j + 1) * 15]
        row = coarse.row(j, named=True)
        assert row["open"] == chunk["open"][0]
        assert row["close"] == chunk["close"][-1]
        assert row["high"] == chunk["high"].max()
        assert row["low"] == chunk["low"].min()
        assert row["volume"] == pytest.approx(chunk["volume"].sum())
        assert row["ts_open"] == chunk["ts_open"][0]
        assert row["ts_close"] == chunk["ts_close"][-1]
    assert aggregate_bars(bars, 1) is bars


def test_rebuilding_on_coarse_bars_really_differs():
    """Контроль: пересборка из грубых баров даёт ДРУГУЮ карту — иначе весь
    приём «строить на базовом ТФ» был бы бессмысленным."""
    lm, _, bars = make_history(120)
    coarse = aggregate_bars(bars, 15)
    lm2 = LiqMap(leverage_grid=[5, 10, 25, 50], buckets=PriceBuckets(1.0),
                 weight_fn=StaticWeights(np.array([1.0, 2.0, 2.0, 1.0])),
                 decay_half_life_s=6 * 3_600.0)
    for r in coarse.iter_rows(named=True):
        lm2.step(r["low"], r["high"], r["close"], r["d_oi_usd"], dt_s=15 * 60.0)
    assert lm2.total_heat() != pytest.approx(lm.total_heat(), rel=1e-6)
