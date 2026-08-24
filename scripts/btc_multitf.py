"""Карта ликвидаций BTC на нескольких таймфреймах и длинных периодах.

Реальные данные (на своей машине):
    python scripts/btc_multitf.py --lake data/vision_lake --symbol BTCUSDT
    # предварительно скачать, например год часовок и месяц пятиминуток:
    #   python scripts/download_vision.py --symbols BTCUSDT --kinds klines metrics \
    #       --intervals 1h --start 2025-08-01 --end 2026-08-01

Без лейка скрипт строит сидированный ряд в масштабе BTC (демо-режим):
    python scripts/btc_multitf.py --synthetic

Полураспад тепла по умолчанию тянется за таймфреймом: на 4h фиксированные
24 часа съедали бы пул за 6 баров, и карта старших ТФ вырождалась бы в
«последние несколько свечей».
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog

from trading_system.core.timeutils import TIMEFRAME_NS
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.history import HeatHistory
from trading_system.liqmap.map import LiqMap, StaticWeights
from trading_system.liqmap.terminal import terminal_heat_overlay
from trading_system.viz.style import PALETTE, apply_style, save_fig

log = structlog.get_logger()

GRID = [3, 5, 10, 20, 25, 30, 40, 50, 60, 75, 100, 125]
SEED_W = np.array([1, 2, 4, 6, 5, 5, 4, 4, 3, 2, 2, 1], dtype=float)
# (таймфрейм, сколько баров показываем, подпись периода)
PLAN = [("5m", 8_640, "месяц"), ("1h", 8_760, "год"), ("4h", 4_380, "два года")]


def synth_btc(n: int, bar_s: float, seed: int = 7) -> pl.DataFrame:
    """Сидированный BTC-подобный ряд: тренды, кластеризация волатильности,
    приток открытого интереса на импульсах."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(bar_s / 300.0)  # вола растёт с длиной бара
    vol = 0.0018 * scale * np.exp(np.cumsum(rng.normal(0, 0.04, n)) * 0.25)
    drift = np.cumsum(rng.normal(0, 0.02 * scale, n)) * 0.001
    ret = rng.normal(0, 1, n) * vol + drift / max(n, 1)
    for start in rng.choice(n, size=max(2, n // 900), replace=False):
        end = min(n, start + max(3, n // 400))
        ret[start:end] += rng.choice([-1.0, 1.0]) * vol[start:end] * 2.2
    price = 65_000.0 * np.exp(np.cumsum(ret))
    hi = price * (1 + np.abs(rng.normal(0, 0.45, n)) * vol)
    lo = price * (1 - np.abs(rng.normal(0, 0.45, n)) * vol)
    op = np.concatenate([[price[0]], price[:-1]])
    hi, lo = np.maximum(hi, np.maximum(op, price)), np.minimum(lo, np.minimum(op, price))
    impulse = np.abs(ret) / (vol + 1e-12)
    d_oi = (impulse - 0.75) * 6.0e5 * scale + rng.normal(0, 1.6e5 * scale, n)
    ts = np.arange(1, n + 1, dtype=np.int64) * int(bar_s * 1e9)
    tr = np.maximum(hi - lo, np.abs(hi - op))
    atr = pl.Series(tr).rolling_mean(14, min_samples=1).to_numpy()
    ls = np.clip(
        0.5 + 0.3 * np.tanh(pl.Series(ret).rolling_mean(24, min_samples=1).to_numpy()
                            / (vol + 1e-12)), 0.15, 0.85)
    return pl.DataFrame({
        "symbol": ["BTCUSDT"] * n, "ts_open": ts - int(bar_s * 1e9), "ts_close": ts,
        "open": op, "high": hi, "low": lo, "close": price,
        "volume": np.abs(rng.normal(900, 200, n)),
        "quote_volume": np.abs(rng.normal(900, 200, n)) * price,
        "d_oi_usd": d_oi, "atr": atr, "long_share": ls,
    })


def bars_from_lake(lake: Path, symbol: str, timeframe: str, limit: int) -> pl.DataFrame:
    """Реальные бары: klines нужного интервала + OI + ATR (как в real_heatmap)."""
    from scripts.real_heatmap import bars_from_klines
    from trading_system.core.io import read_stream
    from trading_system.features import join_open_interest, with_atr

    klines = read_stream(lake, "kline", symbol=symbol)
    bars = bars_from_klines(klines, timeframe)
    if bars.is_empty():
        raise SystemExit(f"в лейке нет {timeframe}-клайнов для {symbol}")
    oi = read_stream(lake, "open_interest", symbol=symbol)
    bars = with_atr(join_open_interest(bars, oi), period=14).tail(limit)
    if "d_oi_usd" not in bars.columns:
        raise SystemExit("нет колонки d_oi_usd — не хватает потока open_interest в лейке")
    return bars.with_columns(pl.col("d_oi_usd").fill_null(0.0))


def build_map(bars: pl.DataFrame, bar_s: float, half_life_s: float, **kw):
    lm = LiqMap(
        leverage_grid=GRID,
        buckets=PriceBuckets.from_atr(float(np.nanmedian(bars["atr"])), 0.1),
        weight_fn=StaticWeights(SEED_W),
        decay_half_life_s=half_life_s,
        **kw,
    )
    hist = HeatHistory(lm)
    ls = "long_share" in bars.columns
    for r in bars.iter_rows(named=True):
        lm.step(r["low"], r["high"], r["close"], r["d_oi_usd"] or 0.0, dt_s=bar_s,
                long_share=r["long_share"] if ls else None)
        hist.record(r["ts_close"])
    return lm, hist


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lake", default=None, help="путь к parquet-лейку с klines+metrics")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--synthetic", action="store_true", help="демо без данных")
    ap.add_argument("--out", default="reports/btc")
    ap.add_argument("--half-life-bars", type=float, default=48.0,
                    help="полураспад в барах ТФ (нижняя граница — 24 часа)")
    args = ap.parse_args()
    out = Path(args.out)
    if not args.lake and not args.synthetic:
        raise SystemExit("укажите --lake с данными или --synthetic для демо")

    summary = []
    for tf, n_bars, period in PLAN:
        bar_s = TIMEFRAME_NS[tf] / 1e9
        half_life = max(86_400.0, args.half_life_bars * bar_s)
        bars = (synth_btc(n_bars, bar_s) if args.synthetic
                else bars_from_lake(Path(args.lake), args.symbol, tf, n_bars))
        lm, hist = build_map(bars, bar_s, half_life)
        title = (f"{args.symbol} · {tf} · {bars.height} баров ({period}) · "
                 f"полураспад {half_life / 3600:.0f} ч"
                 + ("" if args.lake else " · демо-ряд"))
        path = terminal_heat_overlay(bars, hist, name=f"btc_map_{tf}", out_dir=out, title=title)
        occ = sum(len(h) for h in lm.heat.values())
        summary.append((tf, bars.height, period, lm.total_heat(), occ,
                        float(bars["close"][-1]), path))
        log.info("map_built", tf=tf, bars=bars.height, heat_usd=round(lm.total_heat()),
                 occupied=occ, path=str(path))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    tfs = [s[0] for s in summary]
    axes[0].bar(tfs, [s[3] / 1e6 for s in summary], color=PALETTE["accent"])
    axes[0].set_ylabel("тепло на карте, млн USD")
    axes[0].set_title("Сколько денег «висит» на карте к концу периода")
    axes[1].bar(tfs, [s[4] for s in summary], color=PALETTE["neutral"])
    axes[1].set_ylabel("занятых ценовых бакетов")
    axes[1].set_title("Насколько карта разрежена")
    for ax in axes:
        ax.set_xlabel("таймфрейм")
    fig.suptitle(f"{args.symbol}: карта на разных горизонтах", y=1.02)
    summary_path = save_fig(fig, "btc_map_summary", out)
    for tf, n, period, heat, occ, last, path in summary:
        print(f"{tf:>3} | {n:>5} баров ({period:>8}) | тепло {heat/1e6:8.1f} млн USD | "
              f"бакетов {occ:>5} | цена {last:,.0f} | {path}")
    print(summary_path)


if __name__ == "__main__":
    main()
