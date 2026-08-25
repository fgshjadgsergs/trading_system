"""Карта ликвидаций на нескольких таймфреймах и длинных периодах.

Реальные данные (на своей машине):
    python scripts/map_multitf.py --lake data/vision_lake --symbol BTCUSDT \
        --timeframes 1h 4h 1d --days 180

Без лейка — сидированный демо-ряд заданного масштаба:
    python scripts/map_multitf.py --synthetic --symbol HYPEUSDT --price 35 --days 180

Полураспад ОДИН для всех таймфреймов и не масштабируется под них: он
описывает, как устаревают позиции на рынке, а не нашу сетку баров. Проверено
численно — один и тот же ряд, агрегированный в 1h и 4h, при равном T½ даёт
совпадающее состояние карты (масса на конец расходится на 2%). Меняется при
переходе на старший ТФ не физика затухания, а горизонт показа.
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
from trading_system.liqmap.mixture import MixtureLiqMap
from trading_system.liqmap.terminal import terminal_heat_overlay
from trading_system.viz.style import PALETTE, apply_style, save_fig

log = structlog.get_logger()

GRID = [3, 5, 10, 20, 25, 30, 40, 50, 60, 75, 100, 125]
SEED_W = np.array([1, 2, 4, 6, 5, 5, 4, 4, 3, 2, 2, 1], dtype=float)
BARS_PER_DAY = {"1m": 1440, "5m": 288, "15m": 96, "1h": 24, "4h": 6, "1d": 1}


def synth_series(n: int, bar_s: float, price0: float = 65_000.0, symbol: str = "BTCUSDT",
                 daily_vol: float = 0.028, oi_daily_usd: float = 50e6,
                 seed: int = 7) -> pl.DataFrame:
    """Сидированный ряд заданного масштаба: кластеризация волатильности,
    режимные тренды без накопленного сноса, приток открытого интереса на
    импульсах (задаётся в USD за сутки и делится по барам)."""
    rng = np.random.default_rng(seed)
    scale = np.sqrt(bar_s / 86_400.0)          # вола за бар из дневной
    vol = daily_vol * scale * np.exp(np.cumsum(rng.normal(0, 0.05, n)) * 0.2)
    ret = rng.normal(0, 1, n) * vol
    # режимы: несколько трендовых участков, но без общего сноса за период
    for start_i in rng.choice(n, size=max(2, n // 300), replace=False):
        end_i = min(n, start_i + max(3, n // 60))
        ret[start_i:end_i] += rng.choice([-1.0, 1.0]) * vol[start_i:end_i] * 0.9
    ret -= ret.mean()
    price = price0 * np.exp(np.cumsum(ret))
    hi = price * (1 + np.abs(rng.normal(0, 0.45, n)) * vol)
    lo = price * (1 - np.abs(rng.normal(0, 0.45, n)) * vol)
    op = np.concatenate([[price[0]], price[:-1]])
    hi, lo = np.maximum(hi, np.maximum(op, price)), np.minimum(lo, np.minimum(op, price))
    impulse = np.abs(ret) / (vol + 1e-12)
    per_bar = oi_daily_usd * bar_s / 86_400.0   # средний приток OI на бар
    d_oi = (impulse - 0.7) * per_bar * 1.6 + rng.normal(0, per_bar * 0.5, n)
    ts = np.arange(1, n + 1, dtype=np.int64) * int(bar_s * 1e9)
    tr = np.maximum(hi - lo, np.abs(hi - op))
    atr = pl.Series(tr).rolling_mean(14, min_samples=1).to_numpy()
    ls = np.clip(
        0.5 + 0.3 * np.tanh(pl.Series(ret).rolling_mean(24, min_samples=1).to_numpy()
                            / (vol + 1e-12)), 0.15, 0.85)
    return pl.DataFrame({
        "symbol": [symbol] * n, "ts_open": ts - int(bar_s * 1e9), "ts_close": ts,
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


def parse_mixture(spec: str) -> list[tuple[float, float]]:
    """`0.75:4,0.25:336` -> [(0.75, 4ч в секундах), (0.25, 336ч)]."""
    out = []
    for part in spec.split(","):
        w, hl = part.split(":")
        out.append((float(w), float(hl) * 3600.0))
    return out


def build_map(bars: pl.DataFrame, bar_s: float, half_life_s: float,
              bucket_bps: float = 0.0, mixture: list[tuple[float, float]] | None = None,
              lev_tiers: list[tuple[float, float]] | None = None, **kw):
    # Сетка бакетов — свойство ИНСТРУМЕНТА, а не нарезки баров (тот же довод,
    # что и для полураспада). ATR минутного бара в ~40 раз меньше часового,
    # и карта на 1m вырождается в сплошную заливку из тысяч волосяных уровней.
    # bucket_bps задаёт шаг в долях цены (10 bps = 0.1%), единый для всех ТФ.
    buckets = (PriceBuckets(float(bars["close"][-1]) * bucket_bps * 1e-4)
               if bucket_bps > 0 else
               PriceBuckets.from_atr(float(np.nanmedian(bars["atr"])), 0.1))
    common = dict(leverage_grid=GRID, buckets=buckets, weight_fn=StaticWeights(SEED_W))
    if lev_tiers is not None:
        lm = MixtureLiqMap.by_leverage(lev_tiers, **common, **kw)
    elif mixture is not None:
        lm = MixtureLiqMap(mixture, **common, **kw)
    else:
        lm = LiqMap(decay_half_life_s=half_life_s, **common, **kw)
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
    ap.add_argument("--synthetic", action="store_true", help="демо-ряд без данных")
    ap.add_argument("--price", type=float, default=65_000.0, help="масштаб цены для демо-ряда")
    ap.add_argument("--daily-vol", type=float, default=0.028, help="дневная вола демо-ряда")
    ap.add_argument("--oi-daily", type=float, default=50e6,
                    help="средний приток открытого интереса за сутки, USD (демо-ряд)")
    ap.add_argument("--timeframes", nargs="+", default=["1h", "4h", "1d"])
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--half-life-h", type=float, default=24.0,
                    help="полураспад тепла в ЧАСАХ — один для всех таймфреймов")
    ap.add_argument("--close-out-fraction", type=float, default=1.0)
    ap.add_argument("--mixture", default=None,
                    help="смесь экспонент вместо одного полураспада: "
                         "«доля:часы,доля:часы» (напр. 0.75:4,0.25:336)")
    ap.add_argument("--lev-tiers", default=None,
                    help="полураспад по группам плеч: «макс_плечо:часы,...» "
                         "по возрастанию (напр. 10:336,50:48,1e9:4)")
    ap.add_argument("--bucket-bps", type=float, default=0.0,
                    help="шаг ценового бакета в bps от цены, единый для всех ТФ "
                         "(0 = из ATR своего таймфрейма, прежнее поведение)")
    ap.add_argument("--out", default="reports/multitf")
    args = ap.parse_args()
    out = Path(args.out)
    if not args.lake and not args.synthetic:
        raise SystemExit("укажите --lake с данными или --synthetic для демо")
    half_life = args.half_life_h * 3600.0

    summary = []
    for tf in args.timeframes:
        if tf not in BARS_PER_DAY:
            raise SystemExit(f"неизвестный таймфрейм {tf}; известны: {sorted(BARS_PER_DAY)}")
        bar_s = TIMEFRAME_NS[tf] / 1e9
        n_bars = BARS_PER_DAY[tf] * args.days
        bars = (synth_series(n_bars, bar_s, price0=args.price, symbol=args.symbol,
                             daily_vol=args.daily_vol, oi_daily_usd=args.oi_daily)
                if args.synthetic
                else bars_from_lake(Path(args.lake), args.symbol, tf, n_bars))
        lm, hist = build_map(
            bars, bar_s, half_life, bucket_bps=args.bucket_bps,
            mixture=parse_mixture(args.mixture) if args.mixture else None,
            lev_tiers=parse_mixture(args.lev_tiers) if args.lev_tiers else None,
            close_out_fraction=args.close_out_fraction)
        law = (f"тиры плеч {args.lev_tiers}" if args.lev_tiers else
               f"смесь {args.mixture}" if args.mixture else
               f"полураспад {args.half_life_h:.0f} ч")
        title = (f"{args.symbol} · {tf} · {bars.height} баров ({args.days} дней) · {law}"
                 + (f" · бакет {args.bucket_bps:g} bps" if args.bucket_bps > 0 else "")
                 + ("" if args.lake else " · демо-ряд"))
        path = terminal_heat_overlay(bars, hist, name=f"map_{args.symbol.lower()}_{tf}",
                                     out_dir=out, title=title)
        occ = sum(len(h) for h in lm.heat.values())
        summary.append((tf, bars.height, lm.total_heat(), occ, float(bars["close"][-1]), path))
        log.info("map_built", tf=tf, bars=bars.height, heat_usd=round(lm.total_heat()),
                 occupied=occ, path=str(path))

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    tfs = [s[0] for s in summary]
    axes[0].bar(tfs, [s[2] / 1e6 for s in summary], color=PALETTE["accent"])
    axes[0].set_ylabel("тепло на карте, млн USD")
    axes[0].set_title("Актуальное тепло к концу периода")
    axes[1].bar(tfs, [s[3] for s in summary], color=PALETTE["neutral"])
    axes[1].set_ylabel("занятых ценовых бакетов")
    axes[1].set_title("Разрежённость карты")
    for ax in axes:
        ax.set_xlabel("таймфрейм")
    fig.suptitle(f"{args.symbol}: один закон затухания на всех ТФ", y=1.02)
    summary_path = save_fig(fig, f"map_{args.symbol.lower()}_summary", out)
    for tf, n, heat, occ, last, path in summary:
        print(f"{tf:>3} | {n:>6} баров | тепло {heat/1e6:8.2f} млн USD | бакетов {occ:>5} | "
              f"цена {last:,.2f} | {path}")
    print(summary_path)


if __name__ == "__main__":
    main()
