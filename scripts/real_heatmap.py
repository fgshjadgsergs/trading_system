"""Liquidation heat map on REAL market data for any USDT-M symbol.

One command (needs network access to data.binance.vision):

    PYTHONPATH=. python3 scripts/real_heatmap.py --symbol ETHUSDT --days 3

Downloads aggTrades + metrics (open interest, L/S ratios) + liquidationSnapshot
daily archives via scripts/download_vision.py, ingests them into a parquet
lake, drives the liquidation map bar-by-bar with causal ratio-derived side
shares, and saves two figures into reports/:

    <symbol>_real_heat_overlay.png  — candles + H(time x price) + real
                                      liquidation prints overlaid as dots
    <symbol>_real_heat_slice.png    — final long/short heat slice

Re-runs reuse the already-downloaded archives and the lake (append-only).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import structlog

import scripts.download_vision as dv
from scripts.run_pipeline import stage_map
from trading_system.core.config import load_config, seed_everything
from trading_system.core.io import read_stream
from trading_system.features import join_open_interest, time_bars, with_atr, with_cvd
from trading_system.viz.overlay import overlay_chart
from trading_system.viz.style import PALETTE, apply_style, save_fig

log = structlog.get_logger()

KINDS = ["aggTrades", "metrics", "liquidationSnapshot"]


def build_heatmap(
    lake: Path,
    symbol: str,
    cfg: dict,
    out: Path,
    timeframe: str = "5m",
    brackets_path: str | None = None,
) -> list[Path]:
    """Bars + map + figures from an already-ingested lake (offline part)."""
    trades = read_stream(lake, "trade", symbol=symbol)
    if trades.is_empty():
        raise SystemExit(f"lake has no trades for {symbol} — run the download step first")
    oi = read_stream(lake, "open_interest", symbol=symbol)
    ratios = read_stream(lake, "ratio", symbol=symbol)
    liqs = read_stream(lake, "liquidation", symbol=symbol)
    bars = with_atr(join_open_interest(with_cvd(time_bars(trades, timeframe)), oi), period=14)
    lm, hist = stage_map(bars, cfg, symbol, ratios=ratios, brackets_path=brackets_path)

    name = f"{symbol.lower()}_real_heat_overlay"
    p1 = overlay_chart(
        bars,
        heat=hist.matrix(),
        name=name,
        out_dir=out,
        title=f"{symbol}: реальные данные — свечи + карта ликвидаций",
    )
    if liqs.height:  # real liquidation prints on top of the heat
        idx_of_ts = bars["ts_open"].to_numpy()
        apply_style()
        fig, ax = plt.subplots(figsize=(14, 8))
        ts_arr, prices, H = hist.matrix()
        if H.size:
            ax.imshow(
                np.log1p(H),
                aspect="auto",
                origin="lower",
                extent=(0, bars.height, float(prices[0]), float(prices[-1])),
                cmap=PALETTE["heat"],
                alpha=0.7,
            )
        from trading_system.viz.overlay import draw_candles

        draw_candles(ax, bars)
        xs = np.searchsorted(idx_of_ts, liqs["ts_event"].to_numpy()) - 1
        ax.scatter(
            xs + 0.5,
            liqs["price"].to_numpy(),
            s=np.clip(liqs["qty_usd"].to_numpy() / 5_000, 4, 80),
            facecolors="none",
            edgecolors="#00e5ff",
            linewidths=0.8,
            zorder=6,
            label="реальные ликвидации (forceOrder)",
        )
        ax.legend(loc="upper left")
        ax.set_title(f"{symbol}: карта vs реальные ликвидации")
        p1b = save_fig(fig, f"{symbol.lower()}_real_heat_vs_liqs", out)
    else:
        p1b = None

    snap = lm.snapshot()
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(snap["prices"], snap["long"], color=PALETTE["long"], label="long pools",
            height=lm.buckets.bucket_size * 0.9)
    ax.barh(snap["prices"], -snap["short"], color=PALETTE["short"], label="short pools",
            height=lm.buckets.bucket_size * 0.9)
    ax.axvline(0, color=PALETTE["neutral"], lw=1)
    ax.set_title(f"{symbol}: срез H на конец периода (USD)")
    ax.legend()
    p2 = save_fig(fig, f"{symbol.lower()}_real_heat_slice", out)
    paths = [p1, p2] + ([p1b] if p1b else [])
    log.info("real_heatmap.done", bars=bars.height, liq_prints=liqs.height, figures=len(paths))
    return paths


def main(argv: list[str] | None = None, fetch=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--end", default=None, help="last day YYYY-MM-DD (default: позавчера UTC)")
    parser.add_argument("--lake", default="data/vision_lake")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--brackets", default=None)
    parser.add_argument("--skip-download", action="store_true", help="lake уже наполнен")
    args = parser.parse_args(argv)

    cfg = load_config()
    seed_everything(int(cfg["project"]["seed"]))
    lake = Path(args.lake)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        end = (
            datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
            if args.end
            else datetime.now(tz=UTC) - timedelta(days=2)  # last surely-published day
        )
        start = end - timedelta(days=args.days - 1)
        dv_args = [
            "--lake", str(lake),
            "--symbols", args.symbol,
            "--kinds", *KINDS,
            "--start", start.strftime("%Y-%m-%d"),
            "--end", end.strftime("%Y-%m-%d"),
        ]
        log.info("real_heatmap.download", args=dv_args)
        dv.main(dv_args, fetch=fetch) if fetch is not None else dv.main(dv_args)

    build_heatmap(lake, args.symbol, cfg, out, timeframe=args.timeframe, brackets_path=args.brackets)


if __name__ == "__main__":
    main()
