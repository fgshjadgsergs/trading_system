"""End-to-end pipeline (Gate 2): запись → стакан → фичи → карта → профиль →
сигналы → бэктест → отчёт, одним скриптом без ручных шагов.

By default the "recording" stage synthesizes a seeded market (no network),
writes it through the real batch writer into a parquet lake, and every later
stage reads ONLY from that lake through the unified reader — the same code
path a live recording or a normalized Vision archive would take.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import polars as pl
import structlog

from trading_system.backtest import (
    BacktestConfig,
    Bar,
    Context,
    Order,
    OrderType,
    run_backtest,
)
from trading_system.backtest.metrics import cost_waterfall, summary
from trading_system.book import BookReplayer
from trading_system.book.reports import book_heatmap, spread_depth_figure
from trading_system.collectors.recorder import BatchWriter
from trading_system.core.config import load_config, seed_everything
from trading_system.core.io import read_stream, write_batch
from trading_system.core.schema import Side, records_to_frame
from trading_system.core.synth import (
    synth_book_stream,
    synth_mark_prices,
    synth_open_interest,
    synth_trades,
)
from trading_system.features import (
    build_multitf,
    join_context,
    join_open_interest,
    time_bars,
    with_atr,
    with_cvd,
)
from trading_system.liqmap import HeatHistory, LiqMap, PriceBuckets, StaticWeights
from trading_system.profile import equal_extremes, fractal_swings, profile
from trading_system.signals import s1_magnet, s2_sweep_reversal, s3_filter
from trading_system.viz import build_report, dist_plot, overlay_chart
from trading_system.viz.style import PALETTE, apply_style, save_fig

log = structlog.get_logger()


# -- stage 1: запись ----------------------------------------------------------


def stage_record(lake: Path, symbol: str, seed: int, n_trades: int) -> None:
    trades = synth_trades(n=n_trades, symbol=symbol, mean_gap_ms=200.0, seed=seed)
    start = trades[0].ts_event
    writer = BatchWriter(lake, max_rows=20_000, max_age_s=3600.0)
    for t in trades:
        writer.add(t)
    writer.flush_all()
    oi = synth_open_interest(symbol=symbol, start_ts=start, n=1_500, step_s=7, seed=seed)
    write_batch(lake, "open_interest", records_to_frame(oi, "open_interest"))
    marks = synth_mark_prices(trades, every_s=5)
    write_batch(lake, "mark_price", records_to_frame(marks, "mark_price"))
    book = synth_book_stream(n_diffs=1_500, symbol=symbol, start_ts=start, seed=seed)
    write_batch(lake, "book_snapshot", records_to_frame([book.snapshot], "book_snapshot"))
    write_batch(lake, "depth_diff", records_to_frame(book.diffs, "depth_diff"))
    log.info("record.done", trades=len(trades), oi=len(oi), diffs=len(book.diffs))


# -- stage 2: стакан ----------------------------------------------------------


def stage_book(lake: Path, symbol: str, out: Path) -> list[tuple[Path, str]]:
    snap = read_stream(lake, "book_snapshot", symbol=symbol)
    diffs = read_stream(lake, "depth_diff", symbol=symbol)
    replayer = BookReplayer.from_frames(snap, diffs)
    grid = replayer.sample_grid(interval_ns=10 * 1_000_000_000, n=40)
    p1 = book_heatmap(grid, tick=0.1, name="pipeline_book_heatmap", out_dir=out)
    replayer2 = BookReplayer.from_frames(snap, diffs)
    metrics = replayer2.sample_metrics(interval_ns=10 * 1_000_000_000)
    p2 = spread_depth_figure(metrics, depth_pct=0.005, name="pipeline_spread_depth", out_dir=out)
    log.info("book.done", grid_rows=grid.height)
    return [(p1, "Стакан: heatmap глубины"), (p2, "Спред и глубина ±0.5%")]


# -- stage 3-6: фичи → карта → профиль → сигналы ------------------------------


def stage_features(lake: Path, symbol: str, timeframe: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    trades = read_stream(lake, "trade", symbol=symbol)
    oi = read_stream(lake, "open_interest", symbol=symbol)
    bars = with_atr(join_open_interest(with_cvd(time_bars(trades, timeframe)), oi), period=14)
    mtf = build_multitf(trades, oi, ["1m", "5m", "15m"], zscore_window=24)
    bars = join_context(bars, mtf, ["5m", "15m"])
    log.info("features.done", bars=bars.height)
    return trades, bars


def stage_map(bars: pl.DataFrame, cfg: dict) -> tuple[LiqMap, HeatHistory, pl.DataFrame]:
    lm_cfg = cfg["liqmap"]
    atr = float(bars["atr"].drop_nulls().median())
    grid = [float(x) for x in lm_cfg["leverage_grid"]]
    weights = np.array([1, 2, 4, 6, 5, 4, 2, 2, 1], dtype=float)[: len(grid)]
    lm = LiqMap(
        leverage_grid=grid,
        buckets=PriceBuckets.from_atr(atr, float(lm_cfg["bucket_atr_fraction"])),
        weight_fn=StaticWeights(weights),
        long_share=float(lm_cfg["long_share_default"]),
        decay_half_life_s=float(lm_cfg["decay_half_life_s"]),
    )
    hist = HeatHistory(lm)
    for row in bars.iter_rows(named=True):
        d_oi = row["d_oi_usd"]
        if d_oi is None:
            d_oi = row["quote_volume"] * 0.05
        lm.step(
            bar_low=row["low"],
            bar_high=row["high"],
            bar_close=row["close"],
            d_oi_usd=d_oi,
            dt_s=(row["ts_close"] - row["ts_open"]) / 1e9,
        )
        hist.record(row["ts_close"])
    pools = pl.DataFrame(
        {
            "price": [p for p, _, _ in lm.top_pools(8)],
            "heat_usd": [h for _, h, _ in lm.top_pools(8)],
            "touched_ts": pl.Series([None] * len(lm.top_pools(8)), dtype=pl.Int64),
        }
    )
    log.info("map.done", total_heat=round(lm.total_heat()), pools=pools.height)
    return lm, hist, pools


def heat_zones(lm: LiqMap, quantile: float) -> pl.DataFrame:
    snap = lm.snapshot()
    heat = snap["long"] + snap["short"]
    if heat.size == 0:
        return pl.DataFrame({"lo": [], "hi": [], "heat_usd": []})
    half = lm.buckets.bucket_size / 2
    return pl.DataFrame(
        {
            "lo": (snap["prices"] - half).tolist(),
            "hi": (snap["prices"] + half).tolist(),
            "heat_usd": heat.tolist(),
        }
    ).filter(pl.col("heat_usd") > 0)


def stage_signals(
    bars: pl.DataFrame, pools: pl.DataFrame, lm: LiqMap, trades: pl.DataFrame, cfg: dict
) -> pl.DataFrame:
    s_cfg = cfg["signals"]
    span = float(bars["high"].max() - bars["low"].min())
    swings = fractal_swings(bars, n=2)
    clusters = equal_extremes(swings, eps=span / 100)
    ev1 = s1_magnet(bars, pools, k_atr=float(s_cfg["s1_magnet_max_dist_atr"]), min_heat_share=0.2)
    ev2 = s2_sweep_reversal(bars, clusters, return_bars=int(s_cfg["s2_sweep_return_bars"]))
    events = pl.concat([e for e in (ev1, ev2) if e.height]) if (ev1.height or ev2.height) else ev1
    events = s3_filter(events, heat_zones(lm, 0.9), dense_quantile=float(s_cfg["s3_dense_zone_quantile"]))
    log.info("signals.done", s1=ev1.height, s2=ev2.height, blocked=int(events["blocked"].sum()) if events.height else 0)
    return events


# -- stage 7: бэктест ---------------------------------------------------------


class EventStrategy:
    """Enter on unblocked signal events at bar close; exit at target or timeout."""

    def __init__(self, events: pl.DataFrame, notional_usd: float = 10_000.0, max_hold_bars: int = 24) -> None:
        self._events = {int(r["ts"]): r for r in events.filter(~pl.col("blocked")).iter_rows(named=True)}
        self._notional = notional_usd
        self._max_hold = max_hold_bars
        self._entry_bar: int | None = None
        self._target: float | None = None

    def on_bar(self, bar: Bar, ctx: Context):
        orders: list[Order] = []
        if ctx.position_qty != 0.0 and self._entry_bar is not None:
            hit = self._target is not None and (
                (ctx.position_qty > 0 and bar.close >= self._target)
                or (ctx.position_qty < 0 and bar.close <= self._target)
            )
            if hit or bar.index - self._entry_bar >= self._max_hold:
                side = Side.SELL if ctx.position_qty > 0 else Side.BUY
                orders.append(Order(side=side, qty=abs(ctx.position_qty), order_type=OrderType.MARKET))
                self._entry_bar, self._target = None, None
                return orders
        ev = self._events.get(bar.ts_close)
        if ev is not None and ctx.position_qty == 0.0 and ctx.n_pending == 0:
            side = Side.BUY if ev["side"] > 0 else Side.SELL
            qty = self._notional / bar.close
            orders.append(Order(side=side, qty=qty, order_type=OrderType.MARKET))
            self._entry_bar = bar.index
            self._target = ev["target"]
        return orders

    def on_trade(self, trade, ctx):
        return None


def stage_backtest(
    trades: pl.DataFrame, events: pl.DataFrame, lake: Path, symbol: str, cfg: dict, seed: int, out: Path
) -> list[tuple[Path, str]]:
    marks = read_stream(lake, "mark_price", symbol=symbol)
    bt_cfg = BacktestConfig.from_config(cfg, timeframe="1m", seed=seed)
    result = run_backtest(trades, EventStrategy(events), bt_cfg, mark_prices=marks)
    stats = summary(result)
    log.info("backtest.done", **{k: round(v, 2) for k, v in stats.items()})
    apply_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    eq = result.equity_curve
    ax.plot(np.asarray(eq["ts"]), np.asarray(eq["equity"]), color=PALETTE["neutral"])
    ax.set_title(f"Equity ({symbol}, {len(result.fills)} fills)")
    p1 = save_fig(fig, "pipeline_equity", out)
    wf = cost_waterfall(result)
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.bar(wf["step"].to_list(), wf["value"].to_list(), color=PALETTE["accent"])
    ax2.set_title("PnL decomposition / cost waterfall")
    ax2.tick_params(axis="x", rotation=30)
    p2 = save_fig(fig2, "pipeline_costs", out)
    figs = [(p1, "Бэктест: equity"), (p2, "Бэктест: водопад издержек")]
    if len(result.fills) >= 2:
        pnls = np.array([f.slippage_usd for f in result.fills])
        figs.append((dist_plot(pnls, "pipeline_slippage", out, title="Slippage per fill"), "Проскальзывание"))
    return figs


# -- main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--lake", default=None, help="existing lake dir (skips synthetic recording)")
    parser.add_argument("--out", default="reports", help="report output dir")
    parser.add_argument("--n-trades", type=int, default=150_000)
    parser.add_argument("--timeframe", default="5m")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["project"]["seed"])
    seed_everything(seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.lake:
        lake = Path(args.lake)
    else:
        import tempfile

        lake = Path(tempfile.mkdtemp(prefix="pipeline_lake_"))
        stage_record(lake, args.symbol, seed, args.n_trades)

    figures: list[tuple[Path, str]] = []
    figures += stage_book(lake, args.symbol, out)
    trades, bars = stage_features(lake, args.symbol, args.timeframe)
    lm, hist, pools = stage_map(bars, cfg)
    events = stage_signals(bars, pools, lm, trades, cfg)
    span = float(bars["high"].max() - bars["low"].min())
    prof = profile(trades, bin_size=span / 60)
    p_overlay = overlay_chart(
        bars,
        heat=hist.matrix(),
        events=events,
        profile=prof,
        levels=pools.select("price"),
        name="pipeline_overlay",
        out_dir=out,
        title=f"{args.symbol}: свечи + карта ликвидаций + сигналы",
    )
    figures.append((p_overlay, "Свечи + карта + сигналы + профиль"))
    figures += stage_backtest(trades, events, lake, args.symbol, cfg, seed, out)
    index = build_report(
        "pipeline",
        figures,
        out_root=out,
        intro="Сквозной прогон: запись → стакан → фичи → карта → профиль → сигналы → бэктест.",
    )
    log.info("pipeline.done", report=str(index), figures=len(figures))


if __name__ == "__main__":
    main()
