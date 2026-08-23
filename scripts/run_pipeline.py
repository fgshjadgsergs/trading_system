"""End-to-end pipeline (Gate 2): запись → стакан → фичи → карта → профиль →
сигналы → бэктест → отчёт, одним скриптом без ручных шагов.

By default the "recording" stage synthesizes a seeded market (no network),
writes it through the real batch writer into a parquet lake, and every later
stage reads ONLY from that lake through the unified reader — the same code
path a live recording or a normalized Vision archive would take.
"""

from __future__ import annotations

import argparse
import math
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
from trading_system.collectors.brackets import bracket_liq_price_fn, load_brackets
from trading_system.collectors.recorder import BatchWriter
from trading_system.core.config import load_config, seed_everything
from trading_system.core.io import read_stream, write_batch
from trading_system.core.schema import Side, Trade, records_to_frame
from trading_system.core.synth import (
    synth_book_stream,
    synth_mark_prices,
    synth_open_interest,
    synth_ratios,
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
from trading_system.liqmap.sides import join_long_share
from trading_system.profile import equal_extremes, fractal_swings, profile
from trading_system.signals import s2_sweep_reversal, s3_filter
from trading_system.signals.detectors import EVENT_SCHEMA
from trading_system.spoof.lifecycle import BookState, LevelJournal
from trading_system.spoof.metrics import annotate_episodes
from trading_system.spoof.score import score_episodes
from trading_system.spoof.walls import merge_zones, wall_zones_at
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
    ratios = synth_ratios(symbol=symbol, start_ts=start, n=300, step_s=300, seed=seed)
    write_batch(lake, "ratio", records_to_frame(ratios, "ratio"))
    log.info("record.done", trades=len(trades), oi=len(oi), diffs=len(book.diffs), ratios=len(ratios))


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


def stage_map(
    bars: pl.DataFrame,
    cfg: dict,
    symbol: str,
    ratios: pl.DataFrame | None = None,
    brackets_path: str | None = None,
    entry_price: str = "close",
    sides_kwargs: dict | None = None,
) -> tuple[LiqMap, HeatHistory]:
    """`entry_price` (M4): "close" (default, бит-в-бит старое поведение) or
    "vwap" — allocate each bar's ΔOI at the bar VWAP (vwap_bar column, or
    quote_volume/volume when present; an honest error otherwise). Bars with
    zero volume fall back to close."""
    if entry_price not in ("close", "vwap"):
        raise ValueError(f"entry_price must be 'close' or 'vwap', got {entry_price!r}")
    lm_cfg = cfg["liqmap"]
    atr = float(bars["atr"].drop_nulls().median())
    grid = [float(x) for x in lm_cfg["leverage_grid"]]
    # seed hump over [3, 5, 10, 20, 25, 30, 40, 50, 60, 75, 100, 125]:
    # peak at 20x, monotone tails (R4 added the 30/40/60 rungs)
    weights = np.array([1, 2, 4, 6, 5, 5, 4, 4, 3, 2, 2, 1], dtype=float)[: len(grid)]
    weights = weights / weights.sum()
    liq_fn = None
    if brackets_path:  # track B1: real per-symbol margin tiers instead of flat MMR
        liq_fn = bracket_liq_price_fn(load_brackets(brackets_path), symbol)
    lm = LiqMap(
        leverage_grid=grid,
        buckets=PriceBuckets.from_atr(atr, float(lm_cfg["bucket_atr_fraction"])),
        weight_fn=StaticWeights(weights),
        liq_price_fn=liq_fn,
        long_share=float(lm_cfg["long_share_default"]),
        decay_half_life_s=float(lm_cfg["decay_half_life_s"]),
    )
    if ratios is not None and ratios.height:  # track B2: causal per-bar side shares
        bars = join_long_share(bars, ratios, **(sides_kwargs or {}))
    if entry_price == "vwap":
        if "vwap_bar" in bars.columns:
            bars = bars.with_columns(pl.col("vwap_bar").alias("_entry"))
        elif "quote_volume" in bars.columns and "volume" in bars.columns:
            bars = bars.with_columns(
                (pl.col("quote_volume") / pl.col("volume")).alias("_entry")
            )
        else:
            raise ValueError(
                "entry_price='vwap' requires a vwap_bar column or quote_volume+volume"
            )
    hist = HeatHistory(lm)
    for row in bars.iter_rows(named=True):
        d_oi = row["d_oi_usd"]
        if d_oi is None:
            d_oi = row["quote_volume"] * 0.05
        entry = row["close"]
        if entry_price == "vwap":
            vw = row["_entry"]
            if vw is not None and math.isfinite(vw) and vw > 0:
                entry = vw  # zero-volume bars keep the close entry
        lm.step(
            bar_low=row["low"],
            bar_high=row["high"],
            bar_close=entry,
            d_oi_usd=d_oi,
            dt_s=(row["ts_close"] - row["ts_open"]) / 1e9,
            long_share=row.get("long_share"),
        )
        hist.record(row["ts_close"])
    log.info(
        "map.done",
        total_heat=round(lm.total_heat()),
        snapshots=len(hist),
        brackets=bool(brackets_path),
        ratio_shares=ratios is not None and ratios.height > 0,
    )
    return lm, hist


def causal_s1_events(
    bars: pl.DataFrame, hist: HeatHistory, k_atr: float, min_heat_share: float
) -> pl.DataFrame:
    """S1 over the CONCURRENT map snapshot of each bar (никакого будущего тепла).

    Snapshot i is the map state at bar i's close; the fire-once rule matches
    s1_magnet's edge trigger. A consumed pool disappears from later snapshots,
    which is exactly the "нетронутый пул" condition.
    """
    rows: list[dict] = []
    fired: set[float] = set()
    for i, bar in enumerate(bars.iter_rows(named=True)):
        atr = bar["atr"]
        if atr is None or i >= len(hist):
            continue
        total = hist.total_at(i)
        if total <= 0:
            continue
        close, ts = bar["close"], bar["ts_close"]
        in_range = [(p, h) for p, h in hist.pools_at(i, k=8) if abs(p - close) <= k_atr * atr]
        if not in_range:
            continue
        price, heat = max(in_range, key=lambda ph: ph[1])
        if heat < min_heat_share * total or price in fired:
            continue
        fired.add(price)
        rows.append(
            {
                "ts": ts,
                "signal": "s1",
                "side": 1 if price > close else -1,
                "price": close,
                "target": price,
                "meta": heat,
            }
        )
    return pl.DataFrame(rows, schema=EVENT_SCHEMA).sort("ts")


def causal_s3_filter(
    events: pl.DataFrame,
    bars: pl.DataFrame,
    hist: HeatHistory,
    dense_quantile: float,
    walls: pl.DataFrame | None = None,
    wall_band: float = 0.0,
) -> pl.DataFrame:
    """S3 veto per event against the map snapshot CONCURRENT with that event.

    With `walls` (scored M6 episodes), stability-weighted wall zones as of the
    event moment join the map zones — track B3.
    """
    if events.is_empty():
        return s3_filter(events, pl.DataFrame({"lo": [], "hi": [], "heat_usd": []}))
    index_of_ts = {int(t): i for i, t in enumerate(bars["ts_close"].to_list())}
    parts: list[pl.DataFrame] = []
    for ev in events.iter_rows(named=True):
        i = index_of_ts.get(ev["ts"])
        one = pl.DataFrame([ev], schema_overrides=EVENT_SCHEMA)
        if i is None or i >= len(hist):
            zones = pl.DataFrame({"lo": [], "hi": [], "heat_usd": []})
        else:
            lo, hi, heat = hist.zones_at(i)
            zones = pl.DataFrame({"lo": lo, "hi": hi, "heat_usd": heat})
        if walls is not None and walls.height and wall_band > 0:
            zones = merge_zones(zones, wall_zones_at(walls, ev["ts"], band=wall_band))
        parts.append(s3_filter(one, zones, dense_quantile=dense_quantile))
    return pl.concat(parts).sort("ts")


def stage_walls(lake: Path, symbol: str) -> pl.DataFrame | None:
    """Scored level episodes from the recorded book + tape (M6 -> S3 zones)."""
    snap = read_stream(lake, "book_snapshot", symbol=symbol)
    diffs = read_stream(lake, "depth_diff", symbol=symbol)
    if snap.is_empty() or diffs.is_empty():
        return None
    replayer = BookReplayer.from_frames(snap, diffs)
    states = []
    for ts, book in replayer.replay():
        bids, asks = book.top_n(20)
        states.append(BookState(ts=ts, bids=bids, asks=asks))
    if not states:
        return None
    lo_ts, hi_ts = states[0].ts, states[-1].ts
    tape = [
        Trade(
            exchange=r["exchange"],
            symbol=r["symbol"],
            ts_event=r["ts_event"],
            ts_recv=r["ts_recv"],
            price=r["price"],
            qty=r["qty"],
            qty_usd=r["qty_usd"],
            side=Side(r["side"]),
            trade_id=r["trade_id"],
        )
        for r in read_stream(
            lake, "trade", symbol=symbol, ts_from=lo_ts, ts_to=hi_ts + 1
        ).iter_rows(named=True)
    ]
    journal = LevelJournal().run(states, tape)
    episodes = score_episodes(annotate_episodes(journal))
    log.info("walls.done", states=len(states), episodes=episodes.height)
    return episodes


def stage_signals(
    bars: pl.DataFrame,
    hist: HeatHistory,
    cfg: dict,
    walls: pl.DataFrame | None = None,
    wall_band: float = 0.0,
) -> pl.DataFrame:
    s_cfg = cfg["signals"]
    # eps scale from the EARLY window's ATR — no full-sample statistics
    warmup_atr = float(bars["atr"].drop_nulls().head(50).median())
    eps = float(cfg["profile"]["equal_extreme_eps_atr"]) * warmup_atr
    swings = fractal_swings(bars, n=2)
    clusters = equal_extremes(swings, eps=eps).with_columns(
        pl.col("ts_last").alias("ts_formed")  # level exists only once fully formed
    )
    ev1 = causal_s1_events(
        bars, hist, k_atr=float(s_cfg["s1_magnet_max_dist_atr"]), min_heat_share=0.1
    )
    ev2 = s2_sweep_reversal(bars, clusters, return_bars=int(s_cfg["s2_sweep_return_bars"]))
    events = pl.concat([e for e in (ev1, ev2) if e.height]) if (ev1.height or ev2.height) else ev1
    events = causal_s3_filter(
        events,
        bars,
        hist,
        dense_quantile=float(s_cfg["s3_dense_zone_quantile"]),
        walls=walls,
        wall_band=wall_band,
    )
    log.info(
        "signals.done",
        s1=ev1.height,
        s2=ev2.height,
        blocked=int(events["blocked"].sum()) if events.height else 0,
    )
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
    parser.add_argument("--brackets", default=None, help="json with per-symbol leverage brackets")
    parser.add_argument(
        "--entry-price", default="close", choices=("close", "vwap"),
        help="M4: цена входа для раскладки ΔOI (vwap = bar VWAP; default close)",
    )
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
    ratios = read_stream(lake, "ratio", symbol=args.symbol)
    lm, hist = stage_map(
        bars, cfg, args.symbol, ratios=ratios, brackets_path=args.brackets,
        entry_price=args.entry_price,
    )
    walls = stage_walls(lake, args.symbol)
    events = stage_signals(bars, hist, cfg, walls=walls, wall_band=lm.buckets.bucket_size)
    span = float(bars["high"].max() - bars["low"].min())
    prof = profile(trades, bin_size=span / 60)
    final_pools = pl.DataFrame({"price": [p for p, _ in hist.pools_at(len(hist) - 1, k=8)]})
    p_overlay = overlay_chart(
        bars,
        heat=hist.matrix(),
        events=events,
        profile=prof,
        levels=final_pools,  # reporting only: final-state pool lines for context
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
