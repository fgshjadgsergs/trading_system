"""Gate A/B report on REAL data: does the liquidation map beat the naive
baseline at predicting real liquidations, and does price react to its pools?

    python scripts/stage3_report.py --symbol ETHUSDT --days 60

Downloads klines + metrics + liquidationSnapshot for the range, builds bars,
calibrates static leverage weights on the TRAIN window (capture-rate objective),
scores naive vs static (optionally rolling) STRICTLY out-of-sample with an
embargo gap, runs the reversal and magnet event studies on the test window,
and writes reports/stage3-<symbol>/README.md with an explicit verdict.

The Gate A verdict (headline) is the capture rate at a lag of --lag-bars bars
(default 1: the map is scored as a forecast made one bar before each print,
so hugging the current price does not pay) and, when the prints carry sides,
side-aware: a print counts only if the heat half-matrix of ITS side is hot in
its cell. lag=0 capture is reported alongside as reference.

Ground truth = real liquidation prints. Vision does NOT publish them (the
S3 listing has no liquidationSnapshot dataset), so the truth comes from a
live forceOrder recording: run scripts/record_live.py, then point --lake at
that lake with --skip-download. Without prints the verdict is "нет данных".
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import structlog

import scripts.download_vision as dv
from scripts.real_heatmap import bars_from_klines
from trading_system.calibration.event_studies import (
    magnet_study,
    reversal_study,
    top_decile_touch_events,
)
from trading_system.calibration.real_data import (
    bars_to_arrays,
    exact_bucket_grid,
    make_real_heat_builder,
)
from trading_system.calibration.weights import (
    RollingCalibrator,
    StaticWeightCalibrator,
    capture_details,
    capture_rate,
    naive_baseline_heat,
)
from trading_system.collectors.brackets import bracket_liq_price_fn, load_brackets
from trading_system.core.config import load_config, seed_everything
from trading_system.core.io import read_stream
from trading_system.core.timeutils import NS_PER_S, TIMEFRAME_NS
from trading_system.features import join_open_interest, with_atr
from trading_system.liqmap.sides import join_long_share
from trading_system.viz import build_report
from trading_system.viz.style import PALETTE, apply_style, save_fig
from trading_system.viz.templates import event_study_plot

log = structlog.get_logger()

# liquidationSnapshot удалён: датасета нет на Vision (S3-листинг, 20.08.2026);
# истина о ликвидациях приходит только из живой записи forceOrder (record_live)
KINDS = ["klines", "metrics"]
NS_PER_DAY = 86_400 * NS_PER_S


def analyze(
    bars: pl.DataFrame,
    liqs: pl.DataFrame,
    cfg: dict,
    out_dir: Path,
    symbol: str,
    timeframe: str = "5m",
    test_frac: float = 0.25,
    embargo_days: float = 2.0,
    top_decile: float = 0.1,
    brackets_path: str | None = None,
    rolling: bool = False,
    n_candidates: int = 32,
    seed: int = 42,
    lag_bars: int = 1,
) -> dict:
    """Core Gate A/B analysis over prepared bars + real liquidation prints.

    The HEADLINE Gate A metric is the capture rate at a lag of `lag_bars`
    bars (the map must predict, not hug the price) and — when the prints
    carry sides — side-aware: each print is scored only against the heat
    half-matrix of its own side. lag=0 capture is reported as reference.
    """
    lm_cfg = cfg["liqmap"]
    grid = np.array([float(x) for x in lm_cfg["leverage_grid"]])
    arr = bars_to_arrays(bars)
    liq_fn = None
    if brackets_path:
        liq_fn = bracket_liq_price_fn(load_brackets(brackets_path), symbol)
    # N5: span from the actually computed liquidation prices, no edge clamp
    edges = exact_bucket_grid(
        arr,
        grid,
        float(lm_cfg["bucket_atr_fraction"]),
        mmr=float(lm_cfg["maint_margin_rate_flat"]),
        liq_fn=liq_fn,
    )
    bar_ns = int(TIMEFRAME_NS[timeframe])
    lag_ns = int(lag_bars) * bar_ns
    builder_kwargs = dict(
        bar_s=bar_ns / NS_PER_S,
        decay_half_life_s=float(lm_cfg["decay_half_life_s"]),
        mmr=float(lm_cfg["maint_margin_rate_flat"]),
        liq_fn=liq_fn,
    )
    build = make_real_heat_builder(arr, grid, edges, **builder_kwargs)
    use_sides = "side" in liqs.columns
    build_split = (
        make_real_heat_builder(arr, grid, edges, split_sides=True, **builder_kwargs)
        if use_sides
        else None
    )

    t0, t1 = int(arr.ts[0]), int(arr.ts[-1])
    embargo = int(embargo_days * NS_PER_DAY)
    t_test = int(t1 - (t1 - t0) * test_frac)
    train_range = (t0, t_test - embargo)
    test_range = (t_test, t1 + 1)
    if train_range[1] <= train_range[0]:
        raise SystemExit("выборка короче эмбарго — увеличьте --days")

    n_liq_train = liqs.filter(
        (pl.col("ts_event") >= train_range[0]) & (pl.col("ts_event") < train_range[1])
    ).height
    n_liq_test = liqs.filter(
        (pl.col("ts_event") >= test_range[0]) & (pl.col("ts_event") < test_range[1])
    ).height
    log.info(
        "split",
        train_days=round((train_range[1] - t0) / NS_PER_DAY, 1),
        embargo_days=embargo_days,
        test_days=round((t1 - t_test) / NS_PER_DAY, 1),
        liq_train=n_liq_train,
        liq_test=n_liq_test,
    )

    capture: dict[str, float | None] = {}
    capture_lag0: dict[str, float] = {}
    capture_by_side: dict[str, float | None] = {}
    weights = None
    if n_liq_train == 0 or n_liq_test == 0:
        log.warning("no_liquidation_truth", train=n_liq_train, test=n_liq_test)
    else:
        # the baseline is an OPPONENT: score both the side-agnostic levels and
        # the same levels split by side geometry (long below price, short
        # above) and keep the stronger one. Either can win — under a lag the
        # split halves are cut at the snapshot's price, not the event's — and
        # beating only the weaker variant would be self-deception.
        side_aware = build_split is not None and "side" in liqs.columns
        naive_variants = [naive_baseline_heat(arr.close, edges)]
        if side_aware:
            naive_variants.append(naive_baseline_heat(arr.close, edges, split_sides=True))
        capture["naive"] = max(
            capture_rate(h, arr.ts, edges, liqs, top_decile, test_range, lag_ns)
            for h in naive_variants
        )
        capture_lag0["naive"] = max(
            capture_rate(h, arr.ts, edges, liqs, top_decile, test_range, 0)
            for h in naive_variants
        )
        cal = StaticWeightCalibrator(
            n_weights=len(grid), seed=seed, n_candidates=n_candidates,
            top_decile=top_decile, lag_ns=lag_ns,
        )
        log.info("calibrating_static", candidates=n_candidates, lag_bars=lag_bars)
        fit = cal.fit(build, arr.ts, edges, liqs, arr.close, ts_range=train_range)
        weights = fit.weights
        static_heat = build_split(weights) if build_split is not None else build(weights)
        capture["static"] = capture_rate(
            static_heat, arr.ts, edges, liqs, top_decile, test_range, lag_ns
        )
        capture_lag0["static"] = capture_rate(
            static_heat, arr.ts, edges, liqs, top_decile, test_range, 0
        )
        if build_split is not None:
            details = capture_details(
                static_heat, arr.ts, edges, liqs, top_decile, test_range, lag_ns
            )
            capture_by_side = {
                s: (c / t if t > 0 else None) for s, (c, t) in details[2].items()
            }
        if rolling:
            roll = RollingCalibrator(cal, train_window_ns=21 * NS_PER_DAY)
            capture["rolling"], _ = roll.oos_capture(
                build, arr.ts, edges, liqs, arr.close, test_range, top_decile, lag_ns
            )
        log.info("capture", **{key: round(v, 4) for key, v in capture.items() if v is not None})

    # event studies on the TEST window with the calibrated (or prior) map
    w_events = weights if weights is not None else np.ones(len(grid)) / len(grid)
    heat = build(w_events)
    events_all = top_decile_touch_events(arr.close, heat, edges, top_decile)
    test_mask = arr.ts[events_all] >= t_test
    events = events_all[test_mask]
    figures: list[tuple[Path, str]] = []
    rev = mag = None
    horizon = 36  # bars; 3h on 5m
    if len(events) >= 10:
        rev = reversal_study(arr.close, arr.atr, events, k_atr=1.0, horizon=horizon, seed=seed)
        p = event_study_plot(
            rev.event_paths,
            f"{symbol.lower()}_s3_reversal_paths",
            out_dir=out_dir,
            baseline=np.nanmean(rev.control_paths, axis=0),
            title=f"{symbol}: путь после касания пула (n={rev.stats.n_events}) vs контроль",
            seed=seed,
        )
        figures.append((p, "Разворот у пулов: событийный путь с ДИ vs контроль"))
        mag = magnet_study(arr.close, arr.atr, heat, edges, horizon=horizon, seed=seed)
        apply_style()
        fig, ax = plt.subplots(figsize=(10, 6))
        centers = (mag.bin_edges_atr[:-1] + mag.bin_edges_atr[1:]) / 2
        ax.errorbar(
            centers,
            mag.p_reach,
            yerr=[mag.p_reach - mag.ci_low, mag.ci_high - mag.p_reach],
            marker="o",
            color=PALETTE["accent"],
            capsize=3,
        )
        ax.set_xlabel("дистанция до пула, ATR")
        ax.set_ylabel(f"P(дойти за {horizon} баров)")
        ax.set_title(f"{symbol}: магнит — вероятность дойти до пула")
        figures.append((save_fig(fig, f"{symbol.lower()}_s3_magnet", out_dir), "Магнит P(d, T)"))
    else:
        log.warning("too_few_events_for_studies", n=len(events))

    if capture:
        apply_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        keys = [key for key in ("naive", "static", "rolling") if capture.get(key) is not None]
        ax.bar(keys, [capture[key] for key in keys],
               color=[PALETTE["neutral"], PALETTE["accent"], PALETTE["long"]][: len(keys)])
        ax.set_ylabel("OOS capture rate (доля liq$ в топ-дециле тепла)")
        ax.set_title(f"{symbol}: Gate A — карта против наивного бейзлайна")
        figures.append(
            (save_fig(fig, f"{symbol.lower()}_s3_capture_ladder", out_dir), "Gate A: capture ladder")
        )

    gate_a = (
        capture.get("static") is not None
        and capture.get("naive") is not None
        and capture["static"] > capture["naive"]
    )
    gate_b_sig = rev is not None and rev.stats.p_value < 0.05 and rev.stats.effect > 0
    return {
        "capture": capture,  # HEADLINE: lag = lag_bars, side-aware when sides present
        "capture_lag0": capture_lag0,  # reference only
        "capture_by_side": capture_by_side,
        "lag_bars": int(lag_bars),
        "side_aware": build_split is not None,
        "weights": weights.tolist() if weights is not None else None,
        "gate_a": gate_a if capture else None,
        "reversal_effect": rev.stats.effect if rev else None,
        "reversal_p": rev.stats.p_value if rev else None,
        "gate_b_significant": gate_b_sig if rev else None,
        "n_events_test": int(len(events)),
        "n_liq_train": n_liq_train,
        "n_liq_test": n_liq_test,
        "figures": figures,
    }


def main(argv: list[str] | None = None, fetch=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="ETHUSDT")
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--end", default=None, help="последний день YYYY-MM-DD (default: позавчера)")
    parser.add_argument("--lake", default="data/vision_lake")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--out", default="reports")
    parser.add_argument("--brackets", default=None)
    parser.add_argument("--rolling", action="store_true", help="добавить скользящую ступень (дольше)")
    parser.add_argument("--test-frac", type=float, default=0.25)
    parser.add_argument("--embargo-days", type=float, default=2.0)
    parser.add_argument("--candidates", type=int, default=32, help="кандидатов в калибраторе")
    parser.add_argument(
        "--lag-bars", type=int, default=1,
        help="лаг (в барах) headline-метрики Gate A; 0 = скоринг по текущему снапшоту",
    )
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config()
    seed = int(cfg["project"]["seed"])
    seed_everything(seed)
    lake = Path(args.lake)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        end = (
            datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
            if args.end
            else datetime.now(UTC) - timedelta(days=2)
        )
        start = end - timedelta(days=args.days - 1)
        dv_args = [
            "--lake", str(lake),
            "--symbols", args.symbol,
            "--kinds", *KINDS,
            "--intervals", args.timeframe,
            "--start", start.strftime("%Y-%m-%d"),
            "--end", end.strftime("%Y-%m-%d"),
        ]
        log.info("stage3.download", args=dv_args)
        dv.main(dv_args, fetch=fetch) if fetch is not None else dv.main(dv_args)

    klines = read_stream(lake, "kline", symbol=args.symbol)
    bars = bars_from_klines(klines, args.timeframe)
    if bars.is_empty():
        raise SystemExit(f"нет {args.timeframe}-клайнов для {args.symbol} в лейке")
    oi = read_stream(lake, "open_interest", symbol=args.symbol)
    ratios = read_stream(lake, "ratio", symbol=args.symbol)
    bars = with_atr(join_open_interest(bars, oi), period=14)
    if ratios.height:
        bars = join_long_share(bars, ratios)
    liqs = read_stream(lake, "liquidation", symbol=args.symbol)
    liq_days = (
        liqs.with_columns((pl.col("ts_event") // NS_PER_DAY).alias("_d"))["_d"].n_unique()
        if liqs.height
        else 0
    )
    total_days = bars.with_columns((pl.col("ts_open") // NS_PER_DAY).alias("_d"))["_d"].n_unique()
    log.info("data", bars=bars.height, liq_prints=liqs.height, liq_days=liq_days, days=total_days)

    res = analyze(
        bars,
        liqs,
        cfg,
        out,
        args.symbol,
        timeframe=args.timeframe,
        test_frac=args.test_frac,
        embargo_days=args.embargo_days,
        brackets_path=args.brackets,
        rolling=args.rolling,
        n_candidates=args.candidates,
        seed=seed,
        lag_bars=args.lag_bars,
    )

    cap = res["capture"]
    lines = [
        f"Символ: {args.symbol}, {total_days} дней {args.timeframe}-баров; "
        f"принтов ликвидаций: train {res['n_liq_train']} / test {res['n_liq_test']} "
        f"({liq_days} дней с данными liquidationSnapshot).",
        "",
    ]
    if res["gate_a"] is None:
        verdict = (
            "**Gate A: нет вердикта — в выборке нет реальных принтов ликвидаций.** "
            "Vision не публикует liquidationSnapshot за эти даты; варианты: другой период "
            "(--end), другой символ, либо живая запись forceOrder (scripts/record_live.py)."
        )
    else:
        naive_s = f"{cap['naive']:.4f}"
        static_s = f"{cap['static']:.4f}"
        metric_s = (
            f"headline-метрика: OOS capture с lag={res['lag_bars']} бар"
            + (", side-aware по сторонам принтов" if res["side_aware"] else "")
        )
        if res["gate_a"]:
            verdict = (
                f"**Gate A: ПРОЙДЕН — карта бьёт наивный бейзлайн OOS: "
                f"static {static_s} > naive {naive_s}** ({metric_s})."
            )
        else:
            verdict = (
                f"**Gate A: НЕ пройден — static {static_s} vs naive {naive_s} OOS** "
                f"({metric_s}). "
                "По чеклисту это стоп и возврат к модели карты, не повод подгонять параметры."
            )
        if cap.get("rolling") is not None:
            verdict += f" Rolling: {cap['rolling']:.4f}."
        cap0 = res["capture_lag0"]
        verdict += (
            f" Справочно lag=0: static {cap0['static']:.4f} vs naive {cap0['naive']:.4f}."
        )
        by_side = {s: v for s, v in res["capture_by_side"].items() if v is not None}
        if by_side:
            verdict += " По сторонам (static, lag): " + ", ".join(
                f"{s} {v:.4f}" for s, v in sorted(by_side.items())
            ) + "."
    lines.append(verdict)
    if res["gate_b_significant"] is not None:
        lines.append("")
        eff = res["reversal_effect"]
        pv = res["reversal_p"]
        state = "значим" if res["gate_b_significant"] else "НЕ значим"
        lines.append(
            f"Event study (к Gate B): эффект разворота у пулов {eff:+.4f} "
            f"(p={pv:.3f}, {res['n_events_test']} событий OOS) — {state} "
            "(издержки ещё не учтены — это следующий шаг Gate B)."
        )
    if res["weights"] is not None:
        grid = cfg["liqmap"]["leverage_grid"]
        w_str = ", ".join(
            f"{int(g)}x:{w:.2f}" for g, w in zip(grid, res["weights"], strict=True)
        )
        lines.append("")
        lines.append(f"Калиброванные веса плеч (train): {w_str}")

    index = build_report(
        f"stage3-{args.symbol.lower()}",
        res["figures"],
        out_root=out,
        intro="\n".join(lines),
    )
    for line in lines:
        if line:
            log.info("verdict", text=line.replace("**", ""))
    log.info("stage3.done", report=str(index), figures=len(res["figures"]))


if __name__ == "__main__":
    main()
