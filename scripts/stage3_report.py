"""Gate A/B report on REAL data: does the liquidation map beat the naive
baseline at predicting real liquidations, and does price react to its pools?

    python scripts/stage3_report.py --symbol ETHUSDT --days 60

Downloads klines + metrics + liquidationSnapshot for the range, builds bars,
calibrates static leverage weights on the TRAIN window (capture-rate objective),
scores naive vs static (optionally rolling) STRICTLY out-of-sample with an
embargo gap, runs the reversal and magnet event studies on the test window,
and writes reports/stage3-<symbol>/README.md with an explicit verdict.

Ground truth = real liquidation prints from Vision liquidationSnapshot
archives. Days without that archive carry no truth; if the whole range lacks
it, the verdict is "нет данных" — record forceOrder live or pick другой период.
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
    bucket_grid,
    make_real_heat_builder,
)
from trading_system.calibration.weights import (
    RollingCalibrator,
    StaticWeightCalibrator,
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

KINDS = ["klines", "metrics", "liquidationSnapshot"]
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
) -> dict:
    """Core Gate A/B analysis over prepared bars + real liquidation prints."""
    lm_cfg = cfg["liqmap"]
    grid = np.array([float(x) for x in lm_cfg["leverage_grid"]])
    arr = bars_to_arrays(bars)
    edges = bucket_grid(arr, float(lm_cfg["bucket_atr_fraction"]))
    liq_fn = None
    if brackets_path:
        liq_fn = bracket_liq_price_fn(load_brackets(brackets_path), symbol)
    build = make_real_heat_builder(
        arr,
        grid,
        edges,
        bar_s=TIMEFRAME_NS[timeframe] / NS_PER_S,
        decay_half_life_s=float(lm_cfg["decay_half_life_s"]),
        mmr=float(lm_cfg["maint_margin_rate_flat"]),
        liq_fn=liq_fn,
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
    weights = None
    if n_liq_train == 0 or n_liq_test == 0:
        log.warning("no_liquidation_truth", train=n_liq_train, test=n_liq_test)
    else:
        naive = naive_baseline_heat(arr.close, edges)
        capture["naive"] = capture_rate(naive, arr.ts, edges, liqs, top_decile, test_range)
        cal = StaticWeightCalibrator(
            n_weights=len(grid), seed=seed, n_candidates=n_candidates, top_decile=top_decile
        )
        log.info("calibrating_static", candidates=n_candidates)
        fit = cal.fit(build, arr.ts, edges, liqs, arr.close, ts_range=train_range)
        weights = fit.weights
        capture["static"] = capture_rate(
            build(weights), arr.ts, edges, liqs, top_decile, test_range
        )
        if rolling:
            roll = RollingCalibrator(cal, train_window_ns=21 * NS_PER_DAY)
            capture["rolling"], _ = roll.oos_capture(
                build, arr.ts, edges, liqs, arr.close, test_range, top_decile, 0
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
        "capture": capture,
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
        if res["gate_a"]:
            verdict = (
                f"**Gate A: ПРОЙДЕН — карта бьёт наивный бейзлайн OOS: "
                f"static {static_s} > naive {naive_s}.**"
            )
        else:
            verdict = (
                f"**Gate A: НЕ пройден — static {static_s} vs naive {naive_s} OOS.** "
                "По чеклисту это стоп и возврат к модели карты, не повод подгонять параметры."
            )
        if cap.get("rolling") is not None:
            verdict += f" Rolling: {cap['rolling']:.4f}."
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
