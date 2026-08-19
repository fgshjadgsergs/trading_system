"""M3 report figures: candles+CVD+ΔOI, feature correlations, z-volume distributions."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import polars as pl
import seaborn as sns

from trading_system.core.schema import records_to_frame
from trading_system.core.synth import synth_open_interest, synth_trades
from trading_system.core.timeutils import NS_PER_S
from trading_system.features.bars import time_bars, with_cvd
from trading_system.features.joins import join_open_interest
from trading_system.features.multitf import build_multitf
from trading_system.viz.style import apply_style, save_fig


def bars_to_pandas(bars: pl.DataFrame) -> pd.DataFrame:
    """Kline frame -> mplfinance-ready OHLCV with a UTC DatetimeIndex."""
    df = bars.select(
        pl.from_epoch("ts_open", time_unit="ns").alias("dt"),
        pl.col("open").alias("Open"),
        pl.col("high").alias("High"),
        pl.col("low").alias("Low"),
        pl.col("close").alias("Close"),
        pl.col("volume").alias("Volume"),
    ).to_pandas()
    return df.set_index("dt")


def candles_cvd_doi(
    bars: pl.DataFrame, name: str = "m3_candles_cvd_doi", out_dir: Path | None = None
) -> Path:
    """5m candles with CVD and ΔOI panels (mplfinance)."""
    pdf = bars_to_pandas(bars)
    adds = [
        mpf.make_addplot(bars["cvd_usd"].to_numpy(), panel=1, ylabel="CVD $"),
    ]
    if "d_oi_usd" in bars.columns:
        adds.append(
            mpf.make_addplot(
                bars["d_oi_usd"].fill_null(0.0).to_numpy(), panel=2, type="bar", ylabel="ΔOI $"
            )
        )
    fig, _ = mpf.plot(
        pdf,
        type="candle",
        style="yahoo",
        addplot=adds,
        volume=False,
        returnfig=True,
        figsize=(14, 9),
    )
    return save_fig(fig, name, out_dir)


def feature_corr_heatmap(
    joined: pl.DataFrame, cols: list[str], name: str = "m3_feature_corr", out_dir: Path | None = None
) -> Path:
    apply_style()
    corr = joined.select(cols).drop_nulls().to_pandas().corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Feature correlations")
    return save_fig(fig, name, out_dir)


def zvol_distributions(
    mtf: pl.DataFrame, name: str = "m3_zvol_by_tf", out_dir: Path | None = None
) -> Path:
    apply_style()
    pdf = mtf.select("tf", "vol_z").drop_nulls().to_pandas()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.kdeplot(data=pdf, x="vol_z", hue="tf", common_norm=False, ax=ax, clip=(-4, 8))
    ax.set_title("Volume z-score distribution per timeframe")
    return save_fig(fig, name, out_dir)


def demo_reports(out_dir: Path, seed: int = 42) -> list[Path]:
    trades = records_to_frame(synth_trades(n=40_000, mean_gap_ms=250.0, seed=seed), "trade")
    start = int(trades["ts_event"].min())
    oi = records_to_frame(
        synth_open_interest(n=2_000, step_s=7, start_ts=start, seed=seed), "open_interest"
    )
    bars5 = join_open_interest(with_cvd(time_bars(trades, "5m")), oi)
    mtf = build_multitf(trades, oi, ["1m", "5m", "15m"], zscore_window=24)
    paths = [candles_cvd_doi(bars5, out_dir=out_dir)]
    corr_cols = ["quote_volume", "delta_usd", "cvd_usd", "d_oi_usd", "taker_buy_volume"]
    paths.append(feature_corr_heatmap(bars5, corr_cols, out_dir=out_dir))
    paths.append(zvol_distributions(mtf, out_dir=out_dir))
    # keep figures meaningful: at least ~2h of 5m bars
    assert (int(trades["ts_event"].max()) - start) > 2 * 3600 * NS_PER_S
    return paths
