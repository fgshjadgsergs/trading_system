"""M3: bars (time & volume), delta/CVD, OI joins, indicators, multi-TF features."""

from trading_system.features.bars import compare_klines, time_bars, volume_bars, with_cvd
from trading_system.features.indicators import with_atr, with_vwap
from trading_system.features.joins import asof_join_backward, join_open_interest
from trading_system.features.multitf import build_multitf, join_context, tf_features

__all__ = [
    "asof_join_backward",
    "build_multitf",
    "compare_klines",
    "join_context",
    "join_open_interest",
    "tf_features",
    "time_bars",
    "volume_bars",
    "with_atr",
    "with_cvd",
    "with_vwap",
]
