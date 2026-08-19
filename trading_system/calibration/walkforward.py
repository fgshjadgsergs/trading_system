"""Stage 3.3 walk-forward: embargoed splits, regime tagging, multi-symbol runs.

Splits carry an embargo gap between train and test so that features computed
with trailing windows on train data cannot leak into the test window. Regimes
are tagged from the price series alone (trend vs chop via rolling directional
ratio, high vs low vol via realized-vol median split) and results aggregate
per regime and per symbol.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import polars as pl
import structlog

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Embargoed walk-forward splitter
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardSplit:
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_range: tuple[int, int]  # half-open ns interval
    test_range: tuple[int, int]

    def __post_init__(self) -> None:
        if len(self.train_idx) and len(self.test_idx):
            assert self.train_idx.max() < self.test_idx.min(), "train/test order violated"
        assert self.train_range[1] <= self.test_range[0], "train must end before test"


class WalkForwardSplitter:
    """Rolling train/test windows in time units with an embargo gap.

    Windows: train [a, a + train_ns) -> gap of embargo_ns -> test
    [a + train_ns + embargo_ns, ... + test_ns); the anchor advances by
    step_ns (default: test_ns, contiguous non-overlapping test windows).
    """

    def __init__(
        self,
        train_ns: int,
        test_ns: int,
        embargo_ns: int,
        step_ns: int | None = None,
    ) -> None:
        if train_ns <= 0 or test_ns <= 0:
            raise ValueError("train_ns and test_ns must be positive")
        if embargo_ns < 0:
            raise ValueError("embargo_ns must be >= 0")
        self.train_ns = train_ns
        self.test_ns = test_ns
        self.embargo_ns = embargo_ns
        self.step_ns = step_ns if step_ns is not None else test_ns
        if self.step_ns <= 0:
            raise ValueError("step_ns must be positive")

    def split(self, ts: np.ndarray) -> list[WalkForwardSplit]:
        ts = np.asarray(ts)
        if len(ts) == 0:
            return []
        if np.any(np.diff(ts) < 0):
            raise ValueError("ts must be sorted ascending")
        t0, t_last = int(ts[0]), int(ts[-1])
        out: list[WalkForwardSplit] = []
        fold = 0
        a = t0
        while True:
            train_lo, train_hi = a, a + self.train_ns
            test_lo = train_hi + self.embargo_ns
            test_hi = test_lo + self.test_ns
            if test_lo > t_last:
                break
            train_idx = np.flatnonzero((ts >= train_lo) & (ts < train_hi))
            test_idx = np.flatnonzero((ts >= test_lo) & (ts < test_hi))
            if len(train_idx) and len(test_idx):
                # embargo respected by construction; assert on real timestamps too
                gap = int(ts[test_idx[0]]) - int(ts[train_idx[-1]])
                assert gap >= self.embargo_ns, "embargo violated"
                assert not np.intersect1d(train_idx, test_idx).size, "index overlap"
                out.append(
                    WalkForwardSplit(
                        fold=fold,
                        train_idx=train_idx,
                        test_idx=test_idx,
                        train_range=(train_lo, train_hi),
                        test_range=(test_lo, test_hi),
                    )
                )
                fold += 1
            a += self.step_ns
        return out


# ---------------------------------------------------------------------------
# Regime tagging
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeTags:
    trend: np.ndarray  # bool: True = trend, False = chop
    high_vol: np.ndarray  # bool: True = high vol
    label: np.ndarray  # str: "trend/high_vol" etc.
    dir_ratio: np.ndarray
    realized_vol: np.ndarray


def _rolling_sum(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing rolling sum with an expanding warmup (causal)."""
    c = np.concatenate([[0.0], np.cumsum(x)])
    lo = np.maximum(np.arange(1, len(x) + 1) - window, 0)
    return c[1:] - c[lo]


def _expanding_median(x: np.ndarray) -> np.ndarray:
    """median(x[: t + 1]) for every t — the causal split threshold."""
    out = np.empty_like(x)
    for t in range(len(x)):
        out[t] = np.median(x[: t + 1])
    return out


def tag_regimes(
    prices: np.ndarray, window: int = 50, trend_threshold: float = 0.3
) -> RegimeTags:
    """Trend/chop by rolling directional ratio, high/low vol by expanding median.

    Directional ratio = |sum of log returns| / sum |log returns| over the
    trailing window; it has an absolute scale (1 = straight line, ~1/sqrt(window)
    = pure chop), so trend uses a fixed threshold. Volatility has no absolute
    scale, so its split is the EXPANDING median (median of rv[: t + 1]). Both
    classifiers are causal: tags at t use prices[: t + 1] only — an early
    fold's label cannot depend on later data.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 3:
        raise ValueError("series too short to tag")
    r = np.diff(np.log(prices), prepend=np.log(prices[0]))
    num = np.abs(_rolling_sum(r, window))
    den = _rolling_sum(np.abs(r), window)
    dir_ratio = num / np.maximum(den, 1e-12)
    sq = _rolling_sum(r**2, window)
    cnt = np.minimum(np.arange(1, len(r) + 1), window)
    rv = np.sqrt(sq / np.maximum(cnt, 1))
    trend = dir_ratio > trend_threshold
    high_vol = rv > _expanding_median(rv)
    label = np.array(
        [
            f"{'trend' if t else 'chop'}/{'high_vol' if v else 'low_vol'}"
            for t, v in zip(trend, high_vol, strict=True)
        ]
    )
    return RegimeTags(
        trend=trend, high_vol=high_vol, label=label, dir_ratio=dir_ratio, realized_vol=rv
    )


def majority_label(labels: np.ndarray, idx: np.ndarray) -> str:
    """Most frequent regime label over the given indices (stable tie-break)."""
    if len(idx) == 0:
        return "unknown"
    vals, counts = np.unique(labels[idx], return_counts=True)
    return str(vals[np.argmax(counts)])


def aggregate_by_regime(metric: np.ndarray, labels: np.ndarray) -> pl.DataFrame:
    """Mean/std/count of a metric grouped by regime label."""
    df = pl.DataFrame({"regime": list(map(str, labels)), "metric": np.asarray(metric, float)})
    return (
        df.group_by("regime")
        .agg(
            pl.col("metric").mean().alias("mean"),
            pl.col("metric").std().alias("std"),
            pl.len().alias("n"),
        )
        .sort("regime")
    )


# ---------------------------------------------------------------------------
# Multi-symbol harness
# ---------------------------------------------------------------------------


@dataclass
class SymbolData:
    """Plain-data bundle per symbol: timestamps, prices, arbitrary payload."""

    ts: np.ndarray
    prices: np.ndarray
    payload: dict[str, Any] = field(default_factory=dict)


EvalFn = Callable[[str, SymbolData, WalkForwardSplit], float]


def run_walkforward(
    data: dict[str, SymbolData],
    splitter: WalkForwardSplitter,
    eval_fn: EvalFn,
    regime_window: int = 50,
) -> pl.DataFrame:
    """Evaluate every symbol on every embargoed split; tag test-window regimes.

    Returns one row per (symbol, fold): metric, majority regime of the test
    window, test range. Aggregate with `summarize_walkforward`.
    """
    rows: list[dict[str, Any]] = []
    for symbol, sd in data.items():
        tags = tag_regimes(sd.prices, window=regime_window)
        for split in splitter.split(sd.ts):
            metric = eval_fn(symbol, sd, split)
            rows.append(
                {
                    "symbol": symbol,
                    "fold": split.fold,
                    "metric": float(metric),
                    "regime": majority_label(tags.label, split.test_idx),
                    "test_start": split.test_range[0],
                    "test_end": split.test_range[1],
                }
            )
        log.debug("walkforward_symbol_done", symbol=symbol, folds=len(rows))
    schema = {
        "symbol": pl.Utf8,
        "fold": pl.Int64,
        "metric": pl.Float64,
        "regime": pl.Utf8,
        "test_start": pl.Int64,
        "test_end": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


def summarize_walkforward(results: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Per-symbol, per-regime and overall aggregates of walk-forward metrics."""
    agg = [
        pl.col("metric").mean().alias("mean"),
        pl.col("metric").std().alias("std"),
        pl.len().alias("n"),
    ]
    return {
        "by_symbol": results.group_by("symbol").agg(agg).sort("symbol"),
        "by_regime": results.group_by("regime").agg(agg).sort("regime"),
        "overall": results.select(agg),
    }
