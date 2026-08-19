"""Stage 3.2 leverage-weight calibration ladder: static -> rolling -> contextual.

Headline metric is the capture rate: the share of real liquidation USD volume
that lands inside the top-decile-heat cells of the concurrent map snapshot
(optionally a lagged snapshot, which makes the metric predictive and immune to
maps that merely hug the current price). The calibrated map must beat a naive
baseline (round numbers + recent swing extremes) or the map does not work.
Extra ladder complexity must improve the out-of-sample capture rate or it is
rolled back.

The calibrators maximize train-window capture plus a flow-matching term:
KL between the realized per-bar liquidation USD flow and the heat mass the
price path consumes per bar. Capture alone is a rank statistic with large flat
plateaus (and rewards price-hugging); the consumed-flow term is the part that
actually identifies the leverage mixture.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import polars as pl
import structlog

from trading_system.calibration.event_studies import price_bucket, top_decile_mask

log = structlog.get_logger(__name__)

HeatBuilder = Callable[[np.ndarray], np.ndarray]
"""Callback: weights -> heat matrix (n_snapshots, n_buckets).

Accepts either a (K,) global weight vector or an (n_snapshots, K) matrix of
per-snapshot weights (weights apply at position-open time). Rows must be
causal: heat[t] uses information up to and including snapshot t only.
"""

NS_PER_DAY = 86_400 * 1_000_000_000


# ---------------------------------------------------------------------------
# Metrics: capture rate (headline), flow divergence, calibration curve
# ---------------------------------------------------------------------------


def _liq_arrays(
    liquidations: pl.DataFrame, ts_range: tuple[int, int] | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts = liquidations["ts_event"].to_numpy()
    price = liquidations["price"].to_numpy()
    usd = liquidations["qty_usd"].to_numpy()
    if ts_range is not None:
        sel = (ts >= ts_range[0]) & (ts < ts_range[1])
        ts, price, usd = ts[sel], price[sel], usd[sel]
    return ts, price, usd


def capture_details(
    heat: np.ndarray,
    heat_ts: np.ndarray,
    bucket_edges: np.ndarray,
    liquidations: pl.DataFrame,
    top_decile: float = 0.1,
    ts_range: tuple[int, int] | None = None,
    lag_ns: int = 0,
) -> tuple[float, float]:
    """(captured_usd, total_usd) against the as-of snapshot per event.

    Each liquidation is matched to the latest snapshot with
    heat_ts <= ts_event - lag_ns (as-of backward: no lookahead; lag_ns > 0
    scores the map as a forecast made `lag_ns` before the event). Events
    with no prior snapshot are excluded (the map did not exist yet); events
    outside the bucket range count as not captured.
    """
    heat_ts = np.asarray(heat_ts)
    ts, price, usd = _liq_arrays(liquidations, ts_range)
    snap = np.searchsorted(heat_ts, ts - lag_ns, side="right") - 1
    alive = snap >= 0
    price, usd, snap = price[alive], usd[alive], snap[alive]
    buckets = price_bucket(price, bucket_edges)
    captured = 0.0
    masks: dict[int, np.ndarray] = {}
    for i in range(len(price)):
        b = buckets[i]
        if b < 0:
            continue
        r = int(snap[i])
        if r not in masks:
            masks[r] = top_decile_mask(heat[r], top_decile)
        if masks[r][b]:
            captured += usd[i]
    return float(captured), float(usd.sum())


def capture_rate(
    heat: np.ndarray,
    heat_ts: np.ndarray,
    bucket_edges: np.ndarray,
    liquidations: pl.DataFrame,
    top_decile: float = 0.1,
    ts_range: tuple[int, int] | None = None,
    lag_ns: int = 0,
) -> float:
    """Share of real liquidation USD landing in top-decile heat cells."""
    captured, total = capture_details(
        heat, heat_ts, bucket_edges, liquidations, top_decile, ts_range, lag_ns
    )
    return captured / total if total > 0 else 0.0


def flow_divergence(
    heat: np.ndarray,
    heat_ts: np.ndarray,
    bucket_edges: np.ndarray,
    prices: np.ndarray,
    liquidations: pl.DataFrame,
    ts_range: tuple[int, int] | None = None,
    floor: float = 1e-12,
) -> float:
    """KL(realized per-bar liq USD flow || heat mass consumed per bar).

    The heat a price bar sweeps (cells of heat[t-1] intersecting the
    [low, high] of bar t) is the mass the map predicts to be liquidated in
    that bar. Comparing that flow with the realized per-bar liquidation USD
    is what identifies the leverage mixture; lower is better. `prices` rows
    align 1:1 with heat_ts.
    """
    heat_ts = np.asarray(heat_ts)
    prices = np.asarray(prices, dtype=float)
    edges = np.asarray(bucket_edges, dtype=float)
    n, nb = len(heat_ts), len(edges) - 1
    if len(prices) != n or heat.shape[0] != n:
        raise ValueError("prices and heat rows must align with heat_ts")
    pred = np.zeros(n)
    for t in range(1, n):
        p_lo = min(prices[t - 1], prices[t])
        p_hi = max(prices[t - 1], prices[t])
        lo = max(int(np.searchsorted(edges, p_lo, side="right") - 1), 0)
        hi = min(int(np.searchsorted(edges, p_hi, side="right") - 1), nb - 1)
        if hi >= lo:
            pred[t] = heat[t - 1, lo : hi + 1].sum()
    ts, _, usd = _liq_arrays(liquidations, None)
    bar_of = np.clip(np.searchsorted(heat_ts, ts, side="left"), 0, n - 1)
    obs = np.bincount(bar_of, weights=usd, minlength=n)
    bars = np.arange(1, n)
    if ts_range is not None:
        bars = bars[(heat_ts[bars] >= ts_range[0]) & (heat_ts[bars] < ts_range[1])]
    o, f = obs[bars], pred[bars]
    if o.sum() <= 0 or f.sum() <= 0:
        return float("inf")
    o, f = o / o.sum(), f / f.sum()
    m = o > 0
    return float(np.sum(o[m] * np.log(o[m] / np.maximum(f[m], floor))))


def calibration_curve(
    heat: np.ndarray,
    heat_ts: np.ndarray,
    bucket_edges: np.ndarray,
    liquidations: pl.DataFrame,
    n_bins: int = 10,
    ts_range: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(predicted_share, realized_share) of liquidation intensity by heat decile.

    Cells of each concurrent snapshot are ranked into `n_bins` quantile bins by
    heat (bin n_bins-1 = hottest). Predicted share = heat mass per bin;
    realized share = liquidation USD per bin. A calibrated map has the two
    profiles close to each other.
    """
    heat_ts = np.asarray(heat_ts)
    ts, price, usd = _liq_arrays(liquidations, ts_range)
    snap = np.searchsorted(heat_ts, ts, side="right") - 1
    alive = snap >= 0
    price, usd, snap = price[alive], usd[alive], snap[alive]
    buckets = price_bucket(price, bucket_edges)
    n_cells = heat.shape[1]
    per_cell = n_cells / n_bins
    realized = np.zeros(n_bins)
    predicted = np.zeros(n_bins)
    for r in np.unique(snap):
        row = heat[r]
        rank = np.argsort(np.argsort(row, kind="stable"), kind="stable")
        bin_of = np.minimum((rank / per_cell).astype(int), n_bins - 1)
        predicted += np.bincount(bin_of, weights=row, minlength=n_bins)
        for i in np.flatnonzero(snap == r):
            if buckets[i] >= 0:
                realized[bin_of[buckets[i]]] += usd[i]
    if predicted.sum() > 0:
        predicted /= predicted.sum()
    if realized.sum() > 0:
        realized /= realized.sum()
    return predicted, realized


# ---------------------------------------------------------------------------
# Naive baseline: round numbers + recent swing extremes
# ---------------------------------------------------------------------------


def naive_baseline_heat(
    prices: np.ndarray,
    bucket_edges: np.ndarray,
    swing_window: int = 50,
    round_weight: float = 1.0,
    big_round_weight: float = 2.5,
    swing_weight: float = 2.0,
) -> np.ndarray:
    """Heat at round price levels plus trailing swing extremes.

    Round levels use a grid set by the price magnitude (e.g. every 1000 for a
    5-figure price, every-5000 levels weighted higher). Swing extremes are the
    trailing rolling max/min of the price. Fully causal: row t depends on
    prices[: t + 1] only. This is the baseline the calibrated map must beat.
    """
    prices = np.asarray(prices, dtype=float)
    edges = np.asarray(bucket_edges, dtype=float)
    n, nb = len(prices), len(edges) - 1
    heat = np.zeros((n, nb))
    step = 10.0 ** (np.floor(np.log10(np.median(prices))) - 1.0)
    base = np.zeros(nb)
    for m in range(int(np.ceil(edges[0] / step)), int(np.floor(edges[-1] / step)) + 1):
        b = int(np.searchsorted(edges, m * step, side="right") - 1)
        if 0 <= b < nb:
            base[b] += big_round_weight if m % 5 == 0 else round_weight
    heat[:] = base
    for t in range(n):
        window = prices[max(0, t - swing_window + 1) : t + 1]
        for level in (window.max(), window.min()):
            b = int(np.searchsorted(edges, level, side="right") - 1)
            if 0 <= b < nb:
                heat[t, b] += swing_weight
    return heat


# ---------------------------------------------------------------------------
# Rung 1: static global weights
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    weights: np.ndarray
    objective: float
    capture: float
    flow_kl: float


class StaticWeightCalibrator:
    """Global leverage-mixture weights maximizing train-window capture rate.

    Weights live on the simplex (softmax-style parametrization): seeded
    Dirichlet candidate search followed by multiplicative coordinate
    refinement. The objective is capture rate (at `lag_ns`) minus
    `flow_weight` times the flow divergence; the smooth flow term breaks the
    plateaus of the rank-based capture statistic and pins the optimum to the
    mixture that actually produced the liquidations.
    """

    def __init__(
        self,
        n_weights: int,
        seed: int = 42,
        n_candidates: int = 48,
        refine_sweeps: int = 3,
        refine_factors: Sequence[float] = (0.5, 0.7, 1.4, 2.0),
        top_decile: float = 0.1,
        lag_ns: int = 0,
        flow_weight: float = 0.5,
        weight_floor: float = 1e-4,
    ) -> None:
        if n_weights < 1:
            raise ValueError("n_weights must be >= 1")
        self.n_weights = n_weights
        self.seed = seed
        self.n_candidates = n_candidates
        self.refine_sweeps = refine_sweeps
        self.refine_factors = tuple(refine_factors)
        self.top_decile = top_decile
        self.lag_ns = lag_ns
        self.flow_weight = flow_weight
        self.weight_floor = weight_floor

    def _objective(
        self,
        w: np.ndarray,
        build_heat: HeatBuilder,
        heat_ts: np.ndarray,
        bucket_edges: np.ndarray,
        liquidations: pl.DataFrame,
        prices: np.ndarray,
        ts_range: tuple[int, int] | None,
    ) -> tuple[float, float, float]:
        heat = build_heat(w)
        cap = capture_rate(
            heat, heat_ts, bucket_edges, liquidations, self.top_decile, ts_range, self.lag_ns
        )
        fkl = flow_divergence(heat, heat_ts, bucket_edges, prices, liquidations, ts_range)
        return cap - self.flow_weight * fkl, cap, fkl

    def _normalize(self, w: np.ndarray) -> np.ndarray:
        w = np.maximum(np.asarray(w, dtype=float), self.weight_floor)
        return w / w.sum()

    def fit(
        self,
        build_heat: HeatBuilder,
        heat_ts: np.ndarray,
        bucket_edges: np.ndarray,
        liquidations: pl.DataFrame,
        prices: np.ndarray,
        ts_range: tuple[int, int] | None = None,
    ) -> FitResult:
        rng = np.random.default_rng(self.seed)
        k = self.n_weights
        half = self.n_candidates // 2
        candidates = [np.full(k, 1.0 / k)]
        candidates += [rng.dirichlet(np.ones(k)) for _ in range(half)]
        candidates += [rng.dirichlet(np.full(k, 0.3)) for _ in range(self.n_candidates - half)]
        args = (build_heat, heat_ts, bucket_edges, liquidations, prices, ts_range)
        best: tuple[float, float, float, np.ndarray] | None = None
        for cand in candidates:
            w = self._normalize(cand)
            obj, cap, fkl = self._objective(w, *args)
            if best is None or obj > best[0]:
                best = (obj, cap, fkl, w)
        assert best is not None
        for _ in range(self.refine_sweeps):
            improved = False
            for j in range(k):
                for f in self.refine_factors:
                    trial = best[3].copy()
                    trial[j] *= f
                    trial = self._normalize(trial)
                    obj, cap, fkl = self._objective(trial, *args)
                    if obj > best[0] + 1e-12:
                        best = (obj, cap, fkl, trial)
                        improved = True
            if not improved:
                break
        log.debug("static_fit", capture=best[1], flow_kl=best[2], weights=best[3].tolist())
        return FitResult(
            weights=best[3], objective=float(best[0]), capture=float(best[1]), flow_kl=float(best[2])
        )


# ---------------------------------------------------------------------------
# Rung 2: rolling weekly re-fit (walk-forward application)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightSegment:
    """Weights fitted on [apply_start - train_window, apply_start),
    applied to [apply_start, apply_end)."""

    apply_start_ts: int
    apply_end_ts: int
    weights: np.ndarray
    train_capture: float


class RollingCalibrator:
    """Weekly (configurable) re-fit on a trailing window, walk-forward apply."""

    def __init__(
        self,
        calibrator: StaticWeightCalibrator,
        train_window_ns: int,
        refit_every_ns: int = 7 * NS_PER_DAY,
    ) -> None:
        if train_window_ns <= 0 or refit_every_ns <= 0:
            raise ValueError("train_window_ns and refit_every_ns must be positive")
        self.calibrator = calibrator
        self.train_window_ns = train_window_ns
        self.refit_every_ns = refit_every_ns

    def fit_segments(
        self,
        build_heat: HeatBuilder,
        heat_ts: np.ndarray,
        bucket_edges: np.ndarray,
        liquidations: pl.DataFrame,
        prices: np.ndarray,
        apply_range: tuple[int, int],
    ) -> list[WeightSegment]:
        """Fit one weight vector per refit interval covering `apply_range`.

        Each segment's weights see only liquidations strictly before its
        apply_start (trailing window), so the application is walk-forward.
        """
        start, end = apply_range
        segments: list[WeightSegment] = []
        t = start
        while t < end:
            t_end = min(t + self.refit_every_ns, end)
            res = self.calibrator.fit(
                build_heat,
                heat_ts,
                bucket_edges,
                liquidations,
                prices,
                ts_range=(t - self.train_window_ns, t),
            )
            segments.append(
                WeightSegment(
                    apply_start_ts=int(t),
                    apply_end_ts=int(t_end),
                    weights=res.weights,
                    train_capture=res.capture,
                )
            )
            t = t_end
        return segments

    @staticmethod
    def applied_heat(
        build_heat: HeatBuilder, heat_ts: np.ndarray, segments: list[WeightSegment]
    ) -> np.ndarray:
        """Stitch per-segment heats: row t takes the segment covering heat_ts[t].

        Rows before the first segment fall back to the first segment's weights;
        they are outside the walk-forward apply range and must not be scored.
        """
        if not segments:
            raise ValueError("no segments")
        heat_ts = np.asarray(heat_ts)
        out: np.ndarray | None = None
        for si, seg in enumerate(segments):
            heat = build_heat(seg.weights)
            if out is None:
                out = heat.copy()
            lo = np.searchsorted(heat_ts, seg.apply_start_ts, side="left")
            hi = len(heat_ts) if si == len(segments) - 1 else np.searchsorted(
                heat_ts, seg.apply_end_ts, side="left"
            )
            out[lo:hi] = heat[lo:hi]
        assert out is not None
        return out

    def oos_capture(
        self,
        build_heat: HeatBuilder,
        heat_ts: np.ndarray,
        bucket_edges: np.ndarray,
        liquidations: pl.DataFrame,
        prices: np.ndarray,
        apply_range: tuple[int, int],
        top_decile: float = 0.1,
        lag_ns: int = 0,
    ) -> tuple[float, list[WeightSegment]]:
        segments = self.fit_segments(
            build_heat, heat_ts, bucket_edges, liquidations, prices, apply_range
        )
        heat = self.applied_heat(build_heat, heat_ts, segments)
        cap = capture_rate(
            heat, heat_ts, bucket_edges, liquidations, top_decile, apply_range, lag_ns
        )
        return cap, segments


# ---------------------------------------------------------------------------
# Rung 3: contextual weights (pure-numpy multinomial softmax regression)
# ---------------------------------------------------------------------------


class ContextualWeights:
    """w(context): softmax regression from context features to a leverage
    distribution, fitted by full-batch gradient descent with L2 regularization.

    Pure numpy, deterministic (zero init, fixed iteration count, seeded API
    for stability). Targets may be hard class labels (n,) or soft
    distributions (n, K).
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        l2: float = 1e-3,
        lr: float = 0.5,
        n_iter: int = 800,
        seed: int = 42,
    ) -> None:
        self.n_features = n_features
        self.n_classes = n_classes
        self.l2 = l2
        self.lr = lr
        self.n_iter = n_iter
        self.seed = seed
        self.coef_ = np.zeros((n_features + 1, n_classes))
        self._mu = np.zeros(n_features)
        self._sigma = np.ones(n_features)
        self.loss_history_: list[float] = []

    def _design(self, x: np.ndarray) -> np.ndarray:
        z = (np.asarray(x, dtype=float) - self._mu) / self._sigma
        return np.hstack([np.ones((len(z), 1)), z])

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        m = logits - logits.max(axis=1, keepdims=True)
        e = np.exp(m)
        return e / e.sum(axis=1, keepdims=True)

    def fit(self, x: np.ndarray, y: np.ndarray) -> ContextualWeights:
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError("X must be (n_samples, n_features)")
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            hot = np.zeros((len(y), self.n_classes))
            hot[np.arange(len(y)), y.astype(int)] = 1.0
            y = hot
        if y.shape != (len(x), self.n_classes):
            raise ValueError("Y must be (n_samples,) labels or (n_samples, n_classes)")
        self._mu = x.mean(axis=0)
        self._sigma = np.where(x.std(axis=0) > 1e-12, x.std(axis=0), 1.0)
        xd = self._design(x)
        n = len(xd)
        w = np.zeros((self.n_features + 1, self.n_classes))
        self.loss_history_ = []
        for _ in range(self.n_iter):
            p = self._softmax(xd @ w)
            loss = -np.sum(y * np.log(np.maximum(p, 1e-300))) / n
            loss += 0.5 * self.l2 * np.sum(w[1:] ** 2)
            self.loss_history_.append(float(loss))
            grad = xd.T @ (p - y) / n
            grad[1:] += self.l2 * w[1:]
            w -= self.lr * grad
        self.coef_ = w
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._softmax(self._design(x) @ self.coef_)


# ---------------------------------------------------------------------------
# Ladder comparison with the rollback selection rule
# ---------------------------------------------------------------------------

_COMPLEXITY_ORDER = ("static", "rolling", "contextual")


@dataclass
class LadderResult:
    capture: dict[str, float]  # rung -> OOS capture rate (includes "naive")
    selected: str
    beats_naive: bool
    static_weights: np.ndarray
    segments: list[WeightSegment] = field(default_factory=list)
    contextual_model: ContextualWeights | None = None
    tolerance: float = 0.0


def select_rung(capture: dict[str, float], tolerance: float) -> str:
    """Simplest rung within `tolerance` of the best OOS capture.

    Implements the checklist rule: extra complexity that does not improve OOS
    by more than the tolerance is rolled back.
    """
    rungs = [r for r in _COMPLEXITY_ORDER if r in capture]
    if not rungs:
        raise ValueError("no ladder rungs in capture dict")
    best = max(capture[r] for r in rungs)
    for r in rungs:
        if capture[r] >= best - tolerance:
            return r
    return rungs[-1]


def compare_ladder(
    build_heat: HeatBuilder,
    heat_ts: np.ndarray,
    bucket_edges: np.ndarray,
    liquidations: pl.DataFrame,
    prices: np.ndarray,
    train_range: tuple[int, int],
    test_range: tuple[int, int],
    n_weights: int,
    context: np.ndarray | None = None,
    context_labels: np.ndarray | None = None,
    calibrator: StaticWeightCalibrator | None = None,
    rolling: RollingCalibrator | None = None,
    contextual: ContextualWeights | None = None,
    tolerance: float = 0.01,
    top_decile: float = 0.1,
    lag_ns: int = 0,
) -> LadderResult:
    """OOS capture for naive/static/rolling/contextual and the rung selection.

    train_range/test_range are half-open ns intervals; the caller is
    responsible for leaving an embargo gap between them (see walkforward).
    `context`/`context_labels` rows align with heat_ts; when omitted the
    contextual rung is skipped. Rolling is skipped when `rolling` is None.
    """
    if train_range[1] > test_range[0]:
        raise ValueError("train_range must end before test_range starts")
    heat_ts = np.asarray(heat_ts)
    cal = calibrator or StaticWeightCalibrator(n_weights=n_weights, lag_ns=lag_ns)
    capture: dict[str, float] = {}

    naive = naive_baseline_heat(prices, bucket_edges)
    capture["naive"] = capture_rate(
        naive, heat_ts, bucket_edges, liquidations, top_decile, test_range, lag_ns
    )

    static_fit = cal.fit(
        build_heat, heat_ts, bucket_edges, liquidations, prices, ts_range=train_range
    )
    static_heat = build_heat(static_fit.weights)
    capture["static"] = capture_rate(
        static_heat, heat_ts, bucket_edges, liquidations, top_decile, test_range, lag_ns
    )

    segments: list[WeightSegment] = []
    if rolling is not None:
        capture["rolling"], segments = rolling.oos_capture(
            build_heat, heat_ts, bucket_edges, liquidations, prices, test_range, top_decile, lag_ns
        )

    model: ContextualWeights | None = None
    if context is not None and context_labels is not None:
        context = np.asarray(context, dtype=float)
        train_rows = (heat_ts >= train_range[0]) & (heat_ts < train_range[1])
        model = contextual or ContextualWeights(n_features=context.shape[1], n_classes=n_weights)
        model.fit(context[train_rows], np.asarray(context_labels)[train_rows])
        ctx_heat = build_heat(model.predict_proba(context))
        capture["contextual"] = capture_rate(
            ctx_heat, heat_ts, bucket_edges, liquidations, top_decile, test_range, lag_ns
        )

    selected = select_rung(capture, tolerance)
    result = LadderResult(
        capture=capture,
        selected=selected,
        beats_naive=capture[selected] > capture["naive"],
        static_weights=static_fit.weights,
        segments=segments,
        contextual_model=model,
        tolerance=tolerance,
    )
    log.info("ladder_compare", capture=capture, selected=selected)
    return result
