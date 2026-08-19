"""Stage 3.3 walk-forward: embargoed splits, regime tags, multi-symbol harness,
leakage canary (shuffled labels must show no OOS edge)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_system.calibration.walkforward import (
    SymbolData,
    WalkForwardSplitter,
    aggregate_by_regime,
    majority_label,
    run_walkforward,
    summarize_walkforward,
    tag_regimes,
)
from trading_system.calibration.weights import ContextualWeights

SEED = 42


# ---------------------------------------------------------------------------
# splitter
# ---------------------------------------------------------------------------


def test_splitter_basic_embargo_and_order():
    ts = np.arange(1000, dtype=np.int64) * 10
    sp = WalkForwardSplitter(train_ns=2000, test_ns=1000, embargo_ns=300)
    splits = sp.split(ts)
    assert len(splits) > 3
    for s in splits:
        assert len(s.train_idx) and len(s.test_idx)
        assert s.train_idx.max() < s.test_idx.min()
        gap = ts[s.test_idx[0]] - ts[s.train_idx[-1]]
        assert gap >= 300
        assert s.train_range[1] + 300 == s.test_range[0]
        assert np.intersect1d(s.train_idx, s.test_idx).size == 0
    # folds advance by step (= test_ns by default)
    assert splits[1].test_range[0] - splits[0].test_range[0] == 1000


def test_splitter_validation():
    with pytest.raises(ValueError):
        WalkForwardSplitter(0, 10, 1)
    with pytest.raises(ValueError):
        WalkForwardSplitter(10, 10, -1)
    with pytest.raises(ValueError):
        WalkForwardSplitter(10, 10, 1).split(np.array([3, 2, 1]))
    assert WalkForwardSplitter(10, 10, 1).split(np.array([], dtype=np.int64)) == []


@settings(max_examples=60, deadline=None)
@given(
    n=st.integers(20, 400),
    train=st.integers(1, 50),
    test=st.integers(1, 30),
    embargo=st.integers(0, 20),
    step=st.integers(1, 40),
    seed=st.integers(0, 1000),
)
def test_splitter_invariants_property(n, train, test, embargo, step, seed):
    rng = np.random.default_rng(seed)
    ts = np.cumsum(rng.integers(1, 5, n)).astype(np.int64)
    sp = WalkForwardSplitter(train_ns=train, test_ns=test, embargo_ns=embargo, step_ns=step)
    for s in sp.split(ts):
        assert np.intersect1d(s.train_idx, s.test_idx).size == 0
        assert ts[s.test_idx[0]] - ts[s.train_idx[-1]] >= embargo
        assert s.train_range[1] <= s.test_range[0]
        assert np.all(np.diff(s.train_idx) > 0) and np.all(np.diff(s.test_idx) > 0)
        assert np.all(ts[s.train_idx] >= s.train_range[0])
        assert np.all(ts[s.train_idx] < s.train_range[1])
        assert np.all(ts[s.test_idx] >= s.test_range[0])
        assert np.all(ts[s.test_idx] < s.test_range[1])


# ---------------------------------------------------------------------------
# regimes
# ---------------------------------------------------------------------------


def test_tag_regimes_trend_vs_chop_and_vol():
    rng = np.random.default_rng(SEED)
    n_half = 600
    trend_part = np.cumsum(np.full(n_half, 20.0) + rng.normal(0, 4.0, n_half))
    chop_part = trend_part[-1] + np.cumsum(rng.normal(0, 60.0, n_half) * np.sign(
        np.sin(np.arange(n_half))
    ))
    prices = 50_000.0 + np.concatenate([trend_part, chop_part])
    tags = tag_regimes(prices, window=50)
    assert tags.trend[100:n_half].mean() > 0.9  # steady climb tagged as trend
    assert tags.trend[n_half + 100 :].mean() < 0.3  # oscillation tagged as chop
    assert tags.high_vol[n_half + 100 :].mean() > 0.7  # wild half is high vol
    assert set(np.unique(tags.label)) <= {
        "trend/high_vol", "trend/low_vol", "chop/high_vol", "chop/low_vol",
    }


def test_aggregate_by_regime_and_majority():
    labels = np.array(["a", "a", "b", "b", "b"])
    df = aggregate_by_regime(np.array([1.0, 3.0, 10.0, 10.0, 10.0]), labels)
    row_a = df.filter(pl.col("regime") == "a")
    assert row_a["mean"][0] == pytest.approx(2.0)
    assert row_a["n"][0] == 2
    assert majority_label(labels, np.array([0, 2, 3])) == "b"
    assert majority_label(labels, np.array([], dtype=int)) == "unknown"


# ---------------------------------------------------------------------------
# multi-symbol harness
# ---------------------------------------------------------------------------


def test_run_walkforward_multi_symbol():
    rng = np.random.default_rng(SEED)
    data = {}
    for i, sym in enumerate(["BTCUSDT", "SOLUSDT"]):
        n = 800
        ts = np.arange(n, dtype=np.int64) * 100
        prices = 100.0 * (i + 1) * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
        data[sym] = SymbolData(ts=ts, prices=prices, payload={"k": i})

    def eval_fn(symbol, sd, split):
        return float(sd.prices[split.test_idx].mean()) + sd.payload["k"]

    sp = WalkForwardSplitter(train_ns=20_000, test_ns=10_000, embargo_ns=1_000)
    res = run_walkforward(data, sp, eval_fn, regime_window=50)
    assert set(res["symbol"].unique()) == {"BTCUSDT", "SOLUSDT"}
    assert res.filter(pl.col("symbol") == "BTCUSDT").height == len(sp.split(data["BTCUSDT"].ts))
    assert res["regime"].null_count() == 0
    summary = summarize_walkforward(res)
    assert set(summary) == {"by_symbol", "by_regime", "overall"}
    assert summary["by_symbol"].height == 2
    assert summary["overall"]["n"][0] == res.height


# ---------------------------------------------------------------------------
# leakage canary: shuffled labels must show no OOS edge
# ---------------------------------------------------------------------------


def _canary_data(n=3000, seed=SEED):
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.int64) * 60_000_000_000
    x = rng.normal(0, 1, (n, 3))
    y = (x[:, 0] + 0.3 * rng.normal(0, 1, n) > 0).astype(int)
    return ts, x, y


def _oos_accuracy(ts, x, y, shuffle_train: bool, seed=SEED) -> float:
    sp = WalkForwardSplitter(
        train_ns=1000 * 60_000_000_000, test_ns=400 * 60_000_000_000,
        embargo_ns=20 * 60_000_000_000,
    )
    rng = np.random.default_rng(seed)
    accs = []
    for split in sp.split(ts):
        y_train = y[split.train_idx]
        if shuffle_train:
            y_train = y_train[rng.permutation(len(y_train))]
        model = ContextualWeights(n_features=3, n_classes=2, n_iter=300)
        model.fit(x[split.train_idx], y_train)
        pred = model.predict_proba(x[split.test_idx]).argmax(axis=1)
        accs.append(float((pred == y[split.test_idx]).mean()))
    return float(np.mean(accs))


def test_leakage_canary_shuffled_labels_have_no_oos_edge():
    ts, x, y = _canary_data()
    acc_real = _oos_accuracy(ts, x, y, shuffle_train=False)
    acc_shuffled = _oos_accuracy(ts, x, y, shuffle_train=True)
    assert acc_real > 0.8  # signal is there and survives the embargoed split
    assert abs(acc_shuffled - 0.5) < 0.1  # canary: shuffling kills the edge
