"""Track B4: boosted contextual weights — beats linear softmax on interactions."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lightgbm")

from trading_system.calibration.boosted import BoostedWeights  # noqa: E402
from trading_system.calibration.weights import ContextualWeights  # noqa: E402

SEED = 42


def _xor_dataset(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """High leverage (class 1) iff impulse XOR rising-OI — pure interaction."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    x = np.column_stack([a + rng.normal(0, 0.1, n), b + rng.normal(0, 0.1, n)])
    y = (a ^ b).astype(int)
    return x, y


def test_boosted_captures_xor_where_softmax_cannot():
    x_tr, y_tr = _xor_dataset(1_500, SEED)
    x_te, y_te = _xor_dataset(500, SEED + 1)

    softmax = ContextualWeights(n_features=2, n_classes=2).fit(x_tr, y_tr)
    boosted = BoostedWeights(n_classes=2, seed=SEED).fit(x_tr, y_tr)

    acc_soft = float((softmax.predict_proba(x_te).argmax(axis=1) == y_te).mean())
    acc_boost = float((boosted.predict_proba(x_te).argmax(axis=1) == y_te).mean())
    assert acc_boost > 0.9  # trees learn the interaction
    assert acc_soft < 0.65  # a linear model cannot
    assert acc_boost - acc_soft > 0.25


def test_soft_targets_and_distribution_output():
    rng = np.random.default_rng(SEED)
    x = rng.normal(0, 1, (400, 3))
    # soft target: class shares depend nonlinearly on x0
    p1 = 1 / (1 + np.exp(-3 * np.sin(x[:, 0] * 2)))
    y = np.column_stack([1 - p1, p1])
    model = BoostedWeights(n_classes=2, seed=SEED).fit(x, y)
    w = model.predict_proba(x)
    assert w.shape == (400, 2)
    assert np.all(w >= 0)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-9)


def test_deterministic_across_fits():
    x, y = _xor_dataset(600, SEED)
    a = BoostedWeights(n_classes=2, seed=SEED).fit(x, y).predict_proba(x)
    b = BoostedWeights(n_classes=2, seed=SEED).fit(x, y).predict_proba(x)
    assert np.array_equal(a, b)


def test_ladder_contract_shapes():
    """Duck-type contract used by compare_ladder(contextual=...)."""
    x, y = _xor_dataset(300, SEED)
    model = BoostedWeights(n_classes=2, seed=SEED)
    assert model.fit(x, y) is model
    w = model.predict_proba(x[:7])
    assert w.shape == (7, 2)
    with pytest.raises(ValueError):
        model.fit(x, np.zeros((len(x), 3)))
    fresh = BoostedWeights(n_classes=2, seed=SEED)
    with pytest.raises(RuntimeError):
        fresh.predict_proba(x)
