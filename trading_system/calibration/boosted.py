"""Track B4: boosted-tree contextual weights — the checklist's alternative to
softmax regression for ladder rung 3.

Same duck-typed contract as ContextualWeights (fit(x, y), predict_proba(x)),
so it drops straight into compare_ladder(contextual=...). One deterministic
LightGBM regressor per leverage class predicts that class's share; predictions
are clipped and renormalized into a distribution. Trees capture interaction
effects (e.g. "high leverage only when impulse volume AND rising OI") that a
linear softmax cannot.

lightgbm is an optional dependency (pyproject extra "boost"); constructing
BoostedWeights without it raises ImportError with an install hint.
"""

from __future__ import annotations

import numpy as np


class BoostedWeights:
    """w(context) via per-class gradient-boosted trees, normalized rows."""

    def __init__(
        self,
        n_classes: int,
        n_estimators: int = 200,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_child_samples: int = 5,
        seed: int = 42,
    ) -> None:
        try:
            import lightgbm  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "BoostedWeights needs lightgbm: pip install 'trading-system[boost]'"
            ) from exc
        self.n_classes = n_classes
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbosity=-1,
        )
        self._models: list = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> BoostedWeights:
        """x: (n, F) context; y: hard labels (n,) or soft distributions (n, K)."""
        from lightgbm import LGBMRegressor

        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            hot = np.zeros((len(y), self.n_classes))
            hot[np.arange(len(y)), y.astype(int)] = 1.0
            y = hot
        if y.shape != (len(x), self.n_classes):
            raise ValueError("targets must be (n,) labels or (n, n_classes) shares")
        self._models = []
        for k in range(self.n_classes):
            model = LGBMRegressor(**self.params)
            model.fit(x, y[:, k])
            self._models.append(model)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self._models:
            raise RuntimeError("fit first")
        x = np.asarray(x, dtype=float)
        raw = np.column_stack([m.predict(x) for m in self._models])
        raw = np.clip(raw, 0.0, None)
        rows = raw.sum(axis=1, keepdims=True)
        uniform = np.full_like(raw, 1.0 / self.n_classes)
        with np.errstate(invalid="ignore"):
            out = np.where(rows > 1e-12, raw / np.where(rows > 0, rows, 1.0), uniform)
        return out
