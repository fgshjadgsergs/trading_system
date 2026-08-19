"""M6: seeded labeled day with ~25 planted episodes; precision/recall >= 0.8."""

from __future__ import annotations

import polars as pl
import pytest

from trading_system.spoof.lifecycle import LevelJournal
from trading_system.spoof.metrics import annotate_episodes
from trading_system.spoof.score import score_episodes
from trading_system.spoof.synth import evaluate_spoof_flags, labeled_day

SEED = 42
N_PATTERNS = 25


@pytest.fixture(scope="module")
def day():
    states, trades, truth = labeled_day(seed=SEED, n_patterns=N_PATTERNS)
    journal = LevelJournal(large_k=3.0, iceberg_refill_ms=300).run(states, trades)
    annotated = annotate_episodes(journal, flicker_k=3, flicker_window_s=60)
    return journal, annotated, truth


def test_truth_has_all_pattern_types(day):
    _, _, truth = day
    assert truth.height == N_PATTERNS
    counts = dict(truth.group_by("label").len().iter_rows())
    assert set(counts) == {"honest", "spoof", "iceberg"}
    assert min(counts.values()) >= 5


def test_spoof_flag_precision_and_recall(day):
    _, annotated, truth = day
    res = evaluate_spoof_flags(truth, annotated, flag="flicker", positive_label="spoof")
    assert res["tp"] + res["fn"] >= 5  # spoof patterns actually planted
    assert res["precision"] >= 0.8
    assert res["recall"] >= 0.8


def test_iceberg_flag_precision_and_recall(day):
    _, annotated, truth = day
    res = evaluate_spoof_flags(truth, annotated, flag="iceberg", positive_label="iceberg")
    assert res["precision"] >= 0.8
    assert res["recall"] >= 0.8


def test_honest_patterns_score_above_spoofers(day):
    _, annotated, truth = day
    scored = score_episodes(annotated)

    def pattern_scores(label: str) -> list[float]:
        out = []
        for pat in truth.filter(pl.col("label") == label).iter_rows(named=True):
            eps = scored.filter(
                (pl.col("side") == pat["side"])
                & ((pl.col("price") - pat["price"]).abs() < 0.5)
                & pl.col("was_large")
            )
            if not eps.is_empty():
                out.append(float(eps["score"].max()))
        return out

    honest = pattern_scores("honest")
    spoof = pattern_scores("spoof")
    assert honest and spoof
    assert min(honest) > max(spoof)
    assert min(honest) >= 0.5
    assert max(spoof) < 0.2


def test_labeled_day_is_deterministic():
    _, _, t1 = labeled_day(seed=SEED, n_patterns=10, duration_s=60.0)
    _, _, t2 = labeled_day(seed=SEED, n_patterns=10, duration_s=60.0)
    assert t1.equals(t2)
    _, _, t3 = labeled_day(seed=SEED + 1, n_patterns=10, duration_s=60.0)
    assert not t1.equals(t3)
