"""Shared fixtures: fixed seeds, temp report/data dirs."""

from __future__ import annotations

import numpy as np
import pytest

from trading_system.core.config import load_config, seed_everything

SEED = 42


@pytest.fixture(autouse=True)
def _fixed_seed():
    seed_everything(SEED)


@pytest.fixture()
def rng() -> np.random.Generator:
    return np.random.default_rng(SEED)


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture()
def tmp_reports(tmp_path):
    d = tmp_path / "reports"
    d.mkdir()
    return d


@pytest.fixture()
def tmp_data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d
