"""YAML config loading with fixed seeds everywhere."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "base.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else DEFAULT_CONFIG
    with open(p) as f:
        cfg = yaml.safe_load(f)
    return cfg


def seed_everything(seed: int) -> np.random.Generator:
    """Seed stdlib and numpy; return a fresh Generator for local use."""
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def reports_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    d = REPO_ROOT / cfg["project"]["reports_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def data_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    d = REPO_ROOT / cfg["project"]["data_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d
