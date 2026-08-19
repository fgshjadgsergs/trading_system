"""Grafana dashboard asset: loader and structural validation."""

from __future__ import annotations

import json
from pathlib import Path

GRAFANA_DASHBOARD_PATH = Path(__file__).with_name("grafana_dashboard.json")

REQUIRED_PANEL_TITLES = (
    "Stream freshness age (s)",
    "Gap events per stream",
    "PnL divergence z-score",
)


def load_dashboard(path: str | Path = GRAFANA_DASHBOARD_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_dashboard(dashboard: dict) -> list[str]:
    """Structural checks; returns a list of problems (empty == valid)."""
    problems: list[str] = []
    for field in ("title", "uid", "schemaVersion", "panels"):
        if field not in dashboard:
            problems.append(f"missing top-level field: {field}")
    panels = dashboard.get("panels", [])
    if not isinstance(panels, list) or not all(isinstance(p, dict) for p in panels):
        problems.append("panels must be a list of objects")
        panels = []
    titles = {p.get("title") for p in panels}
    for required in REQUIRED_PANEL_TITLES:
        if required not in titles:
            problems.append(f"missing panel: {required}")
    for p in panels:
        for field in ("id", "type", "title", "targets", "gridPos"):
            if field not in p:
                problems.append(f"panel {p.get('title', '?')}: missing {field}")
    return problems
