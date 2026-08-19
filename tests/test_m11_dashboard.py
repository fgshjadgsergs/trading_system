"""M11 Grafana dashboard asset: valid JSON, passes the structural validator."""

from __future__ import annotations

import copy
import json

from trading_system.monitoring.dashboard import (
    GRAFANA_DASHBOARD_PATH,
    REQUIRED_PANEL_TITLES,
    load_dashboard,
    validate_dashboard,
)


def test_shipped_dashboard_is_valid_json_and_passes_validation():
    raw = GRAFANA_DASHBOARD_PATH.read_text(encoding="utf-8")
    dashboard = json.loads(raw)  # must be valid JSON
    assert dashboard == load_dashboard()
    assert validate_dashboard(dashboard) == []


def test_dashboard_covers_all_monitored_signals():
    titles = {p["title"] for p in load_dashboard()["panels"]}
    assert set(REQUIRED_PANEL_TITLES) <= titles


def test_validator_flags_missing_top_level_field():
    d = copy.deepcopy(load_dashboard())
    del d["uid"]
    assert any("uid" in p for p in validate_dashboard(d))


def test_validator_flags_missing_panel():
    d = copy.deepcopy(load_dashboard())
    d["panels"] = [p for p in d["panels"] if p["title"] != REQUIRED_PANEL_TITLES[0]]
    problems = validate_dashboard(d)
    assert any(REQUIRED_PANEL_TITLES[0] in p for p in problems)


def test_validator_flags_incomplete_panel():
    d = copy.deepcopy(load_dashboard())
    del d["panels"][0]["targets"]
    problems = validate_dashboard(d)
    assert any("targets" in p for p in problems)


def test_validator_flags_non_list_panels():
    d = copy.deepcopy(load_dashboard())
    d["panels"] = "oops"
    problems = validate_dashboard(d)
    assert any("panels" in p for p in problems)
