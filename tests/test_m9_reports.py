"""M9 visualization: state machine diagram + crash-drill incident log figures."""

from __future__ import annotations

from trading_system.risk.reports import demo_reports

EXPECTED = {"m9_order_state_machine.png", "m9_crash_drill_log.png"}


def test_demo_reports_generates_all_figures(tmp_path):
    paths = demo_reports(tmp_path, seed=42)
    assert len(paths) == 2
    assert {p.name for p in paths} == EXPECTED
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 5_000  # non-trivial png
    # no stray artifacts (the drill journal must not land in the reports dir)
    assert {p.name for p in tmp_path.iterdir()} == EXPECTED
