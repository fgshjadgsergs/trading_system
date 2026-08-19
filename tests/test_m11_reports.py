"""M11 visualization: fire-drill timeline and PnL envelope figures."""

from __future__ import annotations

from trading_system.monitoring.reports import demo_reports

EXPECTED = {"m11_drill_timeline.png", "m11_pnl_envelope.png"}


def test_demo_reports_generates_all_figures(tmp_path):
    paths = demo_reports(tmp_path, seed=42)
    assert len(paths) == 2
    assert {p.name for p in paths} == EXPECTED
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 5_000  # non-trivial png
