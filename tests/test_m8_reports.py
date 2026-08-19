"""M8 visualization: demo_reports produces all checklist figures from synthetic data."""

from __future__ import annotations

from trading_system.backtest.reports import demo_reports

EXPECTED = {"m8_equity_curve.png", "m8_trade_pnl_dist.png", "m8_cost_waterfall.png"}


def test_demo_reports_generates_all_figures(tmp_path):
    paths = demo_reports(tmp_path, seed=42)
    assert len(paths) == 3
    assert {p.name for p in paths} == EXPECTED
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 5_000  # non-trivial png
