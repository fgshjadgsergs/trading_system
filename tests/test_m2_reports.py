"""M2 viz tests: demo_reports produces the checklist figures as real pngs."""

from __future__ import annotations

from trading_system.book.reports import demo_reports


def test_demo_reports_generates_all_figures(tmp_path):
    paths = demo_reports(tmp_path, seed=42)
    names = {p.name for p in paths}
    assert names == {"m2_book_heatmap_1h.png", "m2_spread_depth_time.png"}
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 5_000, f"{p.name} is trivially small"
