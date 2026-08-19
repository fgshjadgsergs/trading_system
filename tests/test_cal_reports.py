"""Stage 3 demo report figures: all pngs exist and are non-trivial."""

from __future__ import annotations

from trading_system.calibration.reports import demo_reports

EXPECTED = {
    "cal_event_reversal_paths.png",
    "cal_event_magnet_curve.png",
    "cal_event_lvn_paths.png",
    "cal_ladder_capture_bars.png",
    "cal_calibration_curve.png",
}


def test_demo_reports_generates_all_figures(tmp_path):
    paths = demo_reports(tmp_path, seed=42)
    assert {p.name for p in paths} == EXPECTED
    for p in paths:
        assert p.exists()
        assert p.stat().st_size > 5 * 1024, f"{p.name} too small"
        assert p.parent == tmp_path


def test_demo_reports_deterministic_names(tmp_path):
    a = [p.name for p in demo_reports(tmp_path / "a", seed=42)]
    b = [p.name for p in demo_reports(tmp_path / "b", seed=42)]
    assert a == b
