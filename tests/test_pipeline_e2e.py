"""Gate 2 smoke: the full pipeline runs as one script with no manual steps."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]


def _map_bars(vwap_shift: float) -> pl.DataFrame:
    """Two flat bars; the second bar's VWAP is close + vwap_shift."""
    n = 2
    close = 100.0
    ns = 60_000_000_000
    return pl.DataFrame(
        {
            "ts_open": [i * ns for i in range(n)],
            "ts_close": [(i + 1) * ns for i in range(n)],
            "low": [close] * n,
            "high": [close] * n,
            "close": [close] * n,
            "volume": [10.0] * n,
            "quote_volume": [10.0 * close, 10.0 * (close + vwap_shift)],
            "d_oi_usd": [1_000.0] * n,
            "atr": [10.0] * n,
        }
    )


CFG_MAP = {
    "liqmap": {
        "leverage_grid": [10],
        "bucket_atr_fraction": 0.5,  # бакет = 5.0
        "long_share_default": 0.5,
        "decay_half_life_s": 86_400.0,
    }
}


def test_stage_map_vwap_entry_capability():
    """M4: entry_price='vwap' меняет карту на баре с |close-vwap| > bucket_size
    и бит-в-бит совпадает с 'close' там, где vwap == close."""
    from scripts.run_pipeline import stage_map

    bars_far = _map_bars(vwap_shift=10.0)  # |close - vwap| = 10 > бакет 5
    lm_close, _ = stage_map(bars_far, CFG_MAP, "X")
    lm_vwap, _ = stage_map(bars_far, CFG_MAP, "X", entry_price="vwap")
    assert lm_close.heat != lm_vwap.heat
    bars_same = _map_bars(vwap_shift=0.0)  # vwap == close на каждом баре
    lm_c2, _ = stage_map(bars_same, CFG_MAP, "X")
    lm_v2, _ = stage_map(bars_same, CFG_MAP, "X", entry_price="vwap")
    assert lm_c2.heat == lm_v2.heat
    # честные ошибки: нет vwap-колонок / неизвестный entry_price
    with pytest.raises(ValueError):
        stage_map(bars_far.drop("volume", "quote_volume"), CFG_MAP, "X", entry_price="vwap")
    with pytest.raises(ValueError):
        stage_map(bars_far, CFG_MAP, "X", entry_price="open")


def test_pipeline_end_to_end(tmp_path):
    out = tmp_path / "reports"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_pipeline.py"),
            "--out",
            str(out),
            "--n-trades",
            "60000",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=420,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = out / "pipeline" / "README.md"
    assert report.exists()
    pngs = list((out / "pipeline").glob("*.png"))
    assert len(pngs) >= 5
    for p in pngs:
        assert p.stat().st_size > 5_000
