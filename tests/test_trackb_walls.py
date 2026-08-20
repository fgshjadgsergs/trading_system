"""Track B3: stability-weighted wall zones feeding the S3 veto — causal by design."""

from __future__ import annotations

import polars as pl
import pytest

from trading_system.core.timeutils import NS_PER_S
from trading_system.signals.detectors import s3_filter
from trading_system.spoof.score import W_LIFE
from trading_system.spoof.walls import merge_zones, wall_zones_at

T0 = 1_755_600_000 * NS_PER_S


def _episode(
    price: float,
    birth_s: float,
    death_s: float | None,
    score: float,
    max_qty: float = 10.0,
    lifetime_ms: float | None = None,
) -> dict:
    death = None if death_s is None else T0 + int(death_s * NS_PER_S)
    birth = T0 + int(birth_s * NS_PER_S)
    return {
        "price": price,
        "birth_ts": birth,
        "death_ts": death if death is not None else T0 + 10**15,
        "alive": death_s is None,
        "lifetime_ms": lifetime_ms
        if lifetime_ms is not None
        else ((death - birth) / 1e6 if death_s is not None else 0.0),
        "max_qty": max_qty,
        "score": score,
    }


def _frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_dead_proven_wall_makes_zone_with_age_decay():
    eps = _frame([_episode(100.0, 0, 60, score=0.9)])
    ts = T0 + 3_660 * NS_PER_S  # one hour after death
    zones = wall_zones_at(eps, ts, band=2.0, min_score=0.1, dead_half_life_s=3_600.0)
    assert zones.height == 1
    row = zones.row(0, named=True)
    assert (row["lo"], row["hi"]) == (99.0, 101.0)
    # heat = price * qty * score * 0.5^(age/T)
    assert row["heat_usd"] == pytest.approx(100.0 * 10.0 * 0.9 * 0.5, rel=1e-6)
    # far in the future the wall fades below the score gate
    far = wall_zones_at(eps, T0 + 100_000 * NS_PER_S, band=2.0, min_score=0.1)
    assert far.height == 0


def test_live_wall_uses_lifetime_prior_not_full_score():
    """An episode still alive at ts must NOT leak its eventual full score."""
    # dies later with a perfect score; at ts it has lived briefly
    eps = _frame(
        [
            _episode(100.0, 0, 600, score=1.0),  # death AFTER ts -> live at ts
            _episode(90.0, -500, -100, score=0.8, lifetime_ms=400_000),  # resolved reference
        ]
    )
    ts = T0 + 200 * NS_PER_S
    zones = wall_zones_at(eps, ts, band=2.0, min_score=0.0, dead_half_life_s=1e12)
    live_zone = zones.filter(pl.col("lo") == 99.0)
    assert live_zone.height == 1
    # lived 200s < reference 400s -> percentile 0, then eff = W_LIFE * 0 = 0?  searchsorted right on [400000] for 200000 -> 0
    assert live_zone["heat_usd"][0] <= 100.0 * 10.0 * W_LIFE + 1e-9
    # and never anywhere near the full-score heat
    assert live_zone["heat_usd"][0] < 100.0 * 10.0 * 1.0 * 0.5


def test_unborn_walls_excluded():
    eps = _frame([_episode(100.0, 500, 600, score=0.9)])
    assert wall_zones_at(eps, T0 + 100 * NS_PER_S, band=2.0).height == 0


def test_merge_zones_and_s3_veto_on_wall():
    wall = pl.DataFrame({"lo": [104.0], "hi": [106.0], "heat_usd": [5e6]})
    map_zones = pl.DataFrame({"lo": [140.0], "hi": [142.0], "heat_usd": [1e4]})
    zones = merge_zones(map_zones, wall)
    assert zones.height == 2
    events = pl.DataFrame(
        {
            "ts": [T0],
            "signal": ["s1"],
            "side": [1],
            "price": [100.0],
            "target": [110.0],
            "meta": [0.0],
        },
        schema_overrides={"side": pl.Int8},
    )
    out = s3_filter(events, zones, dense_quantile=0.5)
    assert out["blocked"][0]  # the honest wall at 105 stands in the way


def test_empty_inputs():
    assert wall_zones_at(pl.DataFrame(), T0, band=1.0).height == 0
    assert merge_zones().height == 0
