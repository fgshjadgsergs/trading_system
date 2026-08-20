"""Track B1: real leverage-bracket tables — parse, cache, wire into the map."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest

from trading_system.collectors.brackets import (
    bracket_liq_price_fn,
    fetch_leverage_brackets,
    load_brackets,
    parse_leverage_brackets,
    save_brackets,
)
from trading_system.core.liquidation import BinanceUsdmLiquidation
from trading_system.core.schema import Side
from trading_system.liqmap.buckets import PriceBuckets
from trading_system.liqmap.map import LiqMap, StaticWeights

FIXTURE = Path(__file__).parent / "fixtures" / "brackets" / "leverage_bracket.json"


def _payload() -> list[dict]:
    return json.loads(FIXTURE.read_text())


def test_parse_real_payload_shape():
    tables = parse_leverage_brackets(_payload())
    assert set(tables) == {"BTCUSDT", "SOLUSDT"}
    btc = tables["BTCUSDT"]
    assert [b.max_notional_usd for b in btc] == [50_000, 250_000, 1_000_000, 10_000_000, 20_000_000]
    assert btc[2].mmr == 0.01 and btc[2].cum == 1_300.0
    # single-symbol object form also parses
    one = parse_leverage_brackets(_payload()[1])
    assert set(one) == {"SOLUSDT"} and len(one["SOLUSDT"]) == 3


def test_save_load_roundtrip(tmp_path):
    tables = parse_leverage_brackets(_payload())
    p = save_brackets(tmp_path / "brackets.json", tables)
    back = load_brackets(p)
    assert back == tables


def test_fetch_with_injected_transport():
    seen: dict = {}

    async def http_get(url, params):
        seen["url"], seen["params"] = url, params
        return _payload()

    tables = asyncio.new_event_loop().run_until_complete(
        fetch_leverage_brackets(http_get, "https://fapi.test", symbol=None)
    )
    assert seen["url"].endswith("/fapi/v1/leverageBracket")
    assert "BTCUSDT" in tables


def test_bracket_fn_matches_formula_and_tier_selfconsistency():
    tables = parse_leverage_brackets(_payload())
    fn = bracket_liq_price_fn(tables, "BTCUSDT")
    formula = BinanceUsdmLiquidation(brackets=tables)
    # tier-crossing long from the review case: 6 BTC @50k 2x -> tier-2 solution
    assert fn(50_000.0, 2, Side.BUY, 6.0) == pytest.approx(
        formula.liq_price(50_000.0, 2, Side.BUY, symbol="BTCUSDT", qty=6.0)
    )
    assert fn(50_000.0, 2, Side.BUY, 6.0) == pytest.approx(
        (50_000.0 * 0.5 - 50.0 / 6.0) / (1 - 0.005), rel=1e-12
    )
    # unknown symbol falls back to flat mmr
    flat = bracket_liq_price_fn(tables, "DOGEUSDT")
    assert flat(100.0, 10, Side.BUY, 1.0) == pytest.approx(100.0 * 0.9 / 0.995, rel=1e-12)


def test_liqmap_passes_position_qty_to_bracket_fn():
    """allocate must hand the slice's coin size to a 4-arg liq fn."""
    seen: list[float] = []

    def spy_fn(entry, lev, side, qty):
        seen.append(qty)
        return entry * 0.9 if side is Side.BUY else entry * 1.1

    lm = LiqMap([10.0, 20.0], PriceBuckets(10.0), StaticWeights(np.array([3.0, 1.0])), liq_price_fn=spy_fn)
    lm.allocate(1_000_000.0, 50_000.0)
    # 2 sides x 2 leverages; qty = amount / price
    assert sorted(seen) == pytest.approx(
        sorted(
            [
                1_000_000.0 * 0.5 * 0.75 / 50_000.0,
                1_000_000.0 * 0.5 * 0.25 / 50_000.0,
            ]
            * 2
        )
    )
    assert lm.total_heat() == pytest.approx(1_000_000.0)
    assert lm.mass_balance_error() < 1e-6


def test_liqmap_backward_compatible_with_3arg_fn():
    lm = LiqMap([10.0], PriceBuckets(10.0), StaticWeights(np.array([1.0])),
                liq_price_fn=lambda entry, lev, side: entry * (0.9 if side is Side.BUY else 1.1))
    lm.allocate(100_000.0, 50_000.0)
    assert lm.total_heat() == pytest.approx(100_000.0)


def test_bracket_map_places_large_flow_deeper_than_flat():
    """With cum deductions, big long slices liquidate at a HIGHER price than
    flat-MMR pretends (tier cum shrinks the numerator) — heat sits closer."""
    tables = parse_leverage_brackets(_payload())
    fn = bracket_liq_price_fn(tables, "BTCUSDT")
    flat = BinanceUsdmLiquidation().liq_price  # flat 0.005
    big_qty = 40.0  # 2M USD at 50k -> tier 4
    lp_bracket = fn(50_000.0, 5, Side.BUY, big_qty)
    lp_flat = flat(50_000.0, 5, Side.BUY)
    assert lp_bracket > lp_flat  # closer to entry: liquidated earlier than naive flat model
