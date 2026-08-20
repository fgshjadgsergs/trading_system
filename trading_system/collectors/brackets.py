"""Per-symbol margin bracket tables from Binance /fapi/v1/leverageBracket.

The endpoint is SIGNED (USER_DATA), so the transport is injected: pass an
authenticated ``http_get(url, params) -> json`` from the operator's client.
This module owns parsing, validation and the local json cache; everything
downstream (BinanceUsdmLiquidation, LiqMap) consumes the parsed tables.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from trading_system.core.liquidation import BinanceUsdmLiquidation, MarginBracket
from trading_system.core.schema import Side

BracketTables = dict[str, tuple[MarginBracket, ...]]


def parse_leverage_brackets(payload: list[dict] | dict) -> BracketTables:
    """Real /fapi/v1/leverageBracket shape -> {symbol: sorted bracket tuple}.

    Accepts both the full list and the single-symbol object. Brackets are
    sorted by notionalCap; continuity of the maintenance function is the
    exchange's contract, not re-derived here.
    """
    items = payload if isinstance(payload, list) else [payload]
    tables: BracketTables = {}
    for item in items:
        rows = sorted(item["brackets"], key=lambda b: float(b["notionalCap"]))
        if not rows:
            continue
        tables[item["symbol"]] = tuple(
            MarginBracket(
                max_notional_usd=float(b["notionalCap"]),
                mmr=float(b["maintMarginRatio"]),
                cum=float(b["cum"]),
            )
            for b in rows
        )
    return tables


async def fetch_leverage_brackets(
    http_get: Callable[[str, dict | None], Awaitable[list | dict]],
    rest_base: str = "https://fapi.binance.com",
    symbol: str | None = None,
) -> BracketTables:
    """Fetch and parse bracket tables via an injected (signed) transport."""
    params = {"symbol": symbol} if symbol else None
    payload = await http_get(f"{rest_base}/fapi/v1/leverageBracket", params)
    return parse_leverage_brackets(payload)


def save_brackets(path: str | Path, tables: BracketTables) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        sym: [
            {"notionalCap": b.max_notional_usd, "maintMarginRatio": b.mmr, "cum": b.cum}
            for b in rows
        ]
        for sym, rows in tables.items()
    }
    p.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    return p


def load_brackets(path: str | Path) -> BracketTables:
    """Read tables from disk: accepts both the cache format written by
    save_brackets and a raw /fapi/v1/leverageBracket response dump."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(doc, list) or "brackets" in doc:
        return parse_leverage_brackets(doc)
    return {
        sym: tuple(
            MarginBracket(
                max_notional_usd=float(b["notionalCap"]),
                mmr=float(b["maintMarginRatio"]),
                cum=float(b["cum"]),
            )
            for b in sorted(rows, key=lambda r: float(r["notionalCap"]))
        )
        for sym, rows in doc.items()
        if rows  # mirror parse_leverage_brackets: empty tables are dropped
    }


def bracket_liq_price_fn(
    tables: BracketTables, symbol: str, flat_mmr_fallback: float = 0.005
) -> Callable[[float, float, Side, float], float]:
    """(entry, leverage, side, qty) -> tier-self-consistent liquidation price.

    Plugs straight into LiqMap.liq_price_fn; symbols missing from the tables
    fall back to the flat maintenance rate.
    """
    formula = BinanceUsdmLiquidation(flat_mmr=flat_mmr_fallback, brackets=tables)

    def fn(entry: float, leverage: float, side: Side, qty: float) -> float:
        return formula.liq_price(entry, leverage, side, symbol=symbol, qty=qty)

    return fn
