"""Binance USDT-M liquidation price model: flat MMR v1, margin brackets v2.

Isolated one-way position of qty q at entry E with leverage L liquidates when
wallet margin + uPnL hits maintenance margin (MMR * notional - cum):

    long:  P = (E * (1 - 1/L) - cum/q) / (1 - MMR)
    short: P = (E * (1 + 1/L) + cum/q) / (1 + MMR)

which matches the exchange calculator's (WB + cum -/+ q*E) / (q*(MMR -/+ 1)).
"""

from __future__ import annotations

from dataclasses import dataclass

from trading_system.core.adapter import LiquidationFormula
from trading_system.core.schema import Side


@dataclass(frozen=True, slots=True)
class MarginBracket:
    max_notional_usd: float
    mmr: float
    cum: float  # maintenance amount deduction, USD


# Reference bracket table (BTCUSDT-like tiers). Tables drift over time and per
# symbol; pass a current table per symbol for production accuracy.
DEFAULT_BRACKETS: tuple[MarginBracket, ...] = (
    MarginBracket(50_000, 0.004, 0),
    MarginBracket(250_000, 0.005, 50),
    MarginBracket(1_000_000, 0.010, 1_300),
    MarginBracket(10_000_000, 0.025, 16_300),
    MarginBracket(20_000_000, 0.050, 266_300),
    MarginBracket(50_000_000, 0.100, 1_266_300),
    MarginBracket(100_000_000, 0.125, 2_516_300),
    MarginBracket(200_000_000, 0.150, 5_016_300),
    MarginBracket(float("inf"), 0.250, 25_016_300),
)


def bracket_for(notional_usd: float, brackets: tuple[MarginBracket, ...]) -> MarginBracket:
    for b in brackets:
        if notional_usd <= b.max_notional_usd:
            return b
    return brackets[-1]


def liq_price(
    entry: float,
    leverage: float,
    side: Side,
    mmr: float,
    cum: float = 0.0,
    qty: float = 1.0,
) -> float:
    """Isolated liquidation price with an explicit maintenance margin rate."""
    if entry <= 0 or leverage <= 0 or qty <= 0:
        raise ValueError("entry, leverage and qty must be positive")
    if not 0 <= mmr < 1:
        raise ValueError("mmr must be in [0, 1)")
    if side is Side.BUY:  # long
        return max(0.0, (entry * (1 - 1 / leverage) - cum / qty) / (1 - mmr))
    return (entry * (1 + 1 / leverage) + cum / qty) / (1 + mmr)


class BinanceUsdmLiquidation(LiquidationFormula):
    """Flat-MMR by default; bracket table per symbol when provided."""

    def __init__(
        self,
        flat_mmr: float = 0.005,
        brackets: dict[str, tuple[MarginBracket, ...]] | None = None,
    ) -> None:
        self._flat_mmr = flat_mmr
        self._brackets = brackets or {}

    def maint_margin_rate(self, symbol: str, notional_usd: float) -> float:
        table = self._brackets.get(symbol)
        if table is None:
            return self._flat_mmr
        return bracket_for(notional_usd, table).mmr

    def liq_price(
        self,
        entry: float,
        leverage: float,
        side: Side,
        *,
        symbol: str | None = None,
        qty: float | None = None,
    ) -> float:
        table = self._brackets.get(symbol) if symbol else None
        if table is None or qty is None:
            return liq_price(entry, leverage, side, self._flat_mmr)
        # The exchange applies the tier containing the notional AT the mark
        # price, so the liquidation price must be tier-self-consistent: solve
        # per tier and accept the tier whose notional range contains qty * LP.
        # Bracket tables are continuous in notional, so exactly one tier fits
        # (up to a boundary); fall back to the entry-notional tier otherwise.
        prev_max = 0.0
        for b in table:
            lp = liq_price(entry, leverage, side, b.mmr, cum=b.cum, qty=qty)
            notional = lp * qty
            if prev_max < notional <= b.max_notional_usd or (
                notional <= b.max_notional_usd and prev_max == 0.0
            ):
                return lp
            prev_max = b.max_notional_usd
        b = bracket_for(entry * qty, table)
        return liq_price(entry, leverage, side, b.mmr, cum=b.cum, qty=qty)
