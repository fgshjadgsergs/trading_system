"""Event-driven backtest engine over unified-schema trade prints.

Replays a ``POLARS_SCHEMAS['trade']`` frame in ts_event order, builds time
bars on the fly, dispatches strategy callbacks and matches orders through the
fill models in :mod:`trading_system.backtest.fills`. Optional inputs: a
``mark_price`` frame (adds funding accrual on the open position at each
funding event), a best bid/ask frame (real spreads for taker fills), and a
``book_provider`` callback (market orders walk the provided L2 side).

Anti-lookahead ordering per print i at time t:

    funding events <= t  ->  bar close (strategy.on_bar)  ->  market fills
    (priced on PRE-print state)  ->  limit matching against print i (strict
    cross, pro-rata partials)  ->  apply print to state  ->  strategy.on_trade
    (orders it emits are first considered at print i+1)

so no order acts on a print before its activation and every taker fill is
priced only on information available before the triggering print.
"""

from __future__ import annotations

import enum
import math
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import polars as pl

from trading_system.backtest.fills import (
    LatencyModel,
    fee_usd,
    impact_bps,
    limit_crossed,
    market_fill_price,
    walk_book,
)
from trading_system.core.schema import Side
from trading_system.core.timeutils import NS_PER_S, TIMEFRAME_NS

BookProvider = Callable[[int, Side], Sequence[tuple[float, float]]]


class OrderType(enum.StrEnum):
    MARKET = "market"
    LIMIT = "limit"


@dataclass(slots=True)
class Order:
    """Strategy order intent; ``ts_placed``/``order_id``/``ts_active`` are set by the engine."""

    side: Side
    qty: float
    order_type: OrderType = OrderType.MARKET
    limit_price: float | None = None
    ts_placed: int = 0
    order_id: int = -1
    ts_active: int = -1


@dataclass(frozen=True, slots=True)
class Fill:
    order_id: int
    ts: int
    side: Side
    qty: float
    price: float
    ref_mid: float  # frictionless reference price at fill time (pre-print mid)
    maker: bool
    fee_usd: float
    slippage_usd: float  # (price - ref_mid) * qty * sign; >= 0 for takers


@dataclass(slots=True)
class Bar:
    """Time bar built from prints; ``ts_close`` is the bucket boundary."""

    index: int
    ts_open: int
    ts_close: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    volume_usd: float
    n_trades: int


@dataclass(frozen=True, slots=True)
class TradePrint:
    ts: int
    price: float
    qty: float
    qty_usd: float
    side: Side


@dataclass(slots=True)
class Context:
    """Read-only view of engine state passed to strategy callbacks."""

    ts: int = 0
    bar_index: int = -1
    position_qty: float = 0.0
    cash: float = 0.0
    equity: float = 0.0
    last_price: float = math.nan
    n_pending: int = 0
    pending_qty_signed: float = 0.0  # signed unfilled qty of resting orders


class Strategy(Protocol):
    """Implement either or both callbacks; each returns order intents (or None)."""

    def on_bar(self, bar: Bar, ctx: Context) -> Iterable[Order] | None: ...

    def on_trade(self, trade: TradePrint, ctx: Context) -> Iterable[Order] | None: ...


@dataclass(slots=True)
class BacktestConfig:
    latency_ms_min: float = 200.0
    latency_ms_max: float = 500.0
    taker_fee: float = 5e-4  # fraction of notional
    maker_fee: float = 2e-4
    half_spread_bps: float = 0.5
    impact_coef_bps: float = 10.0
    impact_cap_bps: float = 25.0
    impact_window_s: float = 60.0
    bar_ns: int = TIMEFRAME_NS["1m"]
    init_cash: float = 100_000.0
    limit_participation: float = 1.0  # cap of limit fill vs print size
    seed: int = 42

    @classmethod
    def from_config(
        cls, cfg: dict[str, Any], *, timeframe: str = "1m", seed: int = 42, **overrides: Any
    ) -> BacktestConfig:
        """Build from the repo yaml config's ``backtest`` section."""
        bt = cfg.get("backtest", {})
        kwargs: dict[str, Any] = {
            "latency_ms_min": float(bt.get("latency_ms_min", 200)),
            "latency_ms_max": float(bt.get("latency_ms_max", 500)),
            "taker_fee": float(bt.get("taker_fee", 5e-4)),
            "maker_fee": float(bt.get("maker_fee", 2e-4)),
            "half_spread_bps": float(bt.get("half_spread_bps", 0.5)),
            "bar_ns": TIMEFRAME_NS[timeframe],
            "seed": seed,
        }
        kwargs.update(overrides)
        return cls(**kwargs)


@dataclass(slots=True)
class BacktestResult:
    equity_curve: pl.DataFrame  # ts, price, cash, position_qty, equity, cum costs
    fills: list[Fill]
    orders: list[Order]
    latencies_ns: list[int]
    init_cash: float
    final_equity: float
    net_pnl: float
    gross_pnl: float  # fills valued at ref mid, no fees/funding
    fees_usd: float
    slippage_usd: float
    funding_usd: float  # positive = net funding paid
    n_bars: int

    @property
    def total_costs_usd(self) -> float:
        return self.fees_usd + self.slippage_usd + self.funding_usd


@dataclass(slots=True)
class _Pending:
    order: Order
    remaining: float
    min_print_i: int  # first print index the order may interact with


_EQUITY_SCHEMA = {
    "ts": pl.Int64,
    "price": pl.Float64,
    "cash": pl.Float64,
    "position_qty": pl.Float64,
    "equity": pl.Float64,
    "fees_cum": pl.Float64,
    "slippage_cum": pl.Float64,
    "funding_cum": pl.Float64,
}


def funding_events(mark_prices: pl.DataFrame, ts_lo: int, ts_hi: int) -> list[tuple[int, float, float]]:
    """(funding_ts, rate, mark_price) for each distinct next_funding_ts in range.

    Rate and mark price come from the last mark row at or before the funding
    time, so accrual uses only information available at the event.
    """
    if mark_prices.is_empty():
        return []
    mp = mark_prices.sort("ts_event")
    ts = mp["ts_event"].to_numpy()
    rate = mp["funding_rate"].to_numpy()
    px = mp["mark_price"].to_numpy()
    times = np.unique(mp["next_funding_ts"].to_numpy())
    out: list[tuple[int, float, float]] = []
    for f_ts in times:
        if not ts_lo <= f_ts <= ts_hi:
            continue
        j = int(np.searchsorted(ts, f_ts, side="right")) - 1
        if j < 0:
            continue
        out.append((int(f_ts), float(rate[j]), float(px[j])))
    return out


class _Engine:
    def __init__(
        self,
        trades: pl.DataFrame,
        strategy: Strategy,
        config: BacktestConfig,
        mark_prices: pl.DataFrame | None,
        bbo: pl.DataFrame | None,
        book_provider: BookProvider | None,
    ) -> None:
        self.cfg = config
        self.strategy = strategy
        self.book_provider = book_provider
        frame = trades.sort("ts_event", maintain_order=True)
        self.ts = frame["ts_event"].to_numpy()
        self.px = frame["price"].to_numpy()
        self.qty = frame["qty"].to_numpy()
        if "qty_usd" in frame.columns:
            self.usd = frame["qty_usd"].to_numpy()
        else:
            self.usd = self.px * self.qty
        self.side = frame["side"].to_numpy() if "side" in frame.columns else np.ones(len(frame))
        self.n = len(frame)
        if self.n == 0:
            raise ValueError("empty trade frame")

        if bbo is not None and not bbo.is_empty():
            b = bbo.sort("ts_event") if "ts_event" in bbo.columns else bbo.sort("ts")
            tcol = "ts_event" if "ts_event" in b.columns else "ts"
            self.bbo_ts = b[tcol].to_numpy()
            self.bbo_bid = b["bid"].to_numpy()
            self.bbo_ask = b["ask"].to_numpy()
        else:
            self.bbo_ts = None
            self.bbo_bid = None
            self.bbo_ask = None
        self.bbo_i = -1  # last bbo row with ts strictly < current print ts

        if mark_prices is not None:
            self.funding = funding_events(mark_prices, int(self.ts[0]), int(self.ts[-1]))
        else:
            self.funding = []
        self.f_i = 0

        rng = np.random.default_rng(config.seed)
        self.latency = LatencyModel(config.latency_ms_min, config.latency_ms_max, rng)

        self.cash = config.init_cash
        self.gross_cash = config.init_cash
        self.pos = 0.0
        self.last_price = math.nan
        self.fees = 0.0
        self.slip = 0.0
        self.fund = 0.0
        self.fills: list[Fill] = []
        self.orders: list[Order] = []
        self.latencies: list[int] = []
        self.pending: list[_Pending] = []
        self.next_id = 0
        self.cur_bar: Bar | None = None
        self.bar_seq = 0
        self.ctx = Context(cash=self.cash)
        self.eq_rows: list[tuple] = []
        self.vol_window: deque[tuple[int, float]] = deque()
        self.vol_sum = 0.0
        self.window_ns = int(config.impact_window_s * NS_PER_S)
        self.has_on_bar = callable(getattr(strategy, "on_bar", None))
        self.has_on_trade = callable(getattr(strategy, "on_trade", None))

    # -- state helpers -----------------------------------------------------

    def _update_ctx(self, ts: int) -> None:
        c = self.ctx
        c.ts = ts
        c.bar_index = self.bar_seq - 1
        c.position_qty = self.pos
        c.cash = self.cash
        c.last_price = self.last_price
        c.equity = self.cash + (self.pos * self.last_price if not math.isnan(self.last_price) else 0.0)
        c.n_pending = len(self.pending)
        c.pending_qty_signed = sum(
            p.remaining * (1.0 if p.order.side is Side.BUY else -1.0) for p in self.pending
        )

    def _mid_spread(self) -> tuple[float, float]:
        """Pre-print (mid, half_spread_bps) from bbo when present, else last price."""
        if self.bbo_ts is not None and self.bbo_i >= 0:
            bid = float(self.bbo_bid[self.bbo_i])
            ask = float(self.bbo_ask[self.bbo_i])
            mid = 0.5 * (bid + ask)
            return mid, (ask - bid) / 2.0 / mid * 1e4
        return self.last_price, self.cfg.half_spread_bps

    def _accept(self, intents: Iterable[Order] | None, ts_placed: int, min_print_i: int) -> None:
        for o in intents or ():
            if not (o.qty > 0.0 and math.isfinite(o.qty)):
                raise ValueError(f"order qty must be positive and finite, got {o.qty}")
            if o.order_type is OrderType.LIMIT and not (
                o.limit_price is not None and o.limit_price > 0.0
            ):
                raise ValueError("limit order requires a positive limit_price")
            o.order_id = self.next_id
            self.next_id += 1
            o.ts_placed = ts_placed
            lat = self.latency.draw_ns()
            o.ts_active = ts_placed + lat
            self.latencies.append(lat)
            self.orders.append(o)
            self.pending.append(_Pending(order=o, remaining=o.qty, min_print_i=min_print_i))

    def _fill(self, p: _Pending, ts: int, price: float, qty: float, ref_mid: float, maker: bool) -> None:
        o = p.order
        sign = 1.0 if o.side is Side.BUY else -1.0
        fee = fee_usd(price * qty, maker, self.cfg.maker_fee, self.cfg.taker_fee)
        slippage = (price - ref_mid) * qty * sign
        self.cash -= sign * qty * price + fee
        self.gross_cash -= sign * qty * ref_mid
        self.pos += sign * qty
        self.fees += fee
        self.slip += slippage
        p.remaining -= qty
        self.fills.append(
            Fill(
                order_id=o.order_id,
                ts=ts,
                side=o.side,
                qty=qty,
                price=price,
                ref_mid=ref_mid,
                maker=maker,
                fee_usd=fee,
                slippage_usd=slippage,
            )
        )

    # -- per-print steps ---------------------------------------------------

    def _process_funding(self, t: int) -> None:
        while self.f_i < len(self.funding) and self.funding[self.f_i][0] <= t:
            _, rate, mark_px = self.funding[self.f_i]
            cost = rate * self.pos * mark_px  # long pays positive rate, short receives
            self.cash -= cost
            self.fund += cost
            self.f_i += 1

    def _append_equity(self, ts: int, price: float) -> None:
        eq = self.cash + self.pos * price
        self.eq_rows.append(
            (ts, price, self.cash, self.pos, eq, self.fees, self.slip, self.fund)
        )

    def _close_bar(self, ts_now: int, min_print_i: int) -> None:
        bar = self.cur_bar
        assert bar is not None
        self.cur_bar = None
        self.bar_seq += 1
        self._append_equity(bar.ts_close, bar.close)
        if self.has_on_bar:
            self._update_ctx(ts_now)
            self._accept(self.strategy.on_bar(bar, self.ctx), ts_now, min_print_i)

    def _maybe_close_bar(self, i: int, t: int) -> None:
        if self.cur_bar is not None and t - t % self.cfg.bar_ns > self.cur_bar.ts_open:
            self._close_bar(t, i)

    def _market_price(self, o: Order, qty: float, t: int, mid: float, hs_bps: float) -> float:
        if self.book_provider is not None:
            levels = self.book_provider(t, o.side)
            return walk_book(levels, qty)
        imp = impact_bps(
            qty * mid, self.vol_sum, self.cfg.impact_coef_bps, self.cfg.impact_cap_bps
        )
        return market_fill_price(o.side, mid, hs_bps, imp)

    def _exec_market(self, i: int, t: int) -> None:
        if math.isnan(self.last_price):
            return
        mid, hs_bps = self._mid_spread()
        for p in [q for q in self.pending if q.order.order_type is OrderType.MARKET]:
            if p.order.ts_active <= t and p.min_print_i <= i:
                price = self._market_price(p.order, p.remaining, t, mid, hs_bps)
                self._fill(p, t, price, p.remaining, mid, maker=False)
        self.pending = [p for p in self.pending if p.remaining > 1e-12]

    def _match_limits(self, i: int, t: int, print_price: float, print_qty: float) -> None:
        filled_any = False
        for p in self.pending:
            o = p.order
            if o.order_type is not OrderType.LIMIT or o.ts_active > t or p.min_print_i > i:
                continue
            assert o.limit_price is not None
            if not limit_crossed(o.side, o.limit_price, print_price):
                continue
            qty = min(p.remaining, print_qty * self.cfg.limit_participation)
            if qty <= 0.0:
                continue
            ref = self.last_price if not math.isnan(self.last_price) else o.limit_price
            self._fill(p, t, o.limit_price, qty, ref, maker=True)
            filled_any = True
        if filled_any:
            self.pending = [p for p in self.pending if p.remaining > 1e-12]

    def _apply_print(self, i: int, t: int) -> None:
        px = float(self.px[i])
        q = float(self.qty[i])
        usd = float(self.usd[i])
        self.last_price = px
        self.vol_window.append((t, usd))
        self.vol_sum += usd
        cutoff = t - self.window_ns
        while self.vol_window and self.vol_window[0][0] < cutoff:
            self.vol_sum -= self.vol_window.popleft()[1]
        bucket = t - t % self.cfg.bar_ns
        bar = self.cur_bar
        if bar is None:
            self.cur_bar = Bar(
                index=self.bar_seq,
                ts_open=bucket,
                ts_close=bucket + self.cfg.bar_ns,
                open=px,
                high=px,
                low=px,
                close=px,
                volume=q,
                volume_usd=usd,
                n_trades=1,
            )
        else:
            bar.high = max(bar.high, px)
            bar.low = min(bar.low, px)
            bar.close = px
            bar.volume += q
            bar.volume_usd += usd
            bar.n_trades += 1

    def _advance_bbo(self, t: int) -> None:
        if self.bbo_ts is None:
            return
        j = self.bbo_i
        n = len(self.bbo_ts)
        while j + 1 < n and self.bbo_ts[j + 1] < t:  # strictly pre-print quotes only
            j += 1
        self.bbo_i = j

    def run(self) -> BacktestResult:
        for i in range(self.n):
            t = int(self.ts[i])
            self._advance_bbo(t)
            self._process_funding(t)
            self._maybe_close_bar(i, t)
            self._exec_market(i, t)
            self._match_limits(i, t, float(self.px[i]), float(self.qty[i]))
            self._apply_print(i, t)
            if self.has_on_trade:
                ev = TradePrint(
                    ts=t,
                    price=float(self.px[i]),
                    qty=float(self.qty[i]),
                    qty_usd=float(self.usd[i]),
                    side=Side(int(self.side[i])),
                )
                self._update_ctx(t)
                # orders emitted here are first considered at print i + 1
                self._accept(self.strategy.on_trade(ev, self.ctx), t, i + 1)

        last_t = int(self.ts[-1])
        end_ts = last_t
        if self.cur_bar is not None:
            end_ts = max(end_ts, self.cur_bar.ts_close)
            self._close_bar(last_t, self.n)
        # Market orders still pending at end of data execute at the final mid
        # (spread and impact applied); resting limit orders expire unfilled.
        mid, hs_bps = self.last_price, self.cfg.half_spread_bps
        if self.bbo_ts is not None and self.bbo_i >= 0:
            mid, hs_bps = self._mid_spread()
        for p in [q for q in self.pending if q.order.order_type is OrderType.MARKET]:
            price = self._market_price(p.order, p.remaining, last_t, mid, hs_bps)
            self._fill(p, last_t, price, p.remaining, mid, maker=False)
        self.pending = [p for p in self.pending if p.remaining > 1e-12]
        self._append_equity(end_ts, self.last_price)

        equity_curve = pl.DataFrame(self.eq_rows, schema=_EQUITY_SCHEMA, orient="row")
        final_equity = self.cash + self.pos * self.last_price
        gross_final = self.gross_cash + self.pos * self.last_price
        return BacktestResult(
            equity_curve=equity_curve,
            fills=self.fills,
            orders=self.orders,
            latencies_ns=self.latencies,
            init_cash=self.cfg.init_cash,
            final_equity=final_equity,
            net_pnl=final_equity - self.cfg.init_cash,
            gross_pnl=gross_final - self.cfg.init_cash,
            fees_usd=self.fees,
            slippage_usd=self.slip,
            funding_usd=self.fund,
            n_bars=self.bar_seq,
        )


def run_backtest(
    trades: pl.DataFrame,
    strategy: Strategy,
    config: BacktestConfig,
    *,
    mark_prices: pl.DataFrame | None = None,
    bbo: pl.DataFrame | None = None,
    book_provider: BookProvider | None = None,
) -> BacktestResult:
    """Replay a unified-schema trade frame through the strategy.

    ``bbo`` (optional): frame with ts_event/bid/ask giving real spreads.
    ``book_provider`` (optional): (ts, side) -> L2 levels; market orders then
    walk the book instead of the bps impact model.
    """
    return _Engine(trades, strategy, config, mark_prices, bbo, book_provider).run()
