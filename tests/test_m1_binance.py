"""M1: Binance USDT-M adapter — stream names, ws/REST parsers on real formats."""

from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path

import pytest

from trading_system.collectors import binance as bn
from trading_system.collectors.ws_client import TransportExhausted
from trading_system.core.adapter import RawMessage
from trading_system.core.liquidation import BinanceUsdmLiquidation
from trading_system.core.schema import (
    BookSnapshot,
    DepthDiff,
    Kline,
    Liquidation,
    MarkPrice,
    RatioMetric,
    Side,
    Trade,
)
from trading_system.core.timeutils import NS_PER_MS

FIX = Path(__file__).parent / "fixtures" / "m1"
TS_RECV = 1_755_600_010_000 * NS_PER_MS


def load(name: str):
    return json.loads((FIX / name).read_text())


def raw(name: str) -> RawMessage:
    return RawMessage(stream="test", payload=(FIX / name).read_text(), ts_recv=TS_RECV)


@pytest.fixture()
def adapter() -> bn.BinanceUsdmAdapter:
    return bn.BinanceUsdmAdapter()


# --------------------------------------------------------------------------- #
# stream names / urls
# --------------------------------------------------------------------------- #
def test_stream_name_builders():
    assert bn.depth_stream("BTCUSDT") == "btcusdt@depth@100ms"
    assert bn.agg_trade_stream("BTCUSDT") == "btcusdt@aggTrade"
    assert bn.force_order_stream("SOLUSDT") == "solusdt@forceOrder"
    assert bn.mark_price_stream("DOGEUSDT") == "dogeusdt@markPrice@1s"
    assert bn.kline_1m_stream("BTCUSDT") == "btcusdt@kline_1m"


def test_combined_ws_url():
    url = bn.combined_ws_url(["btcusdt@aggTrade", "btcusdt@depth@100ms"])
    assert url == "wss://fstream.binance.com/stream?streams=btcusdt@aggTrade/btcusdt@depth@100ms"
    with pytest.raises(ValueError):
        bn.combined_ws_url([])


def test_adapter_stream_names(adapter):
    names = adapter.stream_names(["BTCUSDT", "SOLUSDT"], ["depth", "agg_trade"])
    assert names == [
        "btcusdt@depth@100ms",
        "solusdt@depth@100ms",
        "btcusdt@aggTrade",
        "solusdt@aggTrade",
    ]
    with pytest.raises(KeyError):
        adapter.stream_names(["BTCUSDT"], ["nope"])


# --------------------------------------------------------------------------- #
# websocket normalizers, field by field
# --------------------------------------------------------------------------- #
def test_agg_trade_maker_buyer_means_taker_sell(adapter):
    recs = adapter.normalize(raw("agg_trade.json"))
    assert len(recs) == 1
    t = recs[0]
    assert isinstance(t, Trade)
    assert t.exchange == "binance_usdm"
    assert t.symbol == "BTCUSDT"
    assert t.ts_event == 1755600001085 * NS_PER_MS  # T (trade time), not E
    assert t.ts_recv == TS_RECV
    assert t.price == 50123.10
    assert t.qty == 0.250
    assert t.qty_usd == pytest.approx(50123.10 * 0.250)
    assert t.side is Side.SELL  # m=true: buyer is maker => taker sold
    assert t.trade_id == 5933014


def test_agg_trade_combined_stream_taker_buy(adapter):
    recs = adapter.normalize(raw("agg_trade_combined.json"))
    assert len(recs) == 1
    t = recs[0]
    assert t.side is Side.BUY  # m=false => taker bought
    assert t.trade_id == 5933015
    assert t.ts_event == 1755600002190 * NS_PER_MS


def test_depth_update_fields(adapter):
    recs = adapter.normalize(raw("depth_update.json"))
    assert len(recs) == 1
    d = recs[0]
    assert isinstance(d, DepthDiff)
    assert d.symbol == "BTCUSDT"
    assert d.ts_event == 1755600003295 * NS_PER_MS  # T (transaction time)
    assert d.first_update_id == 157
    assert d.final_update_id == 160
    assert d.prev_final_update_id == 149
    assert d.bids == ((50120.50, 10.043), (50119.00, 0.0))  # qty 0 = level removal
    assert d.asks == ((50121.00, 2.510),)


def test_force_order_sell_is_long_liquidation(adapter):
    recs = adapter.normalize(raw("force_order.json"))
    assert len(recs) == 1
    liq = recs[0]
    assert isinstance(liq, Liquidation)
    assert liq.symbol == "BTCUSDT"
    assert liq.ts_event == 1755600004395 * NS_PER_MS  # o.T
    assert liq.price == 49905.30  # average fill price ap, not order price p
    assert liq.qty == 0.014
    assert liq.qty_usd == pytest.approx(49905.30 * 0.014)
    assert liq.side is Side.SELL
    assert liq.liquidated_long is True


def test_force_order_buy_side():
    msg = load("force_order.json")
    msg["o"]["S"] = "BUY"
    m = RawMessage(stream="t", payload=json.dumps(msg), ts_recv=TS_RECV)
    liq = bn.BinanceUsdmAdapter().normalize(m)[0]
    assert liq.side is Side.BUY
    assert liq.liquidated_long is False  # short position liquidated


def test_mark_price_fields(adapter):
    recs = adapter.normalize(raw("mark_price.json"))
    assert len(recs) == 1
    mp = recs[0]
    assert isinstance(mp, MarkPrice)
    assert mp.symbol == "BTCUSDT"
    assert mp.ts_event == 1755600005000 * NS_PER_MS
    assert mp.mark_price == 50130.15
    assert mp.index_price == 50128.62659091
    assert mp.funding_rate == 0.00038167
    assert mp.next_funding_ts == 1755628800000 * NS_PER_MS


def test_kline_open_and_closed(adapter):
    k_open = adapter.normalize(raw("kline_1m_open.json"))[0]
    assert isinstance(k_open, Kline)
    assert k_open.closed is False  # k.x
    assert k_open.ts_open == 1755600000000 * NS_PER_MS
    assert k_open.ts_close == 1755600059999 * NS_PER_MS
    assert (k_open.open, k_open.high, k_open.low, k_open.close) == (
        50100.00,
        50140.00,
        50095.50,
        50131.20,
    )
    assert k_open.volume == 123.456
    assert k_open.quote_volume == 6187654.32
    assert k_open.taker_buy_volume == 70.100
    assert k_open.taker_buy_quote_volume == 3513210.11
    assert k_open.n_trades == 100

    k_closed = adapter.normalize(raw("kline_1m_closed.json"))[0]
    assert k_closed.closed is True
    assert k_closed.n_trades == 132


def test_normalize_bytes_payload_and_unknown_event(adapter):
    as_bytes = RawMessage(
        stream="t", payload=(FIX / "agg_trade.json").read_bytes(), ts_recv=TS_RECV
    )
    assert len(adapter.normalize(as_bytes)) == 1
    unknown = RawMessage(stream="t", payload='{"e": "weirdEvent"}', ts_recv=TS_RECV)
    assert adapter.normalize(unknown) == []
    sub_reply = RawMessage(stream="t", payload='{"result": null, "id": 1}', ts_recv=TS_RECV)
    assert adapter.normalize(sub_reply) == []


# --------------------------------------------------------------------------- #
# REST normalizers
# --------------------------------------------------------------------------- #
def test_depth_snapshot_parse():
    snap = bn.parse_depth_snapshot(load("depth_snapshot.json"), symbol="BTCUSDT", ts_recv=TS_RECV)
    assert isinstance(snap, BookSnapshot)
    assert snap.exchange == "binance_usdm"
    assert snap.symbol == "BTCUSDT"
    assert snap.ts_event == 1755600000959 * NS_PER_MS  # T
    assert snap.last_update_id == 1027024
    assert snap.bids == ((50120.50, 10.043), (50119.00, 4.310))  # best bid first
    assert snap.asks == ((50121.00, 12.000), (50122.50, 0.500))  # best ask first


def test_open_interest_parse():
    oi = bn.parse_open_interest(load("open_interest.json"), ts_recv=TS_RECV, price=50000.0)
    assert oi.symbol == "BTCUSDT"
    assert oi.ts_event == 1755600007011 * NS_PER_MS
    assert oi.open_interest == 10659.509
    assert oi.open_interest_usd == pytest.approx(10659.509 * 50000.0)
    no_price = bn.parse_open_interest(load("open_interest.json"), ts_recv=TS_RECV)
    assert math.isnan(no_price.open_interest_usd)  # unknown, never a fake 0


def test_global_ls_account_ratio():
    pts = bn.parse_global_ls_account(load("global_ls_account.json"), ts_recv=TS_RECV)
    assert len(pts) == 2
    p = pts[0]
    assert p.metric == RatioMetric.GLOBAL_LS_ACCOUNT.value
    assert p.long_share == 0.6622
    assert p.short_share == 0.3378
    assert p.ratio == 1.9603
    assert p.ts_event == 1755599700000 * NS_PER_MS
    assert pts[1].ratio == 1.8105


def test_top_ls_position_ratio():
    pts = bn.parse_top_ls_position(load("top_ls_position.json"), ts_recv=TS_RECV)
    assert len(pts) == 1
    assert pts[0].metric == RatioMetric.TOP_LS_POSITION.value
    assert pts[0].long_share == 0.5891
    assert pts[0].short_share == 0.4109
    assert pts[0].ratio == 1.4337


def test_taker_ls_ratio_shares_from_volumes():
    pts = bn.parse_taker_ls(load("taker_ls.json"), ts_recv=TS_RECV, symbol="BTCUSDT")
    assert len(pts) == 1
    p = pts[0]
    assert p.metric == RatioMetric.TAKER_LS.value
    assert p.symbol == "BTCUSDT"  # endpoint payload has no symbol field
    total = 387.3300 + 248.5030
    assert p.long_share == pytest.approx(387.3300 / total)
    assert p.short_share == pytest.approx(248.5030 / total)
    assert p.ratio == 1.5586
    with pytest.raises(ValueError):
        bn.parse_taker_ls(load("taker_ls.json"), ts_recv=TS_RECV)  # symbol required


# --------------------------------------------------------------------------- #
# snapshot / liq_formula / subscribe with injected transports
# --------------------------------------------------------------------------- #
def test_snapshot_uses_injected_http_get():
    calls: list[tuple[str, dict]] = []

    async def fake_http_get(url: str, params: dict):
        calls.append((url, params))
        return load("depth_snapshot.json")

    adapter = bn.BinanceUsdmAdapter(http_get=fake_http_get, clock=lambda: TS_RECV)
    snap = asyncio.run(adapter.snapshot("BTCUSDT", depth=500))
    assert calls == [("https://fapi.binance.com/fapi/v1/depth", {"symbol": "BTCUSDT", "limit": 500})]
    assert snap.last_update_id == 1027024
    assert snap.ts_recv == TS_RECV


def test_snapshot_without_transport_raises():
    with pytest.raises(ValueError, match="http_get"):
        asyncio.run(bn.BinanceUsdmAdapter().snapshot("BTCUSDT"))


def test_liq_formula_is_binance_model():
    f = bn.BinanceUsdmAdapter().liq_formula()
    assert isinstance(f, BinanceUsdmLiquidation)
    assert f.liq_price(50_000.0, 10.0, Side.BUY) < 50_000.0


def test_subscribe_yields_raw_messages():
    combined = (FIX / "agg_trade_combined.json").read_text()

    class FakeTransport:
        def __init__(self):
            self.sent = 0

        async def recv(self, timeout=None):
            if self.sent >= 2:
                raise TransportExhausted
            self.sent += 1
            return combined

        async def close(self):
            pass

    async def factory(url: str):
        assert "stream?streams=btcusdt@aggTrade" in url
        return FakeTransport()

    adapter = bn.BinanceUsdmAdapter(ws_connect=factory, clock=lambda: TS_RECV)

    async def collect():
        out = []
        async for raw_msg in adapter.subscribe(["BTCUSDT"], ["agg_trade"]):
            out.append(raw_msg)
        return out

    msgs = asyncio.run(collect())
    assert len(msgs) == 2
    assert msgs[0].stream == "btcusdt@aggTrade"
    assert msgs[0].ts_recv == TS_RECV
    recs = adapter.normalize(msgs[0])
    assert len(recs) == 1 and recs[0].side is Side.BUY
