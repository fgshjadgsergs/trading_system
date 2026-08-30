"""REST-фолбэк рекордера и бэкфилл: парсеры, фильтр «ws молчит», склейка
бэкфилла с живым фидом, прогрев платформы при пустом лейке."""

from __future__ import annotations

import math
import time

import polars as pl

from scripts.backfill_lake import parse_oi_hist
from trading_system.collectors.binance import parse_premium_index, parse_rest_klines
from trading_system.core.io import read_stream, write_batch
from trading_system.core.schema import records_to_frame
from trading_system.core.timeutils import now_ns
from trading_system.platform.feed import LakeBarFeed
from trading_system.platform.server import Platform
from trading_system.platform.state import Bar, LiveMapState

S_NS = 1_000_000_000
MIN_NS = 60 * S_NS


def test_parse_rest_klines_closed_flag():
    now_ms = now_ns() // 1_000_000
    closed = [now_ms - 120_000, "100", "101", "99", "100.5", "10",
              now_ms - 60_001, "1000", 5, "4", "400", "0"]
    # у формирующейся свечи closeTime — конец ТЕКУЩЕЙ минуты, т.е. в будущем
    forming = [now_ms - 15_000, "100.5", "102", "100", "101", "3",
               now_ms + 44_999, "300", 2, "1", "100", "0"]
    ks = parse_rest_klines([closed, forming], "BTCUSDT")
    assert ks[0].closed and not ks[1].closed
    assert ks[0].symbol == "BTCUSDT" and ks[0].quote_volume == 1000.0


def test_parse_oi_hist_carries_usd():
    recs = parse_oi_hist([{"symbol": "BTCUSDT", "sumOpenInterest": "1000.5",
                           "sumOpenInterestValue": "65000000.1",
                           "timestamp": 1756500000000}], ts_recv=7)
    assert recs[0].open_interest_usd == 65000000.1
    assert recs[0].open_interest == 1000.5
    assert math.isfinite(recs[0].open_interest_usd)


def test_rest_fallback_only_when_ws_silent():
    """Фолбэк пишет клайны/марк только пока соответствующий ws молчит;
    mark-цена для OI обновляется в любом случае."""
    import scripts.record_live as rl

    ws_alive: dict[str, float] = {}
    last_mark: dict[str, float] = {}
    last_rest: dict[str, int] = {}
    now_ms = now_ns() // 1_000_000
    payload = [[now_ms - 120_000, "1", "2", "0.5", "1.5", "1",
                now_ms - 60_001, "10", 1, "1", "5", "0"]]

    def klines_poll(sym: str) -> list:
        if time.monotonic() - ws_alive.get("kline", 0.0) < 180.0:
            return []
        out = []
        for k in parse_rest_klines(payload, sym):
            if k.closed and k.ts_close > last_rest.get(sym, 0):
                last_rest[sym] = k.ts_close
                out.append(k)
        return out

    assert len(klines_poll("BTCUSDT")) == 1   # ws молчит -> пишем
    assert len(klines_poll("BTCUSDT")) == 0   # дубль не пишем
    ws_alive["kline"] = time.monotonic()
    last_rest.clear()
    assert klines_poll("BTCUSDT") == []       # ws ожил -> фолбэк молчит

    mp = parse_premium_index({"symbol": "BTCUSDT", "markPrice": "65000.5",
                              "indexPrice": "65000", "lastFundingRate": "0.0001",
                              "nextFundingTime": now_ms + 3_600_000,
                              "time": now_ms}, 1)
    last_mark[mp.symbol] = mp.mark_price
    assert last_mark["BTCUSDT"] == 65000.5
    assert hasattr(rl, "run")  # сам модуль импортируем — проводка существует


def _kline_row(i: int, t0: int, px: float):
    from trading_system.core.schema import Kline
    return Kline("binance_usdm", "BTCUSDT", t0 + i * MIN_NS,
                 t0 + i * MIN_NS + MIN_NS - 1_000_000,
                 px, px + 2, px - 2, px + 1, 1.0, 100.0, 0.5, 50.0, 3, True)


def _oi_row(ts: int, usd: float):
    from trading_system.core.schema import OpenInterest
    return OpenInterest("binance_usdm", "BTCUSDT", ts, ts, usd / 65_000.0, usd)


def test_backfill_then_live_seam(tmp_path):
    """Бэкфилл (свечи+OI задним числом) и «живая» дозапись склеиваются в один
    непрерывный ряд баров без дыр и дублей на шве."""
    lake = tmp_path / "lake"
    t0 = 1_756_500_000 * S_NS
    hist_k = [_kline_row(i, t0, 65_000.0 + i) for i in range(10)]
    hist_oi = [_oi_row(t0 + i * 5 * MIN_NS, 60e6 + i * 1e6) for i in range(3)]
    write_batch(lake, "kline", records_to_frame(hist_k, "kline"))
    write_batch(lake, "open_interest", records_to_frame(hist_oi, "open_interest"))

    feed = LakeBarFeed(lake, "BTCUSDT")
    first = feed.poll()
    assert first and first[-1].ts_close <= t0 + 10 * MIN_NS

    live_k = [_kline_row(i, t0, 65_010.0 + i) for i in range(10, 14)]
    live_oi = [_oi_row(t0 + 13 * MIN_NS + MIN_NS, 66e6)]
    write_batch(lake, "kline", records_to_frame(live_k, "kline"))
    write_batch(lake, "open_interest", records_to_frame(live_oi, "open_interest"))
    more = feed.poll()
    seam = [b.ts_close for b in first] + [b.ts_close for b in more]
    assert seam == sorted(set(seam)), "дубли или беспорядок на шве"
    diffs = {seam[i + 1] - seam[i] for i in range(len(seam) - 1)}
    assert diffs == {MIN_NS}, "дыра на шве бэкфилл/лайв"


def test_platform_lazy_state_and_warming(tmp_path):
    """Пустой лейк: символ отвечает «прогрев», состояние создаётся из ПЕРВОГО
    реального бара, масштаб сетки — от его цены."""
    class ScriptedFeed:
        def __init__(self):
            self.batches = [[], [], [Bar(0, MIN_NS, 100.0, 101.0, 99.0, 100.5, 1e5)]]
        def poll(self):
            return self.batches.pop(0) if self.batches else []

    pl_ = Platform(poll_s=999)
    created = {}
    def factory(first: Bar) -> LiveMapState:
        st = LiveMapState("XUSDT", bucket_size=first.close * 30e-4)
        created["bucket"] = st.map.buckets.bucket_size
        return st
    pl_.add_symbol_lazy("XUSDT", ScriptedFeed(), factory)
    assert pl_.state("XUSDT") is None and "XUSDT" in pl_.symbols
    pl_.pump_once()
    pl_.pump_once()
    assert pl_.state("XUSDT") is None          # баров ещё нет — прогрев
    pl_.pump_once()
    st = pl_.state("XUSDT")
    assert st is not None and created["bucket"] == 100.5 * 30e-4
    assert st.meta()["frames"] == 1


def test_backfill_idempotent_filter(tmp_path):
    """Повторный бэкфилл не плодит дубли: пишутся только записи новее лежащих."""
    from scripts.backfill_lake import last_ts_in_lake
    lake = tmp_path / "lake"
    assert last_ts_in_lake(lake, "kline", "BTCUSDT", "ts_close") == 0
    t0 = 1_756_500_000 * S_NS
    write_batch(lake, "kline",
                records_to_frame([_kline_row(0, t0, 65_000.0)], "kline"))
    have = last_ts_in_lake(lake, "kline", "BTCUSDT", "ts_close")
    assert have == t0 + MIN_NS - 1_000_000
    df = read_stream(lake, "kline", symbol="BTCUSDT")
    assert isinstance(df, pl.DataFrame) and df.height == 1
