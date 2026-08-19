"""URL/path builders and download plan for data.binance.vision (1.2)."""

from __future__ import annotations

import datetime as dt

import pytest

from trading_system.collectors.vision import (
    KINDS,
    archive_name,
    checksum_url,
    plan_downloads,
    vision_path,
    vision_url,
)

D = dt.date(2024, 1, 15)
BASE = "https://data.binance.vision"


def test_daily_agg_trades_url():
    assert vision_url("aggTrades", "BTCUSDT", D, "daily") == (
        f"{BASE}/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-15.zip"
    )


def test_monthly_url_uses_month_tag():
    assert vision_url("aggTrades", "SOLUSDT", D, "monthly") == (
        f"{BASE}/data/futures/um/monthly/aggTrades/SOLUSDT/SOLUSDT-aggTrades-2024-01.zip"
    )


def test_klines_have_interval_subdir_and_interval_filename():
    assert vision_path("klines", "BTCUSDT", D, "daily", "1m") == (
        "data/futures/um/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01-15.zip"
    )
    assert vision_path("premiumIndexKlines", "BTCUSDT", D, "monthly", "5m") == (
        "data/futures/um/monthly/premiumIndexKlines/BTCUSDT/5m/BTCUSDT-5m-2024-01.zip"
    )


def test_funding_rate_is_monthly_only():
    assert vision_path("fundingRate", "BTCUSDT", D, "monthly") == (
        "data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip"
    )
    with pytest.raises(ValueError, match="not published daily"):
        vision_path("fundingRate", "BTCUSDT", D, "daily")


@pytest.mark.parametrize("kind", ["metrics", "bookDepth", "liquidationSnapshot"])
def test_daily_only_kinds_reject_monthly(kind):
    with pytest.raises(ValueError, match="not published monthly"):
        vision_path(kind, "BTCUSDT", D, "monthly")


def test_checksum_url_appends_suffix():
    url = checksum_url("metrics", "DOGEUSDT", D, "daily")
    assert url.endswith("DOGEUSDT-metrics-2024-01-15.zip.CHECKSUM")


def test_unknown_kind_and_missing_interval_raise():
    with pytest.raises(ValueError, match="unknown Vision kind"):
        vision_url("depth", "BTCUSDT", D, "daily")
    with pytest.raises(ValueError, match="requires an interval"):
        archive_name("klines", "BTCUSDT", D, "daily")


def test_every_kind_builds_a_url():
    for kind in KINDS:
        period = "daily" if kind != "fundingRate" else "monthly"
        interval = "1m" if kind in ("klines", "premiumIndexKlines") else None
        assert vision_url(kind, "BTCUSDT", D, period, interval).startswith(BASE + "/data/")


def test_plan_counts_and_fallback_periods():
    items = plan_downloads(
        symbols=["BTCUSDT", "SOLUSDT"],
        kinds=["aggTrades", "klines", "fundingRate", "metrics"],
        start=dt.date(2024, 1, 30),
        end=dt.date(2024, 2, 2),
        period="daily",
        intervals=["1m", "5m"],
    )
    # per symbol: 4 aggTrades + 4*2 klines + 2 monthly fundingRate + 4 metrics = 18
    assert len(items) == 36
    funding = [i for i in items if i.kind == "fundingRate"]
    assert {i.period for i in funding} == {"monthly"}
    assert sorted({i.date for i in funding}) == [dt.date(2024, 1, 1), dt.date(2024, 2, 1)]
    klines = [i for i in items if i.kind == "klines" and i.symbol == "BTCUSDT"]
    assert {i.interval for i in klines} == {"1m", "5m"}
    assert all(i.checksum_url == i.url + ".CHECKSUM" for i in items)
    assert all(i.url.endswith(i.rel_path) for i in items)


def test_plan_monthly_request_falls_back_to_daily_for_daily_only_kinds():
    items = plan_downloads(
        symbols=["BTCUSDT"],
        kinds=["aggTrades", "metrics"],
        start=dt.date(2024, 1, 5),
        end=dt.date(2024, 3, 2),
        period="monthly",
    )
    agg = [i for i in items if i.kind == "aggTrades"]
    metrics = [i for i in items if i.kind == "metrics"]
    assert len(agg) == 3  # Jan, Feb, Mar
    assert {i.period for i in metrics} == {"daily"}
    assert len(metrics) == (dt.date(2024, 3, 2) - dt.date(2024, 1, 5)).days + 1


def test_plan_rejects_inverted_range():
    with pytest.raises(ValueError, match="end date before start"):
        plan_downloads(["BTCUSDT"], ["aggTrades"], dt.date(2024, 2, 1), dt.date(2024, 1, 1))
