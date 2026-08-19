"""Offline end-to-end run of scripts/download_vision.py with an injected fetch (1.2)."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

from trading_system.collectors.vision import plan_downloads, read_catalog
from trading_system.core.io import read_stream

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "vision"


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "download_vision_cli", REPO_ROOT / "scripts" / "download_vision.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _zip_bytes(fixture: str, member: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(member, (FIXTURES / fixture).read_bytes())
    return buf.getvalue()


FIXTURE_OF_KIND = {"aggTrades": "aggTrades.csv", "klines": "klines.csv", "fundingRate": "fundingRate.csv"}


@pytest.fixture()
def served():
    items = plan_downloads(
        ["BTCUSDT"],
        ["aggTrades", "klines", "fundingRate"],
        dt.date(2024, 1, 1),
        dt.date(2024, 1, 1),
        intervals=["1m"],
    )
    mapping: dict[str, bytes] = {}
    for item in items:
        name = item.rel_path.rsplit("/", 1)[-1]
        blob = _zip_bytes(FIXTURE_OF_KIND[item.kind], name[:-4] + ".csv")
        mapping[item.url] = blob
        mapping[item.checksum_url] = (
            hashlib.sha256(blob).hexdigest() + "  " + name + "\n"
        ).encode()
    return mapping


def _fetch(mapping):
    def fetch(url: str) -> bytes:
        return mapping[url]

    return fetch


CLI_ARGS = [
    "--symbols",
    "BTCUSDT",
    "--kinds",
    "aggTrades",
    "klines",
    "fundingRate",
    "--start",
    "2024-01-01",
    "--end",
    "2024-01-01",
]


def test_cli_downloads_normalizes_and_catalogs(tmp_path, served):
    cli = _load_cli()
    lake = tmp_path / "lake"
    rc = cli.main([*CLI_ARGS, "--lake", str(lake)], fetch=_fetch(served))
    assert rc == 0
    assert read_stream(lake, "trade", symbol="BTCUSDT").height == 3
    assert read_stream(lake, "kline", symbol="BTCUSDT").height == 2
    funding = read_stream(lake, "mark_price", symbol="BTCUSDT")
    assert funding.height == 2
    assert funding.get_column("funding_rate").to_list() == [0.0001, -0.00005]
    archives = list((lake / "_archive").rglob("*.zip"))
    assert len(archives) == 3
    catalog = read_catalog(lake)
    assert set(catalog.get_column("stream").to_list()) == {"trade", "kline", "mark_price"}


def test_cli_second_run_skips_and_does_not_duplicate(tmp_path, served):
    cli = _load_cli()
    lake = tmp_path / "lake"
    fetch = _fetch(served)
    assert cli.main([*CLI_ARGS, "--lake", str(lake)], fetch=fetch) == 0
    assert cli.main([*CLI_ARGS, "--lake", str(lake)], fetch=fetch) == 0
    assert read_stream(lake, "trade", symbol="BTCUSDT").height == 3  # no duplicates


def test_cli_corrupt_checksum_exits_nonzero_but_ingests_the_rest(tmp_path, served):
    cli = _load_cli()
    lake = tmp_path / "lake"
    bad = dict(served)
    checksum_urls = [u for u in bad if u.endswith(".CHECKSUM") and "aggTrades" in u]
    bad[checksum_urls[0]] = b"0" * 64 + b"  broken.zip\n"
    rc = cli.main([*CLI_ARGS, "--lake", str(lake)], fetch=_fetch(bad))
    assert rc == 2
    assert read_stream(lake, "trade", symbol="BTCUSDT").height == 0
    assert read_stream(lake, "kline", symbol="BTCUSDT").height == 2


def test_cli_no_normalize_only_mirrors_archives(tmp_path, served):
    cli = _load_cli()
    lake = tmp_path / "lake"
    rc = cli.main(
        [*CLI_ARGS, "--lake", str(lake), "--no-normalize", "--no-catalog"],
        fetch=_fetch(served),
    )
    assert rc == 0
    assert list((lake / "_archive").rglob("*.zip"))
    assert not (lake / "trade").exists()
    assert not (lake / "_catalog").exists()
