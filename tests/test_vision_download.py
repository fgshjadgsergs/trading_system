"""Checksum-verified downloader with injectable fetch (1.2)."""

from __future__ import annotations

import datetime as dt
import hashlib
import io
import zipfile

import pytest

from trading_system.collectors.vision import (
    download,
    parse_checksum,
    plan_downloads,
    sha256_hex,
)


def make_zip(csv_text: str, member: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(member, csv_text)
    return buf.getvalue()


def checksum_text(blob: bytes, name: str) -> bytes:
    return f"{hashlib.sha256(blob).hexdigest()}  {name}\n".encode()


def fake_fetch(mapping: dict[str, bytes]):
    def fetch(url: str) -> bytes:
        if url not in mapping:
            raise KeyError(f"404 {url}")
        return mapping[url]

    return fetch


@pytest.fixture()
def plan():
    return plan_downloads(
        ["BTCUSDT"], ["aggTrades"], dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    )


@pytest.fixture()
def served(plan):
    mapping: dict[str, bytes] = {}
    for item in plan:
        name = item.rel_path.rsplit("/", 1)[-1]
        blob = make_zip("1,50000.0,0.001,1,1,1704067200123,true\n", name[:-4] + ".csv")
        mapping[item.url] = blob
        mapping[item.checksum_url] = checksum_text(blob, name)
    return mapping


def test_download_writes_verified_archives(plan, served, tmp_path):
    results = download(plan, tmp_path, fake_fetch(served))
    assert [r.status for r in results] == ["downloaded", "downloaded"]
    for r in results:
        assert r.ok and r.path is not None and r.path.exists()
        assert r.path == tmp_path / r.item.rel_path
        assert r.path.read_bytes() == served[r.item.url]


def test_skip_existing(plan, served, tmp_path):
    download(plan, tmp_path, fake_fetch(served))
    again = download(plan, tmp_path, fake_fetch(served))
    assert [r.status for r in again] == ["skipped_existing", "skipped_existing"]
    assert all(r.ok and r.path is not None for r in again)
    fresh = download(plan, tmp_path, fake_fetch(served), skip_existing=False)
    assert [r.status for r in fresh] == ["downloaded", "downloaded"]


def test_corrupt_checksum_yields_error_record_and_no_file(plan, served, tmp_path):
    bad = dict(served)
    good_sum = parse_checksum(bad[plan[0].checksum_url])
    flipped = ("0" if good_sum[0] != "0" else "1") + good_sum[1:]
    bad[plan[0].checksum_url] = f"{flipped}  x.zip\n".encode()
    results = download(plan, tmp_path, fake_fetch(bad))
    assert results[0].status == "checksum_mismatch"
    assert results[0].path is None
    assert "expected=" in results[0].error
    assert not (tmp_path / plan[0].rel_path).exists()
    assert results[1].status == "downloaded"  # one bad archive does not stop the rest


def test_unparsable_checksum_is_a_mismatch(plan, served, tmp_path):
    bad = dict(served)
    bad[plan[0].checksum_url] = b"this is not a checksum file"
    results = download(plan[:1], tmp_path, fake_fetch(bad))
    assert results[0].status == "checksum_mismatch"


def test_fetch_error_recorded(plan, tmp_path):
    results = download(plan, tmp_path, fake_fetch({}))
    assert [r.status for r in results] == ["fetch_error", "fetch_error"]
    assert all(not r.ok and r.path is None and r.error for r in results)


def test_missing_checksum_file_is_fetch_error(plan, served, tmp_path):
    partial = {plan[0].url: served[plan[0].url]}
    results = download(plan[:1], tmp_path, fake_fetch(partial))
    assert results[0].status == "fetch_error"
    assert "checksum" in results[0].error


def test_no_verify_skips_checksum_fetch(plan, served, tmp_path):
    partial = {i.url: served[i.url] for i in plan}  # no .CHECKSUM entries at all
    results = download(plan, tmp_path, fake_fetch(partial), verify=False)
    assert [r.status for r in results] == ["downloaded", "downloaded"]


def test_parse_checksum_variants():
    digest = sha256_hex(b"payload")
    assert parse_checksum(f"{digest}  file.zip".encode()) == digest
    assert parse_checksum(digest.upper().encode()) == digest
    assert parse_checksum(f"\n{digest} *file.zip\n".encode()) == digest
    assert parse_checksum(b"garbage") is None
    assert parse_checksum(b"") is None
