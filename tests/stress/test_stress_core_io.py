"""Stress: core.io parquet lake — write atomicity under crashes, tmp hygiene, volume.

Scale heavy sizes with env STRESS_SCALE (default 1).
"""

from __future__ import annotations

import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import polars as pl
import pytest

from trading_system.core.io import read_stream, write_batch
from trading_system.core.schema import POLARS_SCHEMAS
from trading_system.core.timeutils import NS_PER_S

pytestmark = pytest.mark.stress

SCALE = float(os.environ.get("STRESS_SCALE", "1"))
REPO_ROOT = Path(__file__).resolve().parents[2]
EXCHANGE = "binance_usdm"
SYMBOL = "BTCUSDT"
HOUR_NS = 3_600 * NS_PER_S
T0 = (1_755_600_000 * NS_PER_S // HOUR_NS) * HOUR_NS  # exact hour boundary


def trade_frame(n: int, ts0: int = T0, step_ns: int = 1_000_000, id0: int = 0) -> pl.DataFrame:
    ts = [ts0 + i * step_ns for i in range(n)]
    return pl.DataFrame(
        {
            "exchange": [EXCHANGE] * n,
            "symbol": [SYMBOL] * n,
            "ts_event": ts,
            "ts_recv": [t + 1_000_000 for t in ts],
            "price": [50_000.0 + (i % 97) for i in range(n)],
            "qty": [0.01] * n,
            "qty_usd": [500.0] * n,
            "side": [1 if i % 2 == 0 else -1 for i in range(n)],
            "trade_id": [id0 + i for i in range(n)],
        },
        schema=POLARS_SCHEMAS["trade"],
    )


# --------------------------------------------------------------------------- #
# atomic publish: failure injected between tmp write and os.replace
# --------------------------------------------------------------------------- #
def test_replace_failure_leaves_no_partial_final_file(tmp_data, monkeypatch):
    write_batch(tmp_data, "trade", trade_frame(1_000))  # baseline batch
    baseline = read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)
    assert len(baseline) == 1_000

    def boom(src, dst):
        raise RuntimeError("disk died between tmp write and publish")

    monkeypatch.setattr("trading_system.core.io.os.replace", boom)
    with pytest.raises(RuntimeError):
        write_batch(tmp_data, "trade", trade_frame(1_000, id0=1_000))

    finals = sorted(tmp_data.glob("trade/**/part-*.parquet"))
    tmps = sorted(tmp_data.glob("trade/**/.part-*.parquet.tmp"))
    assert [f.name for f in finals] == ["part-00000.parquet"]  # no new final file
    assert [f.name for f in tmps] == [".part-00001.parquet.tmp"]  # orphan tmp left behind
    # the lake stays fully readable with the orphan tmp in place
    after = read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)
    assert after.equals(baseline)

    # recovery: next write claims the same part number and succeeds
    monkeypatch.undo()
    write_batch(tmp_data, "trade", trade_frame(1_000, id0=1_000))
    frame = read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)
    assert len(frame) == 2_000
    assert sorted(frame["trade_id"].to_list()) == list(range(2_000))


def test_read_stream_ignores_tmp_leftovers_and_junk(tmp_data):
    write_batch(tmp_data, "trade", trade_frame(500))
    d = next(tmp_data.glob("trade/exchange=*/symbol=*/date=*/hour=*"))
    # truncated parquet bytes under a tmp name: crash leftover
    (d / ".part-00042.parquet.tmp").write_bytes(b"PAR1\x00\x01\x02truncated")
    (d / "notes.txt").write_text("junk")
    frame = read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)
    assert len(frame) == 500
    # part numbering counts only published finals, tmp junk does not shift it
    paths = write_batch(tmp_data, "trade", trade_frame(500, id0=500))
    assert [p.name for p in paths] == ["part-00001.parquet"]
    assert len(read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)) == 1_000


# --------------------------------------------------------------------------- #
# SIGKILL storm: a writer process killed at random moments never corrupts the lake
# --------------------------------------------------------------------------- #
_CHILD = """
import sys
from pathlib import Path
import polars as pl
from trading_system.core.io import write_batch
from trading_system.core.schema import POLARS_SCHEMAS

root, ready, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
T0 = int(sys.argv[4])
ts = [T0 + i * 1_000 for i in range(n)]
base = pl.DataFrame(
    {
        "exchange": ["binance_usdm"] * n,
        "symbol": ["BTCUSDT"] * n,
        "ts_event": ts,
        "ts_recv": [t + 1 for t in ts],
        "price": [50_000.0 + i * 0.007 for i in range(n)],
        "qty": [0.01 + (i % 31) * 1e-4 for i in range(n)],
        "qty_usd": [500.0 + i * 0.35 for i in range(n)],
        "side": [1 if i % 2 == 0 else -1 for i in range(n)],
        "trade_id": list(range(n)),
    },
    schema=POLARS_SCHEMAS["trade"],
)
ready.write_text("ok")
i = 0
while True:
    write_batch(root, "trade", base.with_columns(pl.col("trade_id") + i * n))
    i += 1
"""


def test_sigkill_mid_write_never_leaves_unreadable_lake(tmp_data):
    rows_per_batch = 20_000
    iterations = max(2, int(round(6 * SCALE)))
    rng = random.Random(20_260_820)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    root = tmp_data / "lake"
    for it in range(iterations):
        ready = tmp_data / f"ready-{it}"
        proc = subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(root), str(ready), str(rows_per_batch), str(T0)],
            env=env,
            cwd=str(REPO_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 60.0
        while not ready.exists():
            if proc.poll() is not None:
                raise AssertionError(f"child died early: {proc.stderr.read().decode()}")
            if time.monotonic() > deadline:
                proc.kill()
                raise AssertionError("child never became ready")
            time.sleep(0.005)
        time.sleep(rng.uniform(0.05, 0.30))  # kill at a random point of the write loop
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=30)

        # after every kill: every published file complete, whole lake readable
        frame = read_stream(root, "trade", exchange=EXCHANGE, symbol=SYMBOL)
        parts = sorted(root.glob("trade/**/part-*.parquet"))
        assert len(frame) == rows_per_batch * len(parts)
        for p in parts:
            assert pl.read_parquet(p).height == rows_per_batch  # no partial file published

    parts = sorted(root.glob("trade/**/part-*.parquet"))
    assert parts, "kill storm never let a single batch through — delays too aggressive"


# --------------------------------------------------------------------------- #
# volume: many hourly partitions in one batch, filters still exact
# --------------------------------------------------------------------------- #
def test_write_batch_across_many_hour_partitions(tmp_data):
    n = int(96_000 * SCALE)
    hours = 48
    step = hours * HOUR_NS // n  # spread evenly over 48 hourly partitions
    frame = trade_frame(n, step_ns=step)
    t = time.perf_counter()
    paths = write_batch(tmp_data, "trade", frame)
    wrote_s = time.perf_counter() - t
    assert len(paths) == hours
    assert len({p.parent for p in paths}) == hours

    t = time.perf_counter()
    back = read_stream(tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL)
    read_s = time.perf_counter() - t
    assert len(back) == n
    assert back["ts_event"].is_sorted()
    assert sorted(back["trade_id"].to_list()) == list(range(n))

    window = read_stream(
        tmp_data, "trade", exchange=EXCHANGE, symbol=SYMBOL,
        ts_from=T0 + HOUR_NS, ts_to=T0 + 2 * HOUR_NS,
    )
    expected = sum(1 for i in range(n) if HOUR_NS <= i * step < 2 * HOUR_NS)
    assert len(window) == expected
    assert window["ts_event"].min() >= T0 + HOUR_NS
    assert window["ts_event"].max() < T0 + 2 * HOUR_NS
    # generous sanity bounds only: catches order-of-magnitude regressions
    assert wrote_s < 30.0 and read_s < 30.0
