"""M6: ground-truth scenarios (honest wall / flickering spoofer / iceberg),
cancel-to-fill, flicker/iceberg flags and stability score ordering."""

from __future__ import annotations

import math

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from trading_system.core.schema import Side, Trade
from trading_system.core.timeutils import NS_PER_MS, NS_PER_S
from trading_system.spoof.lifecycle import BookState, LevelJournal
from trading_system.spoof.metrics import annotate_episodes, cancel_to_fill
from trading_system.spoof.score import journal_scores, score_episodes, stability_score

T0 = 1_755_600_000 * NS_PER_S
WALL = 95.5

BIDS = tuple((99.0 - i, 1.0) for i in range(4))
ASKS = tuple((101.0 + i, 1.0) for i in range(4))


def _states(wall_qty: dict[int, float], n: int, step_ns: int) -> list[BookState]:
    out = []
    for i in range(n):
        q = wall_qty.get(i)
        bids = BIDS + (((WALL, q),) if q else ())
        out.append(BookState(ts=T0 + i * step_ns, bids=bids, asks=ASKS))
    return out


def _sell(ts: int, qty: float, tid: int) -> Trade:
    return Trade(
        exchange="binance_usdm",
        symbol="BTCUSDT",
        ts_event=ts,
        ts_recv=ts + NS_PER_MS,
        price=WALL,
        qty=qty,
        qty_usd=WALL * qty,
        side=Side.SELL,
        trade_id=tid,
    )


def _run(wall_qty, trades, n, step_ns):
    j = LevelJournal(large_k=3.0, iceberg_refill_ms=300).run(
        _states(wall_qty, n, step_ns), trades
    )
    ann = annotate_episodes(j, flicker_k=3, flicker_window_s=60)
    return j, score_episodes(ann)


def _honest():
    step = 500 * NS_PER_MS
    wall = {i: 6.0 for i in range(2, 101)}
    trades = []
    for b in range(4):  # eaten in 4 bites of 1.5
        i = 101 + b
        left = 6.0 - 1.5 * (b + 1)
        if left > 1e-9:
            wall[i] = left
        trades.append(_sell(T0 + i * step - 100 * NS_PER_MS, 1.5, b + 1))
    return _run(wall, trades, 110, step)


def _spoofer():
    step = 500 * NS_PER_MS
    wall = {}
    cur = 2
    for c in range(4):  # appears and is pulled 4 times, never filled
        for k in range(3 + c % 2):
            wall[cur + k] = 6.0
        cur += 3 + c % 2 + 5
    return _run(wall, [], 60, step)


def _iceberg():
    step = 200 * NS_PER_MS
    wall = {i: 6.0 for i in range(2, 10)}
    trades = []
    i = 10
    for c in range(4):  # 75% eaten, refilled on the next state
        wall[i] = 1.5
        trades.append(_sell(T0 + i * step - 100 * NS_PER_MS, 4.5, c + 1))
        wall[i + 1] = 6.0
        wall[i + 2] = 6.0
        i += 3
    trades.append(_sell(T0 + i * step - 100 * NS_PER_MS, 6.0, 99))  # final kill
    return _run(wall, trades, i + 5, step)


def _wall_rows(scored: pl.DataFrame) -> pl.DataFrame:
    return scored.filter(pl.col("price") == WALL)


def test_honest_wall_labels_and_score():
    _, scored = _honest()
    (row,) = _wall_rows(scored).to_dicts()
    assert row["outcome"] == "executed"
    assert row["fill_frac"] > 0.95
    assert not row["flicker"]
    assert not row["iceberg"]
    assert row["score"] >= 0.6


def test_spoofer_flicker_and_near_zero_score():
    _, scored = _spoofer()
    rows = _wall_rows(scored)
    assert rows.height == 4
    assert rows["flicker"].all()
    assert (rows["flicker_count"] >= 3).all()
    assert (rows["fill_frac"] == 0.0).all()
    assert (rows["outcome"] == "canceled").all()
    assert float(rows["score"].max()) < 0.1


def test_iceberg_refills_flag_and_high_score():
    _, scored = _iceberg()
    (row,) = _wall_rows(scored).to_dicts()
    assert row["iceberg"]
    assert row["chain_refills"] >= 3
    assert not row["flicker"]
    assert row["outcome"] == "executed"
    assert row["score"] >= 0.6


def test_score_ordering_across_scenarios():
    _, honest = _honest()
    _, spoof = _spoofer()
    _, ice = _iceberg()
    h = float(_wall_rows(honest)["score"][0])
    s = float(_wall_rows(spoof)["score"].max())
    i = float(_wall_rows(ice)["score"][0])
    assert h > 0.6  # honest above the "iceberg-ish" threshold
    assert i > 0.6  # iceberg high: real hidden liquidity
    assert s < 0.1  # spoofer near zero
    assert h > s + 0.4


def test_iceberg_chain_links_full_consumption_episodes():
    step = 200 * NS_PER_MS
    wall = {}
    trades = []
    for c in range(4):  # fully eaten, reborn on the very next state
        i = 2 + 2 * c
        wall[i] = 6.0
        trades.append(_sell(T0 + (i + 1) * step - 100 * NS_PER_MS, 6.0, c + 1))
    j = LevelJournal(large_k=3.0, iceberg_refill_ms=300).run(
        _states(wall, 14, step), trades
    )
    ann = annotate_episodes(j, flicker_k=3, flicker_window_s=60)
    rows = ann.filter(pl.col("price") == WALL)
    assert rows.height == 4
    assert (rows["refill_chain_id"] >= 0).all()
    assert rows["refill_chain_id"].n_unique() == 1
    assert int(rows["chain_refills"].max()) == 3
    assert rows["iceberg"].all()
    # fully-filled deaths must not read as flicker
    assert not rows["flicker"].any()


def test_cancel_to_fill_separates_populations():
    hj, _ = _honest()
    sj, _ = _spoofer()
    window = 3_600 * NS_PER_S
    h = cancel_to_fill(hj.events_frame(), window).filter(pl.col("side") == "bid")
    s = cancel_to_fill(sj.events_frame(), window).filter(pl.col("side") == "bid")
    h_ratio = float(h["cancel_to_fill"].sum())
    assert h_ratio < 0.1
    s_row = s.to_dicts()[0]
    assert s_row["canceled"] > 20.0
    assert s_row["filled"] == 0.0
    assert math.isinf(s_row["cancel_to_fill"]) or s_row["cancel_to_fill"] is None


@settings(max_examples=100, deadline=None)
@given(
    p=st.floats(0.0, 1.0),
    f=st.floats(0.0, 1.0),
    c=st.integers(0, 6),
    r=st.integers(0, 8),
    d=st.floats(0.0, 1.0),
)
def test_stability_score_bounded_and_monotone(p, f, c, r, d):
    s = stability_score(p, f, c, r)
    assert 0.0 <= s <= 1.0
    assert stability_score(min(p + d, 1.0), f, c, r) >= s - 1e-12
    assert stability_score(p, min(f + d, 1.0), c, r) >= s - 1e-12
    assert stability_score(p, f, c + 1, r) <= s + 1e-12
    assert stability_score(p, f, c, r + 1) >= s - 1e-12


def test_journal_scores_grid_frame():
    j, _ = _honest()
    scored, grid = journal_scores(j, flicker_k=3, flicker_window_s=60)
    assert grid.columns == ["ts", "side", "price", "qty", "score"]
    assert grid["score"].null_count() == 0
    assert float(grid["score"].min()) >= 0.0
    assert float(grid["score"].max()) <= 1.0
    wall_grid = grid.filter(pl.col("price") == WALL)
    wall_score = float(_wall_rows(scored)["score"][0])
    assert set(wall_grid["score"].unique().to_list()) == {wall_score}
