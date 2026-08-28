"""Адверсариальные тесты контракта LiveMapState и дельта-протокола платформы.

Контракты: идемпотентность по ts_close, строгая монотонность кадров и
корректность delta(since), кольцо кадров с признаком gap, эпоха как
идентичность параметров + первого бара, атомарность отказа на битых барах,
потокобезопасность apply_bar и детерминизм реплея порциями.
"""

from __future__ import annotations

import json
import math
import threading

import pytest

from trading_system.platform.state import Bar, LiveMapState

NS = 1_000_000_000  # нс в секунде: бары идут раз в секунду


def mk_bar(
    i: int,
    *,
    o: float = 100.0,
    h: float = 101.0,
    lo: float = 99.0,
    c: float = 100.0,
    d_oi: float = 10_000.0,
    long_share: float | None = None,
) -> Bar:
    return Bar(
        ts_open=i * NS, ts_close=(i + 1) * NS,
        open=o, high=h, low=lo, close=c,
        d_oi_usd=d_oi, long_share=long_share,
    )


def walk_bar(i: int, *, d_oi: float = 10_000.0) -> Bar:
    """Детерминированный блуждающий бар: разные пути, знак ΔOI, доли сторон."""
    p0 = 100.0 + 7.0 * math.sin(i * 0.7)
    p1 = 100.0 + 7.0 * math.sin((i + 1) * 0.7)
    step_d_oi = d_oi * (1.0 if i % 4 else -0.5)
    return Bar(
        ts_open=i * NS, ts_close=(i + 1) * NS,
        open=p0, high=max(p0, p1) + 1.0, low=min(p0, p1) - 1.0, close=p1,
        d_oi_usd=step_d_oi, long_share=0.3 + 0.4 * ((i * 37) % 11) / 10.0,
    )


def frame_ts(state: LiveMapState) -> list[int]:
    return [f["ts"] for f in state.snapshot()["frames"]]


@pytest.fixture()
def state() -> LiveMapState:
    return LiveMapState("TESTUSDT", 1.0)


# -- 1. идемпотентность --------------------------------------------------------


def test_duplicate_bar_is_noop(state: LiveMapState) -> None:
    assert state.apply_bar(mk_bar(0)) is True
    assert state.apply_bar(mk_bar(1)) is True
    heat = state.map.total_heat()
    frames = frame_ts(state)
    assert state.apply_bar(mk_bar(1)) is False  # тот же ts_close
    assert state.map.total_heat() == heat
    assert frame_ts(state) == frames


def test_past_bar_rejected_even_with_new_payload(state: LiveMapState) -> None:
    state.apply_bar(mk_bar(0))
    state.apply_bar(mk_bar(5))
    heat = state.map.total_heat()
    frames = frame_ts(state)
    # прошлое с другим содержимым (перезапуск фида, пересчитанный бар)
    old = Bar(ts_open=2 * NS, ts_close=3 * NS, open=50.0, high=200.0,
              low=10.0, close=150.0, d_oi_usd=1e9)
    assert state.apply_bar(old) is False
    assert state.map.total_heat() == heat
    assert frame_ts(state) == frames
    assert state.meta()["dropped_old_bars"] == 1


def test_stale_broken_bar_is_dropped_not_raised(state: LiveMapState) -> None:
    """Дедупликация по ts идёт раньше валидации: битый дубль тихо False."""
    state.apply_bar(mk_bar(0))
    heat = state.map.total_heat()
    broken_dup = Bar(ts_open=0, ts_close=1 * NS, open=100.0, high=90.0,
                     low=110.0, close=float("nan"), d_oi_usd=1.0)
    assert state.apply_bar(broken_dup) is False
    assert state.map.total_heat() == heat


# -- 2. монотонность кадров и delta(since) ------------------------------------


def test_frames_strictly_increasing(state: LiveMapState) -> None:
    for i in [0, 1, 3, 2, 7, 7, 4, 8]:  # с перемешанным/повторным ts
        state.apply_bar(walk_bar(i))
    ts = frame_ts(state)
    assert ts == sorted(set(ts))
    assert all(b > a for a, b in zip(ts, ts[1:], strict=False))


def test_delta_returns_exactly_frames_after_since(state: LiveMapState) -> None:
    for i in range(10):
        state.apply_bar(walk_bar(i))
    all_ts = frame_ts(state)
    since = all_ts[4]
    d = state.delta(since, state.epoch)
    assert d["gap"] is False
    got = [f["ts"] for f in d["frames"]]
    assert got == [t for t in all_ts if t > since]  # строго >, порядок возр.
    assert since not in got
    assert [b["ts"] for b in d["bars"]] == got
    # since между кадрами (не равен ни одному ts) — та же выдача
    d2 = state.delta(since + 1, state.epoch)
    assert [f["ts"] for f in d2["frames"]] == [t for t in all_ts if t > since + 1]
    assert d2["gap"] is False


def test_delta_from_the_future_is_empty_no_gap(state: LiveMapState) -> None:
    for i in range(5):
        state.apply_bar(walk_bar(i))
    d = state.delta(state.meta()["last_ts"] + NS, state.epoch)
    assert d["frames"] == [] and d["bars"] == [] and d["gap"] is False


def test_snapshot_plus_deltas_equals_final_snapshot_tail() -> None:
    state = LiveMapState("TESTUSDT", 1.0, max_frames=10)
    for i in range(8):
        state.apply_bar(walk_bar(i))
    snap = state.snapshot()
    client_frames = list(snap["frames"])
    epoch, since = snap["epoch"], snap["last_ts"]
    # дальше сервер живёт порциями, клиент догоняется дельтами
    nxt = 8
    for chunk in (3, 1, 7, 5, 4):  # кольцо (10) переполняется по ходу
        for i in range(nxt, nxt + chunk):
            state.apply_bar(walk_bar(i))
        nxt += chunk
        d = state.delta(since, epoch)
        assert d["gap"] is False, "клиент, тянущий каждую порцию, в кольце"
        client_frames.extend(d["frames"])
        since = d["frames"][-1]["ts"] if d["frames"] else since
    final = state.snapshot()["frames"]
    ts = [f["ts"] for f in client_frames]
    assert ts == sorted(set(ts))  # склейка без дырок и дублей
    tail = client_frames[-len(final):]
    assert json.dumps(tail, sort_keys=True) == json.dumps(final, sort_keys=True)


# -- 3. кольцо и gap -----------------------------------------------------------


def test_ring_gap_for_lagging_client_and_full_data_inside_ring() -> None:
    state = LiveMapState("TESTUSDT", 1.0, max_frames=5)
    for i in range(12):
        state.apply_bar(walk_bar(i))
    stored = frame_ts(state)
    assert len(stored) == 5  # кольцо: 8..12 c
    oldest = stored[0]
    # отстал дальше кольца: его ts уже выпал
    d = state.delta(oldest - NS, state.epoch)
    assert d["gap"] is True
    # ровно на границе выпадения (на 1 нс старше самого старого хранимого)
    assert state.delta(oldest - 1, state.epoch)["gap"] is True
    # внутри кольца: видел самый старый хранимый кадр — полные данные без gap
    d = state.delta(oldest, state.epoch)
    assert d["gap"] is False
    assert [f["ts"] for f in d["frames"]] == stored[1:]
    # видел предпоследний — получает ровно последний
    d = state.delta(stored[-2], state.epoch)
    assert d["gap"] is False and [f["ts"] for f in d["frames"]] == stored[-1:]


def test_since_zero_bootstraps_whole_ring_without_gap() -> None:
    state = LiveMapState("TESTUSDT", 1.0, max_frames=4)
    for i in range(9):
        state.apply_bar(walk_bar(i))
    d = state.delta(0, state.epoch)
    assert d["gap"] is False
    assert [f["ts"] for f in d["frames"]] == frame_ts(state)


def test_delta_on_empty_state(state: LiveMapState) -> None:
    d = state.delta(123, None)
    assert d["frames"] == [] and d["bars"] == []
    assert d["gap"] is False and d["epoch"] is None and d["last_ts"] is None


# -- 4. эпоха ------------------------------------------------------------------


def test_epoch_deterministic_for_same_params_and_first_bar() -> None:
    a = LiveMapState("TESTUSDT", 1.0)
    b = LiveMapState("TESTUSDT", 1.0)
    assert a.epoch is None and b.epoch is None  # до первого бара не запечатана
    a.apply_bar(mk_bar(0))
    b.apply_bar(mk_bar(0))
    assert a.epoch == b.epoch and a.epoch is not None
    a.apply_bar(walk_bar(1))  # дальнейшие бары эпоху не меняют
    assert a.epoch == b.epoch


def test_epoch_differs_by_params_symbol_and_first_bar() -> None:
    base = LiveMapState("TESTUSDT", 1.0)
    base.apply_bar(mk_bar(0))
    variants = [
        LiveMapState("TESTUSDT", 2.0),                          # bucket_size
        LiveMapState("TESTUSDT", 1.0, close_out_fraction=0.5),  # close_out
        LiveMapState("TESTUSDT", 1.0, decay_half_life_s=60.0),  # half-life
        LiveMapState("OTHERUSDT", 1.0),                         # символ
    ]
    for v in variants:
        v.apply_bar(mk_bar(0))
    other_first = LiveMapState("TESTUSDT", 1.0)
    other_first.apply_bar(mk_bar(7))  # другой первый бар (другой ts_close)
    epochs = [base.epoch, other_first.epoch, *(v.epoch for v in variants)]
    assert len(set(epochs)) == len(epochs)


def test_delta_with_foreign_epoch_forces_gap(state: LiveMapState) -> None:
    for i in range(3):
        state.apply_bar(walk_bar(i))
    d = state.delta(0, "deadbeefdeadbeef")
    assert d["gap"] is True and d["frames"] == [] and d["bars"] == []
    assert d["epoch"] == state.epoch  # клиенту сообщают актуальную эпоху
    # None (клиент без эпохи) проверку пропускает
    assert state.delta(0, None)["gap"] is False


# -- 5. битые бары -------------------------------------------------------------


def broken_bars(i: int) -> list[Bar]:
    nan, inf = float("nan"), float("inf")
    return [
        mk_bar(i, o=nan), mk_bar(i, h=nan), mk_bar(i, lo=nan), mk_bar(i, c=nan),
        mk_bar(i, h=inf), mk_bar(i, c=-inf),
        mk_bar(i, lo=101.0, h=99.0),                      # low > high
        mk_bar(i, lo=50.0, h=150.0, d_oi=nan),            # ΔOI ломается в allocate
        mk_bar(i, lo=50.0, h=150.0, d_oi=inf),
        mk_bar(i, lo=50.0, h=150.0, long_share=1.5),      # доля вне [0, 1]
        mk_bar(i, lo=50.0, h=150.0, long_share=nan),
    ]


def test_broken_bar_raises_and_leaves_state_untouched(state: LiveMapState) -> None:
    state.apply_bar(mk_bar(0))
    state.apply_bar(mk_bar(1))
    meta = state.meta()
    frames = json.dumps(state.snapshot(), sort_keys=True)
    for bad in broken_bars(2):  # пути 50..150 накрывают всё тепло карты
        with pytest.raises(ValueError, match="broken bar"):
            state.apply_bar(bad)
        assert state.meta() == meta, bad  # ни heat, ни кадров, ни last_ts
        assert json.dumps(state.snapshot(), sort_keys=True) == frames, bad
    # ts не продвинулся: честный бар с тем же ts_close всё ещё применяется
    assert state.apply_bar(mk_bar(2)) is True


def test_broken_first_bar_does_not_seal_epoch() -> None:
    for bad in broken_bars(0):
        state = LiveMapState("TESTUSDT", 1.0)
        with pytest.raises(ValueError, match="broken bar"):
            state.apply_bar(bad)
        assert state.epoch is None, bad
        assert state.meta()["frames"] == 0 and state.map.total_heat() == 0.0


# -- 6. потокобезопасность -----------------------------------------------------


def test_parallel_apply_bar_keeps_mass_and_frame_invariants(state: LiveMapState) -> None:
    n_threads, n_bars = 4, 240
    bars = [walk_bar(i) for i in range(n_bars)]
    # перемешанные шарды: каждый поток подаёт свою псевдослучайную подпоследовательность
    shards = [sorted(range(t, n_bars, n_threads), key=lambda i: (i * 73) % n_bars)
              for t in range(n_threads)]
    applied = [0] * n_threads
    reader_errors: list[str] = []
    start = threading.Barrier(n_threads + 1)
    stop = threading.Event()

    def writer(t: int) -> None:
        start.wait()
        applied[t] = sum(state.apply_bar(bars[i]) for i in shards[t])

    def reader() -> None:
        start.wait()
        while not stop.is_set():
            for resp in (state.snapshot(), state.delta(100 * NS, state.epoch)):
                ts = [f["ts"] for f in resp["frames"]]
                if ts != sorted(set(ts)):
                    reader_errors.append(f"non-monotone frames: {ts}")
                    return

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(n_threads)]
    rd = threading.Thread(target=reader)
    for th in [*threads, rd]:
        th.start()
    for th in threads:
        th.join(timeout=20)
    stop.set()
    rd.join(timeout=20)
    assert not reader_errors
    ts = frame_ts(state)
    assert ts == sorted(set(ts))  # ни дублей, ни перестановок по ts
    assert len(ts) == sum(applied)  # один принятый бар = ровно один кадр
    assert state.meta()["dropped_old_bars"] == n_bars - sum(applied)
    assert state.map.mass_balance_error() < 1e-6


# -- 7. детерминизм реплея -----------------------------------------------------


def test_replay_in_chunks_is_byte_identical() -> None:
    bars = [walk_bar(i) for i in range(60)]
    one = LiveMapState("TESTUSDT", 1.0, close_out_fraction=0.3)
    for b in bars:
        one.apply_bar(b)  # по одному
    batch = LiveMapState("TESTUSDT", 1.0, close_out_fraction=0.3)
    i = 0
    for chunk in (7, 1, 13, 20, 5, 14):
        # перекрытие окон опроса: хвост прошлой порции пересылается повторно
        for b in bars[max(i - 2, 0): i + chunk]:
            batch.apply_bar(b)
        i += chunk
    assert one.epoch == batch.epoch
    assert json.dumps(one.snapshot()["frames"], sort_keys=True) == json.dumps(
        batch.snapshot()["frames"], sort_keys=True
    )
    assert json.dumps(one.snapshot()["bars"], sort_keys=True) == json.dumps(
        batch.snapshot()["bars"], sort_keys=True
    )
    assert one.map.total_heat() == batch.map.total_heat()
