"""健康事件与状态推导测试。"""
import numpy as np
import pytest

from app.events import (
    ChangeType,
    HealthEvent,
    derive_state,
    failed_elements,
    health_factors,
)

N = 8


def ev(eid, t, element, ctype, value=None, seq=0):
    return HealthEvent(eid, t, element - 1, ChangeType(ctype), value, seq)


def test_failed_and_recovered():
    events = [
        ev("e1", 100.0, 4, "failed", seq=1),
        ev("e2", 200.0, 4, "recovered", seq=2),
    ]
    st = derive_state(events, N, t=150.0)
    assert failed_elements(st) == [3]
    st = derive_state(events, N, t=250.0)
    assert failed_elements(st) == []
    assert abs(health_factors(st)[3] - 1.0) < 1e-12


def test_amplitude_phase_error():
    events = [
        ev("e1", 100.0, 2, "amplitude_error", -3.0, seq=1),
        ev("e2", 100.0, 2, "phase_error", 10.0, seq=2),
    ]
    st = derive_state(events, N)
    f = health_factors(st)[1]
    assert abs(f) == pytest.approx(10 ** (-3.0 / 20.0), rel=1e-9)
    assert np.degrees(np.angle(f)) == pytest.approx(10.0, abs=1e-9)


def test_out_of_order_events_placed_by_timestamp():
    """晚到的事件按其时刻归位，回放结果与到达顺序无关。"""
    e1 = ev("e1", 100.0, 4, "failed", seq=1)
    e2 = ev("e2", 50.0, 4, "amplitude_error", -6.0, seq=2)  # 晚到但时刻更早
    e3 = ev("e3", 150.0, 4, "recovered", seq=3)
    arrival_order = [e1, e2, e3]
    # t=75：只有 e2 生效（幅度误差）
    st = derive_state(arrival_order, N, t=75.0)
    assert not st[3].failed
    assert st[3].amplitude_error_db == -6.0
    # t=120：e2 然后 e1（失效）
    st = derive_state(arrival_order, N, t=120.0)
    assert st[3].failed
    # t=200：全部，已恢复
    st = derive_state(arrival_order, N, t=200.0)
    assert not st[3].failed and st[3].amplitude_error_db == 0.0
    # 同一集合按不同到达顺序回放，结果一致
    import itertools
    for perm in itertools.permutations([e1, e2, e3]):
        for t in (75.0, 120.0, 200.0):
            s1 = derive_state(list(perm), N, t=t)
            s2 = derive_state(arrival_order, N, t=t)
            assert [es.as_dict() for es in s1] == [es.as_dict() for es in s2]


def test_same_timestamp_ordered_by_arrival():
    e1 = ev("e1", 100.0, 3, "amplitude_error", -1.0, seq=5)
    e2 = ev("e2", 100.0, 3, "amplitude_error", -2.0, seq=9)
    st = derive_state([e2, e1], N, t=100.0)
    assert st[2].amplitude_error_db == -2.0  # 后到的覆盖


def test_recovered_clears_all_errors():
    events = [
        ev("e1", 1.0, 5, "amplitude_error", -2.0, seq=1),
        ev("e2", 2.0, 5, "phase_error", 15.0, seq=2),
        ev("e3", 3.0, 5, "failed", seq=3),
        ev("e4", 4.0, 5, "recovered", seq=4),
    ]
    st = derive_state(events, N)
    es = st[4]
    assert not es.failed and es.amplitude_error_db == 0.0 and es.phase_error_deg == 0.0
