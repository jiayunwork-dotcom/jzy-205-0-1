"""接口端到端测试：定义、事件、查询、作业、版本、回放、恢复。"""
import time

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import ARRAY_DEF, JOB_CONFIG, TEMPLATE, wait_job


# ----------------------------------------------------------------------
# 阵列定义与参考指标
# ----------------------------------------------------------------------
def test_array_factor_endpoint(array_client):
    """任意方向上的复阵因子查询。"""
    r = array_client.get("/array-factor", params=[("theta", 0.0), ("theta", 30.0)])
    assert r.status_code == 200
    af = r.json()["array_factor"]
    assert len(af) == 2
    # 等幅同相 8 单元在法线方向相干叠加，阵因子 = 8
    assert af[0]["real"] == pytest.approx(8.0, abs=1e-9)
    assert af[0]["imag"] == pytest.approx(0.0, abs=1e-9)


def test_define_array_and_reference_pattern(array_client):
    r = array_client.get("/pattern")
    assert r.status_code == 200
    m = r.json()["metrics"]
    assert m["main_lobe_direction_deg"] == pytest.approx(0.0, abs=0.05)
    assert abs(m["first_null_deg"]) == pytest.approx(14.48, abs=0.05)
    assert m["max_sidelobe_db"] == pytest.approx(-12.80, abs=0.05)
    assert m["half_power_beamwidth_deg"] == pytest.approx(12.80, abs=0.1)


def test_steered_array_pattern(client):
    x_phase = [-360.0 * i * 0.5 * 0.5 for i in range(8)]  # -2pi*x*sin30° 换算成度
    body = {
        "n_elements": 8,
        "spacing_wavelengths": 0.5,
        "weights": [{"amplitude": 1.0, "phase_deg": p} for p in x_phase],
        "pointing_deg": 30.0,
    }
    assert client.put("/array", json=body).status_code == 200
    m = client.get("/pattern").json()["metrics"]
    assert m["main_lobe_direction_deg"] == pytest.approx(30.0, abs=0.05)


def test_pattern_sampling(array_client):
    r = array_client.get("/pattern", params={"include_pattern": True, "step_deg": 1.0})
    data = r.json()
    assert len(data["pattern"]) == 181
    peak = max(p["db"] for p in data["pattern"])
    assert peak == pytest.approx(0.0, abs=1e-6)


# ----------------------------------------------------------------------
# 拒收规则
# ----------------------------------------------------------------------
def test_reject_invalid_array_definitions(client):
    bad = dict(ARRAY_DEF)
    bad["n_elements"] = 1
    bad["weights"] = bad["weights"][:1]
    assert client.put("/array", json=bad).status_code == 422

    bad = dict(ARRAY_DEF)
    bad["spacing_wavelengths"] = 0.0
    assert client.put("/array", json=bad).status_code == 422

    bad = dict(ARRAY_DEF)
    bad["spacing_wavelengths"] = -0.5
    assert client.put("/array", json=bad).status_code == 422

    bad = dict(ARRAY_DEF)
    bad["weights"] = [{"amplitude": -1.0, "phase_deg": 0.0}] * 8
    assert client.put("/array", json=bad).status_code == 422

    # 非有限数（NaN）：httpx 的 json 编码器拒发，改发原始报文
    import json as _json
    raw = _json.dumps({**ARRAY_DEF, "weights": [{"amplitude": float("nan"), "phase_deg": 0.0}] * 8})
    r = client.put("/array", content=raw, headers={"content-type": "application/json"})
    assert r.status_code == 422

    bad = dict(ARRAY_DEF)
    bad["weights"] = [{"amplitude": 1.0, "phase_deg": 0.0}] * 7
    assert client.put("/array", json=bad).status_code == 422


def test_reject_event_with_bad_element(array_client):
    r = array_client.post("/events", json={
        "event_id": "x1", "timestamp": 1.0, "element": 99, "change_type": "failed"})
    assert r.status_code == 400
    r = array_client.post("/events", json={
        "event_id": "x2", "timestamp": 1.0, "element": 0, "change_type": "failed"})
    assert r.status_code == 422


def test_reject_event_missing_value(array_client):
    r = array_client.post("/events", json={
        "event_id": "x3", "timestamp": 1.0, "element": 2, "change_type": "amplitude_error"})
    assert r.status_code == 400


def test_reject_bad_template(array_client):
    def job_with(template, loss=1.0):
        body = dict(JOB_CONFIG)
        body["template"] = template
        body["max_mainlobe_loss_db"] = loss
        return array_client.post("/jobs", json=body)

    # 区间越出 ±90°
    assert job_with({"sectors": [{"theta_min": -120.0, "theta_max": 0.0, "max_sll_db": -20.0}]}).status_code == 422
    # 区间相互重叠
    assert job_with({"sectors": [
        {"theta_min": -90.0, "theta_max": 0.0, "max_sll_db": -20.0},
        {"theta_min": -10.0, "theta_max": 90.0, "max_sll_db": -20.0},
    ]}).status_code == 422
    # 允许的主瓣损失为负
    assert job_with(TEMPLATE, loss=-1.0).status_code == 422


# ----------------------------------------------------------------------
# 事件：乱序、重复、状态查询
# ----------------------------------------------------------------------
def _fold(events, n, t):
    """测试侧独立折叠，用于和服务的回放对照。"""
    sel = [e for e in events if e["timestamp"] <= t]
    sel.sort(key=lambda e: (e["timestamp"], e["arrival_seq"]))
    st = [{"failed": False, "amplitude_error_db": 0.0, "phase_error_deg": 0.0} for _ in range(n)]
    for e in sel:
        s = st[e["element"] - 1]
        c = e["change_type"]
        if c == "failed":
            s["failed"] = True
        elif c == "amplitude_error":
            s["amplitude_error_db"] = e["value"]
        elif c == "phase_error":
            s["phase_error_deg"] = e["value"]
        elif c == "recovered":
            s.update(failed=False, amplitude_error_db=0.0, phase_error_deg=0.0)
    return st


def test_events_out_of_order_and_duplicate(array_client):
    ev = [
        {"event_id": "e1", "timestamp": 100.0, "element": 4, "change_type": "failed"},
        {"event_id": "e2", "timestamp": 50.0, "element": 4, "change_type": "amplitude_error", "value": -6.0},
        {"event_id": "e3", "timestamp": 150.0, "element": 4, "change_type": "recovered"},
    ]
    # 乱序到达
    for e in (ev[0], ev[2], ev[1]):
        r = array_client.post("/events", json=e)
        assert r.status_code == 201
        assert r.json()["applied"] is True
    # 重复事件只生效一次
    r = array_client.post("/events", json=ev[0])
    assert r.json()["applied"] is False

    events = array_client.get("/events").json()["events"]
    assert len(events) == 3

    # 逐时刻回放与独立折叠逐项一致
    for t in (25.0, 75.0, 125.0, 200.0, 1e12):
        replay = array_client.get("/replay", params={"t": t}).json()
        got = [{k: el[k] for k in ("failed", "amplitude_error_db", "phase_error_deg")}
               for el in replay["state"]["elements"]]
        assert got == _fold(events, 8, t), f"t={t} 回放不一致"

    # 在线"当前状态"与 t=+inf 的回放一致
    online = array_client.get("/state").json()
    replay_inf = array_client.get("/replay", params={"t": 1e12}).json()
    assert online["elements"] == replay_inf["state"]["elements"]
    # 方向图查询与回放指标一致
    pat = array_client.get("/pattern").json()
    assert pat["version_id"] == replay_inf["active_version_id"]
    assert pat["metrics"] == replay_inf["metrics"]


def test_replay_active_version_consistent(array_client):
    """每一时刻的生效方案，回放与在线一致。"""
    # 先做一次重构（健康状态），产生带模板的版本
    job = array_client.post("/jobs", json=JOB_CONFIG).json()
    rec = wait_job(array_client, job["job_id"])
    assert rec["status"] == "succeeded"
    v_healthy = rec["result"]["version_id"]

    # t=100 失效 → 自动重构
    r = array_client.post("/events", json={
        "event_id": "f1", "timestamp": 100.0, "element": 4, "change_type": "failed"})
    auto = r.json()["auto_job"]
    assert auto is not None
    rec = wait_job(array_client, auto["job_id"])
    assert rec["status"] == "succeeded"
    v_failed = rec["result"]["version_id"]
    assert v_failed != v_healthy

    # t=50（失效前）：生效的是健康时的方案；t=150：失效后的方案
    assert array_client.get("/replay", params={"t": 50.0}).json()["active_version_id"] == v_healthy
    assert array_client.get("/replay", params={"t": 150.0}).json()["active_version_id"] == v_failed
    # 在线当前生效 = 回放 t=+inf
    assert array_client.get("/replay", params={"t": 1e12}).json()["active_version_id"] == v_failed


# ----------------------------------------------------------------------
# 重构作业
# ----------------------------------------------------------------------
def test_manual_job_and_version_binding(array_client):
    job = array_client.post("/jobs", json=JOB_CONFIG).json()
    rec = wait_job(array_client, job["job_id"])
    assert rec["status"] == "succeeded", rec
    assert rec["result"]["feasible"] is True
    vid = rec["result"]["version_id"]

    v = array_client.get(f"/versions/{vid}").json()
    assert v["template"] == TEMPLATE
    assert v["basis_event_ids"] == []
    assert v["solve"]["stop_reason"] == "converged"
    # 生效版本已切换
    pat = array_client.get("/pattern").json()
    assert pat["version_id"] == vid


def test_failure_triggers_auto_reconfigure(array_client):
    # 健康时先绑定模板
    job = array_client.post("/jobs", json=JOB_CONFIG).json()
    assert wait_job(array_client, job["job_id"])["status"] == "succeeded"

    # 失效前：健康阵列重构后副瓣已被压到模板限附近
    before = array_client.get("/pattern").json()["metrics"]["max_sidelobe_db"]
    assert before <= -10.0 + 0.1

    # 4 号单元失效：副瓣抬升 → 自动重构
    r = array_client.post("/events", json={
        "event_id": "f4", "timestamp": 10.0, "element": 4, "change_type": "failed"})
    auto = r.json()["auto_job"]
    assert auto is not None
    assert auto["trigger"] == "auto"
    assert "f4" in auto["basis"]["event_ids"]

    rec = wait_job(array_client, auto["job_id"])
    assert rec["status"] == "succeeded", rec
    vid = rec["result"]["version_id"]
    v = array_client.get(f"/versions/{vid}").json()
    # 失效单元加权严格为零
    assert v["weights"][3] == [0.0, 0.0]
    # 实际方向图满足模板（±18° 以外 ≤ -10 dB）
    samples = array_client.get("/pattern", params={
        "include_pattern": True, "step_deg": 0.5}).json()["pattern"]
    worst = max(p["db"] for p in samples if abs(p["theta_deg"]) >= 18.0)
    assert worst <= -10.0 + 0.1
    assert v["solve"]["max_violation_db"] <= 0.05


def test_iteration_cap_honest_return_via_api(array_client):
    # 健康阵列对 -10 dB 模板本就满足，改用 -20 dB 紧模板并只给 5 次迭代
    body = dict(JOB_CONFIG)
    body["template"] = {"sectors": [
        {"theta_min": -90.0, "theta_max": -18.0, "max_sll_db": -20.0},
        {"theta_min": 18.0, "theta_max": 90.0, "max_sll_db": -20.0},
    ]}
    body["max_iterations"] = 5
    job = array_client.post("/jobs", json=body).json()
    rec = wait_job(array_client, job["job_id"])
    assert rec["status"] == "infeasible"
    res = rec["result"]
    assert res["feasible"] is False
    assert res["worst_angle_deg"] is not None
    assert res["worst_excess_db"] > 0.0
    # 最优加权仍存档为版本，但不生效
    v = array_client.get(f"/versions/{res['version_id']}").json()
    assert v["solve"]["feasible"] is False
    pat = array_client.get("/pattern").json()
    assert pat["version_id"] != res["version_id"]


def test_both_start_modes_via_api(array_client):
    # 先制造失效（此时无模板，不触发自动重构）
    array_client.post("/events", json={
        "event_id": "f4", "timestamp": 1.0, "element": 4, "change_type": "failed"})
    versions = []
    for mode in ("cold", "warm"):
        body = dict(JOB_CONFIG)
        body["start_mode"] = mode
        job = array_client.post("/jobs", json=body).json()
        rec = wait_job(array_client, job["job_id"])
        assert rec["status"] == "succeeded", f"{mode} 起点未满足模板"
        versions.append(rec["result"]["version_id"])
    # 两个版本都满足模板：±18° 以外 ≤ -10 dB，失效位严格为零
    for vid in versions:
        v = array_client.get(f"/versions/{vid}").json()
        assert v["solve"]["max_violation_db"] <= 0.05
        assert v["weights"][3] == [0.0, 0.0]
        samples = array_client.get("/pattern", params={
            "version_id": vid, "include_pattern": True, "step_deg": 0.5}).json()["pattern"]
        worst = max(p["db"] for p in samples if abs(p["theta_deg"]) >= 18.0)
        assert worst <= -10.0 + 0.1


# 让作业长时间运行的配置：-20 dB 紧模板（不可满足）+ 大迭代上限 + 人为延时
LONG_JOB = dict(JOB_CONFIG, max_iterations=2_000_000, iteration_delay_ms=2.0)
LONG_JOB["template"] = {"sectors": [
    {"theta_min": -90.0, "theta_max": -18.0, "max_sll_db": -20.0},
    {"theta_min": 18.0, "theta_max": 90.0, "max_sll_db": -20.0},
]}


def test_job_superseded_by_new_event(array_client):
    job = array_client.post("/jobs", json=LONG_JOB).json()
    jid = job["job_id"]
    # 等它跑起来
    for _ in range(200):
        if array_client.get(f"/jobs/{jid}").json()["status"] == "running":
            break
        time.sleep(0.02)
    # 新事件到达 → 旧作业作废，按最新状态自动重新发起
    r = array_client.post("/events", json={
        "event_id": "f4", "timestamp": 1.0, "element": 4, "change_type": "failed"})
    auto = r.json()["auto_job"]
    assert array_client.get(f"/jobs/{jid}").json()["status"] == "superseded"
    assert auto is not None and auto["job_id"] != jid
    # 自动作业也在跑同样的模板；清理
    array_client.post(f"/jobs/{auto['job_id']}/cancel")


def test_cancel_job(array_client):
    job = array_client.post("/jobs", json=LONG_JOB).json()
    jid = job["job_id"]
    for _ in range(200):
        if array_client.get(f"/jobs/{jid}").json()["status"] == "running":
            break
        time.sleep(0.02)
    rec = array_client.post(f"/jobs/{jid}/cancel").json()
    assert rec["status"] == "cancelled"
    # 已结束的作业不能再取消
    assert array_client.post(f"/jobs/{jid}/cancel").status_code == 404


def test_version_compare(array_client):
    job = array_client.post("/jobs", json=JOB_CONFIG).json()
    rec = wait_job(array_client, job["job_id"])
    versions = array_client.get("/versions").json()["versions"]
    v1 = next(v["version_id"] for v in versions if v["kind"] == "initial")
    v2 = rec["result"]["version_id"]
    r = array_client.get("/versions/compare", params={"a": v1, "b": v2})
    assert r.status_code == 200
    data = r.json()
    assert data["a"]["version_id"] == v1 and data["b"]["version_id"] == v2
    assert data["a"]["max_sidelobe_db"] == pytest.approx(-12.80, abs=0.05)
    assert len(data["elements"]) == 8
    for el in data["elements"]:
        assert "delta_amplitude_db" in el and "delta_phase_deg" in el


# ----------------------------------------------------------------------
# 重启恢复
# ----------------------------------------------------------------------
def test_restart_recovery(redis_client):
    app1 = create_app(redis_client)
    with TestClient(app1) as c1:
        assert c1.put("/array", json=ARRAY_DEF).status_code == 200
        c1.post("/events", json={"event_id": "e1", "timestamp": 5.0, "element": 2,
                                 "change_type": "amplitude_error", "value": -3.0})
        job = c1.post("/jobs", json=JOB_CONFIG).json()
        rec = wait_job(c1, job["job_id"])
        assert rec["status"] == "succeeded"
        versions_before = c1.get("/versions").json()["versions"]
        state_before = c1.get("/state").json()
        pattern_before = c1.get("/pattern").json()

    # 模拟重启：同一 Redis，新建应用实例
    app2 = create_app(redis_client)
    with TestClient(app2) as c2:
        assert c2.get("/array").json()["n_elements"] == 8
        assert c2.get("/state").json() == state_before
        assert c2.get("/pattern").json() == pattern_before
        assert c2.get("/versions").json()["versions"] == versions_before
        # 回放也一致
        assert c2.get("/replay", params={"t": 1e12}).json()["state"]["elements"] == state_before["elements"]


def test_restart_aborts_unfinished_job(redis_client):
    app1 = create_app(redis_client)
    with TestClient(app1) as c1:
        c1.put("/array", json=ARRAY_DEF)
        job = c1.post("/jobs", json=LONG_JOB).json()
        jid = job["job_id"]
        for _ in range(200):
            if c1.get(f"/jobs/{jid}").json()["status"] == "running":
                break
            time.sleep(0.02)

        app2 = create_app(redis_client)
        with TestClient(app2) as c2:
            assert c2.get(f"/jobs/{jid}").json()["status"] == "aborted"
        # 清理旧实例的作业线程
        c1.post(f"/jobs/{jid}/cancel")
