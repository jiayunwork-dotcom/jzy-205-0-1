import time

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

# 重构测试用模板：8 单元缺 4 号单元时物理可达（边界约 -12 dB）
TEMPLATE = {
    "sectors": [
        {"theta_min": -90.0, "theta_max": -18.0, "max_sll_db": -10.0},
        {"theta_min": 18.0, "theta_max": 90.0, "max_sll_db": -10.0},
    ]
}

JOB_CONFIG = {
    "template": TEMPLATE,
    "pointing_deg": 0.0,
    "max_mainlobe_loss_db": 3.0,
    "amplitude_cap": 1.0,
    "max_iterations": 10000,
}

ARRAY_DEF = {
    "n_elements": 8,
    "spacing_wavelengths": 0.5,
    "weights": [{"amplitude": 1.0, "phase_deg": 0.0}] * 8,
    "pointing_deg": 0.0,
}


@pytest.fixture()
def redis_client():
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture()
def client(redis_client):
    app = create_app(redis_client)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def array_client(client):
    r = client.put("/array", json=ARRAY_DEF)
    assert r.status_code == 200, r.text
    return client


def wait_job(client, job_id, timeout=60.0):
    """轮询作业直到终态。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        rec = client.get(f"/jobs/{job_id}").json()
        if rec["status"] not in ("queued", "running"):
            return rec
        time.sleep(0.05)
    raise TimeoutError(f"作业 {job_id} 在 {timeout}s 内未结束")
