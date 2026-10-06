import time

import pytest

from tests.conftest import failed_element_template, interval, uniform_array, weight


def wait_status(client, job_id, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/reconstruction-jobs/{job_id}")
        payload = response.json()
        if payload["status"] in {"completed", "failed", "cancelled", "superseded"}:
            return payload
        time.sleep(0.01)
    raise AssertionError("job did not finish")


def test_api_reference_pattern(client):
    response = client.put("/array", json=uniform_array().model_dump())
    assert response.status_code == 200
    pattern = client.get("/pattern?step_deg=0.01").json()
    metrics = pattern["metrics"]
    assert metrics["main_direction_deg"] == pytest.approx(0.0, abs=1e-8)
    assert metrics["first_null_offset_deg"] == pytest.approx(14.48, abs=0.02)
    assert metrics["peak_sidelobe_db"] == pytest.approx(-12.80, abs=0.02)
    assert metrics["half_power_beamwidth_deg"] == pytest.approx(12.80, abs=0.03)


def test_rejects_invalid_inputs(client):
    bad_array = uniform_array().model_dump()
    bad_array["element_count"] = 1
    bad_array["weights"] = []
    assert client.put("/array", json=bad_array).status_code == 422
    bad_spacing = uniform_array().model_dump()
    bad_spacing["spacing_wavelengths"] = 0
    assert client.put("/array", json=bad_spacing).status_code == 422
    assert client.put(
        "/array",
        json=uniform_array().model_dump(),
    ).status_code == 200
    assert client.post("/events", json=[{"event_id": "bad", "timestamp": 1, "element": 9, "kind": "failed"}]).status_code == 400
    bad_template = failed_element_template().model_dump()
    bad_template["intervals"] = [
        interval(-90.0, -10.0, -8).model_dump(),
        interval(-20.0, 90.0, -8).model_dump(),
    ]
    assert client.post("/reconstruction-jobs", json={"template": bad_template}).status_code == 422
    bad_loss = failed_element_template().model_dump()
    bad_loss["allowed_mainlobe_gain_loss_db"] = -1
    assert client.post("/reconstruction-jobs", json={"template": bad_loss}).status_code == 422


def test_failure_reconstruction_compare_and_versions(client):
    client.put("/array", json=uniform_array().model_dump())
    failed = {"event_id": "f4", "timestamp": 1.0, "element": 4, "kind": "failed"}
    assert client.post("/events", json=[failed]).status_code == 200
    failed_pattern = client.get("/pattern?timestamp=1&step_deg=.02").json()
    assert failed_pattern["metrics"]["peak_sidelobe_db"] == pytest.approx(-8.17, abs=0.1)
    template = failed_element_template().model_dump()
    first = client.post(
        "/reconstruction-jobs",
        json={"template": template, "timestamp": 1.0, "warm_start": False},
    ).json()["job_id"]
    second = client.post(
        "/reconstruction-jobs",
        json={"template": template, "timestamp": 1.0, "warm_start": True},
    ).json()["job_id"]
    completed = [wait_status(client, first), wait_status(client, second)]
    assert all(job["status"] == "completed" for job in completed)
    version_ids = [job["version_id"] for job in completed]
    comparison = client.post("/weight-versions/compare", json={"version_ids": version_ids}).json()
    assert len(comparison["weight_changes"]) == 8
    assert comparison["left_metrics"]["worst_template_response_db"] <= -8.0
    assert comparison["right_metrics"]["worst_template_response_db"] <= -8.0


def test_event_rejection_does_not_mutate_and_duplicates(client):
    client.put("/array", json=uniform_array().model_dump())
    bad_batch = [
        {"event_id": "ok", "timestamp": 1.0, "element": 1, "kind": "phase_error", "phase_error_deg": 5},
        {"event_id": "bad", "timestamp": 2.0, "element": 99, "kind": "failed"},
    ]
    assert client.post("/events", json=bad_batch).status_code == 400
    assert client.get("/events").json()["events"] == []
    event = {"event_id": "dup", "timestamp": 1.0, "element": 1, "kind": "failed"}
    assert client.post("/events", json=[event]).status_code == 200
    assert client.post("/events", json=[event]).json()["results"][0]["added"] is False
    assert len(client.get("/events").json()["events"]) == 1
