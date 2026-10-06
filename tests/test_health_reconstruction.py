import time

import numpy as np
import pytest

from app.models import EventType, HealthEvent
from tests.conftest import failed_element_template


def wait_for_job(engine, job_id: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = engine.store.load_job(job_id)
        if job["status"] in {"completed", "failed", "cancelled", "superseded"}:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not finish: {job}")


def failed_event(timestamp: float, event_id: str = "fail-1", element: int = 4) -> HealthEvent:
    return HealthEvent(
        event_id=event_id,
        timestamp=timestamp,
        element=element,
        kind=EventType.failed,
    )


def test_failed_element_raises_sidelobe(engine):
    engine.submit_events([failed_event(1.0)])
    pattern = engine.service.pattern_payload(timestamp=1.0, step_deg=0.02)
    assert pattern["metrics"]["peak_sidelobe_db"] == pytest.approx(-8.17, abs=0.1)
    actual = engine.service.actual_weights_at(1.0)
    assert actual[3] == 0.0


def test_gain_phase_and_recovery_state(engine):
    events = [
        HealthEvent(event_id="g", timestamp=1.0, element=2, kind=EventType.gain_error, gain_error_db=-3.0),
        HealthEvent(event_id="p", timestamp=2.0, element=2, kind=EventType.phase_error, phase_error_deg=45.0),
        HealthEvent(event_id="ok", timestamp=3.0, element=2, kind=EventType.recovered),
    ]
    engine.submit_events(events)
    state_2 = engine.service.state_at(2.0)[1]
    assert state_2.channels[1].gain_error_db == -3.0
    assert state_2.channels[1].phase_error_deg == 45.0
    state_3 = engine.service.state_at(3.0)[1]
    assert state_3.channels[1].healthy
    assert state_3.channels[1].multiplier == 1.0 + 0j


def test_late_and_duplicate_events_order_by_timestamp(engine):
    late = HealthEvent(event_id="late", timestamp=0.5, element=1, kind=EventType.phase_error, phase_error_deg=10)
    duplicate = failed_event(1.0)
    engine.submit_events([failed_event(1.0), HealthEvent(event_id="recover", timestamp=2.0, element=4, kind=EventType.recovered)])
    engine.submit_events([late, duplicate])
    state_1 = engine.service.state_at(1.0)[1]
    assert set(state_1.event_ids) == {"fail-1", "late"}
    state_now = engine.service.state_at(None)[1]
    assert state_now.channels[3].healthy
    assert len(engine.store.load_events()) == 3


def test_reconstruction_satisfies_template_from_both_starts(engine):
    engine.submit_events([failed_event(1.0)])
    template = failed_element_template()
    current_job = engine.submit_job(template, timestamp=1.0, warm_start=True)
    uniform_job = engine.submit_job(template, timestamp=1.0, warm_start=False)
    for job_id in (current_job, uniform_job):
        job = wait_for_job(engine, job_id)
        assert job["status"] == "completed"
        version = engine.store.load_version(job["version_id"])
        assert version["feasible"]
        assert version["metrics"]["mainlobe_gain_loss_db"] >= -3.0
        assert version["metrics"]["worst_template_response_db"] <= -8.0
        assert version["weights"][3]["magnitude"] == 0.0


def test_iteration_limit_reports_best_and_violation(engine):
    engine.submit_events([failed_event(1.0)])
    impossible = failed_element_template(limit_db=-80.0, max_iterations=5)
    job_id = engine.submit_job(impossible, timestamp=1.0, warm_start=False)
    job = wait_for_job(engine, job_id)
    assert job["status"] == "completed"
    version = engine.store.load_version(job["version_id"])
    assert not version["feasible"]
    assert version["iterations"] == 5
    assert version["worst_angle_deg"] is not None
    assert version["worst_excess_db"] > 0.0
    assert np.abs(version["weights"][3]["magnitude"]) == 0.0


def complex(item):
    return item["magnitude"] * np.exp(1j * np.deg2rad(item["phase_deg"]))
