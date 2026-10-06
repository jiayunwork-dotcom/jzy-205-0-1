import time

import pytest

from app.engine import RadarEngine
from app.models import EventType, HealthEvent
from app.storage import Store
from tests.conftest import failed_element_template, uniform_array
from tests.test_health_reconstruction import failed_event


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.005)
    raise AssertionError("condition was not reached")


def wait_jobs_idle(engine):
    return wait_until(
        lambda: all(job["status"] not in {"queued", "running"} for job in engine.jobs.list_jobs())
    )


def test_new_event_supersedes_unfinished_job_and_auto_rebuilds():
    store = Store()
    engine = RadarEngine(store)
    engine.define_array(uniform_array())
    engine.submit_events([failed_event(1.0)])
    # Seed a feasible current version manually; the first failure has no plan to
    # invalidate, so it cannot automatically trigger reconstruction.
    template = failed_element_template()
    seeded = engine.submit_job(template, timestamp=1.0, warm_start=False)
    wait_jobs_idle(engine)
    assert engine.store.load_job(seeded)["status"] == "completed"

    second_failure = HealthEvent(event_id="fifth-failure", timestamp=2.0, element=5, kind=EventType.failed)
    engine.submit_events([second_failure])
    wait_jobs_idle(engine)
    jobs = engine.jobs.list_jobs()
    automatic = [job for job in jobs if job.get("auto")]
    assert automatic

    recovery = HealthEvent(event_id="recover", timestamp=3.0, element=4, kind=EventType.recovered)
    engine.submit_events([recovery])
    wait_jobs_idle(engine)
    active_id = store.active_version_id()
    active = store.load_version(active_id)
    assert active["state_timestamp"] == 3.0
    assert set(active["event_ids"]) == {"fail-1", "fifth-failure", "recover"}
    assert active["metrics"]["mainlobe_gain_loss_db"] < 0.0


def test_replay_matches_online_event_order_and_effective_state():
    online = Store()
    engine = RadarEngine(online)
    engine.define_array(uniform_array())
    # Establish a plan for the late-arriving first failure.
    seeded = engine.submit_job(failed_element_template(), timestamp=2.0, warm_start=False)
    wait_jobs_idle(engine)
    assert engine.store.load_job(seeded)["status"] == "completed"
    engine.submit_events([failed_event(2.0, event_id="late-fail")])
    engine.submit_events(
        [
            HealthEvent(
                event_id="late-other-fail",
                timestamp=1.0,
                element=5,
                kind=EventType.failed,
            )
        ]
    )
    wait_jobs_idle(engine)
    online_events = sorted(event["event_id"] for event in online.load_events())
    online_active = engine.store.load_version(engine.store.active_version_id())

    replay = engine.replay()
    assert sorted(event["event_id"] for event in replay["events"]) == online_events
    replayed_active = replay["active_version"]
    assert replayed_active is not None
    assert replayed_active["event_ids"] == online_active["event_ids"]
    assert replayed_active["state_timestamp"] == online_active["state_timestamp"]
    assert replayed_active["weights"] == online_active["weights"]


def test_restart_recovers_events_array_and_active_version():
    store = Store()
    engine = RadarEngine(store)
    engine.define_array(uniform_array())
    engine.submit_events([failed_event(1.0)])
    job_id = engine.submit_job(failed_element_template(), timestamp=1.0, warm_start=False)
    wait_jobs_idle(engine)
    version_id = store.active_version_id()
    assert version_id

    restarted = RadarEngine(store)
    assert restarted.store.active_version_id() == version_id
    _, state = restarted.service.state_at(1.0)
    assert state.failed_elements == [4]
    persisted_version = restarted.store.load_version(version_id)
    assert persisted_version["weights"][3]["magnitude"] == 0.0
    assert restarted.service.template_compliant(persisted_version, 1.0)
