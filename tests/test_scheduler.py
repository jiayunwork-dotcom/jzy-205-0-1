import threading
import time

from app.models import JobStatus
from app.scheduler import JobManager
from app.storage import Store
from conftest import failed_element_template


def test_scheduler_cancel_signals_running_job():
    started = threading.Event()

    def slow_job(*, template, timestamp, start_weights, start_point, job_id, cancellation_token):
        started.set()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if cancellation_token():
                return {"status": "cancelled"}
            time.sleep(0.001)
        raise AssertionError("job was not cancelled")

    manager = JobManager(Store(), slow_job)
    template = failed_element_template(max_iterations=5000)
    job_id = manager.submit(
        template=template,
        timestamp=None,
        start_point="uniform",
        start_weights=None,
    )
    assert started.wait(1.0)
    cancelled = manager.cancel(job_id)
    assert cancelled["status"] == JobStatus.cancelled.value
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        job = manager.store.load_job(job_id)
        if job["status"] == JobStatus.cancelled.value:
            break
        time.sleep(0.005)
    assert manager.store.load_job(job_id)["status"] == JobStatus.cancelled.value
