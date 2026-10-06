"""Asynchronous reconstruction job scheduler.

The numerical solver is NumPy/CPU-bound and therefore runs in a worker thread.
Only one reconstruction is evaluated at a time; a newer event or job cancels and
supersedes the running/queued job before the latest request is queued.
"""
from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

import numpy as np

from .models import JobStatus, ReconstructionTemplate
from .service import _weights_from_payload, new_id


class JobManager:
    def __init__(self, store, run_callable: Callable[..., dict[str, Any]]):
        self.store = store
        self.run = run_callable
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="reconstruction")
        self._future: Future | None = None
        self._current_id: str | None = None
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._restore()

    def _restore(self) -> None:
        for job in self.store.load_jobs():
            if job["status"] in (JobStatus.queued.value, JobStatus.running.value):
                # A process restart interrupts in-memory execution. Re-queue so
                # deterministic state at restart is recovered automatically.
                job["status"] = JobStatus.queued.value
                job["progress"] = {"message": "re-queued after service restart"}
                self.store.save_job(job)
        queued = [
            job
            for job in self.store.load_jobs()
            if job["status"] == JobStatus.queued.value
        ]
        for job in sorted(queued, key=lambda item: item["created_at"])[:1]:
            self._launch(job["job_id"])

    def submit(
        self,
        *,
        template: ReconstructionTemplate,
        timestamp: float | None,
        start_point: str,
        start_weights: np.ndarray | None,
        auto: bool = False,
    ) -> str:
        job_id = new_id(self.store, "job")
        job = {
            "job_id": job_id,
            "status": JobStatus.queued.value,
            "template": template.model_dump(mode="json") if hasattr(template, "model_dump") else vars(template),
            "timestamp": timestamp,
            "start_point": start_point,
            "start_weights": _payload(start_weights),
            "auto": auto,
            "created_at": _now(),
            "progress": {"queued": True},
            "version_id": None,
            "error": None,
        }
        self.store.save_job(job)
        # A newer automatic reconstruction represents the latest event state and
        # supersedes an in-flight automatic reconstruction. Explicit manual jobs
        # remain independent so two starts can be compared.
        if auto:
            self._cancel_active(job_id)
        with self._lock:
            self._launch(job_id)
        return job_id

    def _cancel_active(self, new_job_id: str) -> None:
        with self._lock:
            if self._current_id and self._current_id != new_job_id:
                event = self._cancel_events.get(self._current_id)
                if event is not None:
                    event.set()
            for job in self.store.load_jobs():
                if job["job_id"] != new_job_id and job["status"] in (
                    JobStatus.queued.value,
                    JobStatus.running.value,
                ):
                    job["status"] = JobStatus.superseded.value
                    self.store.save_job(job)

    def _launch(self, job_id: str) -> None:
        job = self.store.load_job(job_id)
        if job is None or job["status"] != JobStatus.queued.value:
            return
        self._cancel_events[job_id] = threading.Event()
        job["status"] = JobStatus.running.value
        self.store.save_job(job)
        self._current_id = job_id
        self._future = self._executor.submit(self._execute, job_id)

    def _execute(self, job_id: str) -> None:
        job = self.store.load_job(job_id)
        if job is None:
            return
        token_event = self._cancel_events[job_id]
        try:
            start_weights = None
            if job.get("start_weights") is not None:
                start_weights = _weights_from_payload(job["start_weights"])
            template = job["template"]
            if isinstance(template, dict):
                template = ReconstructionTemplate.model_validate(template)
            result = self.run(
                template=template,
                timestamp=job["timestamp"],
                start_weights=start_weights,
                start_point=job["start_point"],
                job_id=job_id,
                cancellation_token=token_event.is_set,
            )
            job = self.store.load_job(job_id) or job
            if token_event.is_set():
                refreshed = self.store.load_job(job_id)
                job["status"] = (
                    JobStatus.cancelled.value
                    if refreshed and refreshed["status"] == JobStatus.cancelled.value
                    else JobStatus.superseded.value
                )
                job["progress"] = {
                    "message": "cancelled" if job["status"] == JobStatus.cancelled.value else "superseded by newer state or request"
                }
            else:
                version_id = new_id(self.store, "ver")
                version = {
                    "version_id": version_id,
                    "job_id": job_id,
                    "created_at": _now(),
                    "state_timestamp": result["timestamp"],
                    "event_ids": result["event_ids"],
                    "template": result["template"],
                    "weights": result["weights"],
                    "metrics": result["metrics"],
                    "converged": result["converged"],
                    "feasible": result["feasible"],
                    "solver_status": result["status"],
                    "iterations": result["iterations"],
                    "worst_angle_deg": result["worst_angle_deg"],
                    "worst_excess_db": result["worst_excess_db"],
                    "start_point": result["start_point"],
                    "auto": bool(job.get("auto", False)),
                }
                self.store.save_version(version)
                # The newest completed reconstruction is the effective one.
                self.store.set_active_version(version_id)
                job["status"] = JobStatus.completed.value
                job["version_id"] = version_id
                job["progress"] = {"iterations": result["iterations"], "feasible": result["feasible"]}
            self.store.save_job(job)
        except Exception as exc:  # the API must report solver/worker failures
            failed = self.store.load_job(job_id)
            if failed is not None:
                failed["status"] = JobStatus.failed.value
                failed["error"] = str(exc)
                self.store.save_job(failed)
        finally:
            with self._lock:
                if self._current_id == job_id:
                    self._current_id = None
                    self._future = None
                self._cancel_events.pop(job_id, None)
                self._start_next_queued()

    def cancel(self, job_id: str) -> dict[str, Any] | None:
        job = self.store.load_job(job_id)
        if job is None:
            return None
        if job["status"] in (JobStatus.queued.value, JobStatus.running.value):
            event = self._cancel_events.get(job_id)
            if event is not None:
                event.set()
            job["status"] = JobStatus.cancelled.value
            self.store.save_job(job)
        return self.store.load_job(job_id)

    def _start_next_queued(self) -> None:
        queued = [
            job
            for job in self.store.load_jobs()
            if job["status"] == JobStatus.queued.value
        ]
        if queued and self._current_id is None:
            next_job = sorted(queued, key=lambda item: item["created_at"])[0]
            self._launch(next_job["job_id"])

    def cancel_for_event(self) -> list[dict[str, Any]]:
        """Mark every unfinished reconstruction superseded by a newer event."""
        superseded: list[dict[str, Any]] = []
        for job in self.store.load_jobs():
            if job["status"] not in (JobStatus.queued.value, JobStatus.running.value):
                continue
            event = self._cancel_events.get(job["job_id"])
            if event is not None:
                event.set()
            job["status"] = JobStatus.superseded.value
            job["progress"] = {"message": "superseded by a newer health event"}
            self.store.save_job(job)
            superseded.append(job)
        return superseded

    def list_jobs(self) -> list[dict[str, Any]]:
        return sorted(self.store.load_jobs(), key=lambda job: job["created_at"], reverse=True)


def _payload(weights: np.ndarray | None) -> list[dict[str, float]] | None:
    if weights is None:
        return None
    return [
        {"magnitude": float(abs(value)), "phase_deg": float(np.rad2deg(np.angle(value)))}
        for value in np.asarray(weights, dtype=complex)
    ]


def _now() -> float:
    import time

    return time.time()
