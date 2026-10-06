"""Application orchestration connecting domain services and the job scheduler."""
from __future__ import annotations

from typing import Any

import numpy as np

from .models import ArrayDefinition, HealthEvent, JobRequest, ReconstructionTemplate
from .scheduler import JobManager
from .service import RadarService
from .storage import Store


class RadarEngine:
    def __init__(self, store: Store | None = None):
        self.store = store or Store()
        self.service = RadarService(self.store)
        self.jobs = JobManager(self.store, self.service.run_reconstruction)

    def define_array(self, definition: ArrayDefinition) -> dict[str, Any]:
        # Replacing the array definition invalidates every old state and plan.
        self.store.redis.flushdb()
        payload = self.service.define_array(definition)
        self.store.append_input({"kind": "array", "payload": payload})
        return payload

    def submit_events(self, events: list[HealthEvent]) -> list[dict[str, Any]]:
        # Validate and reject the entire batch before mutating the event log.
        for event in events:
            array = self.service._ensure_array()
            if event.element < 1 or event.element > array.element_count:
                raise ValueError(f"event references non-existent element {event.element}")
        results = [self.service.submit_event(event) for event in events]
        if results:
            self.store.append_input({"kind": "events", "payload": [result["event"] for result in results]})
            self._auto_reconstruct_after_event()
        return results

    def _auto_reconstruct_after_event(self) -> None:
        # Any new health event invalidates work still based on an older state.
        self.jobs.cancel_for_event()
        version_id = self.store.active_version_id()
        if not version_id:
            return
        version = self.store.load_version(version_id)
        if not version:
            return
        template = ReconstructionTemplate.model_validate(version["template"])
        _, state = self.service.state_at(None)
        if self.service.template_compliant(
            version, state.timestamp, health_multipliers=state.multipliers
        ):
            return
        weights = np.asarray(_weights(version["weights"]), dtype=complex)
        self.submit_job(
            ReconstructionTemplate.model_validate(version["template"]),
            timestamp=state.timestamp,
            warm_start=True,
            start_weights=weights,
            auto=True,
            record_input=False,
        )

    def submit_job(
        self,
        template: ReconstructionTemplate,
        *,
        timestamp: float | None = None,
        warm_start: bool = True,
        start_weights: np.ndarray | None = None,
        auto: bool = False,
        record_input: bool = True,
    ) -> str:
        if start_weights is None and warm_start:
            version_id = self.store.active_version_id()
            if version_id:
                version = self.store.load_version(version_id)
                if version is not None:
                    start_weights = np.asarray(_weights(version["weights"]), dtype=complex)
        if start_weights is None:
            start_weights = self.service._ensure_array().beam_direction_weights()
        job_id = self.jobs.submit(
            template=template,
            timestamp=timestamp,
            start_point="current" if warm_start else "uniform",
            start_weights=start_weights,
            auto=auto,
        )
        if record_input:
            self.store.append_input({"kind": "job", "job_id": job_id})
        return job_id

    def job(self, request: JobRequest) -> str:
        warm = self.service._ensure_array() and True
        warm = request.warm_start if request.warm_start is not None else request.template.warm_start
        job_id = self.submit_job(
            request.template,
            timestamp=request.timestamp,
            warm_start=warm,
        )
        return job_id

    def replay(self) -> dict[str, Any]:
        """Replay every submitted array, event batch, and manual job in arrival order.

        Auto-reconstructions are regenerated from the same health transitions, so
        replay does not need to store those generated jobs as inputs.
        """
        inputs = self.store.load_inputs()
        replayed = RadarEngine(Store())
        for item in inputs:
            kind = item["kind"]
            if kind == "array":
                replayed.service.define_array(ArrayDefinition.model_validate(item["payload"]))
            elif kind == "events":
                events = [HealthEvent.model_validate(payload) for payload in item["payload"]]
                for event in events:
                    replayed.service.submit_event(event)
                replayed._auto_reconstruct_after_event()
                _wait_idle(replayed)
            elif kind == "job":
                # Reconstruct the exact manual submission from its recorded request.
                job = self.store.load_job(item["job_id"])
                replayed.submit_job(
                    ReconstructionTemplate.model_validate(job["template"]),
                    timestamp=job["timestamp"],
                    warm_start=job["start_point"] == "current",
                    start_weights=np.asarray(_weights(job["start_weights"]), dtype=complex)
                    if job["start_weights"] is not None
                    else None,
                    auto=False,
                    record_input=False,
                )
                _wait_idle(replayed)
        active_id = replayed.store.active_version_id()
        return {
            "array": replayed.store.load_array(),
            "events": replayed.store.load_events(),
            "active_version_id": active_id,
            "active_version": replayed.store.load_version(active_id) if active_id else None,
            "jobs": replayed.jobs.list_jobs(),
        }


def _wait_idle(engine: "RadarEngine", timeout: float = 5.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(job["status"] not in {"queued", "running"} for job in engine.jobs.list_jobs()):
            return
        time.sleep(0.005)
    raise RuntimeError("replay did not finish a reconstruction")


def _weights(payload):
    return [
        float(item["magnitude"]) * np.exp(1j * np.deg2rad(float(item["phase_deg"])))
        for item in payload
    ]
