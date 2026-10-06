"""HTTP API."""
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .engine import RadarEngine
from .models import (
    ArrayDefinition,
    HealthEvent,
    JobRequest,
)
from .storage import Store

REDIS_URL = os.getenv("REDIS_URL", "")
store = Store.from_url(REDIS_URL) if REDIS_URL else Store()
engine = RadarEngine(store)
app = FastAPI(title="Ground surveillance radar array service", version="1.0.0")


class WeightOverride(BaseModel):
    weights: list[dict[str, float]] = Field(..., min_length=2)


class CompareRequest(BaseModel):
    version_ids: list[str] = Field(..., min_length=2, max_length=2)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "storage": "redis" if store.ping() else "unavailable"}


@app.put("/array")
def define_array(definition: ArrayDefinition) -> dict[str, Any]:
    try:
        return engine.define_array(definition)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/array")
def get_array() -> dict[str, Any]:
    payload = store.load_array()
    if payload is None:
        raise HTTPException(status_code=404, detail="array is not defined")
    return payload


@app.post("/events")
def submit_events(events: list[HealthEvent]) -> dict[str, Any]:
    try:
        return {"results": engine.submit_events(events)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/events")
def list_events() -> dict[str, Any]:
    return {"events": store.load_events()}


@app.get("/state")
def get_state(timestamp: float | None = Query(default=None)) -> dict[str, Any]:
    try:
        array, state = engine.service.state_at(timestamp)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "timestamp": state.timestamp,
        "event_ids": list(state.event_ids),
        "failed_elements": state.failed_elements,
        "channels": [
            {
                "element": index + 1,
                "healthy": channel.healthy,
                "gain_error_db": channel.gain_error_db,
                "phase_error_deg": channel.phase_error_deg,
                "multiplier": {"real": channel.multiplier.real, "imag": channel.multiplier.imag},
                "event_ids": list(channel.event_ids),
            }
            for index, channel in enumerate(state.channels)
        ],
    }


@app.get("/pattern")
def pattern(
    timestamp: float | None = Query(default=None),
    step_deg: float = Query(default=0.02, gt=0.0, le=1.0),
) -> dict[str, Any]:
    try:
        return engine.service.pattern_payload(timestamp=timestamp, step_deg=step_deg)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/pattern")
def pattern_from_weights(
    override: WeightOverride,
    step_deg: float = Query(default=0.02, gt=0.0, le=1.0),
) -> dict[str, Any]:
    try:
        return engine.service.pattern_payload(weights=override.weights, step_deg=step_deg)
    except (RuntimeError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reconstruction-jobs")
def create_job(request: JobRequest) -> dict[str, str]:
    try:
        job_id = engine.job(request)
        return {"job_id": job_id}
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/reconstruction-jobs")
def list_jobs() -> dict[str, Any]:
    return {"jobs": engine.jobs.list_jobs()}


@app.get("/reconstruction-jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = store.load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.post("/reconstruction-jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    job = engine.jobs.cancel(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/weight-versions")
def list_versions() -> dict[str, Any]:
    return {"versions": store.load_versions(), "active_version_id": store.active_version_id()}


@app.get("/weight-versions/{version_id}")
def get_version(version_id: str) -> dict[str, Any]:
    version = store.load_version(version_id)
    if version is None:
        raise HTTPException(status_code=404, detail="version not found")
    return version


@app.post("/weight-versions/compare")
def compare_versions(request: CompareRequest, timestamp: float | None = Query(default=None)) -> dict[str, Any]:
    try:
        return engine.service.compare_versions(request.version_ids[0], request.version_ids[1], timestamp)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="version not found") from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/replay")
def replay() -> dict[str, Any]:
    return engine.replay()
