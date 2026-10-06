"""Data contracts and validation for the radar service."""
from __future__ import annotations

import math
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return value


class ComplexWeight(BaseModel):
    magnitude: float = Field(..., ge=0.0)
    phase_deg: float

    @field_validator("magnitude", "phase_deg")
    @classmethod
    def _finite(cls, value: float) -> float:
        return _finite(value)


class ArrayDefinition(BaseModel):
    element_count: int = Field(..., ge=2)
    spacing_wavelengths: float = Field(..., gt=0.0)
    weights: list[ComplexWeight]
    beam_direction_deg: float = Field(..., ge=-90.0, le=90.0)

    @field_validator("spacing_wavelengths")
    @classmethod
    def _positive_finite(cls, value: float) -> float:
        return _finite(value)

    @field_validator("beam_direction_deg")
    @classmethod
    def _direction_finite(cls, value: float) -> float:
        return _finite(value)

    @model_validator(mode="after")
    def _check_weights(self) -> "ArrayDefinition":
        if len(self.weights) != self.element_count:
            raise ValueError("weights length must equal element_count")
        return self

    @property
    def complex_weights(self) -> np.ndarray:
        phases = np.deg2rad(np.asarray([w.phase_deg for w in self.weights], dtype=float))
        magnitudes = np.asarray([w.magnitude for w in self.weights], dtype=float)
        return magnitudes * np.exp(1j * phases)


class EventType(str, Enum):
    failed = "failed"
    gain_error = "gain_error"
    phase_error = "phase_error"
    recovered = "recovered"


class HealthEvent(BaseModel):
    event_id: str
    timestamp: float
    element: int = Field(..., ge=1)
    kind: EventType
    gain_error_db: float = 0.0
    phase_error_deg: float = 0.0

    @field_validator("timestamp", "gain_error_db", "phase_error_deg")
    @classmethod
    def _finite_values(cls, value: float) -> float:
        return _finite(value)

    @model_validator(mode="after")
    def _check_payload(self) -> "HealthEvent":
        if self.kind == EventType.gain_error and self.gain_error_db == 0.0:
            raise ValueError("gain_error event requires a non-zero gain_error_db")
        if self.kind == EventType.phase_error and self.phase_error_deg == 0.0:
            raise ValueError("phase_error event requires a non-zero phase_error_deg")
        return self


class SidelobeInterval(BaseModel):
    start_deg: float
    end_deg: float
    upper_limit_db: float

    @field_validator("start_deg", "end_deg", "upper_limit_db")
    @classmethod
    def _finite_values(cls, value: float) -> float:
        return _finite(value)

    @model_validator(mode="after")
    def _check_interval(self) -> "SidelobeInterval":
        if not (-90.0 <= self.start_deg < self.end_deg <= 90.0):
            raise ValueError("interval must satisfy -90 <= start < end <= 90")
        return self


class ReconstructionTemplate(BaseModel):
    intervals: list[SidelobeInterval]
    main_direction_deg: float = Field(..., ge=-90.0, le=90.0)
    allowed_mainlobe_gain_loss_db: float = Field(..., ge=0.0)
    angle_step_deg: float = Field(0.1, gt=0.0, le=5.0)
    max_iterations: int = Field(300, ge=1, le=5000)
    convergence_tolerance: float = Field(1e-5, gt=0.0, lt=1.0)
    warm_start: bool = True

    @field_validator("main_direction_deg", "allowed_mainlobe_gain_loss_db")
    @classmethod
    def _finite_values(cls, value: float) -> float:
        return _finite(value)

    @model_validator(mode="after")
    def _check_intervals(self) -> "ReconstructionTemplate":
        if not self.intervals:
            raise ValueError("template must contain at least one interval")
        ordered = sorted(self.intervals, key=lambda item: (item.start_deg, item.end_deg))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_deg < previous.end_deg:
                raise ValueError("template intervals must not overlap")
        return self


class JobRequest(BaseModel):
    template: ReconstructionTemplate
    timestamp: float | None = None
    warm_start: bool | None = None


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"
    superseded = "superseded"


StartPoint = Literal["current", "uniform"]


def public_weight(weight: complex) -> dict[str, float]:
    return {"magnitude": float(abs(weight)), "phase_deg": float(np.rad2deg(np.angle(weight)))}


def public_complex_array(values: np.ndarray) -> list[dict[str, float]]:
    return [public_weight(value) for value in values]
