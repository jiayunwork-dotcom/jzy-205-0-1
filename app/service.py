"""Domain services: array state, pattern metrics, versions, and comparisons."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .array import ArrayModel, PatternResult
from .health import EventLog
from .models import (
    ArrayDefinition,
    EventType,
    HealthEvent,
    JobStatus,
    ReconstructionTemplate,
)
from .reconstruction import reconstruct_weights


def new_id(store, prefix: str) -> str:
    return f"{prefix}_{store.next_id(prefix):08d}"


class RadarService:
    def __init__(self, store):
        self.store = store
        self.event_log: EventLog | None = None
        self.array: ArrayModel | None = None
        self._load()

    def _load(self) -> None:
        payload = self.store.load_array()
        if payload is None:
            self.array = None
            self.event_log = None
        else:
            definition = ArrayDefinition.model_validate(payload)
            self.array = ArrayModel.from_definition(definition)
            self.event_log = EventLog(self.array.element_count)
            for event_payload in self.store.load_events():
                event = HealthEvent.model_validate(event_payload)
                self.event_log.add(event)

    def define_array(self, definition: ArrayDefinition) -> dict[str, Any]:
        self.array = ArrayModel.from_definition(definition)
        self.event_log = EventLog(self.array.element_count)
        payload = definition.model_dump(mode="json")
        self.store.save_array(payload)
        return payload

    def _ensure_array(self) -> ArrayModel:
        if self.array is None or self.event_log is None:
            raise RuntimeError("array has not been defined")
        return self.array

    def submit_event(self, event: HealthEvent) -> dict[str, Any]:
        array = self._ensure_array()
        if event.element < 1 or event.element > array.element_count:
            raise ValueError(f"event references non-existent element {event.element}")
        added = self.event_log.add(event)
        if added:
            self.store.save_event(event.model_dump(mode="json"))
        return {"added": added, "event": event.model_dump(mode="json")}

    def state_at(self, timestamp: float | None = None):
        array = self._ensure_array()
        state = self.event_log.derive_state(timestamp)
        return array, state

    def actual_weights_at(self, timestamp: float | None = None) -> np.ndarray:
        array, state = self.state_at(timestamp)
        return state.actual_weights(self.array.nominal_weights)

    def pattern_payload(
        self,
        *,
        timestamp: float | None = None,
        weights: list[dict[str, float]] | None = None,
        step_deg: float = 0.02,
    ) -> dict[str, Any]:
        array = self._ensure_array()
        if weights is not None:
            chosen_weights = _weights_from_payload(weights)
            if chosen_weights.shape != (array.element_count,):
                raise ValueError("weights length must equal element_count")
            if np.any(~np.isfinite(chosen_weights.real)) or np.any(~np.isfinite(chosen_weights.imag)):
                raise ValueError("weights must be finite")
        else:
            _, state = self.state_at(timestamp)
            chosen_weights = state.actual_weights(array.nominal_weights)
        pattern = array.pattern(chosen_weights, step_deg=step_deg)
        return serialize_pattern(pattern)

    def run_reconstruction(
        self,
        *,
        template: ReconstructionTemplate,
        timestamp: float | None = None,
        start_weights: np.ndarray | None = None,
        start_point: str,
        job_id: str,
        cancellation_token=None,
    ) -> dict[str, Any]:
        array, state = self.state_at(timestamp)
        result = reconstruct_weights(
            nominal_weights=array.nominal_weights,
            actual_multipliers=state.multipliers,
            spacing=array.spacing_wavelengths,
            template=template,
            start_weights=start_weights,
            start_point=start_point,
            cancellation_token=cancellation_token,
        )
        metrics = self.solution_metrics(result.weights, template, timestamp)
        return {
            "job_id": job_id,
            "weights": _weights_to_payload(result.weights),
            "event_ids": list(state.event_ids),
            "timestamp": state.timestamp,
            "template": template.model_dump(mode="json"),
            "iterations": result.iterations,
            "converged": result.converged,
            "feasible": result.feasible,
            "status": result.status,
            "start_point": start_point,
            "metrics": metrics,
            "worst_angle_deg": result.worst_angle_deg,
            "worst_excess_db": result.worst_excess_db,
            "progress": result.progress,
        }

    def solution_metrics(self, weights: np.ndarray, template: ReconstructionTemplate, timestamp: float | None) -> dict[str, Any]:
        array, state = self.state_at(timestamp)
        actual = state.actual_weights(np.asarray(weights, dtype=complex))
        pattern = array.pattern(actual, step_deg=min(0.02, template.angle_step_deg))
        n = np.arange(array.element_count)
        nominal_gain = float(
            np.sum(
                np.abs(array.nominal_weights)
                * np.cos(
                    2.0
                    * np.pi
                    * array.spacing_wavelengths
                    * np.sin(np.deg2rad(template.main_direction_deg))
                    * n
                )
            )
        )
        actual_peak = float(np.max(pattern.magnitude))
        responses, angles = template_response(pattern, template)
        if responses.size:
            worst_index = int(np.argmax(responses))
            worst_angle = float(angles[worst_index])
            worst_response = float(responses[worst_index])
        else:
            worst_angle = None
            worst_response = 0.0
        template_db = min(interval.upper_limit_db for interval in template.intervals)
        return {
            "main_direction_deg": pattern.main_direction_deg,
            "half_power_beamwidth_deg": pattern.half_power_beamwidth_deg,
            "first_null_offset_deg": pattern.first_null_offset_deg,
            "peak_sidelobe_db": pattern.peak_sidelobe_db,
            "mainlobe_gain_loss_db": 20.0 * math.log10(max(actual_peak, np.finfo(float).tiny) / nominal_gain),
            "worst_template_angle_deg": worst_angle,
            "worst_template_response_db": 20.0 * math.log10(max(worst_response, np.finfo(float).tiny) / nominal_gain),
            "worst_template_margin_db": template_db - (20.0 * math.log10(max(worst_response, np.finfo(float).tiny) / nominal_gain)),
        }

    def template_compliant(self, version: dict[str, Any], timestamp: float | None = None,
                         health_multipliers: np.ndarray | None = None) -> bool:
        template = ReconstructionTemplate.model_validate(version["template"])
        weights = _weights_from_payload(version["weights"])
        array, state = self.state_at(timestamp)
        multipliers = state.multipliers if health_multipliers is None else health_multipliers
        actual = multipliers * weights
        pattern = array.pattern(actual, step_deg=min(0.02, template.angle_step_deg))
        responses, _ = template_response(pattern, template)
        if not responses.size:
            return True
        n = np.arange(array.element_count)
        nominal_peak = float(
            np.sum(
                np.abs(array.nominal_weights)
                * np.cos(
                    2.0
                    * np.pi
                    * array.spacing_wavelengths
                    * np.sin(np.deg2rad(template.main_direction_deg))
                    * n
                )
            )
        )
        loss_allowed = template.allowed_mainlobe_gain_loss_db
        loss = 20.0 * math.log10(max(np.max(pattern.magnitude), np.finfo(float).tiny) / nominal_peak)
        if loss < -loss_allowed - 1e-5:
            return False
        for response, interval in zip_grouped_template_responses(pattern, template):
            limit = nominal_peak * (10.0 ** (interval.upper_limit_db / 20.0))
            if response > limit * (1.0 + 1e-7) + 1e-9:
                return False
        return True

    def compare_versions(self, left_id: str, right_id: str, timestamp: float | None = None) -> dict[str, Any]:
        left = self.store.load_version(left_id)
        right = self.store.load_version(right_id)
        if left is None or right is None:
            raise KeyError("version not found")
        template = ReconstructionTemplate.model_validate(left.get("template") or right.get("template"))
        lw = _weights_from_payload(left["weights"])
        rw = _weights_from_payload(right["weights"])
        return {
            "left_version_id": left_id,
            "right_version_id": right_id,
            "left_metrics": left.get("metrics") or self.solution_metrics(lw, template, timestamp),
            "right_metrics": right.get("metrics") or self.solution_metrics(rw, template, timestamp),
            "weight_changes": [
                {
                    "element": i + 1,
                    "magnitude_delta": float(abs(rw[i]) - abs(lw[i])),
                    "phase_delta_deg": float(_phase_delta(np.angle(lw[i]), np.angle(rw[i]))),
                    "left": {"magnitude": float(abs(lw[i])), "phase_deg": float(np.rad2deg(np.angle(lw[i])))},
                    "right": {"magnitude": float(abs(rw[i])), "phase_deg": float(np.rad2deg(np.angle(rw[i])))},
                }
                for i in range(len(lw))
            ],
        }


def zip_grouped_template_responses(pattern: PatternResult, template: ReconstructionTemplate):
    for interval in template.intervals:
        mask = (pattern.angles >= interval.start_deg) & (pattern.angles <= interval.end_deg)
        yield float(np.max(pattern.magnitude[mask]) if np.any(mask) else 0.0), interval


def template_response(pattern: PatternResult, template: ReconstructionTemplate):
    values = []
    angles = []
    for interval in template.intervals:
        mask = (pattern.angles >= interval.start_deg) & (pattern.angles <= interval.end_deg)
        if np.any(mask):
            values.append(float(np.max(pattern.magnitude[mask])))
            index = int(np.argmax(np.where(mask, pattern.magnitude, -np.inf)))
            angles.append(float(pattern.angles[index]))
    return np.asarray(values, dtype=float), np.asarray(angles, dtype=float)


def serialize_pattern(pattern: PatternResult) -> dict[str, Any]:
    return {
        "angles_deg": [float(x) for x in pattern.angles],
        "array_factor": [{"real": float(x.real), "imag": float(x.imag)} for x in pattern.array_factor],
        "magnitude": [float(x) for x in pattern.magnitude],
        "normalized_pattern": [float(x) for x in pattern.normalized],
        "normalized_db": [float(x) for x in pattern.power_db],
        "metrics": {
            "main_direction_deg": pattern.main_direction_deg,
            "half_power_beamwidth_deg": pattern.half_power_beamwidth_deg,
            "first_null_offset_deg": pattern.first_null_offset_deg,
            "peak_sidelobe_db": pattern.peak_sidelobe_db,
        },
    }


def _weights_to_payload(weights: np.ndarray) -> list[dict[str, float]]:
    return [
        {"magnitude": float(abs(weight)), "phase_deg": float(np.rad2deg(np.angle(weight)))}
        for weight in np.asarray(weights, dtype=complex)
    ]


def _weights_from_payload(payload: list[dict[str, float]]) -> np.ndarray:
    values = []
    for item in payload:
        values.append(float(item["magnitude"]) * np.exp(1j * np.deg2rad(float(item["phase_deg"]))))
    return np.asarray(values, dtype=complex)


def _phase_delta(left: float, right: float) -> float:
    return float(np.rad2deg(np.angle(np.exp(1j * (right - left)))))
