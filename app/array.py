"""Array model and deterministic array-factor calculations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArrayModel:
    element_count: int
    spacing_wavelengths: float
    nominal_weights: np.ndarray
    beam_direction_deg: float

    def __post_init__(self) -> None:
        if self.element_count < 2:
            raise ValueError("element_count must be at least 2")
        if not np.isfinite(self.spacing_wavelengths) or self.spacing_wavelengths <= 0:
            raise ValueError("spacing must be a positive finite number")
        weights = np.asarray(self.nominal_weights, dtype=complex)
        if weights.shape != (self.element_count,):
            raise ValueError("weights length must equal element_count")
        if np.any(~np.isfinite(weights.real)) or np.any(~np.isfinite(weights.imag)):
            raise ValueError("weights must be finite")
        if np.any(np.abs(weights) < 0.0):  # abs cannot be negative; retained for API contract
            raise ValueError("weight magnitudes must not be negative")
        if not np.isfinite(self.beam_direction_deg) or not -90.0 <= self.beam_direction_deg <= 90.0:
            raise ValueError("beam direction must be in [-90, 90] degrees")
        object.__setattr__(self, "nominal_weights", weights)

    @classmethod
    def from_definition(cls, definition) -> "ArrayModel":
        return cls(
            definition.element_count,
            definition.spacing_wavelengths,
            definition.complex_weights,
            definition.beam_direction_deg,
        )

    @property
    def positions(self) -> np.ndarray:
        return np.arange(self.element_count, dtype=float) * self.spacing_wavelengths

    def steering_vector(self, theta_deg: float | np.ndarray) -> np.ndarray:
        theta = np.asarray(np.deg2rad(theta_deg), dtype=float)
        sine = np.sin(theta)
        return np.exp(1j * 2.0 * np.pi * self.spacing_wavelengths * sine[..., None] * np.arange(self.element_count))

    def array_factor(self, weights: np.ndarray, theta_deg: float | np.ndarray) -> np.ndarray:
        weights = np.asarray(weights, dtype=complex)
        return np.squeeze(self.steering_vector(theta_deg) @ weights)

    def pattern_grid(self, step_deg: float = 0.02) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        angles = np.arange(-90.0, 90.0 + step_deg / 2.0, step_deg)
        factors = self.array_factor(self.nominal_weights, angles)
        return angles, factors, np.abs(factors)

    def pattern(self, weights: np.ndarray | None = None, step_deg: float = 0.02):
        weights = self.nominal_weights if weights is None else np.asarray(weights, dtype=complex)
        angles = np.arange(-90.0, 90.0 + step_deg / 2.0, step_deg)
        factors = self.array_factor(weights, angles)
        magnitude = np.abs(factors)
        peak_index = int(np.argmax(magnitude))
        peak = max(float(magnitude[peak_index]), np.finfo(float).tiny)
        normalized = magnitude / peak
        power_db = 20.0 * np.log10(np.maximum(normalized, np.finfo(float).tiny))
        return PatternResult(
            angles=angles,
            array_factor=factors,
            magnitude=magnitude,
            normalized=normalized,
            power_db=power_db,
            peak_index=peak_index,
        )

    def beam_direction_weights(self, direction_deg: float | None = None) -> np.ndarray:
        """Apply the linear phase gradient for a commanded broadside-relative beam.

        The stored magnitude taper is preserved. Stored phases describe the
        broadside design; the requested beam direction supplies the steering
        gradient.
        """
        direction = self.beam_direction_deg if direction_deg is None else direction_deg
        n = np.arange(self.element_count)
        steer = np.exp(-1j * 2.0 * np.pi * self.spacing_wavelengths * np.sin(np.deg2rad(self.beam_direction_deg)) * n)
        return np.abs(self.nominal_weights) * steer


@dataclass(frozen=True)
class PatternResult:
    angles: np.ndarray
    array_factor: np.ndarray
    magnitude: np.ndarray
    normalized: np.ndarray
    power_db: np.ndarray
    peak_index: int

    @property
    def main_direction_deg(self) -> float:
        return float(self.angles[self.peak_index])

    @property
    def half_power_beamwidth_deg(self) -> float:
        half = 1.0 / np.sqrt(2.0)
        peak = self.peak_index
        left = np.where(self.normalized[: peak + 1] >= half)[0]
        right = np.where(self.normalized[peak:] >= half)[0]
        if left.size == 0 or right.size == 0:
            return float("nan")
        step = float(self.angles[1] - self.angles[0])

        def threshold_angle(index: int, moving_left: bool) -> float:
            if moving_left:
                x0, x1 = self.normalized[index], self.normalized[index + 1]
                a0, a1 = self.angles[index], self.angles[index + 1]
            else:
                x0, x1 = self.normalized[index - 1], self.normalized[index]
                a0, a1 = self.angles[index - 1], self.angles[index]
            if x1 == x0:
                return float(a1)
            return float(a0 + (half - x0) * (a1 - a0) / (x1 - x0))

        left_index = int(left[0])
        right_index = int(peak + right[-1])
        left_angle = threshold_angle(left_index, True)
        right_angle = threshold_angle(right_index, False)
        return right_angle - left_angle

    def _first_null(self, side: int) -> float | None:
        # A zero is a local minimum well below the peak. The threshold is deliberately
        # permissive because sampled zeros may not exactly reach numerical zero.
        values = self.normalized
        start = self.peak_index
        indices = range(start + 1, len(values) - 1) if side > 0 else range(start - 1, 0, -1)
        for index in indices:
            if values[index] <= values[index - 1] and values[index] <= values[index + 1] and values[index] < 0.25:
                return float(self.angles[index])
        return None

    @property
    def first_null_offset_deg(self) -> float:
        right = self._first_null(1)
        left = self._first_null(-1)
        offsets = [abs(x - self.main_direction_deg) for x in (right, left) if x is not None]
        if not offsets:
            return float("nan")
        return float(min(offsets))

    @property
    def peak_sidelobe_db(self) -> float:
        # Exclude the mainlobe region between its first nulls when they exist.
        left = self._first_null(-1)
        right = self._first_null(1)
        if left is not None and right is not None and left < right:
            mask = np.ones_like(self.angles, dtype=bool)
            mask[(self.angles >= left) & (self.angles <= right)] = False
        else:
            mask = self.angles != self.main_direction_deg
        if not np.any(mask):
            return float("-inf")
        return float(20.0 * np.log10(max(float(np.max(self.normalized[mask])), np.finfo(float).tiny)))


def sidelobe_template_mask(angles: np.ndarray, intervals) -> np.ndarray:
    mask = np.zeros_like(angles, dtype=bool)
    for interval in intervals:
        mask |= (angles >= interval.start_deg) & (angles <= interval.end_deg)
    return mask


def sidelobe_limits(angles: np.ndarray, intervals) -> np.ndarray:
    """Return per-angle absolute magnitude limits; +inf outside template intervals."""
    limits = np.full_like(angles, np.inf, dtype=float)
    for interval in intervals:
        local = (angles >= interval.start_deg) & (angles <= interval.end_deg)
        candidate = np.power(10.0, interval.upper_limit_db / 20.0)
        limits[local] = np.minimum(limits[local], candidate)
    return limits
