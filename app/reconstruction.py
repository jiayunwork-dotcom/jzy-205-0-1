"""Numerical sidelobe-template reconstruction.

The problem is convex over complex weights:
  minimize a small smooth regularization,
  subject to |w^H a(theta)| <= L(theta) on template angles,
             Re(w^H a0) >= Gmin,
             |w_i| <= c_i and w_i=0 for failed elements.

No external optimizer is used. We solve its differentiable exact-penalty form with
projected Adam and continue the penalty coefficient. The disk projection enforces
the per-element amplitude cap exactly, so failed channels remain exactly zero on
every returned iterate, not merely approximately.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ReconstructionResult:
    weights: np.ndarray
    iterations: int
    converged: bool
    feasible: bool
    status: str
    max_violation: float
    mainlobe_gain_loss_db: float
    peak_template_response_db: float
    worst_angle_deg: float | None
    worst_excess_db: float
    progress: list[dict[str, float | int]]
    start_point: str

    @property
    def success(self) -> bool:
        return self.converged and self.feasible


def _steering_matrix(angles_deg: np.ndarray, element_count: int, spacing: float) -> np.ndarray:
    n = np.arange(element_count)
    sine = np.sin(np.deg2rad(angles_deg))
    return np.exp(1j * 2.0 * np.pi * spacing * np.multiply.outer(sine, n))


def evaluate_constraints(
    weights: np.ndarray,
    steering: np.ndarray,
    mask: np.ndarray,
    limits: np.ndarray,
    steer0: np.ndarray,
    minimum_gain: float,
) -> dict[str, Any]:
    responses = steering @ weights
    magnitude = np.abs(responses)
    excess = magnitude - limits
    side_violation = float(np.max(np.where(mask, excess, -np.inf)))
    gain = float(np.real(np.vdot(steer0, weights)))
    gain_violation = minimum_gain - gain
    violation = max(0.0, side_violation, gain_violation)
    if np.isfinite(side_violation) and side_violation > gain_violation:
        index = int(np.argmax(np.where(mask, excess, -np.inf)))
        worst_angle = float(_angles_from_matrix(index, steering))
        excess_db = float(20.0 * np.log10(max(magnitude[index], np.finfo(float).tiny) / max(limits[index], np.finfo(float).tiny)))
    elif gain_violation > 0.0:
        worst_angle = None
        excess_db = float(20.0 * np.log10(minimum_gain / max(abs(gain), np.finfo(float).tiny)))
    else:
        worst_angle = None
        excess_db = 0.0
    return {
        "responses": responses,
        "magnitude": magnitude,
        "gain": gain,
        "violation": float(violation),
        "side_violation": side_violation,
        "gain_violation": float(gain_violation),
        "worst_angle": worst_angle,
        "worst_excess_db": excess_db,
    }


def _angles_from_matrix(index: int, steering: np.ndarray) -> float:
    # The solver normally passes the grid explicitly; this fallback is only for a
    # defensive path where an angle is reconstructed from the first two elements.
    phase = float(np.angle(steering[index, 1]))
    spacing_phase = float(np.angle(steering[index, 1] / steering[index, 0]))
    sine = spacing_phase / (2.0 * np.pi * 0.5)
    sine = np.clip(sine, -1.0, 1.0)
    _ = phase
    return float(np.rad2deg(np.arcsin(sine)))


def project_amplitude(weights: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Projection onto zero and per-element disks |w_i| <= c_i."""
    magnitude = np.abs(weights)
    scale = np.minimum(1.0, caps / np.maximum(magnitude, np.finfo(float).tiny))
    result = weights * scale
    result[~np.isfinite(caps) | (caps == 0.0)] = 0.0
    return result


def reconstruct_weights(
    *,
    nominal_weights: np.ndarray,
    actual_multipliers: np.ndarray,
    spacing: float,
    template,
    start_weights: np.ndarray | None = None,
    start_point: str = "current",
    cancellation_token: Any = None,
) -> ReconstructionResult:
    nominal = np.asarray(nominal_weights, dtype=complex)
    multipliers = np.asarray(actual_multipliers, dtype=complex)
    element_count = nominal.size
    angles = np.arange(-90.0, 90.0 + template.angle_step_deg / 2.0, template.angle_step_deg)
    # Commanded steering matrix and actual channel gains are combined once.
    commanded = _steering_matrix(angles, element_count, spacing)
    steering = commanded * multipliers[None, :]
    steer0_nominal = np.exp(
        -1j
        * 2.0
        * np.pi
        * spacing
        * np.sin(np.deg2rad(template.main_direction_deg))
        * np.arange(element_count)
    )
    steer0 = steer0_nominal * multipliers
    # Nominal coherent gain at the requested direction. Magnitudes preserve the
    # defined taper while the steering vector supplies the required linear phase.
    nominal_aligned = np.abs(nominal) * steer0_nominal
    nominal_gain = float(abs(np.sum(nominal_aligned)))
    if nominal_gain <= np.finfo(float).eps:
        raise ValueError("nominal mainlobe gain is zero; cannot apply a gain-loss limit")
    minimum_gain = nominal_gain * float(np.power(10.0, -template.allowed_mainlobe_gain_loss_db / 20.0))
    mask = np.zeros_like(angles, dtype=bool)
    limits = np.full_like(angles, np.inf, dtype=float)
    for interval in template.intervals:
        local = (angles >= interval.start_deg) & (angles <= interval.end_deg)
        # Template dB is relative to the nominal mainlobe peak.
        absolute_limit = nominal_gain * float(np.power(10.0, interval.upper_limit_db / 20.0))
        limits[local] = np.minimum(limits[local], absolute_limit)
        mask |= local
    if not np.any(mask):
        raise ValueError("template contains no sampled angles")
    caps = np.abs(nominal)
    caps[multipliers == 0.0] = 0.0
    if start_weights is None:
        initial = nominal.copy()
    else:
        initial = np.asarray(start_weights, dtype=complex).copy()
    initial = project_amplitude(initial, caps)
    # Regularization stays toward commanded nominal weights. Its gradient is tiny and
    # mainly removes drift between equally feasible solutions.
    regularize = 1e-5
    best_weights = initial.copy()
    best_violation = np.inf
    best_score = np.inf
    previous_best = np.inf
    progress: list[dict[str, float | int]] = []
    weights = initial.copy()
    m_adam = np.zeros_like(weights)
    v_adam = np.zeros_like(weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-12
    # Penalty continuation. The initial penalty is zero so the first iteration
    # yields the projection of the starting point (and immediately satisfies a
    # loose template). Later finite penalties tighten to feasibility.
    penalties = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)
    iteration = 0
    converged = False
    plateau_count = 0
    learning_rate = 0.02
    max_iterations = int(template.max_iterations)
    for penalty in penalties:
        if iteration >= max_iterations or converged:
            break
        for local_index in range(max_iterations):
            iteration += 1
            if cancellation_token is not None and cancellation_token():
                return ReconstructionResult(
                    best_weights,
                    iteration,
                    False,
                    False,
                    "cancelled",
                    best_violation if np.isfinite(best_violation) else 0.0,
                    0.0,
                    0.0,
                    None,
                    0.0,
                    progress,
                    start_point,
                )
            responses = steering @ weights
            magnitude = np.abs(responses)
            # Subgradient of ReLU(|a^H w|-L). The phase of the response selects
            # the nearest point of each constraint disk.
            active = mask & (magnitude > limits)
            side_gradient = np.zeros(element_count, dtype=complex)
            if np.any(active):
                factor = ((magnitude[active] - limits[active]) / np.maximum(magnitude[active], epsilon))
                side_gradient = (factor[:, None] * responses[active, None].conj() * steering[active, :]).sum(axis=0)
            gain = float(np.real(np.vdot(steer0, weights)))
            gain_short = minimum_gain - gain
            main_gradient = -gain_short * steer0 if gain_short > 0.0 else 0.0j
            # Wirtinger gradient of 0.5||w-nominal||^2 is (w-nominal)^*. NumPy
            # conventions make this the conjugate of the stored variable.
            objective_gradient = weights - nominal
            gradient = regularize * objective_gradient + penalty * (side_gradient + main_gradient)
            m_adam = beta1 * m_adam + (1.0 - beta1) * gradient
            v_adam = beta2 * v_adam + (1.0 - beta2) * (gradient.real**2 + gradient.imag**2)
            m_hat = m_adam / (1.0 - beta1**iteration)
            v_hat = v_adam / (1.0 - beta2**iteration)
            step = learning_rate / (np.sqrt(v_hat) + epsilon)
            weights = project_amplitude(weights - step * m_hat, caps)
            evaluation = _evaluate(weights, steering, mask, limits, steer0, minimum_gain, angles)
            violation = evaluation["violation"]
            # Prefer feasibility; among nearly equal violations, lower regularization score.
            score = float(np.linalg.norm(weights - nominal))
            if violation < best_violation - 1e-11 or (
                abs(violation - best_violation) <= max(1e-11, 1e-10 * (1.0 + abs(best_violation)))
                and score < best_score - 1e-10
            ):
                best_violation = violation
                best_score = score
                best_weights = weights.copy()
            if iteration % max(1, min(25, max_iterations // 10)) == 0 or iteration == 1:
                progress.append(
                    {
                        "iteration": iteration,
                        "violation": float(violation),
                        "gain": float(evaluation["gain"]),
                        "peak_constrained_db": float(
                            20.0
                            * np.log10(
                                max(np.max(magnitude[mask]), np.finfo(float).tiny) / nominal_gain
                            )
                        ),
                    }
                )
            tolerance = float(template.convergence_tolerance)
            if violation <= tolerance and abs(best_violation - violation) <= tolerance:
                converged = True
            if violation <= tolerance and abs(previous_best - best_violation) <= 1e-9 * max(1.0, best_violation):
                plateau_count += 1
            else:
                plateau_count = 0
            previous_best = best_violation
            if converged or iteration >= max_iterations or plateau_count >= 40:
                break
    final = _evaluate(best_weights, steering, mask, limits, steer0, minimum_gain, angles)
    feasible = final["violation"] <= float(template.convergence_tolerance)
    status = "converged" if feasible else ("cancelled" if False else "max_iterations")
    constrained_values = np.abs((commanded[mask] @ best_weights))
    actual_constrained = np.abs((steering[mask] @ best_weights))
    peak_actual = float(np.max(actual_constrained))
    return ReconstructionResult(
        best_weights,
        iteration,
        feasible,
        feasible,
        status,
        float(final["violation"]),
        float(20.0 * np.log10(max(final["gain"], np.finfo(float).tiny) / nominal_gain)),
        float(20.0 * np.log10(max(peak_actual, np.finfo(float).tiny) / nominal_gain)),
        final["worst_angle"],
        float(final["worst_excess_db"]),
        progress,
        start_point,
    )


def _evaluate(weights, steering, mask, limits, steer0, minimum_gain, angles):
    responses = steering @ weights
    magnitude = np.abs(responses)
    excess = magnitude - limits
    side_violation = float(np.max(np.where(mask, excess, -np.inf)))
    gain = float(np.real(np.vdot(steer0, weights)))
    gain_violation = minimum_gain - gain
    violation = max(0.0, side_violation, gain_violation)
    if np.isfinite(side_violation) and side_violation >= gain_violation:
        index = int(np.argmax(np.where(mask, excess, -np.inf)))
        actual_value = magnitude[index]
        worst_angle: float | None = float(angles[index])
        worst_excess_db = float(
            20.0 * np.log10(max(actual_value, np.finfo(float).tiny) / max(limits[index], np.finfo(float).tiny))
        )
    elif gain_violation > 0.0:
        worst_angle = None
        worst_excess_db = float(20.0 * np.log10(minimum_gain / max(abs(gain), np.finfo(float).tiny)))
    else:
        worst_angle = None
        worst_excess_db = 0.0
    return {
        "responses": responses,
        "magnitude": magnitude,
        "gain": gain,
        "violation": float(violation),
        "side_violation": side_violation,
        "gain_violation": float(gain_violation),
        "worst_angle": worst_angle,
        "worst_excess_db": worst_excess_db,
    }
