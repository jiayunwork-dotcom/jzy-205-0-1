import numpy as np
import pytest

from app.array import ArrayModel
from app.models import ArrayDefinition
from tests.conftest import uniform_array


def test_reference_uniform_pattern_metrics():
    model = ArrayModel.from_definition(uniform_array())
    pattern = model.pattern(model.nominal_weights, step_deg=0.01)
    assert pattern.main_direction_deg == pytest.approx(0.0, abs=1e-9)
    assert pattern.first_null_offset_deg == pytest.approx(14.48, abs=0.02)
    assert pattern.peak_sidelobe_db == pytest.approx(-12.80, abs=0.02)
    assert pattern.half_power_beamwidth_deg == pytest.approx(12.80, abs=0.03)


def test_linear_phase_steers_beam():
    definition = uniform_array(beam=30.0)
    model = ArrayModel.from_definition(definition)
    pattern = model.pattern(model.beam_direction_weights(), step_deg=0.01)
    assert pattern.main_direction_deg == pytest.approx(30.0, abs=0.01)


def test_complex_scaling_leaves_normalized_pattern_invariant():
    model = ArrayModel.from_definition(uniform_array())
    base = model.pattern(model.nominal_weights, step_deg=0.05)
    scaled = model.pattern((2.0 - 3.0j) * model.nominal_weights, step_deg=0.05)
    np.testing.assert_allclose(scaled.normalized, base.normalized, rtol=0, atol=1e-12)


def test_reversing_weights_mirrors_pattern_about_boresight():
    model = ArrayModel.from_definition(uniform_array())
    base = model.pattern(np.array([1, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 1], dtype=complex), 0.05)
    weights = np.array([1, 0.8, 1.2, 0.9, 1.1, 0.7, 1.3, 1], dtype=complex)
    reversed_pattern = model.pattern(weights[::-1], 0.05)
    np.testing.assert_allclose(reversed_pattern.normalized, base.normalized[::-1], rtol=0, atol=1e-12)
