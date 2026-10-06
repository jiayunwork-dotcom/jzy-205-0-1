import numpy as np
import pytest

from app.models import ArrayDefinition, ComplexWeight, SidelobeInterval
from app.storage import Store
from app.engine import RadarEngine


def weight(magnitude: float = 1.0, phase: float = 0.0) -> ComplexWeight:
    return ComplexWeight(magnitude=magnitude, phase_deg=phase)


def uniform_array(element_count: int = 8, spacing: float = 0.5, beam: float = 0.0) -> ArrayDefinition:
    return ArrayDefinition(
        element_count=element_count,
        spacing_wavelengths=spacing,
        weights=[weight() for _ in range(element_count)],
        beam_direction_deg=beam,
    )


def interval(start: float, end: float, limit_db: float) -> SidelobeInterval:
    return SidelobeInterval(start_deg=start, end_deg=end, upper_limit_db=limit_db)


def failed_element_template(limit_db: float = -8.0, max_iterations: int = 500):
    from app.models import ReconstructionTemplate

    return ReconstructionTemplate(
        intervals=[interval(-90.0, -15.0, limit_db), interval(15.0, 90.0, limit_db)],
        main_direction_deg=0.0,
        allowed_mainlobe_gain_loss_db=3.0,
        angle_step_deg=0.05,
        max_iterations=max_iterations,
        convergence_tolerance=1e-5,
    )


@pytest.fixture
def store():
    return Store()


@pytest.fixture
def engine(store):
    radar = RadarEngine(store)
    radar.define_array(uniform_array())
    return radar


@pytest.fixture
def client(engine, monkeypatch):
    from fastapi.testclient import TestClient
    import app.api as api

    monkeypatch.setattr(api, "store", engine.store)
    monkeypatch.setattr(api, "engine", engine)
    return TestClient(api.app)
