"""方向图与阵列不变量测试。"""
import numpy as np
import pytest

from app.pattern import array_factor, compute_metrics, normalized_pattern_db

N, D = 8, 0.5


def test_reference_uniform_array():
    """8 单元、半波长、等幅同相的参考指标。"""
    m = compute_metrics(np.ones(N), N, D)
    assert m.main_lobe_direction_deg == pytest.approx(0.0, abs=0.05)
    assert abs(m.first_null_deg) == pytest.approx(14.48, abs=0.05)
    assert m.max_sidelobe_db == pytest.approx(-12.80, abs=0.05)
    assert m.half_power_beamwidth_deg == pytest.approx(12.80, abs=0.1)


def test_steering_to_30deg():
    """线性相位梯度把主瓣移到 30°。"""
    x = np.arange(N) * D
    w = np.exp(-1j * 2 * np.pi * x * np.sin(np.deg2rad(30.0)))
    m = compute_metrics(w, N, D)
    assert m.main_lobe_direction_deg == pytest.approx(30.0, abs=0.05)


def test_complex_scale_invariance():
    """全体加权同乘非零复数，归一化方向图不变。"""
    rng = np.random.default_rng(7)
    w = rng.normal(size=N) + 1j * rng.normal(size=N)
    w[0] += 1.0  # 避免意外零向量
    c = 3.7 * np.exp(1j * 0.9)
    theta = np.arange(-90.0, 90.0, 0.2)
    p1 = normalized_pattern_db(array_factor(w, N, D, theta))
    p2 = normalized_pattern_db(array_factor(c * w, N, D, theta))
    np.testing.assert_allclose(p1, p2, atol=1e-10)
    m1 = compute_metrics(w, N, D)
    m2 = compute_metrics(c * w, N, D)
    assert m1.main_lobe_direction_deg == pytest.approx(m2.main_lobe_direction_deg, abs=1e-6)
    assert m1.max_sidelobe_db == pytest.approx(m2.max_sidelobe_db, abs=1e-6)
    assert m1.half_power_beamwidth_deg == pytest.approx(m2.half_power_beamwidth_deg, abs=1e-6)


def test_mirror_symmetry():
    """加权序列整体反转，方向图关于法线镜像。"""
    rng = np.random.default_rng(3)
    w = rng.normal(size=N) + 1j * rng.normal(size=N)
    w_rev = w[::-1]
    theta = np.arange(-89.9, 90.0, 0.2)
    p = normalized_pattern_db(array_factor(w, N, D, theta))
    p_rev = normalized_pattern_db(array_factor(w_rev, N, D, -theta))
    np.testing.assert_allclose(p, p_rev, atol=1e-10)


def test_linear_phase_gradient_steering():
    """等幅阵列叠加线性相位梯度，主瓣移到对应方向。"""
    for angle in (10.0, -25.0, 45.0):
        x = np.arange(N) * D
        w = np.exp(-1j * 2 * np.pi * x * np.sin(np.deg2rad(angle)))
        m = compute_metrics(w, N, D)
        assert m.main_lobe_direction_deg == pytest.approx(angle, abs=0.05)


def test_element_failure_raises_sidelobe():
    """4 号单元（1 基）失效后最高副瓣抬到约 -8.2 dB。"""
    health = np.ones(N, dtype=complex)
    health[3] = 0.0
    m = compute_metrics(np.ones(N), N, D, health=health)
    assert m.max_sidelobe_db == pytest.approx(-8.2, abs=0.3)
