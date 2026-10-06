"""加权重构求解器测试。"""
import numpy as np
import pytest

from app.reconfigure import (
    MaskSector,
    SolveConfig,
    evaluate_template,
    solve,
    uniform_start_weights,
)

N, D = 8, 0.5
HEALTH_FAILED4 = np.array([1, 1, 1, 0, 1, 1, 1, 1], dtype=complex)
# 缺 4 号单元时物理可达的模板（可行性边界约 -12 dB）
SECTORS = [MaskSector(-90.0, -18.0, -10.0), MaskSector(18.0, 90.0, -10.0)]


def make_cfg(start, sectors=SECTORS, loss_db=3.0, max_iter=10000, health=HEALTH_FAILED4):
    return SolveConfig(
        n_elements=N, spacing_wl=D, pointing_deg=0.0, sectors=sectors,
        max_mainlobe_loss_db=loss_db, amplitude_cap=1.0, reference_gain=8.0,
        health=health, start_weights=start, max_iterations=max_iter,
    )


def test_reconfigure_meets_template_after_failure():
    """4 号单元失效后重构，实际方向图满足模板。"""
    start = uniform_start_weights(N, D, 0.0, 1.0, HEALTH_FAILED4)
    res = solve(make_cfg(start))
    assert res.feasible, f"未收敛: {res.stop_reason}, viol={res.max_violation_db}"
    # 独立复评：实际方向图（含健康因子）满足模板
    excess, angle = evaluate_template(res.weights, HEALTH_FAILED4, N, D, 0.0, SECTORS)
    assert excess <= 0.02
    # 失效单元加权严格为零
    assert res.weights[3] == 0.0
    # 幅度不超过上限
    assert np.abs(res.weights).max() <= 1.0 + 1e-9
    # 主瓣损失在允许范围内
    assert res.mainlobe_loss_db >= -3.0 - 1e-6


def test_both_start_points_satisfy_same_template():
    """从当前方案（warm）与等幅加权（cold）出发都能满足同一模板。"""
    cold = uniform_start_weights(N, D, 0.0, 1.0, HEALTH_FAILED4)
    warm = np.ones(N, dtype=complex)  # 当前生效方案（原等幅加权）
    results = {}
    for name, start in [("cold", cold), ("warm", warm)]:
        res = solve(make_cfg(start))
        assert res.feasible, f"{name} 起点未满足模板"
        excess, _ = evaluate_template(res.weights, HEALTH_FAILED4, N, D, 0.0, SECTORS)
        assert excess <= 0.02
        assert res.weights[3] == 0.0
        results[name] = res
    # 凸问题：两种起点收敛到同一最优（权重差异很小）
    np.testing.assert_allclose(results["cold"].weights, results["warm"].weights, atol=1e-2)


def test_iteration_cap_returns_best_honestly():
    """迭代上限极低时如实返回：不可行标记、当前最优加权、最坏角度与超出量。"""
    start = uniform_start_weights(N, D, 0.0, 1.0, HEALTH_FAILED4)
    res = solve(make_cfg(start, max_iter=5))
    assert not res.feasible
    assert res.stop_reason == "max_iterations"
    assert res.iterations == 5
    assert res.worst_angle_deg is not None
    assert res.worst_excess_db > 0.0
    # 返回的加权仍满足硬约束
    assert res.weights[3] == 0.0
    assert np.abs(res.weights).max() <= 1.0 + 1e-9


def test_infeasible_template_reported():
    """物理上不可满足的模板（-15 dB）被如实判定，不假装收敛。"""
    tight = [MaskSector(-90.0, -18.0, -15.0), MaskSector(18.0, 90.0, -15.0)]
    start = uniform_start_weights(N, D, 0.0, 1.0, HEALTH_FAILED4)
    res = solve(make_cfg(start, sectors=tight, max_iter=8000))
    assert not res.feasible
    assert res.stop_reason in ("stalled", "max_iterations")
    assert res.worst_excess_db > 0.5
    assert res.worst_angle_deg is not None


def test_failed_element_weight_strictly_zero_and_cap():
    """任意（含多个失效单元）情况下失效位严格为零、幅度受限。"""
    health = np.array([1, 0, 1, 0, 1, 1, 1, 1], dtype=complex)
    sectors = [MaskSector(-90.0, -20.0, -8.0), MaskSector(20.0, 90.0, -8.0)]
    start = uniform_start_weights(N, D, 0.0, 0.8, health)
    cfg = SolveConfig(
        n_elements=N, spacing_wl=D, pointing_deg=0.0, sectors=sectors,
        max_mainlobe_loss_db=4.0, amplitude_cap=0.8, reference_gain=8.0,
        health=health, start_weights=start, max_iterations=10000,
    )
    res = solve(cfg)
    assert res.weights[1] == 0.0 and res.weights[3] == 0.0
    assert np.abs(res.weights).max() <= 0.8 + 1e-9
