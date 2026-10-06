"""阵列模型与方向图计算。

约定：
- 均匀线阵，单元位于 x_n = n * d（n = 0..N-1），d 以波长为单位。
- 角度 theta 为偏离阵列法线的角度，单位度，范围 [-90, 90]。
- 阵因子 AF(theta) = sum_n w_n * exp(j * 2pi * x_n * sin(theta))。
- 健康因子 health[n] 为复数：失效单元为 0，幅度/相位误差折算为复增益。
  实际方向图用 w_n * health[n] 计算。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi
# 半功率点电平（相对主瓣峰值）
HALF_POWER_DB = -3.010299956639812
# 判定"深零点"的阈值，用于主瓣区域划分
NULL_THRESHOLD_DB = -20.0
# 指标计算的角度网格步长（度）
METRIC_GRID_STEP_DEG = 0.01


def steering_matrix(n_elements: int, spacing_wl: float, theta_deg) -> np.ndarray:
    """导向矩阵 A，A[g, n] = exp(j*2pi*x_n*sin(theta_g))。"""
    theta = np.asarray(theta_deg, dtype=float).reshape(-1)
    x = np.arange(n_elements, dtype=float) * float(spacing_wl)
    phase = TWO_PI * np.sin(np.deg2rad(theta))[:, None] * x[None, :]
    return np.exp(1j * phase)


def array_factor(
    weights,
    n_elements: int,
    spacing_wl: float,
    theta_deg,
    health=None,
) -> np.ndarray:
    """任意方向上的复阵因子。health 为每单元复健康因子（失效为 0）。"""
    A = steering_matrix(n_elements, spacing_wl, theta_deg)
    w = np.asarray(weights, dtype=complex).reshape(-1)
    if health is not None:
        w = w * np.asarray(health, dtype=complex).reshape(-1)
    return A @ w


def normalized_pattern_db(af) -> np.ndarray:
    """归一化方向图（dB），峰值归一到 0 dB。"""
    mag = np.abs(np.asarray(af, dtype=complex))
    peak = float(mag.max()) if mag.size else 0.0
    if peak <= 0.0:
        return np.full(mag.shape, -300.0)
    return 20.0 * np.log10(np.maximum(mag, peak * 1e-15) / peak)


@dataclass
class PatternMetrics:
    main_lobe_direction_deg: float
    half_power_beamwidth_deg: float | None
    first_null_deg: float | None  # 距主瓣最近的零点（带符号角度）
    max_sidelobe_db: float | None

    def as_dict(self) -> dict:
        return {
            "main_lobe_direction_deg": self.main_lobe_direction_deg,
            "half_power_beamwidth_deg": self.half_power_beamwidth_deg,
            "first_null_deg": self.first_null_deg,
            "max_sidelobe_db": self.max_sidelobe_db,
        }


def _parabolic_offset(ym: float, y0: float, yp: float) -> float:
    """三点抛物线插值的极值位置偏移（以采样间隔为单位）。"""
    denom = ym - 2.0 * y0 + yp
    if abs(denom) < 1e-18:
        return 0.0
    return float(np.clip(0.5 * (ym - yp) / denom, -1.0, 1.0))


def compute_metrics(
    weights,
    n_elements: int,
    spacing_wl: float,
    health=None,
    step_deg: float = METRIC_GRID_STEP_DEG,
) -> PatternMetrics:
    """计算主瓣指向、半功率波束宽度、第一零点和最高副瓣电平。

    主瓣区域由主瓣两侧最近的深零点（低于 NULL_THRESHOLD_DB 的局部极小）
    界定；若无深零点则退化为 -20 dB 穿越点。副瓣电平在主瓣区域外取最大。
    关键位置均用抛物线插值/二分加精确求值细化，避免网格量化误差。
    """
    theta = np.arange(-90.0, 90.0 + 0.5 * step_deg, step_deg)
    af = array_factor(weights, n_elements, spacing_wl, theta, health)
    mag = np.abs(af)
    pk = int(np.argmax(mag))
    peak_grid = mag[pk]
    if peak_grid <= 0.0:
        raise ValueError("方向图恒为零（所有有效加权均为零）")

    w = np.asarray(weights, dtype=complex).reshape(-1)
    h = None if health is None else np.asarray(health, dtype=complex).reshape(-1)

    def exact_mag(t: float) -> float:
        ww = w if h is None else w * h
        x = np.arange(n_elements) * spacing_wl
        return float(abs(np.sum(ww * np.exp(1j * TWO_PI * np.sin(np.deg2rad(t)) * x))))

    # 主瓣指向：抛物线细化后精确求峰值
    if 0 < pk < len(theta) - 1:
        off = _parabolic_offset(mag[pk - 1], mag[pk], mag[pk + 1])
        main_dir = float(theta[pk] + off * step_deg)
    else:
        main_dir = float(theta[pk])
    peak_exact = max(exact_mag(main_dir), peak_grid)

    def exact_db(t: float) -> float:
        return 20.0 * np.log10(max(exact_mag(t), peak_exact * 1e-15) / peak_exact)

    db = 20.0 * np.log10(np.maximum(mag, peak_grid * 1e-15) / peak_grid)

    def find_crossing(level: float, start: int, direction: int):
        """从 start 沿 direction 找 db 跌破 level 的位置，二分细化。"""
        i = start
        while True:
            j = i + direction
            if j < 0 or j >= len(theta):
                return None
            if db[j] < level:
                lo, hi = float(theta[i]), float(theta[j])
                for _ in range(60):
                    mid = 0.5 * (lo + hi)
                    if exact_db(mid) >= level:
                        lo = mid
                    else:
                        hi = mid
                return 0.5 * (lo + hi)
            i = j

    # 半功率波束宽度
    left_hp = find_crossing(HALF_POWER_DB, pk, -1)
    right_hp = find_crossing(HALF_POWER_DB, pk, +1)
    hpbw = (right_hp - left_hp) if (left_hp is not None and right_hp is not None) else None

    # 主瓣两侧最近的局部极小作为主瓣边缘。失效单元会填平零点，
    # 因此边缘判定不苛求深度；"第一零点"仍只报告足够深的真零点。
    is_min = np.zeros(len(theta), dtype=bool)
    is_min[1:-1] = (db[1:-1] <= db[:-2]) & (db[1:-1] < db[2:])
    minima = np.where(is_min & (db < -6.0))[0]
    left_minima = minima[minima < pk]
    right_minima = minima[minima > pk]

    def refine_null(idx: int) -> float:
        off = _parabolic_offset(mag[idx - 1], mag[idx], mag[idx + 1])
        return float(theta[idx] + off * step_deg)

    left_edge = right_edge = None
    null_angles: list[float] = []
    if left_minima.size:
        idx = int(left_minima[-1])
        left_edge = refine_null(idx)
        if db[idx] < NULL_THRESHOLD_DB:
            null_angles.append(left_edge)
    if right_minima.size:
        idx = int(right_minima[0])
        right_edge = refine_null(idx)
        if db[idx] < NULL_THRESHOLD_DB:
            null_angles.append(right_edge)
    # 退化情形：连局部极小都没有，用 -20 dB 穿越点界定主瓣
    if left_edge is None:
        left_edge = find_crossing(NULL_THRESHOLD_DB, pk, -1)
    if right_edge is None:
        right_edge = find_crossing(NULL_THRESHOLD_DB, pk, +1)

    first_null = None
    if null_angles:
        first_null = min(null_angles, key=lambda a: abs(a - main_dir))

    # 最高副瓣：主瓣区域外的最大归一化电平
    max_sll = None
    region = np.zeros(len(theta), dtype=bool)
    if left_edge is not None:
        region |= theta < left_edge - 1e-9
    if right_edge is not None:
        region |= theta > right_edge + 1e-9
    if region.any():
        idx = int(np.argmax(np.where(region, db, -np.inf)))
        if 0 < idx < len(theta) - 1:
            off = _parabolic_offset(db[idx - 1], db[idx], db[idx + 1])
            sll_theta = float(theta[idx] + off * step_deg)
        else:
            sll_theta = float(theta[idx])
        max_sll = exact_db(sll_theta)

    return PatternMetrics(
        main_lobe_direction_deg=main_dir,
        half_power_beamwidth_deg=hpbw,
        first_null_deg=first_null,
        max_sidelobe_db=max_sll,
    )
