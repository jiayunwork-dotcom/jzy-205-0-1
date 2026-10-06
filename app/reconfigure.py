"""加权重构求解器。

问题表述（凸可行性问题）
------------------------
记 a(theta) 为计入健康因子后的有效导向矢量（失效单元对应分量为 0），
w 为待求复加权，theta0 为主瓣指向。由于全体加权同乘非零复数不改变归一化
方向图，可不失一般性地固定 AF(theta0) 的相位，令

    s = Re(AF(theta0)),  Im(AF(theta0)) = 0,  s >= g_min

则副瓣模板约束 |AF(theta)| <= m(theta) * s 是关于 w 的凸约束（二阶锥），
失效单元 w_n = 0 与幅度上限 |w_n| <= cap 也是凸约束。因此"满足模板的
加权"是一个凸可行性问题，不存在局部极小陷阱，这是选择下述投影梯度法
而不是启发式搜索的理由。

求解方法
--------
不调用现成优化库，自行实现 FISTA（带回溯线搜索与自适应重启的加速投影
梯度法），最小化凸罚函数

    P(w) = mean_theta max(|AF(theta)| - m(theta)*s, 0)^2
           + WG * max(g_min - s, 0)^2 + WP * Im(AF(theta0))^2

P 光滑、凸，梯度有解析式；每次迭代把迭代点投影回约束集（失效单元置零、
幅度截断）。模板可满足当且仅当 min P = 0。

终止条件（三者先到为准，绝不无限迭代）
--------------------------------------
1. 收敛：归一化方向图的最大模板违反量与主瓣增益亏量均 <= tolerance_db；
2. 停滞：连续 stall_window 次评估最优评分无可见改善；
3. 达到 max_iterations 上限。
无论以何种方式停止，都如实返回迄今最优加权、违反最严重的角度与超出量。
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .pattern import steering_matrix

# 求解器内部副瓣网格步长（度）
SOLVER_GRID_STEP_DEG = 0.25
# 主瓣增益与相位罚权重（相对副瓣罚）
GAIN_PENALTY_WEIGHT = 50.0
PHASE_PENALTY_WEIGHT = 50.0


@dataclass
class MaskSector:
    """副瓣模板的一个角度区间：theta_min..theta_max 内电平不超过 max_sll_db。"""

    theta_min: float
    theta_max: float
    max_sll_db: float

    def as_dict(self) -> dict:
        return {
            "theta_min": self.theta_min,
            "theta_max": self.theta_max,
            "max_sll_db": self.max_sll_db,
        }

    @staticmethod
    def from_dict(d: dict) -> "MaskSector":
        return MaskSector(float(d["theta_min"]), float(d["theta_max"]), float(d["max_sll_db"]))


def template_grid(
    sectors: list[MaskSector], step_deg: float = SOLVER_GRID_STEP_DEG
) -> tuple[np.ndarray, np.ndarray]:
    """把模板展开为 (角度网格, 各点上限 dB)。"""
    angles: list[np.ndarray] = []
    for s in sectors:
        n = int(np.floor((s.theta_max - s.theta_min) / step_deg)) + 1
        angles.append(s.theta_min + step_deg * np.arange(n))
    theta = np.concatenate(angles) if angles else np.zeros(0)
    mask = np.zeros(theta.shape)
    for s in sectors:
        in_sector = (theta >= s.theta_min - 1e-9) & (theta <= s.theta_max + 1e-9)
        mask[in_sector] = s.max_sll_db
    return theta, mask


def evaluate_template(
    weights,
    health,
    n_elements: int,
    spacing_wl: float,
    pointing_deg: float,
    sectors: list[MaskSector],
    step_deg: float = SOLVER_GRID_STEP_DEG,
) -> tuple[float, float | None]:
    """评估加权在给定健康状态下对模板的违反情况。

    返回 (最大超出量 dB, 违反最严重的角度)。电平以指向处增益归一。
    """
    theta, mask_db = template_grid(sectors, step_deg)
    if theta.size == 0:
        return 0.0, None
    w = np.asarray(weights, dtype=complex) * np.asarray(health, dtype=complex)
    af = steering_matrix(n_elements, spacing_wl, theta) @ w
    ref = abs(steering_matrix(n_elements, spacing_wl, [pointing_deg]) @ w)[0]
    if ref <= 0.0:
        return float("inf"), None
    rel_db = 20.0 * np.log10(np.maximum(np.abs(af), ref * 1e-15) / ref)
    excess = rel_db - mask_db
    i = int(np.argmax(excess))
    return float(excess[i]), float(theta[i])


@dataclass
class SolveConfig:
    n_elements: int
    spacing_wl: float
    pointing_deg: float
    sectors: list[MaskSector]
    max_mainlobe_loss_db: float
    amplitude_cap: float
    reference_gain: float  # 健康阵列定义加权在指向处的增益，作为损失基准
    health: np.ndarray  # 当前健康因子（失效单元为 0）
    start_weights: np.ndarray
    max_iterations: int = 5000
    tolerance_db: float = 0.02
    stall_window: int = 300
    iteration_delay_ms: float = 0.0  # 仅用于演示/测试，拖慢迭代便于观察
    on_progress: object = None  # 回调 (iteration, max_violation_db)
    should_cancel: object = None  # 回调 () -> bool


@dataclass
class SolveResult:
    weights: np.ndarray
    feasible: bool
    iterations: int
    stop_reason: str  # converged | stalled | max_iterations | cancelled
    max_violation_db: float
    worst_angle_deg: float | None
    worst_excess_db: float
    mainlobe_loss_db: float
    penalty: float


def uniform_start_weights(
    n_elements: int, spacing_wl: float, pointing_deg: float, amplitude: float, health
) -> np.ndarray:
    """等幅加权加线性相位梯度（冷启动起点），失效单元置零。"""
    x = np.arange(n_elements) * spacing_wl
    w = amplitude * np.exp(-1j * 2.0 * np.pi * x * np.sin(np.deg2rad(pointing_deg)))
    w = w.astype(complex)
    w[np.asarray(health, dtype=complex) == 0.0] = 0.0
    return w


def solve(cfg: SolveConfig) -> SolveResult:
    n = cfg.n_elements
    health = np.asarray(cfg.health, dtype=complex).reshape(-1)
    failed = health == 0.0

    theta, mask_db = template_grid(cfg.sectors)
    if theta.size == 0:
        raise ValueError("副瓣模板为空")
    m_lin = 10.0 ** (mask_db / 20.0)
    A = steering_matrix(n, cfg.spacing_wl, theta) * health[None, :]
    a0 = (steering_matrix(n, cfg.spacing_wl, [cfg.pointing_deg])[0]) * health
    g_min = cfg.reference_gain * 10.0 ** (-cfg.max_mainlobe_loss_db / 20.0)
    G = float(theta.size)
    WG = GAIN_PENALTY_WEIGHT
    WP = PHASE_PENALTY_WEIGHT

    def project(w: np.ndarray) -> np.ndarray:
        w = w.copy()
        w[failed] = 0.0
        mag = np.abs(w)
        over = mag > cfg.amplitude_cap
        w[over] *= cfg.amplitude_cap / mag[over]
        return w

    def penalty_grad(w: np.ndarray) -> tuple[float, np.ndarray]:
        af = A @ w
        s0 = complex(a0 @ w)
        s, p = s0.real, s0.imag
        mag = np.abs(af)
        v = mag - m_lin * s
        act = v > 0.0
        P = float(np.sum(np.where(act, v, 0.0) ** 2) / G)
        if s < g_min:
            P += WG * (g_min - s) ** 2
        P += WP * p * p
        safe = np.where(mag > 1e-12, mag, 1e-12)
        coef = np.where(act, v / safe, 0.0)
        g = A.conj().T @ (coef * af) / G
        g -= np.conj(a0) * float(np.sum(np.where(act, v, 0.0) * m_lin)) / G
        if s < g_min:
            g -= WG * (g_min - s) * np.conj(a0)
        g += WP * p * 1j * np.conj(a0)
        return P, g

    def evaluate(w: np.ndarray) -> tuple[float, float, float | None, float]:
        """返回 (评分, 最大违反 dB, 最坏角度, 主瓣损失 dB)。评分为违反与增益亏量的较大者。"""
        af = A @ w
        s0 = abs(complex(a0 @ w))
        if s0 <= 1e-15:
            return float("inf"), float("inf"), None, float("inf")
        rel_db = 20.0 * np.log10(np.maximum(np.abs(af), s0 * 1e-15) / s0)
        excess = rel_db - mask_db
        i = int(np.argmax(excess))
        viol_db = float(excess[i])
        shortfall_db = 20.0 * float(np.log10(g_min / s0)) if s0 < g_min else 0.0
        loss_db = 20.0 * float(np.log10(s0 / cfg.reference_gain))
        return max(viol_db, shortfall_db), viol_db, float(theta[i]), loss_db

    # 初始步长：罚函数梯度的 Lipschitz 上界（Frobenius 界，回溯搜索兜底）
    L = (2.0 / G) * (float(np.sum(np.abs(A) ** 2)) + float(np.sum(m_lin**2)) * float(np.sum(np.abs(a0) ** 2)))
    L += 2.0 * (WG + WP) * float(np.sum(np.abs(a0) ** 2))
    mu = 1.0 / max(L, 1e-12)

    w = project(np.asarray(cfg.start_weights, dtype=complex).reshape(-1))
    y = w.copy()
    t_mom = 1.0
    P_w, _ = penalty_grad(w)

    def grad_step(y: np.ndarray, P_y: float, g: np.ndarray, mu: float):
        """回溯线搜索的投影梯度步，返回 (w_cand, P_cand, mu)。"""
        for _ in range(40):
            w_cand = project(y - mu * g)
            diff = w_cand - y
            P_cand, _ = penalty_grad(w_cand)
            sufficient = P_y + float(np.real(np.vdot(g, diff))) + float(np.vdot(diff, diff).real) / (2.0 * mu)
            if P_cand <= sufficient:
                return w_cand, P_cand, mu
            mu *= 0.5
        return w_cand, P_cand, mu

    best_w = w.copy()
    best_score, best_viol, best_angle, best_loss = evaluate(w)
    stall_count = 0
    stop_reason = "max_iterations"
    iterations = 0
    delay = cfg.iteration_delay_ms / 1000.0

    for k in range(1, cfg.max_iterations + 1):
        if cfg.should_cancel is not None and cfg.should_cancel():
            stop_reason = "cancelled"
            break
        if delay > 0.0:
            time.sleep(delay)
        iterations = k

        P_y, g = penalty_grad(y)
        w_cand, P_cand, mu = grad_step(y, P_y, g, mu)
        if P_cand > P_w:
            # 动量帮倒忙：自适应重启，从当前最优点重新走梯度步
            t_mom = 1.0
            y = w.copy()
            P_y, g = penalty_grad(y)
            w_cand, P_cand, mu = grad_step(y, P_y, g, mu)
        # FISTA 动量
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t_mom * t_mom))
        y = project(w_cand + ((t_mom - 1.0) / t_new) * (w_cand - w))
        w, P_w, t_mom = w_cand, P_cand, t_new

        if k % 10 == 0 or k == cfg.max_iterations:
            score, viol, angle, loss = evaluate(w)
            if score < best_score - 1e-9:
                best_score, best_viol, best_angle, best_loss = score, viol, angle, loss
                best_w = w.copy()
                stall_count = 0
            else:
                stall_count += 1
            if cfg.on_progress is not None:
                cfg.on_progress(k, viol)
            if best_score <= cfg.tolerance_db:
                stop_reason = "converged"
                break
            if stall_count >= cfg.stall_window:
                stop_reason = "stalled"
                break

    feasible = best_score <= cfg.tolerance_db
    final_pen, _ = penalty_grad(best_w)
    return SolveResult(
        weights=best_w,
        feasible=feasible,
        iterations=iterations,
        stop_reason=stop_reason,
        max_violation_db=best_viol,
        worst_angle_deg=best_angle,
        worst_excess_db=max(best_viol, 0.0),
        mainlobe_loss_db=best_loss,
        penalty=final_pen,
    )
