"""应用服务层：阵列定义、事件、状态/方向图查询、重构作业、版本与回放。

一致性设计：
- 任意时刻状态由事件折叠唯一确定（见 events.py），在线与回放共用同一函数；
- 每个版本记录 valid_from（其依据事件集合的最大时刻）与 basis_event_ids，
  t 时刻的生效版本 = valid_from <= t 中 (valid_from, 序号) 最大者；
  在线"当前生效"等价于 t=+inf 时的同一规则，因此回放与在线逐项一致；
- 作业完成时若其依据的状态已过期（期间来了新事件），结果被丢弃（stale），
  新事件处理流程已经用最新状态发起了替代作业。
"""
from __future__ import annotations

import threading
import time

import numpy as np

from . import pattern as pat
from .events import ChangeType, HealthEvent, derive_state, failed_elements, health_factors
from .jobs import JobManager
from .reconfigure import (
    MaskSector,
    SolveConfig,
    evaluate_template,
    solve,
    uniform_start_weights,
)
from .storage import Storage

# 自动重构触发阈值：模板违反量超过该值（dB）才发起
AUTO_TRIGGER_TOL_DB = 0.05


class ArrayService:
    def __init__(self, storage: Storage):
        self.storage = storage
        self.lock = threading.RLock()
        self.array_def: dict | None = None
        self.events: list[HealthEvent] = []
        self.versions: dict[str, dict] = {}
        self.active_version_id: str | None = None
        self.jobs = JobManager(storage, self._run_job)
        self._restore()

    # ------------------------------------------------------------------
    # 持久化恢复
    # ------------------------------------------------------------------
    def _restore(self) -> None:
        self.array_def = self.storage.load_array()
        self.events = [HealthEvent.from_dict(d) for d in self.storage.list_events()]
        for v in self.storage.list_versions():
            self.versions[v["version_id"]] = v
        self.active_version_id = self.storage.get_active_version()
        self._epoch = self.storage.get_array_epoch() or 0
        self.jobs.abort_unfinished()
        # 重启兜底：若生效方案在当前状态下违反模板，补发自动重构
        if self.array_def is not None:
            self._maybe_auto_reconfigure(superseded_config=None)

    # ------------------------------------------------------------------
    # 阵列定义
    # ------------------------------------------------------------------
    def define_array(self, defn: dict) -> dict:
        n = int(defn["n_elements"])
        d = float(defn["spacing_wavelengths"])
        if n < 2:
            raise ValueError("单元数必须不少于 2")
        if not np.isfinite(d) or d <= 0:
            raise ValueError("间距必须为正的有限数")
        weights = [complex(float(re), float(im)) for re, im in defn["weights"]]
        if len(weights) != n:
            raise ValueError("加权数量必须与单元数一致")
        for w in weights:
            if not (np.isfinite(w.real) and np.isfinite(w.imag)):
                raise ValueError("加权必须为有限数")
        pointing = float(defn["pointing_deg"])
        if not (-90.0 <= pointing <= 90.0):
            raise ValueError("指向必须在 [-90, 90] 度内")

        with self.lock:
            self.jobs.supersede_running()
            self.storage.clear_all()
            self.events = []
            self.versions = {}
            self.active_version_id = None
            self.array_def = {
                "n_elements": n,
                "spacing_wavelengths": d,
                "weights": [[w.real, w.imag] for w in weights],
                "pointing_deg": pointing,
            }
            self.storage.save_array(self.array_def)
            self._epoch = self.storage.next_seq()
            self.storage.set_array_epoch(self._epoch)
            self._create_version(
                kind="initial", job_id=None, weights=weights,
                basis_event_ids=[], valid_from=None,
                template=None, reconfig=None, solve_info=None,
            )
        return self.array_def

    def get_array(self) -> dict:
        if self.array_def is None:
            raise LookupError("阵列尚未定义")
        return self.array_def

    def _require_array(self) -> dict:
        if self.array_def is None:
            raise LookupError("阵列尚未定义")
        return self.array_def

    # ------------------------------------------------------------------
    # 版本
    # ------------------------------------------------------------------
    def _create_version(self, kind, job_id, weights, basis_event_ids, valid_from,
                        template, reconfig, solve_info) -> dict:
        defn = self._require_array()
        seq = self.storage.next_seq()
        version_id = f"v{seq}"
        health = self._health_at(None)
        metrics = pat.compute_metrics(
            np.array(weights, dtype=complex), defn["n_elements"],
            defn["spacing_wavelengths"], health,
        )
        version = {
            "version_id": version_id,
            "seq": seq,
            "kind": kind,
            "job_id": job_id,
            "weights": [[complex(w).real, complex(w).imag] for w in weights],
            "valid_from": valid_from,
            "basis_event_ids": list(basis_event_ids),
            "template": template,
            "reconfig": reconfig,
            "metrics": metrics.as_dict(),
            "solve": solve_info,
            "created_at": time.time(),
        }
        self.versions[version_id] = version
        self.storage.save_version(version_id, version)
        # 只有满足模板的方案才生效；不可满足时仅存档
        if kind == "initial" or (solve_info or {}).get("feasible", False):
            self.active_version_id = version_id
            self.storage.set_active_version(version_id)
        return version

    def list_versions(self) -> list[dict]:
        return sorted(self.versions.values(), key=lambda v: v["seq"])

    def get_version(self, version_id: str) -> dict:
        v = self.versions.get(version_id)
        if v is None:
            raise LookupError(f"版本不存在: {version_id}")
        return v

    def _active_at(self, t: float | None) -> dict:
        if not self.versions:
            raise LookupError("阵列尚未定义")
        if t is None:
            return self.versions[self.active_version_id]
        def key(v):
            vf = v["valid_from"]
            return (vf if vf is not None else float("-inf"), v["seq"])
        candidates = [v for v in self.versions.values()
                      if v["valid_from"] is None or v["valid_from"] <= t]
        if not candidates:
            candidates = list(self.versions.values())
        return max(candidates, key=key)

    def compare_versions(self, a_id: str, b_id: str) -> dict:
        va, vb = self.get_version(a_id), self.get_version(b_id)
        elements = []
        for i, (wa, wb) in enumerate(zip(va["weights"], vb["weights"])):
            ca, cb = complex(*wa), complex(*wb)
            ma, mb = abs(ca), abs(cb)
            d_amp = None if ma == 0.0 or mb == 0.0 else 20.0 * float(np.log10(mb / ma))
            d_phase = None
            if ma > 0.0 and mb > 0.0:
                d_phase = float(np.degrees(np.angle(cb / ca)))
            elements.append({
                "element": i + 1,
                "weight_a": wa, "weight_b": wb,
                "delta_amplitude_db": d_amp,
                "delta_phase_deg": d_phase,
            })
        def summary(v):
            return {
                "version_id": v["version_id"],
                "kind": v["kind"],
                "max_sidelobe_db": v["metrics"]["max_sidelobe_db"],
                "mainlobe_loss_db": (v.get("solve") or {}).get("mainlobe_loss_db"),
                "feasible": (v.get("solve") or {}).get("feasible"),
                "valid_from": v["valid_from"],
            }
        return {"a": summary(va), "b": summary(vb), "elements": elements}

    # ------------------------------------------------------------------
    # 健康事件
    # ------------------------------------------------------------------
    def add_event(self, data: dict) -> tuple[dict, bool, dict | None]:
        """登记事件。返回 (事件记录, 是否新生效, 触发的自动作业或 None)。"""
        defn = self._require_array()
        element = int(data["element"])
        if element < 1 or element > defn["n_elements"]:
            raise ValueError(f"事件引用了不存在的单元: {element}")
        change = ChangeType(data["change_type"])
        value = data.get("value")
        if change in (ChangeType.AMPLITUDE_ERROR, ChangeType.PHASE_ERROR):
            if value is None or not np.isfinite(float(value)):
                raise ValueError("幅度/相位误差事件必须给出有限数值")
            value = float(value)
        else:
            value = None

        with self.lock:
            if self.storage.has_event(data["event_id"]):
                existing = next(e for e in self.events if e.event_id == data["event_id"])
                return existing.as_dict(), False, None
            ev = HealthEvent(
                event_id=data["event_id"],
                timestamp=float(data["timestamp"]),
                element=element - 1,
                change_type=change,
                value=value,
                arrival_seq=self.storage.next_seq(),
            )
            self.events.append(ev)
            self.storage.append_event(ev.event_id, ev.as_dict())
            auto_job = self._maybe_auto_reconfigure_locked()
        return ev.as_dict(), True, auto_job

    def list_events(self) -> list[dict]:
        return [e.as_dict() for e in self.events]

    # ------------------------------------------------------------------
    # 状态与方向图
    # ------------------------------------------------------------------
    def _health_at(self, t: float | None) -> np.ndarray:
        defn = self._require_array()
        return health_factors(derive_state(self.events, defn["n_elements"], t))

    def state_at(self, t: float | None) -> dict:
        defn = self._require_array()
        state = derive_state(self.events, defn["n_elements"], t)
        factors = health_factors(state)
        return {
            "t": t,
            "elements": [
                {
                    "element": i + 1,
                    **es.as_dict(),
                    "factor": [complex(factors[i]).real, complex(factors[i]).imag],
                }
                for i, es in enumerate(state)
            ],
            "failed_elements": [i + 1 for i in failed_elements(state)],
        }

    def pattern_at(self, t: float | None = None, version_id: str | None = None,
                   include_pattern: bool = False, step_deg: float = 1.0) -> dict:
        defn = self._require_array()
        version = self.get_version(version_id) if version_id else self._active_at(t)
        weights = np.array([complex(*w) for w in version["weights"]])
        health = self._health_at(t)
        metrics = pat.compute_metrics(
            weights, defn["n_elements"], defn["spacing_wavelengths"], health)
        out = {
            "t": t,
            "version_id": version["version_id"],
            "metrics": metrics.as_dict(),
        }
        if include_pattern:
            theta = np.arange(-90.0, 90.0 + 0.5 * step_deg, step_deg)
            af = pat.array_factor(weights, defn["n_elements"],
                                  defn["spacing_wavelengths"], theta, health)
            db = pat.normalized_pattern_db(af)
            out["pattern"] = [
                {"theta_deg": float(th), "db": float(v)} for th, v in zip(theta, db)
            ]
        return out

    def replay(self, t: float) -> dict:
        """按时刻回放：t 时刻的阵列状态、生效方案与实际方向图指标。"""
        version = self._active_at(t)
        return {
            "t": t,
            "state": self.state_at(t),
            "active_version_id": version["version_id"],
            "metrics": self.pattern_at(t=t, version_id=version["version_id"])["metrics"],
        }

    def array_factor_at(self, theta_deg: list[float], t: float | None = None,
                        version_id: str | None = None) -> dict:
        """任意方向上的复阵因子（含健康状态影响）。"""
        defn = self._require_array()
        version = self.get_version(version_id) if version_id else self._active_at(t)
        weights = np.array([complex(*w) for w in version["weights"]])
        health = self._health_at(t)
        af = pat.array_factor(weights, defn["n_elements"], defn["spacing_wavelengths"],
                              np.asarray(theta_deg, dtype=float), health)
        return {
            "t": t,
            "version_id": version["version_id"],
            "array_factor": [
                {"theta_deg": float(th), "real": float(v.real), "imag": float(v.imag)}
                for th, v in zip(theta_deg, af)
            ],
        }

    # ------------------------------------------------------------------
    # 重构作业
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_template(template: dict) -> list[MaskSector]:
        sectors = [MaskSector.from_dict(s) for s in template["sectors"]]
        if not sectors:
            raise ValueError("模板至少需要一个角度区间")
        for s in sectors:
            if not (-90.0 <= s.theta_min < s.theta_max <= 90.0):
                raise ValueError("模板角度区间必须落在 [-90, 90] 内且下限小于上限")
            if not np.isfinite(s.max_sll_db):
                raise ValueError("模板电平必须为有限数")
        ordered = sorted(sectors, key=lambda s: s.theta_min)
        for a, b in zip(ordered, ordered[1:]):
            if b.theta_min < a.theta_max:
                raise ValueError("模板角度区间相互重叠")
        return sectors

    def _reference_gain(self, pointing_deg: float) -> float:
        """健康阵列定义加权在指向处的增益，作为主瓣损失基准。"""
        defn = self._require_array()
        w0 = np.array([complex(*w) for w in defn["weights"]])
        a0 = pat.steering_matrix(defn["n_elements"], defn["spacing_wavelengths"], [pointing_deg])[0]
        return float(abs(a0 @ w0))

    def _validate_job_config(self, config: dict) -> dict:
        defn = self._require_array()
        sectors = self._validate_template(config["template"])
        pointing = float(config["pointing_deg"])
        if not (-90.0 <= pointing <= 90.0):
            raise ValueError("主瓣指向必须在 [-90, 90] 度内")
        loss = float(config["max_mainlobe_loss_db"])
        if not np.isfinite(loss) or loss < 0.0:
            raise ValueError("允许的主瓣损失不能为负")
        cap = config.get("amplitude_cap")
        if cap is None:
            cap = max(abs(complex(*w)) for w in defn["weights"]) or 1.0
        cap = float(cap)
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError("幅度上限必须为正的有限数")
        max_iter = int(config.get("max_iterations", 5000))
        if max_iter < 1:
            raise ValueError("迭代次数上限必须为正")
        start_mode = config.get("start_mode", "warm")
        if start_mode not in ("warm", "cold"):
            raise ValueError("start_mode 必须为 warm 或 cold")
        return {
            "template": {"sectors": [s.as_dict() for s in sectors]},
            "pointing_deg": pointing,
            "max_mainlobe_loss_db": loss,
            "amplitude_cap": cap,
            "start_mode": start_mode,
            "max_iterations": max_iter,
            "tolerance_db": float(config.get("tolerance_db", 0.02)),
            "iteration_delay_ms": float(config.get("iteration_delay_ms", 0.0)),
        }

    def submit_job(self, config: dict, trigger: str = "manual") -> dict:
        config = self._validate_job_config(config)
        with self.lock:
            basis = self._basis_snapshot()
            job_id = f"job{self.storage.next_seq()}"
            return self.jobs.submit(job_id, config, trigger, basis)

    def _basis_snapshot(self) -> dict:
        return {
            "epoch": self._epoch,
            "event_ids": [e.event_id for e in self.events],
            "valid_from": max((e.timestamp for e in self.events), default=None),
            "arrival_seq": max((e.arrival_seq for e in self.events), default=0),
        }

    def _run_job(self, record: dict, should_cancel, on_progress) -> dict:
        """作业执行体（在后台线程中运行）。"""
        config = record["config"]
        basis = record["basis"]
        defn = self._require_array()
        n, d = defn["n_elements"], defn["spacing_wavelengths"]

        with self.lock:
            health = self._health_at(None)
            active = self.versions[self.active_version_id]
            warm = np.array([complex(*w) for w in active["weights"]])

        if config["start_mode"] == "warm":
            start = warm
        else:
            start = uniform_start_weights(n, d, config["pointing_deg"],
                                          config["amplitude_cap"], health)
        sectors = [MaskSector.from_dict(s) for s in config["template"]["sectors"]]
        result = solve(SolveConfig(
            n_elements=n,
            spacing_wl=d,
            pointing_deg=config["pointing_deg"],
            sectors=sectors,
            max_mainlobe_loss_db=config["max_mainlobe_loss_db"],
            amplitude_cap=config["amplitude_cap"],
            reference_gain=self._reference_gain(config["pointing_deg"]),
            health=health,
            start_weights=start,
            max_iterations=config["max_iterations"],
            tolerance_db=config["tolerance_db"],
            iteration_delay_ms=config.get("iteration_delay_ms", 0.0),
            on_progress=on_progress,
            should_cancel=should_cancel,
        ))

        with self.lock:
            snap = self._basis_snapshot()
            if snap["arrival_seq"] != basis["arrival_seq"] or snap["epoch"] != basis["epoch"]:
                # 求解期间来了新事件或阵列被重定义，结果已过期
                return {"status_override": "stale", "feasible": False,
                        "iterations": result.iterations}
            solve_info = {
                "feasible": result.feasible,
                "iterations": result.iterations,
                "stop_reason": result.stop_reason,
                "max_violation_db": result.max_violation_db,
                "worst_angle_deg": result.worst_angle_deg,
                "worst_excess_db": result.worst_excess_db,
                "mainlobe_loss_db": result.mainlobe_loss_db,
                "start_mode": config["start_mode"],
            }
            version = self._create_version(
                kind="reconfigured", job_id=record["job_id"],
                weights=result.weights,
                basis_event_ids=basis["event_ids"],
                valid_from=basis["valid_from"],
                template=config["template"], reconfig=config,
                solve_info=solve_info,
            )
        return {
            "feasible": result.feasible,
            "version_id": version["version_id"],
            "iterations": result.iterations,
            "stop_reason": result.stop_reason,
            "max_violation_db": result.max_violation_db,
            "worst_angle_deg": result.worst_angle_deg,
            "worst_excess_db": result.worst_excess_db,
            "mainlobe_loss_db": result.mainlobe_loss_db,
        }

    def _maybe_auto_reconfigure_locked(self) -> dict | None:
        """（调用方须持锁）新事件后的自动重构判定。"""
        superseded_id = self.jobs.supersede_running()
        superseded_config = None
        if superseded_id is not None:
            rec = self.jobs.storage_job(superseded_id)
            superseded_config = rec["config"] if rec else None
        return self._maybe_auto_reconfigure(superseded_config)

    def _maybe_auto_reconfigure(self, superseded_config: dict | None) -> dict | None:
        if self.array_def is None or self.active_version_id is None:
            return None
        active = self.versions[self.active_version_id]
        # 候选模板：被作废作业的模板优先，其次生效版本绑定的模板
        candidates = []
        if superseded_config is not None:
            candidates.append(superseded_config)
        if active.get("reconfig") is not None and active.get("template") is not None:
            candidates.append(active["reconfig"])
        health = self._health_at(None)
        weights = np.array([complex(*w) for w in active["weights"]])
        defn = self.array_def
        for cfg in candidates:
            sectors = [MaskSector.from_dict(s) for s in cfg["template"]["sectors"]]
            excess, _ = evaluate_template(
                weights, health, defn["n_elements"], defn["spacing_wavelengths"],
                cfg["pointing_deg"], sectors)
            if excess > AUTO_TRIGGER_TOL_DB:
                auto_cfg = dict(cfg)
                auto_cfg["start_mode"] = "warm"  # 自动重构默认从当前方案出发
                return self.submit_job(auto_cfg, trigger="auto")
        return None

    # ------------------------------------------------------------------
    # 作业查询
    # ------------------------------------------------------------------
    def get_job(self, job_id: str) -> dict:
        rec = self.jobs.storage_job(job_id)
        if rec is None:
            raise LookupError(f"作业不存在: {job_id}")
        return rec

    def list_jobs(self) -> list[dict]:
        return self.jobs.list_jobs()

    def cancel_job(self, job_id: str) -> dict:
        if not self.jobs.cancel(job_id):
            raise LookupError(f"作业不存在或已结束: {job_id}")
        # 等待状态落库（取消是协作式的，通常立刻生效）
        for _ in range(500):
            rec = self.jobs.storage_job(job_id)
            if rec["status"] not in ("queued", "running"):
                return rec
            time.sleep(0.02)
        return self.jobs.storage_job(job_id)
