"""健康事件与阵列状态推导（事件溯源）。

每条事件：事件编号、时刻、单元编号（内部 0 基）、变化类型与数值。
任意时刻 t 的阵列状态由所有 timestamp <= t 的事件按 (时刻, 到达序号)
排序后依次折叠得到；同一事件编号只生效一次（存储层去重），晚到的事件
按其时刻归位。折叠是纯函数，在线处理与回放共用同一代码路径，保证一致。
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np


class ChangeType(str, enum.Enum):
    FAILED = "failed"                  # 通道完全失效
    AMPLITUDE_ERROR = "amplitude_error"  # 幅度误差，value 为 dB（负值表示下降）
    PHASE_ERROR = "phase_error"        # 相位误差，value 为度
    RECOVERED = "recovered"            # 恢复正常（清除该单元全部误差状态）


@dataclass
class HealthEvent:
    event_id: str
    timestamp: float
    element: int  # 0 基
    change_type: ChangeType
    value: float | None
    arrival_seq: int  # 到达序号，用于同时刻事件的确定排序

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "element": self.element + 1,  # 对外 1 基
            "change_type": self.change_type.value,
            "value": self.value,
            "arrival_seq": self.arrival_seq,
        }

    @staticmethod
    def from_dict(d: dict) -> "HealthEvent":
        return HealthEvent(
            event_id=d["event_id"],
            timestamp=float(d["timestamp"]),
            element=int(d["element"]) - 1,
            change_type=ChangeType(d["change_type"]),
            value=None if d.get("value") is None else float(d["value"]),
            arrival_seq=int(d["arrival_seq"]),
        )


@dataclass
class ElementState:
    failed: bool = False
    amplitude_error_db: float = 0.0
    phase_error_deg: float = 0.0

    def factor(self) -> complex:
        """该单元的复健康因子；失效单元严格为 0。"""
        if self.failed:
            return 0.0 + 0.0j
        return complex(10.0 ** (self.amplitude_error_db / 20.0)) * np.exp(
            1j * np.deg2rad(self.phase_error_deg)
        )

    def as_dict(self) -> dict:
        return {
            "failed": self.failed,
            "amplitude_error_db": self.amplitude_error_db,
            "phase_error_deg": self.phase_error_deg,
        }


def apply_event(state: list[ElementState], ev: HealthEvent) -> None:
    es = state[ev.element]
    if ev.change_type is ChangeType.FAILED:
        es.failed = True
    elif ev.change_type is ChangeType.AMPLITUDE_ERROR:
        es.amplitude_error_db = float(ev.value)
    elif ev.change_type is ChangeType.PHASE_ERROR:
        es.phase_error_deg = float(ev.value)
    elif ev.change_type is ChangeType.RECOVERED:
        es.failed = False
        es.amplitude_error_db = 0.0
        es.phase_error_deg = 0.0
    else:  # pragma: no cover - 防御
        raise ValueError(f"未知事件类型: {ev.change_type}")


def derive_state(
    events: list[HealthEvent], n_elements: int, t: float | None = None
) -> list[ElementState]:
    """折叠事件得到 t 时刻的阵列状态；t=None 表示取全部已知事件。"""
    selected = [e for e in events if t is None or e.timestamp <= t]
    selected.sort(key=lambda e: (e.timestamp, e.arrival_seq))
    state = [ElementState() for _ in range(n_elements)]
    for ev in selected:
        apply_event(state, ev)
    return state


def health_factors(state: list[ElementState]) -> np.ndarray:
    return np.array([es.factor() for es in state], dtype=complex)


def failed_elements(state: list[ElementState]) -> list[int]:
    """失效单元编号（0 基）。"""
    return [i for i, es in enumerate(state) if es.failed]
