"""接口入参模型（pydantic v2）。拒收规则在这里做第一层校验。"""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .events import ChangeType


class WeightIn(BaseModel):
    amplitude: float
    phase_deg: float = 0.0

    @field_validator("amplitude")
    @classmethod
    def amplitude_ok(cls, v: float) -> float:
        if not math.isfinite(v) or v < 0.0:
            raise ValueError("加权幅度必须为非负有限数")
        return v

    @field_validator("phase_deg")
    @classmethod
    def phase_ok(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("加权相位必须为有限数")
        return v


class ArrayDefIn(BaseModel):
    n_elements: int = Field(ge=2, description="单元数不少于 2")
    spacing_wavelengths: float = Field(gt=0.0, description="间距（波长）必须为正")
    weights: list[WeightIn]
    pointing_deg: float = Field(ge=-90.0, le=90.0)

    @field_validator("spacing_wavelengths")
    @classmethod
    def spacing_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("间距必须为有限数")
        return v

    @model_validator(mode="after")
    def weights_match(self):
        if len(self.weights) != self.n_elements:
            raise ValueError("加权数量必须与单元数一致")
        return self

    def to_service(self) -> dict:
        return {
            "n_elements": self.n_elements,
            "spacing_wavelengths": self.spacing_wavelengths,
            "weights": [
                [
                    w.amplitude * math.cos(math.radians(w.phase_deg)),
                    w.amplitude * math.sin(math.radians(w.phase_deg)),
                ]
                for w in self.weights
            ],
            "pointing_deg": self.pointing_deg,
        }


class EventIn(BaseModel):
    event_id: str = Field(min_length=1)
    timestamp: float
    element: int = Field(ge=1, description="单元编号，从 1 数起")
    change_type: ChangeType
    value: float | None = None

    @field_validator("timestamp")
    @classmethod
    def timestamp_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("时刻必须为有限数")
        return v

    @field_validator("value")
    @classmethod
    def value_finite(cls, v: float | None) -> float | None:
        if v is not None and not math.isfinite(v):
            raise ValueError("数值必须为有限数")
        return v


class SectorIn(BaseModel):
    theta_min: float = Field(ge=-90.0, le=90.0)
    theta_max: float = Field(ge=-90.0, le=90.0)
    max_sll_db: float

    @field_validator("max_sll_db")
    @classmethod
    def sll_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("模板电平必须为有限数")
        return v

    @model_validator(mode="after")
    def ordered(self):
        if self.theta_min >= self.theta_max:
            raise ValueError("模板区间下限必须小于上限")
        return self


class TemplateIn(BaseModel):
    sectors: list[SectorIn] = Field(min_length=1)

    @model_validator(mode="after")
    def non_overlapping(self):
        ordered = sorted(self.sectors, key=lambda s: s.theta_min)
        for a, b in zip(ordered, ordered[1:]):
            if b.theta_min < a.theta_max:
                raise ValueError("模板角度区间相互重叠")
        return self


class JobIn(BaseModel):
    template: TemplateIn
    pointing_deg: float = Field(ge=-90.0, le=90.0)
    max_mainlobe_loss_db: float = Field(ge=0.0, description="允许的主瓣增益损失不能为负")
    amplitude_cap: float | None = Field(default=None, gt=0.0)
    start_mode: Literal["warm", "cold"] = "warm"
    max_iterations: int = Field(default=5000, ge=1, le=2_000_000)
    tolerance_db: float = Field(default=0.02, gt=0.0)
    iteration_delay_ms: float = Field(default=0.0, ge=0.0, le=1000.0)

    @field_validator("max_mainlobe_loss_db")
    @classmethod
    def loss_finite(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("允许的主瓣损失必须为有限数")
        return v
