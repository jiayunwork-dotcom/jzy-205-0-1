"""FastAPI 接口层。

以工厂方式创建应用，便于测试注入 fakeredis：
    uvicorn 'app.main:create_app' --factory
生产环境通过 REDIS_URL 连接 Redis 7（compose 中已配置）。
"""
from __future__ import annotations

import math
import os

import redis
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .schemas import ArrayDefIn, EventIn, JobIn
from .service import ArrayService
from .storage import Storage


def _sanitize(obj):
    """把校验错误详情清洗为可 JSON 序列化的结构。"""
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


def create_app(redis_client=None) -> FastAPI:
    if redis_client is None:
        redis_client = redis.Redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6379/0"),
            decode_responses=True,
        )
    service = ArrayService(Storage(redis_client))
    app = FastAPI(title="阵列重构后端", version="1.0.0")
    app.state.service = service

    @app.exception_handler(RequestValidationError)
    async def validation_handler(request, exc):
        return JSONResponse(status_code=422, content={"detail": _sanitize(exc.errors())})

    def wrap(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except LookupError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    # ---- 阵列定义 ----
    @app.put("/array")
    def define_array(body: ArrayDefIn):
        return wrap(service.define_array, body.to_service())

    @app.get("/array")
    def get_array():
        return wrap(service.get_array)

    # ---- 健康事件 ----
    @app.post("/events", status_code=201)
    def add_event(body: EventIn):
        record, applied, auto_job = wrap(service.add_event, body.model_dump())
        return {"event": record, "applied": applied, "auto_job": auto_job}

    @app.get("/events")
    def list_events():
        return {"events": service.list_events()}

    # ---- 状态与方向图 ----
    @app.get("/state")
    def state(t: float | None = Query(default=None)):
        return wrap(service.state_at, t)

    @app.get("/pattern")
    def pattern(
        t: float | None = Query(default=None),
        version_id: str | None = Query(default=None),
        include_pattern: bool = Query(default=False),
        step_deg: float = Query(default=1.0, gt=0.0, le=10.0),
    ):
        return wrap(service.pattern_at, t, version_id, include_pattern, step_deg)

    @app.get("/replay")
    def replay(t: float):
        return wrap(service.replay, t)

    @app.get("/array-factor")
    def array_factor(
        theta: list[float] = Query(..., description="方向角（度），可重复传参"),
        t: float | None = Query(default=None),
        version_id: str | None = Query(default=None),
    ):
        return wrap(service.array_factor_at, theta, t, version_id)

    # ---- 重构作业 ----
    @app.post("/jobs", status_code=202)
    def submit_job(body: JobIn):
        return wrap(service.submit_job, body.model_dump())

    @app.get("/jobs")
    def list_jobs():
        return {"jobs": service.list_jobs()}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str):
        return wrap(service.get_job, job_id)

    @app.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str):
        return wrap(service.cancel_job, job_id)

    # ---- 版本 ----
    @app.get("/versions")
    def list_versions():
        return {"versions": service.list_versions()}

    @app.get("/versions/compare")
    def compare_versions(a: str = Query(...), b: str = Query(...)):
        return wrap(service.compare_versions, a, b)

    @app.get("/versions/{version_id}")
    def get_version(version_id: str):
        return wrap(service.get_version, version_id)

    @app.get("/health")
    def health():
        return {"status": "ok", "array_defined": service.array_def is not None}

    return app
