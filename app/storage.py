"""Redis 持久化层。

键规划：
  array:def        HASH   阵列定义（JSON）
  events:all       HASH   event_id -> 事件 JSON
  events:arrival   LIST   event_id 按到达顺序
  versions:all     HASH   version_id -> 版本 JSON
  versions:order   LIST   version_id 按创建顺序
  jobs:all         HASH   job_id -> 作业 JSON
  jobs:order       LIST   job_id 按创建顺序
  active_version   STRING 当前生效版本号
  meta:seq         计数器（到达序号 / 版本号 / 作业号）

Redis 侧开启 AOF 持久化（见 docker-compose.yml），服务重启后由此层恢复。
"""
from __future__ import annotations

import json


class Storage:
    def __init__(self, redis_client):
        self.r = redis_client

    # ---- 通用 ----
    def next_seq(self) -> int:
        return int(self.r.incr("meta:seq"))

    def clear_all(self) -> None:
        for key in [
            "array:def", "array:epoch", "events:all", "events:arrival", "versions:all",
            "versions:order", "jobs:all", "jobs:order", "active_version",
        ]:
            self.r.delete(key)

    # ---- 阵列定义 ----
    def save_array(self, array_def: dict) -> None:
        self.r.set("array:def", json.dumps(array_def))

    def load_array(self) -> dict | None:
        raw = self.r.get("array:def")
        return json.loads(raw) if raw else None

    def set_array_epoch(self, epoch: int) -> None:
        self.r.set("array:epoch", str(epoch))

    def get_array_epoch(self) -> int | None:
        raw = self.r.get("array:epoch")
        return int(raw) if raw else None

    # ---- 事件 ----
    def has_event(self, event_id: str) -> bool:
        return bool(self.r.hexists("events:all", event_id))

    def append_event(self, event_id: str, payload: dict) -> None:
        pipe = self.r.pipeline()
        pipe.hset("events:all", event_id, json.dumps(payload))
        pipe.rpush("events:arrival", event_id)
        pipe.execute()

    def list_events(self) -> list[dict]:
        ids = self.r.lrange("events:arrival", 0, -1)
        if not ids:
            return []
        raws = self.r.hmget("events:all", ids)
        return [json.loads(x) for x in raws if x is not None]

    # ---- 版本 ----
    def save_version(self, version_id: str, payload: dict) -> None:
        pipe = self.r.pipeline()
        pipe.hset("versions:all", version_id, json.dumps(payload))
        pipe.rpush("versions:order", version_id)
        pipe.execute()

    def list_versions(self) -> list[dict]:
        ids = self.r.lrange("versions:order", 0, -1)
        if not ids:
            return []
        raws = self.r.hmget("versions:all", ids)
        return [json.loads(x) for x in raws if x is not None]

    def set_active_version(self, version_id: str) -> None:
        self.r.set("active_version", version_id)

    def get_active_version(self) -> str | None:
        return self.r.get("active_version")

    # ---- 作业 ----
    def save_job(self, job_id: str, payload: dict, enqueue: bool = False) -> None:
        pipe = self.r.pipeline()
        pipe.hset("jobs:all", job_id, json.dumps(payload))
        if enqueue:
            pipe.rpush("jobs:order", job_id)
        pipe.execute()

    def list_jobs(self) -> list[dict]:
        ids = self.r.lrange("jobs:order", 0, -1)
        if not ids:
            return []
        raws = self.r.hmget("jobs:all", ids)
        return [json.loads(x) for x in raws if x is not None]
