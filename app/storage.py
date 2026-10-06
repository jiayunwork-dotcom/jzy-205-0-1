"""Persistence layer.

Redis is the production store. JSON payloads are kept under versioned key names;
AOF persistence is enabled in ``redis.conf``. A tiny in-process Redis-compatible
fallback is provided for unit tests that exercise repository logic without a
running Redis server.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any, Iterable

import redis as redis_lib


KEY_ARRAY = "radar:v1:array"
KEY_EVENT_IDS = "radar:v1:event:ids"
KEY_EVENT_PREFIX = "radar:v1:event:"
KEY_JOB_IDS = "radar:v1:job:ids"
KEY_JOB_PREFIX = "radar:v1:job:"
KEY_VERSION_IDS = "radar:v1:version:ids"
KEY_VERSION_PREFIX = "radar:v1:version:"
KEY_ACTIVE = "radar:v1:active-version"
KEY_INPUT_IDS = "radar:v1:input:ids"
KEY_INPUT_PREFIX = "radar:v1:input:"


def dumps(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def loads(value: bytes | str | None) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.encode("utf-8")
    return json.loads(value.decode("utf-8"))


class FakeRedis:
    """Small subset of the redis-py API used by this service, with TTL support."""

    def __init__(self) -> None:
        self._data: dict[bytes, bytes] = {}
        self._expiry: dict[bytes, float] = {}
        self._sets: dict[bytes, set[bytes]] = defaultdict(set)
        self._lists: dict[bytes, list[bytes]] = defaultdict(list)
        self._zsets: dict[bytes, dict[bytes, float]] = defaultdict(dict)

    def _expire(self, key: bytes) -> None:
        if key in self._expiry and self._expiry[key] <= time.time():
            self._data.pop(key, None)
            self._sets.pop(key, None)
            self._zsets.pop(key, None)
            self._expiry.pop(key, None)

    def ping(self) -> bool:
        return True

    def flushdb(self) -> bool:
        self._data.clear()
        self._expiry.clear()
        self._sets.clear()
        self._lists.clear()
        self._zsets.clear()
        return True

    def set(self, key: str, value: bytes) -> bool:
        key = key.encode() if isinstance(key, str) else key
        self._data[key] = value
        self._expire(key)
        return True

    def get(self, key: str) -> bytes | None:
        key = key.encode() if isinstance(key, str) else key
        self._expire(key)
        return self._data.get(key)

    def delete(self, *keys: str) -> int:
        count = 0
        for key in keys:
            raw = key.encode() if isinstance(key, str) else key
            count += raw in self._data or raw in self._sets or raw in self._zsets
            self._data.pop(raw, None)
            self._sets.pop(raw, None)
            self._lists.pop(raw, None)
            self._zsets.pop(raw, None)
            self._expiry.pop(raw, None)
        return count

    def exists(self, *keys: str) -> int:
        return sum(1 for key in keys if self.get(key if isinstance(key, str) else key.decode()) is not None)

    def sadd(self, key: str, *values: bytes | str) -> int:
        raw = key.encode() if isinstance(key, str) else key
        target = self._sets[raw]
        added = 0
        for value in values:
            item = value.encode() if isinstance(value, str) else value
            if item not in target:
                target.add(item)
                added += 1
        return added

    def smembers(self, key: str) -> set[bytes]:
        raw = key.encode() if isinstance(key, str) else key
        self._expire(raw)
        return set(self._sets.get(raw, set()))

    def zadd(self, key: str, mapping: dict[str, float]) -> int:
        raw = key.encode() if isinstance(key, str) else key
        target = self._zsets[raw]
        added = 0
        for member, score in mapping.items():
            item = member.encode() if isinstance(member, str) else member
            if item not in target:
                added += 1
            target[item] = float(score)
        return added

    def zrange(self, key: str, start: int, end: int) -> list[bytes]:
        raw = key.encode() if isinstance(key, str) else key
        ordered = [member for member, _ in sorted(self._zsets.get(raw, {}).items(), key=lambda item: (item[1], item[0]))]
        if end == -1:
            end = len(ordered) - 1
        return ordered[start : end + 1]

    def incr(self, key: str, amount: int = 1) -> int:
        raw = key.encode() if isinstance(key, str) else key
        value = int(self._data.get(raw, b"0")) + amount
        self._data[raw] = str(value).encode()
        return value

    def lpush(self, key: str, *values: bytes | str) -> int:
        raw = key.encode() if isinstance(key, str) else key
        target = self._lists[raw]
        for value in values:
            target.insert(0, value.encode() if isinstance(value, str) else value)
        return len(values)

    def lrange(self, key: str, start: int, end: int) -> list[bytes]:
        raw = key.encode() if isinstance(key, str) else key
        values = self._lists.get(raw, [])
        return values if end == -1 else values[start : end + 1]

    def scan_iter(self, pattern: str) -> Iterable[bytes]:
        prefix = pattern.encode().rstrip(b"*")
        keys = set(self._data) | set(self._sets) | set(self._zsets)
        for key in sorted(keys):
            self._expire(key)
            if key.startswith(prefix):
                yield key


class Store:
    def __init__(self, redis_client=None):
        self.redis = redis_client or FakeRedis()

    @classmethod
    def from_url(cls, url: str) -> "Store":
        return cls(redis_lib.Redis.from_url(url, decode_responses=False))

    def ping(self) -> bool:
        return bool(self.redis.ping())

    def save_array(self, array_payload: dict[str, Any]) -> None:
        self.redis.set(KEY_ARRAY, dumps(array_payload))

    def load_array(self) -> dict[str, Any] | None:
        return loads(self.redis.get(KEY_ARRAY))

    def save_event(self, payload: dict[str, Any]) -> bool:
        event_id = payload["event_id"]
        added = bool(self.redis.sadd(KEY_EVENT_IDS, event_id))
        if added:
            self.redis.set(f"{KEY_EVENT_PREFIX}{event_id}", dumps(payload))
        return added

    def load_events(self) -> list[dict[str, Any]]:
        result = []
        for raw in self.redis.smembers(KEY_EVENT_IDS):
            event_id = raw.decode() if isinstance(raw, bytes) else raw
            payload = loads(self.redis.get(f"{KEY_EVENT_PREFIX}{event_id}"))
            if payload is not None:
                result.append(payload)
        return result

    def delete_event(self, event_id: str) -> None:
        self.redis.delete(f"{KEY_EVENT_PREFIX}{event_id}")

    def next_id(self, kind: str) -> int:
        return int(self.redis.incr(f"radar:v1:counter:{kind}"))

    def append_input(self, payload: dict[str, Any]) -> None:
        self.redis.lpush(KEY_INPUT_IDS, dumps(payload))

    def load_inputs(self) -> list[dict[str, Any]]:
        raw_items = self.redis.lrange(KEY_INPUT_IDS, 0, -1)
        return [loads(raw) for raw in reversed(raw_items)]

    def save_job(self, payload: dict[str, Any]) -> None:
        self.redis.sadd(KEY_JOB_IDS, payload["job_id"])
        self.redis.set(f"{KEY_JOB_PREFIX}{payload['job_id']}", dumps(payload))

    def load_job(self, job_id: str) -> dict[str, Any] | None:
        return loads(self.redis.get(f"{KEY_JOB_PREFIX}{job_id}"))

    def load_jobs(self) -> list[dict[str, Any]]:
        jobs = []
        for raw in self.redis.smembers(KEY_JOB_IDS):
            job_id = raw.decode() if isinstance(raw, bytes) else raw
            payload = self.load_job(job_id)
            if payload is not None:
                jobs.append(payload)
        return jobs

    def save_version(self, payload: dict[str, Any]) -> None:
        self.redis.sadd(KEY_VERSION_IDS, payload["version_id"])
        self.redis.set(f"{KEY_VERSION_PREFIX}{payload['version_id']}", dumps(payload))

    def load_version(self, version_id: str) -> dict[str, Any] | None:
        return loads(self.redis.get(f"{KEY_VERSION_PREFIX}{version_id}"))

    def load_versions(self) -> list[dict[str, Any]]:
        versions = []
        for raw in self.redis.smembers(KEY_VERSION_IDS):
            version_id = raw.decode() if isinstance(raw, bytes) else raw
            payload = self.load_version(version_id)
            if payload is not None:
                versions.append(payload)
        return versions

    def set_active_version(self, version_id: str | None) -> None:
        if version_id is None:
            self.redis.delete(KEY_ACTIVE)
        else:
            self.redis.set(KEY_ACTIVE, version_id.encode())

    def active_version_id(self) -> str | None:
        raw = self.redis.get(KEY_ACTIVE)
        return raw.decode() if raw else None
