"""重构作业调度。

同一时刻至多一个作业在运行。新作业提交或新健康事件到达时，未完成的
作业被作废（superseded）并以最新状态重新发起；用户也可显式取消。
作业在后台线程中执行，进度周期性写回存储，可查询；取消通过协作式
标志位实现，求解器每次迭代检查一次。
"""
from __future__ import annotations

import enum
import threading
import time
import traceback


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"      # 收敛且满足模板
    INFEASIBLE = "infeasible"    # 达到上限/停滞，未满足模板（已如实返回最优解）
    CANCELLED = "cancelled"      # 用户取消
    SUPERSEDED = "superseded"    # 被更新的作业作废
    FAILED = "failed"            # 异常
    ABORTED = "aborted"          # 服务重启时仍未完成
    STALE = "stale"              # 完成时其依据的状态已过期，结果被丢弃


# 终态：作业记录不再变化
TERMINAL = {
    JobStatus.SUCCEEDED, JobStatus.INFEASIBLE, JobStatus.CANCELLED,
    JobStatus.SUPERSEDED, JobStatus.FAILED, JobStatus.ABORTED, JobStatus.STALE,
}


class _Superseded(Exception):
    pass


class _Cancelled(Exception):
    pass


class JobManager:
    def __init__(self, storage, run_fn):
        """run_fn(job_record, should_cancel, on_progress) -> result dict"""
        self.storage = storage
        self.run_fn = run_fn
        self._lock = threading.RLock()
        self._cancel_flags: dict[str, threading.Event] = {}
        self._threads: dict[str, threading.Thread] = {}

    # ---- 生命周期 ----
    def submit(self, job_id: str, config: dict, trigger: str, basis: dict) -> dict:
        """提交作业；若已有作业在运行，先将其作废。"""
        with self._lock:
            self.supersede_running()
            record = {
                "job_id": job_id,
                "status": JobStatus.QUEUED.value,
                "trigger": trigger,  # manual | auto
                "config": config,
                "basis": basis,
                "created_at": time.time(),
                "progress": {"iteration": 0, "max_iterations": config.get("max_iterations"), "max_violation_db": None},
                "result": None,
                "version_id": None,
                "error": None,
            }
            self.storage.save_job(job_id, record, enqueue=True)
            flag = threading.Event()
            self._cancel_flags[job_id] = flag
            th = threading.Thread(target=self._run, args=(job_id, flag), daemon=True)
            self._threads[job_id] = th
            th.start()
            return record

    def _run(self, job_id: str, flag: threading.Event) -> None:
        with self._lock:
            record = self.storage_job(job_id)
            if record is None or record["status"] != JobStatus.QUEUED.value:
                return  # 启动前已被取消或作废
            record["status"] = JobStatus.RUNNING.value
            self.storage.save_job(job_id, record)

        def should_cancel() -> bool:
            if not flag.is_set():
                return False
            # 区分用户取消与被作废
            with self._lock:
                cur = self.storage_job(job_id)
            if cur and cur["status"] == JobStatus.SUPERSEDED.value:
                raise _Superseded()
            raise _Cancelled()

        def on_progress(iteration: int, max_violation_db: float) -> None:
            rec = self.storage_job(job_id)
            if rec is None or rec["status"] != JobStatus.RUNNING.value:
                return
            rec["progress"] = {
                "iteration": iteration,
                "max_iterations": rec["config"].get("max_iterations"),
                "max_violation_db": max_violation_db,
            }
            self.storage.save_job(job_id, rec)

        try:
            result = self.run_fn(record, should_cancel, on_progress)
            if result.get("status_override"):
                self._finish(job_id, JobStatus(result["status_override"]), result)
            else:
                self._finish(job_id, JobStatus.SUCCEEDED if result.get("feasible") else JobStatus.INFEASIBLE, result)
        except _Superseded:
            pass  # 状态已是 superseded
        except _Cancelled:
            self._finish(job_id, JobStatus.CANCELLED, None)
        except Exception:  # noqa: BLE001 - 作业异常必须落库
            self._finish(job_id, JobStatus.FAILED, None, error=traceback.format_exc())

    def _finish(self, job_id: str, status: JobStatus, result: dict | None, error: str | None = None) -> None:
        with self._lock:
            rec = self.storage_job(job_id)
            if rec is None:
                return
            if rec["status"] == JobStatus.SUPERSEDED.value:
                return  # 已被作废，不覆盖
            rec["status"] = status.value
            if result is not None:
                rec["result"] = result
                rec["version_id"] = result.get("version_id")
            if error is not None:
                rec["error"] = error
            self.storage.save_job(job_id, rec)
            self._cancel_flags.pop(job_id, None)

    def supersede_running(self) -> str | None:
        """作废当前未完成的作业（若有），返回其 job_id。"""
        with self._lock:
            for rec in self.list_jobs():
                if rec["status"] in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
                    rec["status"] = JobStatus.SUPERSEDED.value
                    self.storage.save_job(rec["job_id"], rec)
                    flag = self._cancel_flags.get(rec["job_id"])
                    if flag is not None:
                        flag.set()
                    return rec["job_id"]
            return None

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            rec = self.storage_job(job_id)
            if rec is None:
                return False
            if rec["status"] == JobStatus.QUEUED.value:
                rec["status"] = JobStatus.CANCELLED.value
                self.storage.save_job(job_id, rec)
                return True
            if rec["status"] == JobStatus.RUNNING.value:
                flag = self._cancel_flags.get(job_id)
                if flag is not None:
                    flag.set()
                return True
            return False

    def abort_unfinished(self) -> None:
        """服务重启时调用：把存储中仍未完成的作业标记为 aborted。"""
        with self._lock:
            for rec in self.list_jobs():
                if rec["status"] in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
                    rec["status"] = JobStatus.ABORTED.value
                    self.storage.save_job(rec["job_id"], rec)

    # ---- 查询 ----
    def storage_job(self, job_id: str) -> dict | None:
        for rec in self.list_jobs():
            if rec["job_id"] == job_id:
                return rec
        return None

    def list_jobs(self) -> list[dict]:
        return self.storage.list_jobs()
