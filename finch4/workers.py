"""Durable, process-safe queue for orchestrator-supplied native commands.

The queue deliberately treats job kinds, metadata, argv, and process output as
opaque data.  Commands are always launched directly, never through a shell.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import uuid


SCHEMA_VERSION = 1
TERMINAL = {"succeeded", "failed", "timed_out", "crashed", "cancelled"}


def _now():
    return _datetime.datetime.now(_datetime.timezone.utc).isoformat()


class JobQueue:
    """A JSON-backed queue of opaque native-process jobs."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.path = self.root / "jobs.json"
        self.lock_path = self.root / "jobs.json.lock"
        self.logs = self.root / "logs"
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        with self._locked() as data:
            self._write_locked(data)

    class _Lock:
        def __init__(self, queue):
            self.queue = queue

        def __enter__(self):
            self.handle = self.queue.lock_path.open("a+", encoding="utf-8")
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX)
            self.data = self.queue._read_unlocked()
            return self.data

        def __exit__(self, exc_type, exc, tb):
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()

    def _locked(self):
        return self._Lock(self)

    def _read_unlocked(self):
        if not self.path.exists():
            return {"schema_version": SCHEMA_VERSION,
                    "next_id": 1, "jobs": []}
        with self.path.open(encoding="utf-8") as handle:
            data = json.load(handle)
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported jobs.json schema version")
        if not isinstance(data.get("jobs"), list):
            raise ValueError("jobs.json jobs must be a list")
        return data

    def _write_locked(self, data):
        payload = json.dumps(data, indent=2, sort_keys=True,
                             ensure_ascii=False, allow_nan=False) + "\n"
        fd, temporary = tempfile.mkstemp(
            prefix=".jobs.", suffix=".tmp", dir=str(self.root))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _job(data, job_id):
        for job in data["jobs"]:
            if job["id"] == job_id:
                return job
        raise KeyError(f"unknown job {job_id!r}")

    def enqueue(self, *, kind, lane, name, argv, cwd=None, timeout=None,
                metadata=None, max_attempts=1):
        """Append one job and return a detached copy of its durable record."""
        if not all(str(value).strip() for value in (kind, lane, name)):
            raise ValueError("kind, lane, and name must be non-empty")
        if isinstance(argv, (str, bytes)) or not argv:
            raise ValueError("argv must be a non-empty sequence")
        argv = [os.fspath(value) for value in argv]
        if any(not isinstance(value, str) or not value for value in argv):
            raise ValueError("every argv item must be a non-empty string")
        if timeout is not None and float(timeout) <= 0:
            raise ValueError("timeout must be positive")
        max_attempts = int(max_attempts)
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        metadata = {} if metadata is None else metadata
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a JSON object")
        json.dumps(metadata, allow_nan=False)
        cwd = None if cwd is None else str(Path(cwd).resolve())
        with self._locked() as data:
            number = int(data["next_id"])
            data["next_id"] = number + 1
            stamp = _now()
            job = {
                "id": f"J{number:06d}", "created_at": stamp,
                "updated_at": stamp, "kind": str(kind),
                "lane": str(lane), "name": str(name), "argv": argv,
                "cwd": cwd, "timeout": (None if timeout is None
                                           else float(timeout)),
                "metadata": copy.deepcopy(metadata),
                "max_attempts": max_attempts, "status": "queued",
                "attempts": [],
            }
            data["jobs"].append(job)
            self._write_locked(data)
            return copy.deepcopy(job)

    def status(self, job_id=None):
        """Return one job, or the complete schema payload when id is omitted."""
        with self._locked() as data:
            value = data if job_id is None else self._job(data, job_id)
            return copy.deepcopy(value)

    def retry(self, job_id):
        """Requeue a terminal job and grant exactly one additional attempt."""
        with self._locked() as data:
            job = self._job(data, job_id)
            if job["status"] not in TERMINAL:
                raise ValueError("only a terminal job can be retried")
            job["max_attempts"] = max(job["max_attempts"],
                                      len(job["attempts"]) + 1)
            job["status"] = "queued"
            job["updated_at"] = _now()
            self._write_locked(data)
            return copy.deepcopy(job)

    def cancel(self, job_id, reason):
        """Cancel queued work before Finch launches it."""
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("cancelling a job requires a reason")
        with self._locked() as data:
            job = self._job(data, job_id)
            if job["status"] != "queued":
                raise ValueError("only a queued job can be cancelled")
            job["status"] = "cancelled"
            job["cancel_reason"] = reason
            job["updated_at"] = _now()
            self._write_locked(data)
            return copy.deepcopy(job)

    @staticmethod
    def _pid_alive(pid):
        if not pid:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    @classmethod
    def _runner_alive(cls, runner_id):
        try:
            pid = int(str(runner_id).split("-", 2)[1])
        except (IndexError, TypeError, ValueError):
            return False
        return cls._pid_alive(pid)

    def _recover_dead_locked(self, data):
        changed = False
        for job in data["jobs"]:
            if job["status"] != "running" or not job["attempts"]:
                continue
            attempt = job["attempts"][-1]
            # The owning runner closes the small claim-to-Popen race and
            # records normal child completion.  Recovery belongs only to a
            # later runner after both the owner and child have disappeared.
            if (self._runner_alive(attempt.get("runner_id")) or
                    self._pid_alive(attempt.get("pid"))):
                continue
            stamp = _now()
            attempt.update(status="crashed", finished_at=stamp,
                           error="worker or child exited without recording completion")
            job["status"] = ("queued" if len(job["attempts"])
                             < job["max_attempts"] else "crashed")
            job["updated_at"] = stamp
            changed = True
        return changed

    def _claim(self, runner_id, max_concurrency, lane_limits):
        with self._locked() as data:
            changed = self._recover_dead_locked(data)
            running = [j for j in data["jobs"] if j["status"] == "running"]
            lane_running = {}
            for job in running:
                lane_running[job["lane"]] = lane_running.get(job["lane"], 0) + 1
            slots = max_concurrency - len(running)
            claimed = []
            if slots > 0:
                for job in data["jobs"]:
                    if slots <= 0:
                        break
                    if job["status"] != "queued":
                        continue
                    limit = lane_limits.get(job["lane"])
                    if limit is not None and lane_running.get(job["lane"], 0) >= limit:
                        continue
                    number = len(job["attempts"]) + 1
                    base = f"logs/{job['id']}-attempt-{number}"
                    attempt = {
                        "number": number, "runner_id": runner_id,
                        "started_at": _now(), "finished_at": None,
                        "pid": None, "status": "running", "returncode": None,
                        "error": None, "stdout_path": base + ".stdout.log",
                        "stderr_path": base + ".stderr.log",
                    }
                    job["attempts"].append(attempt)
                    job["status"] = "running"
                    job["updated_at"] = attempt["started_at"]
                    claimed.append(copy.deepcopy(job))
                    lane_running[job["lane"]] = lane_running.get(job["lane"], 0) + 1
                    slots -= 1
                    changed = True
            if changed:
                self._write_locked(data)
            return claimed

    def _set_pid(self, job_id, number, pid):
        with self._locked() as data:
            job = self._job(data, job_id)
            attempt = job["attempts"][number - 1]
            attempt["pid"] = pid
            job["updated_at"] = _now()
            self._write_locked(data)

    def _finish(self, job_id, number, outcome, returncode=None, error=None):
        with self._locked() as data:
            job = self._job(data, job_id)
            attempt = job["attempts"][number - 1]
            stamp = _now()
            attempt.update(status=outcome, finished_at=stamp,
                           returncode=returncode, error=error)
            if outcome != "succeeded" and len(job["attempts"]) < job["max_attempts"]:
                job["status"] = "queued"
            else:
                job["status"] = outcome
            job["updated_at"] = stamp
            self._write_locked(data)

    def run(self, *, max_concurrency=1, lane_limits=None, poll_interval=0.02):
        """Run claimable jobs to completion and return the full queue status."""
        max_concurrency = int(max_concurrency)
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one")
        lane_limits = dict(lane_limits or {})
        for lane, limit in lane_limits.items():
            if not str(lane) or int(limit) < 1:
                raise ValueError("lane limits must be positive")
            lane_limits[lane] = int(limit)
        runner_id = f"runner-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        active = {}
        while True:
            for job in self._claim(runner_id, max_concurrency, lane_limits):
                attempt = job["attempts"][-1]
                stdout_path = self.root / attempt["stdout_path"]
                stderr_path = self.root / attempt["stderr_path"]
                stdout_handle = stdout_path.open("wb")
                stderr_handle = stderr_path.open("wb")
                try:
                    process = subprocess.Popen(
                        job["argv"], cwd=job["cwd"], stdin=subprocess.DEVNULL,
                        stdout=stdout_handle, stderr=stderr_handle,
                        shell=False, start_new_session=True)
                except Exception as exc:
                    stdout_handle.close()
                    stderr_handle.close()
                    self._finish(job["id"], attempt["number"], "crashed",
                                 error=f"{type(exc).__name__}: {exc}")
                    continue
                self._set_pid(job["id"], attempt["number"], process.pid)
                active[job["id"]] = {
                    "job": job, "attempt": attempt, "process": process,
                    "started": time.monotonic(), "stdout": stdout_handle,
                    "stderr": stderr_handle,
                }
            for job_id, item in list(active.items()):
                process = item["process"]
                timeout = item["job"]["timeout"]
                timed_out = (timeout is not None and
                             time.monotonic() - item["started"] >= timeout)
                if timed_out and process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    try:
                        process.wait(timeout=0.5)
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                returncode = process.poll()
                if returncode is None:
                    continue
                item["stdout"].close()
                item["stderr"].close()
                if timed_out:
                    outcome = "timed_out"
                    error = f"exceeded timeout of {timeout:g} seconds"
                elif returncode < 0:
                    outcome, error = "crashed", f"terminated by signal {-returncode}"
                elif returncode:
                    outcome, error = "failed", None
                else:
                    outcome, error = "succeeded", None
                self._finish(job_id, item["attempt"]["number"], outcome,
                             returncode=returncode, error=error)
                del active[job_id]
            if not active:
                snapshot = self.status()
                if not any(j["status"] == "queued" for j in snapshot["jobs"]):
                    return snapshot
                # Queued work may be blocked by jobs owned by another runner.
                if any(j["status"] == "running" for j in snapshot["jobs"]):
                    return snapshot
            time.sleep(poll_interval)


def _parser():
    parser = argparse.ArgumentParser(description="Finch native-process worker queue")
    parser.add_argument("--root", default=".finch-workers")
    sub = parser.add_subparsers(dest="command", required=True)
    enqueue = sub.add_parser("enqueue")
    enqueue.add_argument("--kind", required=True)
    enqueue.add_argument("--lane", required=True)
    enqueue.add_argument("--name", required=True)
    enqueue.add_argument("--cwd")
    enqueue.add_argument("--timeout", type=float)
    enqueue.add_argument("--metadata", default="{}")
    enqueue.add_argument("--max-attempts", type=int, default=1)
    enqueue.add_argument("argv", nargs=argparse.REMAINDER)
    run = sub.add_parser("run")
    run.add_argument("--max-concurrency", type=int, default=1)
    run.add_argument("--lane-limit", action="append", default=[], metavar="LANE=N")
    status = sub.add_parser("status")
    status.add_argument("job_id", nargs="?")
    retry = sub.add_parser("retry")
    retry.add_argument("job_id")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("job_id")
    cancel.add_argument("--reason", required=True)
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    queue = JobQueue(args.root)
    if args.command == "enqueue":
        command = list(args.argv)
        if command[:1] == ["--"]:
            command = command[1:]
        result = queue.enqueue(
            kind=args.kind, lane=args.lane, name=args.name, argv=command,
            cwd=args.cwd, timeout=args.timeout, metadata=json.loads(args.metadata),
            max_attempts=args.max_attempts)
    elif args.command == "run":
        limits = {}
        for item in args.lane_limit:
            lane, separator, value = item.rpartition("=")
            if not separator or not lane:
                raise SystemExit("--lane-limit must be LANE=N")
            limits[lane] = int(value)
        result = queue.run(max_concurrency=args.max_concurrency,
                           lane_limits=limits)
    elif args.command == "retry":
        result = queue.retry(args.job_id)
    elif args.command == "cancel":
        result = queue.cancel(args.job_id, args.reason)
    else:
        result = queue.status(args.job_id)
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
