import json
import multiprocessing
import sys
import time

import pytest

from finch4.workers import JobQueue, main


def py(code, *args):
    return [sys.executable, "-c", code, *map(str, args)]


def _enqueue_many(root, prefix, count):
    queue = JobQueue(root)
    for index in range(count):
        queue.enqueue(kind="test", lane="cpu", name=f"{prefix}-{index}",
                      argv=py("pass"), metadata={"writer": prefix})


def test_enqueue_persists_opaque_job_and_status_returns_copy(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    job = queue.enqueue(
        kind="evolve", lane="gpu-a", name="lineage-7",
        argv=py("print('ok')"), cwd=tmp_path, timeout=2,
        metadata={"lineage": "L0007", "score": None}, max_attempts=3)
    assert job["id"] == "J000001"
    assert job["status"] == "queued"
    assert job["metadata"] == {"lineage": "L0007", "score": None}
    assert job["cwd"] == str(tmp_path.resolve())
    assert job["attempts"] == []

    job["metadata"]["lineage"] = "mutated"
    persisted = json.loads((tmp_path / "queue" / "jobs.json").read_text())
    assert persisted["schema_version"] == 1
    assert persisted["next_id"] == 2
    assert persisted["jobs"][0]["metadata"]["lineage"] == "L0007"


def test_concurrent_enqueues_are_fcntl_safe(tmp_path):
    root = tmp_path / "queue"
    context = multiprocessing.get_context("fork")
    processes = [context.Process(target=_enqueue_many,
                                 args=(root, f"p{i}", 8))
                 for i in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join(5)
        assert process.exitcode == 0
    data = JobQueue(root).status()
    ids = [job["id"] for job in data["jobs"]]
    assert len(ids) == 32 and len(set(ids)) == 32
    assert data["next_id"] == 33


def test_run_captures_logs_and_records_success_failure_and_crash(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    success = queue.enqueue(kind="k", lane="a", name="success",
                            argv=py("import sys; print('OUT'); print('ERR', file=sys.stderr)"))
    failure = queue.enqueue(kind="k", lane="a", name="failure",
                            argv=py("import sys; print('bad'); sys.exit(7)"))
    crash = queue.enqueue(kind="k", lane="b", name="crash",
                          argv=[str(tmp_path / "does-not-exist")])
    queue.run(max_concurrency=3)

    a = queue.status(success["id"])
    b = queue.status(failure["id"])
    c = queue.status(crash["id"])
    assert a["status"] == "succeeded" and a["attempts"][0]["returncode"] == 0
    assert b["status"] == "failed" and b["attempts"][0]["returncode"] == 7
    assert c["status"] == "crashed" and "FileNotFoundError" in c["attempts"][0]["error"]
    assert (queue.root / a["attempts"][0]["stdout_path"]).read_text().strip() == "OUT"
    assert (queue.root / a["attempts"][0]["stderr_path"]).read_text().strip() == "ERR"


def test_timeout_is_recorded_and_process_group_is_stopped(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    job = queue.enqueue(kind="k", lane="slow", name="timeout",
                        argv=py("import time; time.sleep(10)"), timeout=0.08)
    started = time.monotonic()
    queue.run(poll_interval=0.005)
    record = queue.status(job["id"])
    assert time.monotonic() - started < 2
    assert record["status"] == "timed_out"
    assert "exceeded timeout" in record["attempts"][0]["error"]
    assert record["attempts"][0]["returncode"] != 0


def test_automatic_and_explicit_retry(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    marker = tmp_path / "marker"
    code = ("from pathlib import Path; import sys; p=Path(sys.argv[1]); "
            "exists=p.exists(); p.write_text('x'); sys.exit(0 if exists else 9)")
    automatic = queue.enqueue(kind="k", lane="a", name="auto",
                              argv=py(code, marker), max_attempts=2)
    queue.run()
    record = queue.status(automatic["id"])
    assert record["status"] == "succeeded"
    assert [a["status"] for a in record["attempts"]] == ["failed", "succeeded"]

    failed = queue.enqueue(kind="k", lane="a", name="manual",
                           argv=py("import sys; sys.exit(4)"))
    queue.run()
    retried = queue.retry(failed["id"])
    assert retried["status"] == "queued" and retried["max_attempts"] == 2
    queue.run()
    assert len(queue.status(failed["id"])["attempts"]) == 2
    with pytest.raises(KeyError):
        queue.retry(automatic["id"][:-1] + "9")


def test_cancel_prevents_queued_job_from_launching_and_records_reason(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    marker = tmp_path / "must-not-exist"
    job = queue.enqueue(
        kind="k", lane="a", name="obsolete",
        argv=py("from pathlib import Path; Path(__import__('sys').argv[1]).touch()",
                marker))
    cancelled = queue.cancel(job["id"], "candidate changed after enqueue")
    assert cancelled["status"] == "cancelled"
    assert cancelled["cancel_reason"] == "candidate changed after enqueue"
    queue.run()
    assert not marker.exists()
    with pytest.raises(ValueError, match="queued"):
        queue.cancel(job["id"], "again")


def test_global_and_per_lane_concurrency_limits(tmp_path):
    queue = JobQueue(tmp_path / "queue")
    state = tmp_path / "state"
    state.mkdir()
    code = """
import os, pathlib, sys, time
root, lane = pathlib.Path(sys.argv[1]), sys.argv[2]
mine = root / (lane + '-' + str(os.getpid()))
mine.write_text('1')
all_count = len(list(root.iterdir()))
lane_count = len(list(root.glob(lane + '-*')))
with (root / 'observed').open('a') as f:
    f.write(f'{all_count},{lane},{lane_count}\\n')
time.sleep(.12)
mine.unlink()
"""
    for index, lane in enumerate(["a", "a", "b", "b"]):
        queue.enqueue(kind="k", lane=lane, name=str(index),
                      argv=py(code, state, lane))
    queue.run(max_concurrency=3, lane_limits={"a": 1, "b": 2},
              poll_interval=0.005)
    observed = (state / "observed").read_text().splitlines()
    values = [line.split(",") for line in observed]
    assert max(int(value[0]) for value in values) <= 4  # includes observed itself
    assert max(int(value[2]) for value in values if value[1] == "a") == 1
    assert max(int(value[2]) for value in values if value[1] == "b") <= 2
    assert all(j["status"] == "succeeded" for j in queue.status()["jobs"])


def test_cli_enqueue_run_and_status(tmp_path, capsys):
    root = tmp_path / "queue"
    assert main(["--root", str(root), "enqueue", "--kind", "test",
                 "--lane", "cpu", "--name", "cli", "--",
                 *py("print('cli-ok')")]) == 0
    enqueued = json.loads(capsys.readouterr().out)
    assert enqueued["id"] == "J000001"
    assert main(["--root", str(root), "run", "--max-concurrency", "2",
                 "--lane-limit", "cpu=1"]) == 0
    capsys.readouterr()
    assert main(["--root", str(root), "status", "J000001"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["status"] == "succeeded"
    assert (root / status["attempts"][0]["stdout_path"]).read_text().strip() == "cli-ok"
