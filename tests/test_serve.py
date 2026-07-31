"""The reporting server driven end to end over localhost HTTP —
concurrent tells included, since removing the single-writer relay is the
server's whole reason to exist."""
import json
import threading
import urllib.request

import pytest

from finch4.agentic import AgenticGA
from finch4.serve import serve


@pytest.fixture
def server(tmp_path):
    srv = serve(str(tmp_path), port=0, tasks=["alpha", "beta"],
                founders=2, children=4, consolidate_every=1, seed=0)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    port = srv.server_address[1]

    def call(name, body=None):
        url = f"http://127.0.0.1:{port}/{name}"
        if body is None:
            req = urllib.request.Request(url)
        else:
            req = urllib.request.Request(
                url, data=json.dumps(body).encode(), method="POST")
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    call.port = port
    yield call, tmp_path
    srv.shutdown()


def test_full_round_over_http(server):
    call, run_dir = server
    jobs = call("ask", {})
    assert len(jobs) == 4 and all(j["kind"] == "found" for j in jobs)

    # concurrent tells: every agent reports the moment it finishes
    def report(job):
        call("tell", {"job_id": job["job_id"], "variation": f"v-{job['job_id']}",
                      "score": float(len(job["job_id"])),
                      "artifact": f"/tmp/{job['job_id']}.py"})
    threads = [threading.Thread(target=report, args=(j,)) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = call("summary")
    assert summary["population"] == 4
    assert summary["tasks_alive"] == ["alpha", "beta"]

    assert call("due")["due"] is True
    batch = call("batch")
    assert set(batch) == {"alpha", "beta"}
    call("audit", {"id": batch["alpha"]["id"], "passed": True})

    survivors = call("consolidated", {})
    assert len(survivors) == 4
    call("rewrite", {"id": survivors[0]["id"], "variation": "deeper"})
    call("rescore", {"id": survivors[1]["id"], "score": 9.0})
    assert len(call("stale")) == 3          # one re-scored, three stale

    # state survived on disk after every mutation
    ga = AgenticGA.load(str(run_dir / "state.json"))
    assert ga.summary() == call("summary")
    # server.json advertises the port for agents
    assert "port" in json.load(open(run_dir / "server.json"))


def test_progress_page_renders(server):
    call, _ = server
    jobs = call("ask", {})
    call("tell", {"job_id": jobs[0]["job_id"], "variation": "v", "score": 1.0})
    import urllib.request
    with urllib.request.urlopen(
            f"http://127.0.0.1:{call.port}/progress") as r:
        page = r.read().decode()
    assert "drawChart" in page and "living population" in page
    data = call("page.json")
    assert data["mode"] == "agentic"
    assert data["living"][0]["id"] == "i0000"
    assert any("tell i0000" in e[1] for e in data["events"])


def test_errors_are_json_not_crashes(server):
    call, _ = server
    with pytest.raises(urllib.error.HTTPError) as e:
        call("nonsense", {})
    assert e.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as e:
        call("tell", {"job_id": "no-such-job", "variation": "x", "score": 1})
    assert e.value.code in (400, 404)


def test_telemetry_only_dashboard_and_live_progress(tmp_path):
    import numpy as np
    from finch4 import live_progress, solve

    cb = live_progress(run_dir=str(tmp_path / "live"))

    def fitness(phenotypes):        # solve()'s contract: batched
        return -(phenotypes.flatten(1) ** 2).mean(dim=1)

    solve(fitness, output_shape=(8,), epochs=4, children=4,
          population_cap=8, founders=2, device="cpu", seed=0,
          progress=cb, progress_every=1)
    with urllib.request.urlopen(cb.url) as r:
        page = r.read().decode()
    assert "drawChart" in page and "fitness over time" in page
    with urllib.request.urlopen(cb.url.replace("/progress",
                                               "/page.json")) as r:
        data = json.loads(r.read())
    assert data["mode"] == "solver" and "fn0" in data["series"]
    # telemetry persisted alongside the run
    lines = open(tmp_path / "live" / "telemetry.jsonl").read().splitlines()
    assert len(lines) >= 2
    assert "best" in json.loads(lines[-1])
    cb.server.shutdown()


def test_agentic_page_uses_multiseries_curve(server):
    call, _ = server
    jobs = call("ask", {})
    for i, job in enumerate(jobs):
        call("tell", {"job_id": job["job_id"], "variation": f"v{i}",
                      "score": float(i)})
    data = call("page.json")
    assert set(data["series"]) == {"alpha", "beta"}   # one line per task
    assert all(len(c) >= 2 for c in data["series"].values())


def test_hub_shows_every_registered_run(tmp_path, monkeypatch):
    import urllib.request
    from finch4 import hub
    from finch4.agentic import AgenticGA
    from finch4.serve import register_run

    monkeypatch.setenv("FINCH4_REGISTRY",
                       str(tmp_path / "registry.jsonl"))
    # a finished agentic run, straight from disk
    ag_dir = tmp_path / "compress_demo"
    ag_dir.mkdir()
    ga = AgenticGA(tasks=["compress"], founders=2, seed=0)
    for job in ga.ask():
        ga.tell(job["job_id"], "v", 1.0)
    ga.save(str(ag_dir / "state.json"))
    register_run(str(ag_dir), port=1)          # dead port -> finished
    # a finished solver run, telemetry only
    so_dir = tmp_path / "apple_solver"
    so_dir.mkdir()
    with open(so_dir / "telemetry.jsonl", "w") as f:
        for e in range(3):
            f.write(json.dumps({"epoch": e, "evaluations": e * 10,
                                "best": {"fn0": -1.0 + e * 0.1}}) + "\n")
    register_run(str(so_dir), port=1)
    data = hub.hub_data()
    names = [c["name"] for c in data["cards"]]
    assert names == ["compress_demo", "apple_solver"] or \
        sorted(names) == ["apple_solver", "compress_demo"]
    assert all(not c["live"] for c in data["cards"])
    assert data["cards"][0]["series"]           # mini curves present
    assert "drawChart" in hub.hub_html()


def test_hub_reads_pre_rename_registry_fallback(tmp_path, monkeypatch):
    """Runs registered by the campaign's pre-rename tooling live in
    ~/.latentspace/registry.jsonl; at the default registry location the
    hub must keep showing them (the fallback the Finch 4 port dropped —
    every other test sets FINCH4_REGISTRY, which is exactly why the drop
    was invisible)."""
    from finch4 import hub

    monkeypatch.delenv("FINCH4_REGISTRY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    new_run = tmp_path / "new_run"
    old_run = tmp_path / "old_run"
    for d, reg in ((new_run, ".finch4"), (old_run, ".latentspace")):
        d.mkdir()
        (tmp_path / reg).mkdir(exist_ok=True)
        with open(tmp_path / reg / "registry.jsonl", "a") as f:
            f.write(json.dumps({"run_dir": str(d), "port": 1,
                                "started": 1.0}) + "\n")
    names = {c["name"] for c in hub.hub_data()["cards"]}
    assert names == {"new_run", "old_run"}


def test_serve_registers_in_global_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("FINCH4_REGISTRY",
                       str(tmp_path / "reg.jsonl"))
    srv = serve(str(tmp_path / "r"), port=0, telemetry_only=True)
    entries = [json.loads(l) for l in open(tmp_path / "reg.jsonl")]
    assert entries[0]["port"] == srv.server_address[1]
    srv.server_close()
