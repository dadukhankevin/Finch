"""The GAR communication plane end to end over localhost HTTP — concurrent
evolver report streams and pairwise judge verdicts included."""
import json
import threading
import urllib.error
import urllib.request

import pytest

from finch4.agentic import Campaign
from finch4.serve import GAService, lineage_curves, serve


def test_lineage_curves_only_plot_the_active_fitness_regime():
    campaign = Campaign(["alpha"])
    line = campaign.found("alpha", "test a mechanism")
    campaign.report(line["id"], "small panel", score=8.0,
                    source="old evaluator", kept=True)
    campaign.start_fitness_regime(
        "alpha", "large-96-v1", "replace the noisy panel")

    series, points = lineage_curves(
        list(campaign.lineages.values()), campaign.fitness_regimes)
    assert series == {}
    assert points == []

    campaign.report(line["id"], "large panel", score=0.5,
                    source="new evaluator", kept=None)
    series, points = lineage_curves(
        list(campaign.lineages.values()), campaign.fitness_regimes)
    assert series == {"alpha": [(2, 0.5)]}
    assert points == [(2, 0.5, line["id"])]


def test_lineage_curves_do_not_treat_search_as_sealed():
    campaign = Campaign(["alpha"])
    line = campaign.found("alpha", "fit the visible cells")
    campaign.report(line["id"], "in-sample perfect", score=1.0,
                    source="search", kept=True)
    series, points = lineage_curves(list(campaign.lineages.values()))
    assert series == {"alpha": [(1, 1.0)]}
    assert "sealed" not in "".join(series)

    campaign.report(line["id"], "hidden set", score=1.0,
                    source="allocator", kept=True, sealed_score=0.4)
    series, points = lineage_curves(list(campaign.lineages.values()))
    assert series["alpha · search"][-1] == (2, 1.0)
    assert series["alpha · sealed"][-1] == (2, 0.4)
    assert points[-1] == (2, 1.0, line["id"])


def test_holdout_is_plotted_best_so_far_and_never_becomes_best():
    campaign = Campaign(["alpha"])
    line = campaign.found("alpha", "held-in vs holdout")
    campaign.report(line["id"], "first", score=0.2, source="allocator",
                    kept=True, sealed_score=0.2, holdout_score=0.05)
    campaign.report(line["id"], "worse holdout, better held-in",
                    score=0.4, source="allocator", kept=True,
                    sealed_score=0.4, holdout_score=-0.1)
    series, _ = lineage_curves(list(campaign.lineages.values()))
    assert series["alpha · sealed"][-1] == (2, 0.4)
    assert series["alpha · holdout"] == [(1, 0.05), (2, 0.05)]
    assert "alpha · search" not in series
    assert campaign.lineages[line["id"]]["best_score"] == 0.4


@pytest.fixture
def server(tmp_path):
    srv = serve(str(tmp_path), port=0, tasks=["alpha", "beta"])
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    port = srv.server_address[1]

    def call(name, body=None):
        url = f"http://127.0.0.1:{port}/{name}"
        request = (urllib.request.Request(url) if body is None else
                   urllib.request.Request(
                       url, data=json.dumps(body).encode(), method="POST"))
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())

    call.port = port
    yield call, tmp_path
    srv.shutdown()


def test_full_campaign_over_http_with_concurrent_reports(server):
    call, run_dir = server
    lines = [call("found", {"task": task, "rationale": f"angle {index}"})
             for index, task in enumerate(["alpha", "alpha", "beta"])]

    artifacts = {}
    for line in lines:
        path = run_dir / f"{line['id']}.py"
        path.write_text(f"# candidate {line['id']}\n")
        artifacts[line["id"]] = str(path)

    def stream(lineage, score):
        call("report", {"lineage": lineage["id"],
                        "summary": f"experiment on {lineage['id']}",
                        "score": score, "source": "protected evaluator",
                        "kept": True,
                        "artifact": artifacts[lineage["id"]]})

    threads = [threading.Thread(target=stream, args=(line, float(i + 1)))
               for i, line in enumerate(lines)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    summary = call("summary")
    assert summary["lineages"] == 3 and summary["reports"] == 3
    assert summary["best"] == {"alpha": 2.0, "beta": 3.0}
    decoder = call("decoder")
    assert decoder["version"] == 3
    disk = (run_dir / "Decoder.md").read_text()
    assert disk == decoder["artifact"]
    for line in lines:
        assert f"experiment on {line['id']}" in disk
        assert f"[{line['id']}#0@" in disk

    call("audit", {"lineage": lines[1]["id"], "report": 0,
                   "outcome": "passed",
                   "rationale": "reproduced", "evidence": {"rerun": 2.0}})
    audited = call("lineages")[1]["reports"][0]
    token = (f"[{lines[1]['id']}#0@"
             f"{audited['artifact_sha256'][:8]}]")
    published = call("incorporate", {
        "artifact": f"Decoder.md v1 text {token}",
        "rationale": "absorb the win"})
    assert published["version"] == 4
    assert call("decoder")["version"] == 4
    assert (run_dir / "Decoder.md").read_text() == (
        f"Decoder.md v1 text {token}")
    # every lineage scored before this incorporate, so every one is stale now
    assert set(call("stale")) == {line["id"] for line in lines}

    call("kill", {"lineage": lines[0]["id"],
                  "rationale": "dominated by its sibling"})
    decisions = call("decisions")
    actions = [d["action"] for d in decisions]
    assert actions[:3] == ["found", "found", "found"]
    assert actions[3:6] == ["share", "share", "share"]
    assert actions[6:] == ["audit", "incorporate", "kill"]

    # state survived on disk after every mutation
    campaign = Campaign.load(run_dir / "state.json")
    assert campaign.summary() == call("summary")
    assert "port" in json.loads((run_dir / "server.json").read_text())


def test_campaign_evidence_and_plateau_routes(server):
    call, run_dir = server
    call("baseline", {"task": "alpha", "score": 1, "source": "control",
                      "rationale": "freeze control"})
    line = call("found", {"task": "alpha", "rationale": "screen"})
    artifact = run_dir / "candidate.py"
    artifact.write_text("# candidate\n")
    call("report", {"lineage": line["id"], "summary": "leader",
                    "score": 2, "source": "screen",
                    "artifact": str(artifact)})
    call("audit", {"lineage": line["id"], "report": 0,
                   "outcome": "passed", "rationale": "reproduced"})
    call("champion", {"lineage": line["id"], "report": 0,
                      "source": "replicate study", "rationale": "won",
                      "evidence": {"replicates": 5}})
    call("plateau", {"task": "alpha", "plateaued": True,
                     "rationale": "allocation exhausted",
                     "evidence": {"since_gain": 8}})
    summary = call("summary")
    assert summary["screening_best"]["alpha"] == 2
    assert summary["baseline"]["alpha"] == 1
    assert summary["verified_champion"]["alpha"] == 2
    assert summary["plateau_alert"] is True


def test_dashboard_mirrors_durable_native_worker_jobs(server):
    call, run_dir = server
    queue = {
        "schema_version": 1, "next_id": 2,
        "jobs": [
            {"id": "J000000", "lane": "evolver", "kind": "sol",
             "name": "L0001 experiment 1", "status": "running",
             "max_attempts": 2, "attempts": [{"number": 1}]},
            {"id": "J000001", "lane": "judge", "kind": "sol",
             "name": "M0000 judge", "status": "queued",
             "max_attempts": 1, "attempts": []},
        ],
    }
    (run_dir / "jobs.json").write_text(json.dumps(queue), encoding="utf-8")

    assert call("jobs") == queue
    page = call("page.json")
    assert page["summary"]["jobs"] == 2
    assert page["summary"]["running_jobs"] == 1
    assert page["summary"]["queued_jobs"] == 1
    assert [job["id"] for job in page["jobs"]] == ["J000000", "J000001"]


def test_pairwise_judge_and_elo_routes(server):
    call, run_dir = server
    call("criteria", {
        "task": "alpha",
        "criteria": {"primary": "canonical score", "secondary": "novelty"},
        "rationale": "give every judge the same fitness contract"})
    lines = [call("found", {"task": "alpha", "rationale": f"angle {i}"})
             for i in range(2)]
    for i, line in enumerate(lines):
        artifact = run_dir / f"pair-{i}.py"
        artifact.write_text(f"VALUE = {i}\n")
        call("report", {
            "lineage": line["id"], "summary": f"candidate {i}",
            "score": float(i), "source": "canonical scorer",
            "evidence": {"cell": i}, "artifact": str(artifact)})
    match = call("pair", {
        "individuals": [[lines[0]["id"], 0], [lines[1]["id"], 0]],
        "rationale": "run one head-to-head", "requested_by": "allocator",
        "judge": "judge-agent"})
    assignment = call(f"match?id={match['id']}")
    assert assignment["id"] == match["id"]
    assert len(assignment["individuals"]) == 2
    assert call("research")["decoder"]["version"] == 2

    decided = call("verdict", {
        "match": match["id"], "winner": lines[1]["id"],
        "judge": "judge-agent", "rationale": "higher score and stronger artifact",
        "evidence": {"score_delta": 1.0}})
    assert decided["elo_update"]["after"][lines[1]["id"]] == 1516
    ratings = call("ratings?task=alpha")
    assert ratings[0]["lineage"] == lines[1]["id"]
    summary = call("summary")
    assert summary["matches"] == 1 and summary["verdicts"] == 1
    page = call("page.json")
    assert page["lineages"][0]["elo"] == 1516
    assert page["matches"][0]["verdict"]["judge"] == "judge-agent"


def test_concurrent_founders_from_separate_services_reserve_unique_ids(tmp_path):
    """Two controllers with stale Campaign copies must not share an ID."""
    state_path = tmp_path / "state.json"
    Campaign(tasks=["alpha"]).save(state_path)
    first = GAService(Campaign.load(state_path), str(state_path),
                      run_dir=str(tmp_path))
    second = GAService(Campaign.load(state_path), str(state_path),
                       run_dir=str(tmp_path))
    barrier = threading.Barrier(2)
    created = []

    def found(service, label):
        barrier.wait()
        created.append(service.handle("found", {
            "task": "alpha", "rationale": f"concurrent {label}"}))

    threads = [threading.Thread(target=found, args=(service, label))
               for service, label in ((first, "one"), (second, "two"))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert {line["id"] for line in created} == {"L0000", "L0001"}
    assert set(Campaign.load(tmp_path / "state.json").lineages) == {
        "L0000", "L0001"}


def test_trust_membrane_over_http(server):
    call, _ = server
    line = call("found", {"task": "alpha", "rationale": "seed"})
    with pytest.raises(urllib.error.HTTPError) as error:
        call("report", {"lineage": line["id"], "summary": "no source",
                        "score": 1.0})
    assert error.value.code == 400
    call("report", {"lineage": line["id"], "summary": "worker claim",
                    "claimed_score": 50.0})
    assert call("summary")["best"]["alpha"] is None
    assert call("summary")["untrusted_reports"] == 1
    with pytest.raises(urllib.error.HTTPError) as error:
        call("incorporate", {"artifact": "x",
                             "rationale": "absorb a claim"})
    assert error.value.code == 400


def test_share_publishes_scored_findings_without_promoting_a_method(server):
    call, run_dir = server
    line = call("found", {"task": "alpha", "rationale": "test it"})
    candidate = run_dir / "negative.py"
    candidate.write_text("VALUE = 0\n")
    report = call("report", {
        "lineage": line["id"], "summary": "did not help", "score": 0.0,
        "source": "protected evaluator", "kept": False,
        "artifact": str(candidate)})
    token = (f"[{line['id']}#0@{report['artifact_sha256'][:8]}]")
    shared = call("share", {
        "artifact": f"# Findings\n\nKnown failure {token}.\n",
        "rationale": "future workers should not repeat it"})
    assert shared["kind"] == "share"
    assert shared["citation_levels"] == {f"{line['id']}#0": "scored"}
    assert (run_dir / "Decoder.md").read_text() == shared["artifact"]
    assert call("decisions")[-1]["action"] == "share"
    compacted = call("compact", {
        "artifact": f"Known failure, concise: {token}.",
        "rationale": "retain the scored finding without upgrading it"})
    assert compacted["citation_levels"] == {f"{line['id']}#0": "scored"}


def test_concurrent_agent_findings_append_without_lost_updates(server):
    call, run_dir = server
    lines = [call("found", {"task": "alpha", "rationale": f"angle {i}"})
             for i in range(2)]
    fragments = []
    for index, line in enumerate(lines):
        artifact = run_dir / f"finding-{index}.py"
        artifact.write_text(f"VALUE = {index}\n")
        report = call("report", {
            "lineage": line["id"], "summary": f"tested {index}",
            "score": float(index), "source": "protected evaluator",
            "artifact": str(artifact)})
        token = f"[{line['id']}#0@{report['artifact_sha256'][:8]}]"
        fragments.append(f"### Finding {index}\n\nObserved result {token}.")

    published = []
    barrier = threading.Barrier(2)

    def append(index):
        barrier.wait()
        published.append(call("finding", {
            "fragment": fragments[index],
            "rationale": f"agent {index} publishes its own result"}))

    threads = [threading.Thread(target=append, args=(index,))
               for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    decoder = call("decoder")
    assert decoder["version"] == 4
    assert all(fragment in decoder["artifact"] for fragment in fragments)
    assert (run_dir / "Decoder.md").read_text() == decoder["artifact"]
    assert sorted(item["version"] for item in published) == [3, 4]


def test_server_refuses_unversioned_decoder_file_drift(tmp_path):
    state = Campaign(["alpha"], decoder="canonical")
    state.save(tmp_path / "state.json")
    (tmp_path / "Decoder.md").write_text("loose manual edit")

    with pytest.raises(ValueError, match="differs from Finch"):
        serve(str(tmp_path), port=0)


def test_revise_score_route_evicts_false_best(server):
    call, _ = server
    line = call("found", {"task": "alpha", "rationale": "seed"})
    other = call("found", {"task": "alpha", "rationale": "seed two"})
    call("report", {"lineage": line["id"], "summary": "suspicious",
                    "score": 1000.0, "source": "evaluator"})
    call("report", {"lineage": other["id"], "summary": "honest",
                    "score": 5.0, "source": "evaluator"})
    call("revise-score", {"lineage": line["id"], "report": 0,
                          "score": -99.0, "source": "audit rerun",
                          "rationale": "fraud exposed"})
    assert call("summary")["best"]["alpha"] == 5.0


def test_report_artifact_is_an_immutable_content_snapshot(server):
    call, run_dir = server
    line = call("found", {"task": "alpha", "rationale": "seed"})
    candidate = run_dir / "candidate.py"
    candidate.write_text("VALUE = 1\n")
    report = call("report", {
        "lineage": line["id"], "summary": "first candidate",
        "score": 1.0, "source": "evaluator", "artifact": str(candidate)})
    snapshot = report["artifact"]
    candidate.write_text("VALUE = 999\n")

    assert snapshot != str(candidate)
    assert open(snapshot).read() == "VALUE = 1\n"
    assert len(report["artifact_sha256"]) == 64
    open(snapshot, "w").write("VALUE = -1\n")
    with pytest.raises(urllib.error.HTTPError) as error:
        call("audit", {"lineage": line["id"], "report": 0,
                       "outcome": "passed", "rationale": "rerun"})
    assert error.value.code == 400


def test_progress_page_and_lineage_rows(server):
    call, _ = server
    line = call("found", {"task": "alpha", "rationale": "seed",
                          "idea": "an angle"})
    call("report", {"lineage": line["id"], "summary": "first experiment",
                    "score": 1.0, "source": "evaluator", "kept": True})
    with urllib.request.urlopen(
            f"http://127.0.0.1:{call.port}/progress") as response:
        page = response.read().decode()
    assert "drawChart" in page and "lineages" in page
    data = call("page.json")
    assert data["mode"] == "agentic"
    row = data["lineages"][0]
    assert row["id"] == line["id"] and row["kind"] == "inject"
    assert row["best"] == 1.0 and row["reports"] == 1
    assert data["decoder"]["version"] == 0
    assert data["tree"]["counts"]["lineages"] == 1
    assert any(node["id"] == f"{line['id']}#0"
               for node in data["tree"]["nodes"])
    assert {"lineages", "running", "reports", "untrusted_reports",
            "audit_passed_reports", "decoder_version", "decisions",
            "stale"} <= set(data["summary"])
    assert any("report" in event[1] for event in data["events"])
    assert set(data["series"]) == {"alpha"}
    tree = call("tree")
    assert tree["counts"]["reports"] == 1


def test_errors_are_json_not_crashes(server):
    call, _ = server
    with pytest.raises(urllib.error.HTTPError) as error:
        call("ask", {})                      # the v2 route is gone
    assert error.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as error:
        call("report", {"lineage": "L9999", "summary": "ghost"})
    assert error.value.code in (400, 404)
    with pytest.raises(urllib.error.HTTPError) as error:
        call("kill", {"lineage": "L9999", "rationale": "ghost"})
    assert error.value.code in (400, 404)


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
    assert "drawChart" in page
    with urllib.request.urlopen(cb.url.replace("/progress",
                                               "/page.json")) as r:
        data = json.loads(r.read())
    assert data["mode"] == "solver" and "fn0" in data["series"]
    lines = open(tmp_path / "live" / "telemetry.jsonl").read().splitlines()
    assert len(lines) >= 2
    assert "best" in json.loads(lines[-1])
    cb.server.shutdown()


def test_hub_shows_agentic_and_archived_runs(tmp_path, monkeypatch):
    from finch4 import hub
    from finch4.serve import register_run

    monkeypatch.setenv("FINCH4_REGISTRY",
                       str(tmp_path / "registry.jsonl"))
    # a finished agentic campaign, straight from disk
    new_dir = tmp_path / "campaign_demo"
    new_dir.mkdir()
    campaign = Campaign(tasks=["compress"])
    line = campaign.found("compress", "seed")
    campaign.report(line["id"], "gain", score=1.0, source="evaluator")
    campaign.report(line["id"], "better", score=2.0, source="evaluator")
    campaign.save(str(new_dir / "state.json"))
    register_run(str(new_dir), port=1)          # dead port -> finished
    # an ARCHIVED pre-high-agent run (old schema) must still render
    old_dir = tmp_path / "archived_v1"
    old_dir.mkdir()
    (old_dir / "state.json").write_text(json.dumps({
        "schema_version": 1,
        "individuals": {
            "i0000": {"id": "i0000", "task": "binpack", "score": 0.9,
                      "alive": True},
            "i0001": {"id": "i0001", "task": "binpack", "score": 0.95,
                      "alive": True}},
        "best": {"binpack": {"id": "i0001", "score": 0.95}}}))
    register_run(str(old_dir), port=1)
    data = hub.hub_data()
    cards = {card["name"]: card for card in data["cards"]}
    assert cards["campaign_demo"]["best"] == {"compress": 2.0}
    assert cards["campaign_demo"]["series"]
    assert cards["archived_v1"]["best"] == {"binpack": 0.95}
    assert cards["archived_v1"]["series"]
    assert "drawChart" in hub.hub_html()


def test_serve_registers_in_global_registry(tmp_path, monkeypatch):
    monkeypatch.setenv("FINCH4_REGISTRY",
                       str(tmp_path / "reg.jsonl"))
    srv = serve(str(tmp_path / "r"), port=0, telemetry_only=True)
    entries = [json.loads(l) for l in open(tmp_path / "reg.jsonl")]
    assert entries[0]["port"] == srv.server_address[1]
    srv.server_close()
