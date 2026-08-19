"""Small invariants for the native Luna trajectory controller."""

import importlib.util
import json
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

import pytest


TASK_DIR = (Path(__file__).resolve().parents[1]
            / "benchmarks/agentic/tasks/luna_harness")
MODULE_PATH = TASK_DIR / "native_control.py"
sys.path.insert(0, str(TASK_DIR))
SPEC = importlib.util.spec_from_file_location("luna_native_control", MODULE_PATH)
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


def test_new_lineage_inherits_frozen_baseline_score(tmp_path, monkeypatch):
    run = tmp_path / "run"
    (run / "controls").mkdir(parents=True)
    (run / "lineages" / "L0000").mkdir(parents=True)
    (run / "starter_harness.json").write_text("{}", encoding="utf-8")
    (run / "controls" / "selection-baseline.json").write_text(
        json.dumps({"score": 44.5, "fitness_score": 0.0}), encoding="utf-8")
    campaign = type("CampaignStub", (), {
        "decoder_version": 0,
        "current_decoder": lambda self: {
            "version": 0, "artifact": "shared baseline",
            "artifact_sha256": "a" * 64},
        "lineages": {"L0000": {
            "parents": [], "best_report": None, "reports": [],
            "idea": "test direction",
        }},
    })()
    monkeypatch.setattr(control, "_line", lambda *_: (campaign, campaign.lineages["L0000"]))

    control._prepare(run, "L0000")

    metadata = json.loads((run / "lineages" / "L0000"
                           / "experiment.json").read_text())
    assert metadata["seed_score"] == 0.0


def test_failed_evaluation_never_becomes_a_lineage_champion():
    assert control._keeps_candidate({"score": 99.0, "failures": 1}, 44.5) is False
    assert control._keeps_candidate({"score": 45.0, "failures": 0}, 44.5) is True


def _block(*deltas, failures=0):
    return {"pairs": [{"language": str(i), "delta": value}
                      for i, value in enumerate(deltas)],
            "candidate_failures": failures}


def test_tournament_stops_early_only_for_broad_clear_results():
    assert control._tournament_decision(
        [_block(-2, -2, -1, 0)])["outcome"] == "clear_loss"
    assert control._tournament_decision(
        [_block(4, 4, 4, -1)])["outcome"] == "large_win"
    assert control._tournament_decision(
        [_block(2, 2, -1, -1)])["outcome"] == "continue"


def test_two_block_promotion_requires_positive_delta_and_win_rate():
    decision = control._tournament_decision(
        [_block(2, 2, -1, -1), _block(2, 2, 2, -1)], final=True)
    assert decision["outcome"] == "win"
    assert decision["comparisons"] == 8
    assert decision["win_rate"] == 0.625


def test_large_decision_only_automates_material_confident_results():
    base = {
        "mean_delta": 0.3, "win_rate": 0.54, "wins": 52,
        "comparisons": 96, "candidate_failures": 0,
        "paired_bootstrap": {"lower_95": -0.8, "upper_95": 1.3},
    }
    assert control._large_decision(base)["outcome"] == "inconclusive"
    win = {**base, "mean_delta": 1.5,
           "paired_bootstrap": {"lower_95": 0.2, "upper_95": 2.8}}
    assert control._large_decision(win)["outcome"] == "win"
    loss = {**base, "mean_delta": -2.0,
            "paired_bootstrap": {"lower_95": -3.1, "upper_95": -0.4}}
    assert control._large_decision(loss)["outcome"] == "clear_loss"


def test_native_sol_evolver_uses_prepared_prompt_and_validates_candidate(
        tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    lineage = run / "lineages" / "L0042"
    lineage.mkdir(parents=True)
    prompt = "Evolve this prepared lineage safely.\n"
    (lineage / "prompt.md").write_text(prompt, encoding="utf-8")
    (lineage / "candidate.json").write_text(
        (TASK_DIR / "starter_harness.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    fake_codex = tmp_path / "fake-codex"
    fake_codex.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "pathlib.Path(os.environ['FAKE_CODEX_RECEIPT']).write_text(json.dumps({\n"
        "  'argv': sys.argv[1:], 'stdin': sys.stdin.read(), 'cwd': os.getcwd()\n"
        "}))\n",
        encoding="utf-8")
    fake_codex.chmod(fake_codex.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("FAKE_CODEX_RECEIPT", str(receipt))

    control.evolve_native(SimpleNamespace(
        run=str(run), lineage="L0042", codex=str(fake_codex), timeout=10))

    invocation = json.loads(receipt.read_text(encoding="utf-8"))
    assert invocation["stdin"] == prompt
    assert invocation["argv"] == [
        "exec", "-m", "gpt-5.6-sol",
        "-c", 'model_reasoning_effort="low"',
        "-c", 'web_search="disabled"',
        "--disable", "multi_agent", "--sandbox", "workspace-write",
        "--ephemeral", "--skip-git-repo-check",
        "-C", str(lineage), "-",
    ]
    assert invocation["cwd"] == str(Path.cwd())
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "ready_for_evaluation"
    assert result["lineage"] == "L0042"


def test_native_sol_evolver_exits_when_generated_candidate_is_invalid(
        tmp_path, monkeypatch):
    run = tmp_path / "run"
    lineage = run / "lineages" / "L0007"
    lineage.mkdir(parents=True)
    (lineage / "prompt.md").write_text("prepared prompt", encoding="utf-8")
    (lineage / "candidate.json").write_text("{}", encoding="utf-8")
    fake_codex = tmp_path / "fake-codex"
    fake_codex.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_codex.chmod(fake_codex.stat().st_mode | stat.S_IXUSR)

    try:
        control.evolve_native(SimpleNamespace(
            run=str(run), lineage="L0007", codex=str(fake_codex), timeout=10))
    except SystemExit as exc:
        assert "candidate validation failed" in str(exc)
    else:
        raise AssertionError("invalid candidate should stop native evolution")


def test_queue_evolver_creates_an_opaque_finch_job_that_chains_laboratory(
        tmp_path, monkeypatch, capsys):
    run = tmp_path / "run"
    lineage = run / "lineages" / "L0009"
    lineage.mkdir(parents=True)
    (lineage / "prompt.md").write_text("prepared", encoding="utf-8")
    (lineage / "experiment.json").write_text(
        json.dumps({"experiment": 3}), encoding="utf-8")
    line = {"idea": "new retrieval interaction"}
    monkeypatch.setattr(control, "_line", lambda *_: (object(), line))

    control.queue_evolver(SimpleNamespace(
        run=str(run), lineage="L0009", agent_timeout=111,
        job_timeout=120, agent_attempts=2, after="lab",
        lab_workers=2, lab_timeout=800, lab_attempts=1, score_workers=2,
        score_timeout=900, score_attempts=1))

    job = json.loads(capsys.readouterr().out)
    assert job["lane"] == "evolver" and job["kind"] == "sol-evolver"
    assert job["metadata"]["lineage"] == "L0009"
    assert job["metadata"]["experiment"] == 3
    assert "--enqueue-lab" in job["argv"]
    assert "--enqueue-score" not in job["argv"]
    assert job["argv"][0] == sys.executable
    queue = json.loads((run / "jobs.json").read_text(encoding="utf-8"))
    assert [item["id"] for item in queue["jobs"]] == [job["id"]]

    # Repeating the task-level request is idempotent even though the generic
    # worker queue itself deliberately has no semantic deduplication policy.
    control.queue_evolver(SimpleNamespace(
        run=str(run), lineage="L0009", agent_timeout=111,
        job_timeout=120, agent_attempts=2, after="lab",
        lab_workers=2, lab_timeout=800, lab_attempts=1, score_workers=2,
        score_timeout=900, score_attempts=1))
    capsys.readouterr()
    assert len(json.loads((run / "jobs.json").read_text())["jobs"]) == 1


def test_prepare_names_visible_lab_feedback_but_keeps_protected_refs_sealed(
        tmp_path, monkeypatch):
    run = tmp_path / "run"
    lineage_dir = run / "lineages" / "L0004"
    lineage_dir.mkdir(parents=True)
    (run / "starter_harness.json").write_text(
        (TASK_DIR / "starter_harness.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    (run / "Decoder.md").write_text("shared", encoding="utf-8")
    (run / "CONTRACT.md").write_text("law", encoding="utf-8")
    (run / "public_fixture").mkdir()
    feedback = {
        "kind": "visible_autoresearch_feedback",
        "mean_sentence_paired_chrf_delta": -1.25,
    }
    feedback_path = lineage_dir / "lab_feedback_e0001_a01_abcdef01.json"
    feedback_path.write_text(json.dumps(feedback), encoding="utf-8")
    line = {"parents": [], "best_report": None, "reports": [],
            "idea": "diagnose actual errors"}
    campaign = type("CampaignStub", (), {
        "decoder_version": 0,
        "current_decoder": lambda self: {
            "version": 0, "artifact": "shared",
            "artifact_sha256": "b" * 64},
        "lineages": {"L0004": line}})()
    monkeypatch.setattr(control, "_line", lambda *_: (campaign, line))

    prepared = control._prepare(run, "L0004")

    prompt = (lineage_dir / "prompt.md").read_text(encoding="utf-8")
    metadata = json.loads((lineage_dir / "experiment.json").read_text())
    assert str(feedback_path) in prompt
    assert "source, your prior" in prompt
    assert "protected development and" in prompt
    assert "sole reference-data" in prompt and "exception and may be used" in prompt
    assert metadata["lab_feedback"] == str(feedback_path)
    assert metadata["lab_feedback_sha256"]
    assert metadata["decoder_sha256"] == "b" * 64
    assert Path(metadata["shared_research"]).read_text() == "shared"
    assert prepared["experiment"] == 1


def test_prepare_snapshots_versioned_research_not_a_drifted_loose_file(
        tmp_path, monkeypatch):
    run = tmp_path / "run"
    lineage_dir = run / "lineages" / "L0002"
    lineage_dir.mkdir(parents=True)
    (run / "starter_harness.json").write_text("{}", encoding="utf-8")
    (run / "Decoder.md").write_text("unversioned drift", encoding="utf-8")
    (run / "CONTRACT.md").write_text("law", encoding="utf-8")
    (run / "public_fixture").mkdir()
    line = {"parents": [], "best_report": None, "reports": [],
            "idea": "read the actual population memory"}
    campaign = type("CampaignStub", (), {
        "decoder_version": 7,
        "current_decoder": lambda self: {
            "version": 7, "artifact": "canonical shared finding",
            "artifact_sha256": "c" * 64},
    })()
    monkeypatch.setattr(control, "_line", lambda *_: (campaign, line))

    control._prepare(run, "L0002")

    metadata = json.loads((lineage_dir / "experiment.json").read_text())
    snapshot = Path(metadata["shared_research"])
    assert snapshot.read_text() == "canonical shared finding"
    assert "current shared research" in (lineage_dir / "prompt.md").read_text()


def test_prepare_gives_continuation_its_latest_score_and_audit(tmp_path,
                                                               monkeypatch):
    run = tmp_path / "run"
    lineage_dir = run / "lineages" / "L0008"
    lineage_dir.mkdir(parents=True)
    artifact = run / "winner.json"
    artifact.write_text("{}", encoding="utf-8")
    (run / "CONTRACT.md").write_text("law", encoding="utf-8")
    (run / "public_fixture").mkdir()
    report = {
        "index": 0, "summary": "dual derivation", "score": 0.326,
        "source": "matched evaluator", "evidence": {"wins": 5},
        "kept": True, "claimed_score": None, "artifact": str(artifact),
        "artifact_sha256": "d" * 64, "experiment_id": "experiment-0001",
        "voided": False, "decoder_version": 4, "audit_status": "passed",
        "audit_history": [{"outcome": "passed", "evidence": {
            "decision": {"mean_delta": 0.326, "wins": 5,
                         "comparisons": 8}}}],
    }
    line = {
        "parents": [], "best_report": 0, "best_score": 0.326,
        "reports": [report], "idea": "continue a verified mechanism",
        "status": "running",
    }
    campaign = type("CampaignStub", (), {
        "decoder_version": 4,
        "current_decoder": lambda self: {
            "version": 4, "artifact": "shared research",
            "artifact_sha256": "e" * 64},
    })()
    monkeypatch.setattr(control, "_line", lambda *_: (campaign, line))

    control._prepare(run, "L0008")

    metadata = json.loads((lineage_dir / "experiment.json").read_text())
    trajectory = json.loads(Path(metadata["trajectory"]).read_text())
    assert trajectory["reports"][0]["audit_status"] == "passed"
    assert trajectory["reports"][0]["audit_history"][0]["evidence"][
        "decision"]["mean_delta"] == 0.326
    prompt = (lineage_dir / "prompt.md").read_text()
    assert str(Path(metadata["trajectory"])) in prompt


def test_reflection_agent_authors_cited_fragment_and_finch_publishes_it(
        tmp_path, monkeypatch, capsys):
    from finch4.agentic import Campaign

    run = tmp_path / "run"
    lineage_dir = run / "lineages" / "L0000"
    lineage_dir.mkdir(parents=True)
    artifact = run / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    campaign = Campaign([control.TASK], decoder="# Shared\n")
    line = campaign.found(control.TASK, "test reflection")
    report = campaign.report(
        line["id"], "tested a sparse tool", score=-1.25,
        source="protected evaluator", kept=False,
        artifact=str(artifact), artifact_sha256="d" * 64,
        evidence={"decision": {"wins": 2, "comparisons": 8}})
    campaign.save(run / "state.json")
    codex = run / "codex"
    codex.write_text("", encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps({"codex_binary": str(codex)}), encoding="utf-8")

    def fake_run(*_args, **_kwargs):
        citation = campaign.citation(line["id"], report["index"])
        (lineage_dir / "finding_r0000.md").write_text(
            f"### Sparse tool loss\n\nThe guarded tool lost {citation}.",
            encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    published = {}

    def fake_call(_run, name, body=None):
        assert name == "finding"
        published.update(body)
        return {"version": 1, "artifact_sha256": "e" * 64}

    monkeypatch.setattr(control.subprocess, "run", fake_run)
    monkeypatch.setattr(control, "_call", fake_call)

    control.reflect_native(SimpleNamespace(
        run=str(run), lineage=line["id"], report=0,
        codex=str(codex), timeout=30))

    result = json.loads(capsys.readouterr().out)
    assert campaign.citation(line["id"], 0) in published["fragment"]
    assert result["decoder_version"] == 1
    assert (lineage_dir / "finding_receipt_r0000.json").is_file()


def test_queue_missing_reflections_skips_receipts_and_untrusted_reports(
        tmp_path, monkeypatch, capsys):
    from finch4.agentic import Campaign

    run = tmp_path / "run"
    run.mkdir()
    campaign = Campaign([control.TASK])
    line = campaign.found(control.TASK, "recover old reports")
    lineage_dir = run / "lineages" / line["id"]
    lineage_dir.mkdir(parents=True)
    artifact = lineage_dir / "artifact.json"
    artifact.write_text("{}")
    digest = control.hashlib.sha256(artifact.read_bytes()).hexdigest()
    campaign.report(
        line["id"], "already reflected", score=1.0, source="evaluator",
        artifact=str(artifact), artifact_sha256=digest)
    campaign.report(
        line["id"], "needs reflection", score=-1.0, source="evaluator",
        artifact=str(artifact), artifact_sha256=digest)
    campaign.report(line["id"], "worker claim only", claimed_score=9.0)
    campaign.save(run / "state.json")
    (lineage_dir / "finding_receipt_r0000.json").write_text("{}")

    queued = []

    def fake_reflection(run_path, lineage_id, report_index, **kwargs):
        queued.append((run_path, lineage_id, report_index, kwargs))
        return {"id": "J000123", "status": "queued"}

    monkeypatch.setattr(control, "_reflection_job", fake_reflection)
    args = SimpleNamespace(
        run=str(run), lineage=[line["id"]], agent_timeout=10,
        job_timeout=20, max_attempts=3)
    control.queue_missing_reflections(args)

    result = json.loads(capsys.readouterr().out)
    assert [item[2] for item in queued] == [1]
    assert result["queued"] == [{
        "report": f"{line['id']}#1", "job": "J000123",
        "status": "queued"}]
    assert result["already_published"] == [f"{line['id']}#0"]
    assert result["ineligible"] == [f"{line['id']}#2"]


def test_interrupted_tournament_resumes_ledgered_hidden_block_without_repeat(
        tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.json"
    baseline = tmp_path / "baseline.json"
    candidate.write_text('{"candidate": 1}', encoding="utf-8")
    baseline.write_text('{"baseline": 1}', encoding="utf-8")
    block = "development-recovery"
    monkeypatch.setattr(control, "DEVELOPMENT_BLOCKS", {block: ("x",)})
    candidate_hash = control.hashlib.sha256(candidate.read_bytes()).hexdigest()
    baseline_hash = control.hashlib.sha256(baseline.read_bytes()).hexdigest()
    public = _block(1.5, 1.0, 0.5, 0.25)
    public.update(candidate_score=40.0, baseline_score=39.0)
    control._append_match(tmp_path, {
        "candidate_sha256": candidate_hash,
        "baseline_sha256": baseline_hash,
        "block": block, "verification": False, "result": public,
    })
    monkeypatch.setattr(
        control, "evaluate_pair",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("ledgered hidden block must not run twice")))

    result = control._run_tournament(
        tmp_path, candidate, baseline,
        {"factory_root": ".", "corpus_dir": ".", "codex_binary": ".",
         "work_root": "."}, workers=1)

    assert result["blocks"] == [public]
    assert result["decision"]["outcome"] == "win"
    assert result["private_record_roots"] == [None]


def test_unambiguous_v2_packaging_is_normalized_without_touching_mechanism(
        tmp_path):
    candidate = json.loads(
        (TASK_DIR / "starter_harness.json").read_text(encoding="utf-8"))
    candidate["schema_version"] = 2
    candidate["skills"] = {}
    candidate["hypothesis"] = "preserve this mechanism exactly"
    candidate["orchestration"] = {
        "topology": "single_session_two_turn",
        "research_phase": "research mechanism",
        "final_phase": "final mechanism",
        "compute_envelope": "descriptive prose",
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")

    assert control._normalize_v2_packaging(path) is True
    repaired = control.validate_harness(path)
    assert repaired["hypothesis"] == "preserve this mechanism exactly"
    assert repaired["orchestration"] == {
        "topology": "single_session_two_turn",
        "research_phase_prompt": "research mechanism",
        "final_phase_prompt": "final mechanism",
        "compute_envelope": control.FIXED_COMPUTE_ENVELOPE,
    }
    assert control._normalize_v2_packaging(path) is False


def test_score_refuses_a_candidate_that_changed_after_enqueue(
        tmp_path, monkeypatch):
    run = tmp_path / "run"
    lineage = run / "lineages" / "L0001"
    lineage.mkdir(parents=True)
    (lineage / "candidate.json").write_text(
        (TASK_DIR / "starter_harness.json").read_text(encoding="utf-8"),
        encoding="utf-8")
    (lineage / "experiment.json").write_text(
        json.dumps({"experiment": 1}), encoding="utf-8")
    monkeypatch.setattr(
        control, "_line", lambda *_: (object(), {"reports": []}))

    with pytest.raises(SystemExit, match="queued candidate changed"):
        control.score(SimpleNamespace(
            run=str(run), lineage="L0001", candidate=None,
            result_json=None, workers=1, expected_sha256="0" * 64))
