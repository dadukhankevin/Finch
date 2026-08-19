"""The Luna judge worker is isolated, native, named, and rating-blind."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks/agentic/tasks/luna_harness/judge_worker.py"
SPEC = importlib.util.spec_from_file_location("luna_judge_worker", MODULE_PATH)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


def assignment(**updates):
    value = {
        "id": "M0007", "task": "luna_harness_translation", "status": "open",
        "criteria": {"primary": "paired chrF++", "secondary": "robustness"},
        "individuals": [
            {"lineage": "L0001", "trajectory": [{"summary": "first"}],
             "evidence": {"languages_won": 3}},
            {"lineage": "L0004", "trajectory": [{"summary": "second"}],
             "evidence": {"languages_won": 6}},
        ],
        "judge": "sol-judge-M0007", "verdict": None, "voided": False,
    }
    value.update(updates)
    return value


def test_native_command_is_safe_argv_and_pins_sol_low_reasoning(tmp_path):
    command = worker.codex_command(Path("/opt/codex"), tmp_path)
    assert command[:4] == ["/opt/codex", "exec", "-m", "gpt-5.6-sol"]
    assert 'model_reasoning_effort="low"' in command
    assert ["--sandbox", "read-only"] == command[command.index("--sandbox"):command.index("--sandbox") + 2]
    assert command[-1] == "-"
    assert all(not any(symbol in item for symbol in (";", "&&", "|")) for item in command)


def test_invoke_requires_strict_verdict_and_passes_exact_assignment(tmp_path):
    seen = {}
    def fake_runner(argv, **kwargs):
        seen.update(argv=argv, kwargs=kwargs)
        return SimpleNamespace(
            returncode=0, stderr="", stdout=json.dumps({
                "winner": "L0004", "rationale": "wins six language cells",
                "evidence": {"languages_won": 6}, "confidence": 0.88}))

    result = worker.invoke_judge(
        assignment(), codex=Path("/opt/codex"), cwd=tmp_path,
        timeout=12, runner=fake_runner)

    assert result["winner"] == "L0004"
    assert json.dumps(assignment(), ensure_ascii=False, indent=2) in seen["kwargs"]["input"]
    assert seen["kwargs"]["capture_output"] is True
    assert seen["kwargs"]["timeout"] == 12


@pytest.mark.parametrize("result", [
    {"winner": "L9999", "rationale": "x", "evidence": {}, "confidence": .5},
    {"winner": "tie", "rationale": "", "evidence": {}, "confidence": .5},
    {"winner": "tie", "rationale": "x", "evidence": [], "confidence": .5},
    {"winner": "tie", "rationale": "x", "evidence": {}, "confidence": 2},
    {"winner": "tie", "rationale": "x", "evidence": {}, "confidence": .5, "extra": 1},
])
def test_strict_result_schema_rejects_bad_outputs(result):
    with pytest.raises(ValueError):
        worker.validate_result(result, assignment())


def test_assignment_rejects_rating_leak():
    leaked = assignment()
    leaked["individuals"][0]["trajectory"][0]["rating"] = 1516
    with pytest.raises(ValueError, match="ratings"):
        worker.validate_assignment(leaked, "M0007")


def test_run_once_fetches_exact_match_and_posts_named_verdict(tmp_path, monkeypatch):
    (tmp_path / "server.json").write_text(json.dumps({"port": 8123}), encoding="utf-8")
    calls = []
    def fake_request(url, body=None):
        calls.append((url, body))
        if body is None:
            return assignment()
        return {"status": "decided", "verdict": body}
    def fake_runner(argv, **kwargs):
        return SimpleNamespace(
            returncode=0, stderr="", stdout=json.dumps({
                "winner": "tie", "rationale": "evidence is balanced",
                "evidence": {"comparison": "matched"}, "confidence": .55}))
    monkeypatch.setattr(worker, "_request_json", fake_request)

    decided = worker.run_once(
        run=tmp_path, match_id="M0007", judge="sol-judge-M0007",
        codex=Path("/fake/codex"), runner=fake_runner)

    assert calls[0][0] == "http://127.0.0.1:8123/match?id=M0007"
    assert calls[1][0] == "http://127.0.0.1:8123/verdict"
    assert calls[1][1] == {
        "match": "M0007", "winner": "tie", "judge": "sol-judge-M0007",
        "rationale": "evidence is balanced",
        "evidence": {"comparison": "matched"}, "confidence": .55,
    }
    assert decided["status"] == "decided"


def test_run_once_refuses_wrong_named_judge(tmp_path, monkeypatch):
    (tmp_path / "server.json").write_text(json.dumps({"port": 8123}), encoding="utf-8")
    monkeypatch.setattr(worker, "_request_json", lambda *_args, **_kwargs: assignment())
    with pytest.raises(ValueError, match="assigned"):
        worker.run_once(
            run=tmp_path, match_id="M0007", judge="some-other-judge",
            codex=Path("/fake/codex"), runner=lambda *_a, **_k: None)
