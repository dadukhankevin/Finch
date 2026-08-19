"""Visible laboratory invariants and feedback shape."""

import importlib.util
import json
from pathlib import Path
import sys


TASK_DIR = (Path(__file__).resolve().parents[1]
            / "benchmarks/agentic/tasks/luna_harness")
MODULE_PATH = TASK_DIR / "lab_evaluator.py"
sys.path.insert(0, str(TASK_DIR))
SPEC = importlib.util.spec_from_file_location("luna_lab_evaluator", MODULE_PATH)
lab = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lab)


def test_laboratory_panel_is_task_disjoint_from_protected_evaluation():
    spec = lab.load_lab_spec()
    assert set(spec["target_indices"]).isdisjoint(lab.PROTECTED_TARGET_INDICES)
    assert len(spec["target_indices"]) == 3
    assert len(spec["languages"]) == 4
    assert "never Finch fitness" in spec["purpose"]


def _session(code, translated, chrf, *, tool_calls=1, runner=True):
    return {
        "language": code,
        "language_name": "Example",
        "score": chrf,
        "failure": "",
        "policy_violations": [],
        "timed_out": False,
        "translations": [{
            "task_id": f"{code}|line-00001",
            "line_number": 1,
            "verse_reference": "TST 1:1",
            "source_text": "source",
            "reference_text": "reference",
            "translated_text": translated,
            "metrics": {"chrf": chrf, "ter": 50.0},
        }],
        "research_phase": {"gate_valid": True},
        "tool_usage": {"declared": [{
            "tool": "tools/critic.py",
            "observed_invocations": tool_calls,
            "commands": ["python3 tools/critic.py"] if tool_calls else [],
        }], "all_commands": []},
        "runner_outputs": {"directory_created": runner, "files": []},
        "input_mutations": [],
    }


def test_feedback_exposes_errors_and_activation_without_becoming_fitness():
    candidate = {
        "candidate_sha256": "c" * 64,
        "lab_spec_sha256": "s" * 64,
        "failures": 0,
        "sessions": [_session("x", "candidate", 30.0, tool_calls=0)],
    }
    baseline = {
        "candidate_sha256": "b" * 64,
        "lab_spec_sha256": "s" * 64,
        "failures": 0,
        "sessions": [_session("x", "baseline", 20.0)],
    }

    feedback = lab.build_feedback_bundle(candidate, baseline)

    assert feedback["fitness"] is None
    assert feedback["mean_sentence_paired_chrf_delta"] == 10.0
    assert feedback["visibility"]["confirmation_references"] == "sealed and untouched"
    language = feedback["languages"][0]
    example = language["examples"][0]
    assert example == {
        "task_id": "x|line-00001", "line_number": 1,
        "verse_reference": "TST 1:1", "source_text": "source",
        "reference_text": "reference", "candidate_translation": "candidate",
        "baseline_translation": "baseline",
        "candidate_metrics": {"chrf": 30.0, "ter": 50.0},
        "baseline_metrics": {"chrf": 20.0, "ter": 50.0},
        "paired_chrf_delta": 10.0,
    }
    assert language["tool_usage"]["declared"][0]["observed_invocations"] == 0


def test_runner_output_snapshot_makes_missing_artifacts_explicit(tmp_path):
    project = tmp_path / "project"
    (project / ".runner").mkdir(parents=True)
    (project / ".runner" / "evidence.json").write_text(
        json.dumps({"ok": True}), encoding="utf-8")
    (project / ".runner" / "final.json").write_text("{}", encoding="utf-8")

    result = lab._runner_outputs(project)

    assert result["directory_created"] is True
    by_path = {row["path"]: row for row in result["files"]}
    assert by_path[".runner/evidence.json"]["text"]
    assert "text" not in by_path[".runner/final.json"]
    assert ".runner/anomaly_report.json" not in by_path
