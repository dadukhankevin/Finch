"""Focused process-watchdog coverage for the protected Luna evaluator."""

import importlib.util
import multiprocessing
import os
import time
from pathlib import Path

import pytest
import json


MODULE_PATH = (Path(__file__).resolve().parents[1]
               / "benchmarks/agentic/tasks/luna_harness/protected_evaluator.py")
SPEC = importlib.util.spec_from_file_location("luna_protected_evaluator", MODULE_PATH)
evaluator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluator)


pytestmark = pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="the native Luna evaluator watchdog uses POSIX fork",
)


def test_watchdog_terminates_hung_session_once_and_reports_zero_evidence():
    calls = multiprocessing.get_context("fork").Value("i", 0)

    def callback_for(code):
        with calls.get_lock():
            calls.value += 1
        time.sleep(5)
        return {"language_code": code, "translations": [{"metrics": {"chrf": 99}}]}

    started = time.monotonic()
    rows = evaluator._run_sessions_with_watchdogs(
        ("aii",), callback_for, timeout_seconds=0.08, workers=1)

    assert time.monotonic() - started < 2
    assert calls.value == 1  # timeout is terminal; it does not retry the session
    assert len(rows) == 1
    assert rows[0]["language_code"] == "aii"
    assert rows[0]["translations"] == []
    assert rows[0]["controller_timeout"] is True
    assert rows[0]["timed_out"] is True
    assert "controller timeout" in rows[0]["failure"]
    assert evaluator._score_session(rows[0])["score"] == 0.0


def test_watchdog_preserves_successful_native_session_result():
    parent_pid = os.getpid()

    def callback_for(code):
        return {
            "language_code": code,
            "worker_pid": os.getpid(),
            "translations": [{"metrics": {"chrf": 12.5}} for _ in range(3)],
        }

    rows = evaluator._run_sessions_with_watchdogs(
        ("lif",), callback_for, timeout_seconds=1, workers=1)

    assert len(rows) == 1
    assert rows[0]["worker_pid"] != parent_pid
    assert rows[0]["controller_timeout"] is False
    assert rows[0].get("failure", "") == ""
    assert evaluator._score_session(rows[0])["score"] == 12.5


def test_paired_summary_pairs_by_language_not_execution_order():
    candidate = {"panel": "development-1", "candidate_sha256": "c", "score": 12,
                 "failures": 0, "sessions": [
                     {"language": "lif", "score": 10}, {"language": "aii", "score": 14}]}
    baseline = {"panel": "development-1", "candidate_sha256": "b", "score": 10,
                "failures": 0, "sessions": [
                    {"language": "aii", "score": 11}, {"language": "lif", "score": 9}]}
    result = evaluator.paired_summary(candidate, baseline)
    assert result["mean_delta"] == 2
    assert result["win_rate"] == 1
    assert [row["language"] for row in result["pairs"]] == ["aii", "lif"]


def test_agent_model_override_is_explicit_and_does_not_mutate_pinned_plan():
    plan = {"execution": {
        "agent_model": "gpt-5.6-luna", "model": "gpt-5.6-luna"}}
    terra = evaluator._plan_for_agent_model(plan, "gpt-5.6-terra")

    assert terra["execution"]["agent_model"] == "gpt-5.6-terra"
    assert terra["execution"]["model"] == "gpt-5.6-terra"
    assert plan["execution"]["agent_model"] == "gpt-5.6-luna"

    with pytest.raises(ValueError, match="unsupported"):
        evaluator._plan_for_agent_model(plan, "gpt-5.6-sol")


def test_large_summary_uses_all_96_sentence_pairs_and_robust_statistics():
    candidate_shards = []
    baseline_shards = []
    for number in range(1, 5):
        shard = f"large-{number}"
        candidate_sessions = []
        baseline_sessions = []
        for code in evaluator.DEVELOPMENT_LANGUAGES:
            baseline_sessions.append({
                "language": code, "score": 40.0,
                "task_scores": [30.0, 40.0, 50.0]})
            candidate_sessions.append({
                "language": code, "score": 42.0,
                "task_scores": [32.0, 42.0, 52.0]})
        common = {"panel": evaluator.LARGE_PANEL_NAME,
                  "shard": shard, "panel_sha256": "p" * 64,
                  "failures": 0}
        candidate_shards.append({
            **common, "candidate_sha256": "c" * 64,
            "sessions": candidate_sessions})
        baseline_shards.append({
            **common, "candidate_sha256": "b" * 64,
            "sessions": baseline_sessions})

    result = evaluator.large_paired_summary(
        candidate_shards, baseline_shards, panel_sha256="p" * 64)

    assert result["comparisons"] == 96
    assert result["mean_delta"] == 2.0
    assert result["median_delta"] == 2.0
    assert result["trimmed_mean_delta"] == 2.0
    assert result["win_rate"] == 1.0
    assert result["language_win_rate"] == 1.0
    assert result["paired_bootstrap"]["lower_95"] == 2.0
    assert result["paired_bootstrap"]["upper_95"] == 2.0


def test_v2_genome_accepts_local_skills_and_fixed_topology(tmp_path):
    candidate = {
        "schema_version": 2, "name": "v2", "hypothesis": "h",
        "activation_condition": "a", "fallback": "f", "falsifier": "x",
        "agent_instructions": "i", "research_prompt": "r",
        "translation_prompt": "t", "tools": {},
        "skills": {".agents/skills/align/SKILL.md": "# Alignment\nUse local corpus evidence."},
        "orchestration": {"topology": "single_session_two_turn",
                          "research_phase_prompt": "Build evidence.",
                          "final_phase_prompt": "Check coverage.",
                          "compute_envelope": {"model": "gpt-5.6-luna",
                                               "sessions_per_language": 1,
                                               "turns_per_session": 2,
                                               "tasks_per_language": 3}},
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    assert evaluator.validate_harness(path)["skills"]

    candidate["orchestration"]["topology"] = "multi_session"
    path.write_text(json.dumps(candidate), encoding="utf-8")
    with pytest.raises(ValueError, match="single_session_two_turn"):
        evaluator.validate_harness(path)
