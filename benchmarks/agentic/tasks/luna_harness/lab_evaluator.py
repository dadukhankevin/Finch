#!/usr/bin/env python3
"""Research-visible laboratory for evolved native-Luna harnesses.

This module deliberately lives beside, rather than inside, the protected
evaluator.  Laboratory references and sentence-level errors are training data;
they never become Finch fitness.  The protected evaluator can therefore remain
byte-for-byte frozen while researchers receive the empirical loop they need.
"""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from pathlib import Path

from protected_evaluator import (
    _load_upstream,
    _materializer,
    _run_sessions_with_watchdogs,
    _score_session,
    _watchdog_timeout,
    validate_harness,
)


HERE = Path(__file__).resolve().parent
LAB_SPEC_PATH = HERE / "laboratory_panel.json"
PROTECTED_TARGET_INDICES = frozenset({27156, 27022, 24200})
MAX_RUNNER_FILE_BYTES = 64 * 1024
MAX_RUNNER_TOTAL_BYTES = 256 * 1024


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_lab_spec(path: Path = LAB_SPEC_PATH) -> dict:
    spec = json.loads(path.read_text(encoding="utf-8"))
    indices = tuple(int(value) for value in spec["target_indices"])
    if len(indices) != 3 or len(set(indices)) != 3:
        raise ValueError("laboratory panel must contain three distinct tasks")
    if set(indices) & PROTECTED_TARGET_INDICES:
        raise ValueError("laboratory tasks overlap protected task indices")
    if len(spec["languages"]) < 2 or len(set(spec["languages"])) != len(
            spec["languages"]):
        raise ValueError("laboratory languages must be distinct")
    if len(spec["verse_references"]) != len(indices):
        raise ValueError("laboratory verse references do not match task count")
    return spec


def build_lab_plan(study, spec: dict) -> dict:
    """Derive an upstream-compatible plan without mutating its frozen plan."""
    plan = copy.deepcopy(study.load_plan())
    language_rows = {row["code"]: row for row in plan["language_panel"]}
    missing = [code for code in spec["languages"] if code not in language_rows]
    if missing:
        raise ValueError(f"laboratory languages absent from pinned plan: {missing}")
    plan["name"] = spec["name"]
    plan["frozen_panel"] = {
        "selection_seed": spec["selection_seed"],
        "target_indices": list(spec["target_indices"]),
        "line_numbers": list(spec["line_numbers"]),
        "verse_references": list(spec["verse_references"]),
        "verse_length_bands": list(spec["verse_length_bands"]),
        "selection_rule": spec["selection_rule"],
    }
    plan["language_panel"] = [language_rows[code] for code in spec["languages"]]
    task_count = len(spec["target_indices"])
    language_count = len(spec["languages"])
    plan["execution"]["raw_calls"] = task_count * language_count
    plan["execution"]["agent_sessions"] = language_count
    plan["execution"]["translations_per_language"] = task_count
    return plan


def _lab_tasks(study, plan: dict, work_root: Path, spec_sha256: str) -> dict:
    cache = work_root / f"_visible_lab_tasks_{spec_sha256[:16]}.json"
    if cache.is_file():
        tasks = json.loads(cache.read_text(encoding="utf-8"))
    else:
        tasks = study.tasks_with_prompts(plan)
        _write_json(cache, tasks)
        cache.chmod(0o600)
    # The pinned project builder asks this module hook for its task payload.
    study.cached_tasks_with_prompts = lambda: tasks
    return tasks


def _runner_outputs(project: Path) -> dict:
    """Capture bounded candidate-produced diagnostics, including absence."""
    root = project / ".runner"
    files = []
    total = 0
    if root.is_dir():
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(project).as_posix()
            data = path.read_bytes()
            record = {
                "path": relative,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            if (path.name != "final.json" and len(data) <= MAX_RUNNER_FILE_BYTES
                    and total + len(data) <= MAX_RUNNER_TOTAL_BYTES):
                record["text"] = data.decode("utf-8", errors="replace")
                total += len(data)
            files.append(record)
    return {"directory_created": root.is_dir(), "files": files,
            "captured_text_bytes": total}


def _tool_usage(candidate: dict, raw: dict) -> dict:
    commands = [str(call.get("command") or "")
                for call in (raw.get("trace_summary") or {}).get("tool_calls", [])]
    declared = []
    for relative in sorted(candidate.get("tools", {})):
        name = Path(relative).name
        matches = [command for command in commands
                   if relative in command or name in command]
        declared.append({"tool": relative, "observed_invocations": len(matches),
                         "commands": matches[:8]})
    return {"declared": declared, "all_commands": commands[:40]}


def _compact_session(candidate: dict, raw: dict, project: Path) -> dict:
    scored = _score_session(raw)
    return {
        "language": raw.get("language_code"),
        "language_name": raw.get("language_name"),
        "score": scored["score"],
        "failure": scored["failure"],
        "policy_violations": scored["policy_violations"],
        "timed_out": scored["timed_out"],
        "translations": raw.get("translations") or [],
        "research_phase": raw.get("research_phase"),
        "tool_usage": _tool_usage(candidate, raw),
        "runner_outputs": _runner_outputs(project),
        "input_mutations": list(raw.get("input_mutations") or []),
    }


def evaluate_lab_artifact(candidate_path: Path, *, factory_root: Path,
                          corpus_dir: Path, codex: Path, work_root: Path,
                          workers: int = 2,
                          lab_spec_path: Path = LAB_SPEC_PATH) -> dict:
    candidate = validate_harness(candidate_path)
    candidate_hash = _sha256(candidate_path)
    spec = load_lab_spec(lab_spec_path)
    spec_sha256 = _sha256(lab_spec_path)
    study, runner, agent_harness = _load_upstream(factory_root, corpus_dir)
    plan = build_lab_plan(study, spec)
    tasks = _lab_tasks(study, plan, work_root, spec_sha256)
    original_build = runner.build_project
    runner.build_project = _materializer(
        study, agent_harness, candidate, candidate_hash)
    evaluation_root = Path(tempfile.mkdtemp(
        prefix="finch-luna-visible-lab-", dir=work_root))

    def one(code: str):
        run_dir = evaluation_root / "records"
        run_dir.mkdir(parents=True, exist_ok=True)
        project_root = evaluation_root / "projects"
        initial = study.initial_agent_prompt(
            tasks[code][0]["language_name"], tasks[code],
            enforce_exploration=True, research_only=True)
        initial += "\n\n# Evolved research method\n\n" + candidate["research_prompt"]
        orchestration = candidate.get("orchestration") or {}
        if orchestration.get("research_phase_prompt"):
            initial += ("\n\n# Evolved research-phase orchestration\n\n"
                        + orchestration["research_phase_prompt"])
        continuation = (
            "The research phase is complete in this same session. Use only the "
            "supplied project, exact examples, local tool results, and evidence "
            "record. Translate all three assigned verses and return only the "
            "required structured JSON.\n\n# Evolved generation method\n\n"
            + candidate["translation_prompt"])
        if orchestration.get("final_phase_prompt"):
            continuation += ("\n\n# Evolved final-phase orchestration\n\n"
                             + orchestration["final_phase_prompt"])
        return runner.run_session(
            arm="agent_luna", code=code, run_dir=run_dir,
            work_root=project_root, codex=codex, plan=plan,
            tasks_by_language=tasks, initial_prompt=initial,
            continuation_prompt=continuation,
            research_gate_path=".runner/evidence.json")

    try:
        raw_results = _run_sessions_with_watchdogs(
            spec["languages"], one, timeout_seconds=_watchdog_timeout(plan),
            workers=workers)
    finally:
        runner.build_project = original_build
    sessions = []
    for raw in sorted(raw_results, key=lambda row: row.get("language_code") or ""):
        code = raw.get("language_code")
        project = evaluation_root / "projects" / "agent_luna" / str(code)
        sessions.append(_compact_session(candidate, raw, project))
    return {
        "schema_version": 1,
        "kind": "visible_autoresearch_laboratory_run",
        "candidate_sha256": candidate_hash,
        "candidate_name": candidate["name"],
        "lab_spec_sha256": spec_sha256,
        "languages": list(spec["languages"]),
        "task_indices": list(spec["target_indices"]),
        "metric": "equal-language mean sentence chrF++",
        "score": sum(float(row["score"]) for row in sessions) / len(sessions),
        "failures": sum(bool(row["failure"] or row["policy_violations"])
                        for row in sessions),
        "sessions": sessions,
        "private_record_root": str(evaluation_root),
    }


def build_feedback_bundle(candidate: dict, baseline: dict) -> dict:
    candidate_sessions = {row["language"]: row for row in candidate["sessions"]}
    baseline_sessions = {row["language"]: row for row in baseline["sessions"]}
    if set(candidate_sessions) != set(baseline_sessions):
        raise ValueError("candidate and laboratory baseline languages differ")
    languages = []
    all_deltas = []
    for code in sorted(candidate_sessions):
        current = candidate_sessions[code]
        control = baseline_sessions[code]
        current_tasks = {row["task_id"]: row for row in current["translations"]}
        control_tasks = {row["task_id"]: row for row in control["translations"]}
        if set(current_tasks) != set(control_tasks):
            raise ValueError(f"candidate and baseline tasks differ for {code}")
        examples = []
        for task_id in sorted(current_tasks):
            candidate_row = current_tasks[task_id]
            baseline_row = control_tasks[task_id]
            candidate_chrf = float(candidate_row["metrics"]["chrf"])
            baseline_chrf = float(baseline_row["metrics"]["chrf"])
            delta = candidate_chrf - baseline_chrf
            all_deltas.append(delta)
            examples.append({
                "task_id": task_id,
                "line_number": candidate_row["line_number"],
                "verse_reference": candidate_row["verse_reference"],
                "source_text": candidate_row["source_text"],
                "reference_text": candidate_row["reference_text"],
                "candidate_translation": candidate_row["translated_text"],
                "baseline_translation": baseline_row["translated_text"],
                "candidate_metrics": candidate_row["metrics"],
                "baseline_metrics": baseline_row["metrics"],
                "paired_chrf_delta": delta,
            })
        languages.append({
            "language": code,
            "language_name": current.get("language_name"),
            "candidate_score": current["score"],
            "baseline_score": control["score"],
            "paired_score_delta": float(current["score"]) - float(control["score"]),
            "candidate_failure": current["failure"],
            "baseline_failure": control["failure"],
            "tool_usage": current["tool_usage"],
            "runner_outputs": current["runner_outputs"],
            "research_phase": current["research_phase"],
            "examples": examples,
        })
    return {
        "schema_version": 1,
        "kind": "visible_autoresearch_feedback",
        "candidate_sha256": candidate["candidate_sha256"],
        "baseline_sha256": baseline["candidate_sha256"],
        "lab_spec_sha256": candidate["lab_spec_sha256"],
        "fitness": None,
        "fitness_status": "diagnostic training data; never Finch selection fitness",
        "mean_sentence_paired_chrf_delta": (
            sum(all_deltas) / len(all_deltas) if all_deltas else 0.0),
        "candidate_failures": candidate["failures"],
        "baseline_failures": baseline["failures"],
        "visibility": {
            "laboratory_references": "visible after each attempted translation",
            "development_1_references": "sealed",
            "development_2_references": "sealed",
            "confirmation_references": "sealed and untouched",
        },
        "research_instructions": [
            "Reason sentence by sentence from source, candidate, baseline, and reference.",
            "Separate a bad hypothesis from a tool that did not run or advice Luna ignored.",
            "Prefer a causal change that addresses observed errors; do not optimize protected panels.",
            "Treat chrF++ as reference agreement evidence, not a complete bilingual judgment.",
        ],
        "languages": languages,
    }


def evaluate_lab_pair(candidate_path: Path, baseline_path: Path, *, run: Path,
                      factory_root: Path, corpus_dir: Path, codex: Path,
                      work_root: Path, workers: int = 2) -> dict:
    """Run one candidate and reuse one frozen, research-visible baseline."""
    lab_spec_path = (run / "laboratory_panel.json"
                     if (run / "laboratory_panel.json").is_file()
                     else LAB_SPEC_PATH)
    baseline_hash = _sha256(baseline_path)
    spec_hash = _sha256(lab_spec_path)
    baseline_cache = (run / "laboratory" / "baselines"
                      / f"{baseline_hash[:16]}-{spec_hash[:16]}.json")
    if baseline_cache.is_file():
        baseline = json.loads(baseline_cache.read_text(encoding="utf-8"))
        if (baseline.get("candidate_sha256") != baseline_hash
                or baseline.get("lab_spec_sha256") != spec_hash):
            raise ValueError("laboratory baseline cache identity mismatch")
    else:
        baseline = evaluate_lab_artifact(
            baseline_path, factory_root=factory_root, corpus_dir=corpus_dir,
            codex=codex, work_root=work_root, workers=workers,
            lab_spec_path=lab_spec_path)
        _write_json(baseline_cache, baseline)
    candidate = evaluate_lab_artifact(
        candidate_path, factory_root=factory_root, corpus_dir=corpus_dir,
        codex=codex, work_root=work_root, workers=workers,
        lab_spec_path=lab_spec_path)
    return build_feedback_bundle(candidate, baseline)
