#!/usr/bin/env python3
"""Rerun one additive evidence-researcher plus full-translator session."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from independent_two_agent_benchmark import independent_agent_topology
from protected_evaluator import (
    _cached_tasks,
    _cardinality_text,
    _load_upstream,
    _materializer,
    _plan_for_agent_model,
    _score_session,
    sha256,
    validate_harness,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--language", required=True)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--factory-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    candidate_path = args.candidate.resolve()
    candidate = validate_harness(candidate_path)
    manifest = json.loads((args.run.resolve() / "manifest.json").read_text())
    corpus_dir = Path(manifest["corpus_dir"])
    codex = Path(manifest["codex_binary"])
    factory_root = args.factory_root.resolve()
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    study, runner, agent_harness = _load_upstream(factory_root, corpus_dir)
    plan = _plan_for_agent_model(study.load_plan(), "gpt-5.6-luna")
    tasks = _cached_tasks(study, work_root)
    code = args.language
    if code not in tasks:
        raise ValueError(f"unknown language: {code}")

    original_build = runner.build_project
    runner.build_project = _materializer(
        study, agent_harness, candidate, sha256(candidate_path)
    )
    evaluation_root = Path(tempfile.mkdtemp(
        prefix=f"finch-luna-additive-rerun-{code}-", dir=work_root
    ))
    try:
        records = evaluation_root / "records"
        records.mkdir(parents=True, exist_ok=True)
        projects = evaluation_root / "projects"
        initial = study.initial_agent_prompt(
            tasks[code][0]["language_name"], tasks[code],
            enforce_exploration=True, research_only=True,
        )
        initial += "\n\n# Independent evidence-research method\n\n" + _cardinality_text(
            candidate["research_prompt"], 10
        )
        continuation = (
            "# Evolved generation method\n\n"
            + _cardinality_text(candidate["translation_prompt"], 10)
        )
        with independent_agent_topology(
            factory_root, corpus_dir, additive=True
        ):
            raw = runner.run_session(
                arm="agent_luna",
                code=code,
                run_dir=records,
                work_root=projects,
                codex=codex,
                plan=plan,
                tasks_by_language=tasks,
                initial_prompt=initial,
                continuation_prompt=continuation,
                research_gate_path=".runner/report.md",
            )
    finally:
        runner.build_project = original_build

    scored = _score_session(raw, retain_translations=True)
    public_items = [
        {
            "task_id": row["task_id"],
            "verse_reference": row["verse_reference"],
            "source_text": row["source_text"],
            "translated_text": row["translated_text"],
            "chrf": row["metrics"]["chrf"],
        }
        for row in scored.get("translations", [])
    ]
    output = {
        "kind": "native_luna_single_language_additive_rerun",
        "candidate_name": candidate["name"],
        "candidate_sha256": sha256(candidate_path),
        "language": code,
        "score": scored["score"],
        "failure": scored["failure"],
        "policy_violations": scored["policy_violations"],
        "items": public_items,
        "private_record_root": str(evaluation_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "language": code,
        "score": scored["score"],
        "items": [
            {"verse": row["verse_reference"], "chrf": row["chrf"]}
            for row in public_items
        ],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
