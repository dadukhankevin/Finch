#!/usr/bin/env python3
"""Compare a fresh evidence-agent -> translator topology with frozen harnesses.

The protected evaluator normally resumes one Codex thread for generation.  This
experiment preserves its project materialization, data boundary, native Luna
command, scoring, and audit machinery, but replaces only that resume command
with a new ephemeral Codex session.  The new session receives no research
conversation; its sole inter-agent bridge is ``.runner/report.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
from pathlib import Path

from protected_evaluator import (
    _load_upstream,
    evaluate,
    paired_summary,
    validate_harness,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


@contextlib.contextmanager
def independent_agent_topology(
    factory_root: Path,
    corpus_dir: Path,
    *,
    additive: bool = False,
    mandatory_report: bool = False,
):
    """Patch the phase handoff to a fresh session plus Markdown report gate.

    In additive mode the fresh translator receives the ordinary complete
    launch prompt, including the exact few-shot examples, and ``report.md`` is
    described only as optional extra evidence.  This tests the marginal value
    of a research specialist without replacing any input or capability of the
    normal translator.
    """
    study, runner, _ = _load_upstream(factory_root, corpus_dir)
    original_resume = runner._resume_command
    original_run_session = runner.run_session

    def fresh_translation_command(
        codex: Path,
        project: Path,
        plan: dict,
        *,
        thread_id: str,
        prompt: str,
    ) -> list[str]:
        del thread_id
        if additive or mandatory_report:
            tasks = study.cached_tasks_with_prompts()
            code = project.name
            language_tasks = tasks[code]
            generation_marker = "# Evolved generation method\n\n"
            generation_prompt = (
                prompt.split(generation_marker, 1)[1]
                if generation_marker in prompt
                else prompt
            )
            baseline_prompt = study.initial_agent_prompt(
                language_tasks[0]["language_name"],
                language_tasks,
                enforce_exploration=False,
                research_only=False,
            )
            if mandatory_report:
                report_path = project / ".runner" / "report.md"
                if not report_path.is_file():
                    raise RuntimeError("mandatory research report is missing")
                report_text = report_path.read_text(encoding="utf-8")
                prompt = (
                    baseline_prompt
                    + "\n\n# Required independent research report\n\n"
                    + "The complete report from the independent evidence "
                    + "researcher is embedded below. Read and consider all of "
                    + "it before translating. It is a required evidence input, "
                    + "not unquestionable authority: verify its claims against "
                    + "the supplied examples and project, reject errors, and "
                    + "extend it with any further research needed. You remain "
                    + "responsible for every translation decision.\n\n"
                    + "<research_report>\n"
                    + report_text
                    + "\n</research_report>\n\n"
                    + "# Treatment-specific translation guidance\n\n"
                    + generation_prompt
                )
                audit_prompt = project / ".runner" / "translator_launch_prompt.txt"
                audit_prompt.write_text(prompt, encoding="utf-8")
            else:
                prompt = (
                    baseline_prompt
                    + "\n\n# Optional independent research report\n\n"
                    + "A separate evidence researcher left `.runner/report.md`. "
                    + "It is optional, fallible supporting context—not a handoff, "
                    + "authority, or substitute for the exact examples and full "
                    + "project available to you. Inspect it when useful, verify its "
                    + "material claims against the supplied evidence, and freely "
                    + "ignore, correct, or extend it with your own research. You "
                    + "remain responsible for every translation decision.\n\n"
                    + "# Treatment-specific translation guidance\n\n"
                    + generation_prompt
                )
        return runner._command(
            codex,
            project,
            plan,
            initial_prompt=prompt,
            ephemeral=True,
            output_schema=True,
            output_last_message="final.json",
        )

    def report_gated_session(**kwargs):
        kwargs["research_gate_path"] = ".runner/report.md"
        return original_run_session(**kwargs)

    runner._resume_command = fresh_translation_command
    runner.run_session = report_gated_session
    try:
        yield
    finally:
        runner._resume_command = original_resume
        runner.run_session = original_run_session


def _thread_ids(trace: Path) -> list[str]:
    ids = []
    if not trace.is_file():
        return ids
    for line in trace.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "thread.started" and event.get("thread_id"):
            value = str(event["thread_id"])
            if value not in ids:
                ids.append(value)
    return ids


def _audit_and_copy_reports(private_root: Path, destination: Path) -> dict:
    reports = {}
    destination.mkdir(parents=True, exist_ok=True)
    projects = private_root / "projects" / "agent_luna"
    traces = private_root / "records" / "traces"
    for project in sorted(projects.iterdir() if projects.is_dir() else []):
        report = project / ".runner" / "report.md"
        translator_prompt = project / ".runner" / "translator_launch_prompt.txt"
        trace = traces / f"agent_luna-{project.name}.jsonl"
        ids = _thread_ids(trace)
        if report.is_file():
            copied = destination / f"{project.name}.md"
            shutil.copy2(report, copied)
            report_hash = _sha256(report)
            report_bytes = report.stat().st_size
        else:
            copied = None
            report_hash = None
            report_bytes = 0
        if translator_prompt.is_file():
            prompt_text = translator_prompt.read_text(encoding="utf-8")
            prompt_hash = _sha256(translator_prompt)
            prompt_bytes = translator_prompt.stat().st_size
            report_embedded = bool(
                report.is_file()
                and report.read_text(encoding="utf-8") in prompt_text
            )
            baseline_blocks = prompt_text.count("# Baseline evidence for ")
        else:
            prompt_hash = None
            prompt_bytes = 0
            report_embedded = False
            baseline_blocks = 0
        reports[project.name] = {
            "report_path": str(copied) if copied else None,
            "report_sha256": report_hash,
            "report_bytes": report_bytes,
            "translator_prompt_sha256": prompt_hash,
            "translator_prompt_bytes": prompt_bytes,
            "report_text_embedded_verbatim": report_embedded,
            "few_shot_task_blocks_in_translator_prompt": baseline_blocks,
            "distinct_native_thread_count": len(ids),
            "independent_sessions_verified": len(ids) >= 2,
        }
    return reports


def benchmark(args: argparse.Namespace) -> dict:
    run = args.run.resolve()
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    factory_root = args.factory_root.resolve()
    corpus_dir = Path(manifest["corpus_dir"])
    codex = Path(manifest["codex_binary"])
    work_root = args.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    work_root.chmod(0o700)

    artifacts = {
        "neutral": args.neutral.resolve(),
        "staged": args.staged.resolve(),
        "independent": args.independent.resolve(),
    }
    for artifact in artifacts.values():
        validate_harness(artifact)

    common = {
        "factory_root": factory_root,
        "corpus_dir": corpus_dir,
        "codex": codex,
        "work_root": work_root,
        "panel": args.panel,
        "workers": args.workers,
        "model": "gpt-5.6-luna",
        "retain_translations": True,
    }
    _write_json(args.status, {
        "status": "running",
        "model": "gpt-5.6-luna",
        "panel": args.panel,
        "completed_arms": [],
    })

    results = {}
    if args.reuse_baselines_from:
        prior = json.loads(
            args.reuse_baselines_from.resolve().read_text(encoding="utf-8")
        )
        if prior.get("panel") != args.panel:
            raise RuntimeError("reused baselines belong to another panel")
        for arm in ("neutral", "staged"):
            expected_hash = _sha256(artifacts[arm])
            observed_hash = ((prior.get("artifacts") or {}).get(arm) or {}).get("sha256")
            if observed_hash != expected_hash:
                raise RuntimeError(f"reused {arm} artifact identity mismatch")
            private_root = Path(prior["private_record_roots"][arm])
            private_result = private_root / "result.json"
            if not private_result.is_file():
                raise RuntimeError(f"reused {arm} private result is missing")
            results[arm] = json.loads(private_result.read_text(encoding="utf-8"))
    else:
        for arm in ("neutral", "staged"):
            results[arm] = evaluate(artifacts[arm], **common)
    _write_json(args.status, {
        "status": "running",
        "model": "gpt-5.6-luna",
        "panel": args.panel,
        "completed_arms": list(results),
        "scores": {name: value["score"] for name, value in results.items()},
        "reused_baselines": bool(args.reuse_baselines_from),
    })

    with independent_agent_topology(
        factory_root,
        corpus_dir,
        additive=args.additive,
        mandatory_report=args.mandatory_report,
    ):
        results["independent"] = evaluate(artifacts["independent"], **common)
    topology = _audit_and_copy_reports(
        Path(results["independent"]["private_record_root"]),
        args.reports.resolve(),
    )

    pairwise = {
        "independent_vs_staged": paired_summary(
            results["independent"], results["staged"]),
        "independent_vs_neutral": paired_summary(
            results["independent"], results["neutral"]),
        "staged_vs_neutral": paired_summary(
            results["staged"], results["neutral"]),
    }
    record = {
        "schema_version": 1,
        "kind": "native_luna_independent_evidence_agent_benchmark",
        "model": "gpt-5.6-luna",
        "panel": args.panel,
        "translation_items_per_arm": sum(
            len(session.get("task_scores") or [])
            for session in results["independent"]["sessions"]
        ),
        "topology": {
            "researcher": "fresh native Luna evidence-only session",
            "bridge": ".runner/report.md",
            "translator": "fresh ephemeral native Luna translation session",
            "handoff_mode": (
                "mandatory_embedded_report"
                if args.mandatory_report
                else (
                    "additive_optional_context" if args.additive else "report_gated"
                )
            ),
            "translator_receives_normal_few_shot_prompt": bool(
                args.additive or args.mandatory_report
            ),
            "translator_receives_report_contents_in_prompt": bool(
                args.mandatory_report
            ),
            "audits": topology,
        },
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in artifacts.items()
        },
        "scores": {name: value["score"] for name, value in results.items()},
        "failures": {name: value["failures"] for name, value in results.items()},
        "pairwise": pairwise,
        "private_record_roots": {
            name: value["private_record_root"] for name, value in results.items()
        },
        "reused_baselines_from": (
            str(args.reuse_baselines_from.resolve())
            if args.reuse_baselines_from else None
        ),
    }
    _write_json(args.output.resolve(), record)
    _write_json(args.status.resolve(), {
        "status": "complete",
        "model": "gpt-5.6-luna",
        "panel": args.panel,
        "scores": record["scores"],
        "result": str(args.output.resolve()),
    })
    return record


def main() -> None:
    here = Path(__file__).resolve().parent
    ethos = here / "experiments" / "aquilla_ethos"
    independent = here / "experiments" / "independent_evidence_agent"
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--factory-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--panel", default="selection")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--additive",
        action="store_true",
        help=(
            "give the fresh translator the complete ordinary few-shot launch "
            "prompt and treat report.md only as optional extra evidence"
        ),
    )
    parser.add_argument(
        "--mandatory-report",
        action="store_true",
        help=(
            "embed the complete report and ordinary few-shot prompt directly "
            "in the fresh translator's required launch context"
        ),
    )
    parser.add_argument("--reuse-baselines-from", type=Path)
    parser.add_argument("--neutral", type=Path, default=ethos / "neutral_harness.json")
    parser.add_argument("--staged", type=Path, default=here / "starter_harness.json")
    parser.add_argument(
        "--independent",
        type=Path,
        default=independent / "independent_evidence_harness.json",
    )
    parser.add_argument("--reports", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(args)
    print(json.dumps({
        "scores": result["scores"],
        "pairwise_mean_deltas": {
            name: value["mean_delta"] for name, value in result["pairwise"].items()
        },
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
