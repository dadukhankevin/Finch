#!/usr/bin/env python3
"""Controls for Sol-evolved, Sol-judged, native-Luna harness GAR."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finch4.agentic import Campaign, report_citation
from finch4.serve import serve
from finch4.workers import JobQueue

from protected_evaluator import (
    DEVELOPMENT_BLOCKS, FIXED_COMPUTE_ENVELOPE, LARGE_PANEL_NAME,
    LARGE_PANEL_VERSION, LARGE_TRANSLATION_ITEMS, create_large_development_panel,
    evaluate, evaluate_pair, large_paired_summary, validate_harness,
)
from lab_evaluator import evaluate_lab_pair


HERE = Path(__file__).resolve().parent
TASK = "luna_harness_translation"
FOUNDER_ANGLES = (
    "Build a corpus-native alignment workbench that exposes repeated-verse anchors and contrastive bilingual correspondences to Luna.",
    "Induce reusable source-to-target templates with typed slots, then make Luna instantiate and verify those templates.",
    "Create an evidence-coverage planner that tracks propositions, roles, negation, quantities, and unsupported target material.",
    "Create an attestation-constrained editor that changes baseline wording only when corpus evidence supports the edit.",
    "Discover useful target morphology and segmentation from the supplied corpus, with concordance evidence for each analysis.",
    "Use contrast sets among nearby source meanings to induce a more discriminating glossary than positive retrieval alone.",
    "Generate multiple corpus-grounded derivations and select a consensus translation without extra Luna sessions.",
    "Add ambiguity-triggered branch-and-test: spend effort only where retrieved evidence conflicts or coverage is uncertain.",
    "Build a reverse-source adequacy critic from corpus-derived correspondences to catch omissions and role reversals.",
    "Build a language-agnostic draft anomaly critic that surfaces corpus-backed warnings, confidence, and counterevidence to Luna without mechanically rewriting the draft.",
    "Use visible laboratory prediction/reference errors to discover one causal harness intervention, and verify that its tool actually activates before claiming its mechanism worked.",
    "Compare independent draft derivations and surface only their consequential disagreements, with a principled abstention path when the corpus cannot resolve them.",
    "Organize the fixed evidence into diverse roles—lexical, syntactic, discourse, and morphological—without changing the data.",
    "Infer a compact language profile from the supplied target corpus and route Luna to language-appropriate analysis steps.",
    "Invent a distinct, falsifiable harness capability that changes Luna's translation decisions using only supplied data.",
)


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _large_regime_files(factory_root: Path, corpus_dir: Path,
                        work_root: Path, starter_sha256: str):
    """Create/reuse the private development panel and return its manifest
    fields.  Both fresh campaigns and live migrations use this one path so
    their fitness contracts cannot silently drift apart."""
    panel_path = work_root / "_protected_large_development_v1.json"
    if not panel_path.is_file():
        create_large_development_panel(
            factory_root=factory_root, corpus_dir=corpus_dir,
            destination=panel_path)
    panel_hash = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    baseline_path = work_root / (
        f"_protected_large_baseline_{panel_hash[:16]}_"
        f"{starter_sha256[:16]}.json")
    return {
        "evaluation_regime": LARGE_PANEL_VERSION,
        "measurement": (
            "one-shot paired chrF++ over 96 distinct verse-language items "
            "versus one frozen reusable pre-GAR baseline"),
        "development_translation_items": LARGE_TRANSLATION_ITEMS,
        "development_verses_per_language": 12,
        "development_shards_per_language": 4,
        "large_development_panel": str(panel_path),
        "large_development_panel_sha256": panel_hash,
        "large_baseline_cache": str(baseline_path),
        "exact_search_repeats": False,
        "verification_repeats_reserved": False,
        "session_budget": (
            "four independent persistent Luna sessions per language; three "
            "translations per session"),
    }


def _call(run: Path, name: str, body=None):
    info = json.loads((run / "server.json").read_text(encoding="utf-8"))
    url = f"http://127.0.0.1:{info['port']}/{name}"
    request = (urllib.request.Request(url) if body is None else
               urllib.request.Request(
                   url, data=json.dumps(body).encode(), method="POST",
                   headers={"Content-Type": "application/json"}))
    try:
        with urllib.request.urlopen(request) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Finch server rejected {name!r} ({error.code}): {detail}") from error


def _git_value(repo: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], text=True,
        capture_output=True, check=True).stdout.strip()


def initialize(args):
    run = Path(args.run).resolve()
    factory = Path(args.factory_root).resolve()
    corpus = Path(args.corpus_dir).resolve()
    codex = Path(args.codex).resolve()
    # Preserve a virtual-environment launcher path instead of resolving its
    # symlink to the base interpreter; the protected evaluator's dependencies
    # belong to that environment.
    evaluator_python = Path(os.path.abspath(os.path.expanduser(
        args.evaluator_python or sys.executable)))
    if run.exists() and any(run.iterdir()):
        raise SystemExit("run directory is not empty")
    if not (factory / "benchmarks/luna-prompt-vs-agent-confirmation").is_dir():
        raise SystemExit("factory root lacks the pinned Luna confirmation benchmark")
    if not (corpus / "eng-engULB.txt").is_file():
        raise SystemExit("corpus directory lacks eng-engULB.txt")
    if not codex.is_file():
        raise SystemExit(f"Codex CLI not found: {codex}")
    if not evaluator_python.is_file():
        raise SystemExit(f"evaluator Python not found: {evaluator_python}")
    run.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "CONTRACT.md", run / "CONTRACT.md")
    shutil.copy2(HERE / "base_playbook.md", run / "Decoder.md")
    shutil.copy2(HERE / "starter_harness.json", run / "starter_harness.json")
    shutil.copy2(HERE / "laboratory_panel.json", run / "laboratory_panel.json")
    shutil.copytree(HERE / "public_fixture", run / "public_fixture")
    validate_harness(run / "starter_harness.json")
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    # Pin the exact worktree used by the evaluator. The source repository may
    # have several divergent research branches, so origin/main is not an
    # adequate identity for a detached benchmark worktree.
    commit = _git_value(factory, "rev-parse", "HEAD")
    version = subprocess.run(
        [str(codex), "--version"], text=True, capture_output=True,
        check=True).stdout.strip()
    starter_sha256 = hashlib.sha256(
        (run / "starter_harness.json").read_bytes()).hexdigest()
    large_regime = _large_regime_files(
        factory, corpus, work_root, starter_sha256)
    manifest = {
        "contract": "native-sol-to-luna-harness-gar-v3-elo",
        "orchestrator": "root Codex allocation policy; Finch durable native-process dispatcher",
        "research_workers": "native Codex gpt-5.6-sol exec jobs launched by Finch, low reasoning effort",
        "translation_workers": "native Codex gpt-5.6-luna exec sessions",
        "model_api_calls": False,
        "population_selection": "named Sol judge verdicts over two trajectories; Finch Elo bookkeeping",
        "selection_languages": ["aii", "lif", "gaq", "usa"],
        "generalization_languages": ["mpj", "kgk", "tzo", "wbi"],
        "confirmation_languages": ["hat", "tam", "fra", "jpn"],
        "confirmation_status": "UNTOUCHED",
        "development_blocks": {key: list(value) for key, value in DEVELOPMENT_BLOCKS.items()},
        "laboratory": {
            "role": "research-visible diagnostic training data; never Finch fitness",
            "panel": "laboratory_panel.json",
            "panel_sha256": hashlib.sha256(
                (run / "laboratory_panel.json").read_bytes()).hexdigest(),
            "references": "visible only after each laboratory translation attempt",
        },
        "same_hidden_block_per_match": True,
        "factory_root": str(factory),
        "factory_commit": commit,
        "corpus_dir": str(corpus),
        "codex_binary": str(codex),
        "codex_version": version,
        "work_root": str(work_root),
        "starter_sha256": starter_sha256,
        "evaluator_sha256": hashlib.sha256(
            (HERE / "protected_evaluator.py").read_bytes()).hexdigest(),
        "evaluator_python": str(evaluator_python),
        "initial_lineage_target": 12,
        "initial_lineage_ceiling": 25,
        **large_regime,
    }
    _write_json(run / "manifest.json", manifest)
    print(json.dumps({"run": str(run), "manifest": manifest},
                     ensure_ascii=False))


def run_server(args):
    run = Path(args.run).resolve()
    server = serve(run, port=args.port, tasks=[TASK])
    settings = _settings(run)
    desired_regime = settings.get("evaluation_regime", "initial")
    active_regime = server.service.campaign.fitness_regime(TASK)
    if desired_regime != active_regime:
        if desired_regime != LARGE_PANEL_VERSION:
            raise RuntimeError(
                f"unsupported manifest fitness regime: {desired_regime}")
        server.service.handle("regime", {
            "task": TASK,
            "name": LARGE_PANEL_VERSION,
            "rationale": (
                "Start this campaign on its manifest's 96-item one-shot "
                "fitness contract."),
            "evidence": {
                "translation_items": LARGE_TRANSLATION_ITEMS,
                "verses_per_language": 12,
                "languages": 8,
                "panel_sha256": settings["large_development_panel_sha256"],
            },
        })
    print(json.dumps({
        "port": server.server_address[1],
        "progress": f"http://127.0.0.1:{server.server_address[1]}/progress",
    }), flush=True)
    server.serve_forever()


def _line(run: Path, lineage_id: str):
    campaign = Campaign.load(run / "state.json")
    try:
        return campaign, campaign.lineages[lineage_id]
    except KeyError:
        raise SystemExit(f"unknown lineage: {lineage_id}")


def _report_ref(value: str):
    try:
        lineage, report = value.split(":", 1)
        return [lineage, int(report)]
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("parent must be LINEAGE:REPORT")


def _latest_lab_feedback(run: Path, lineage_dir: Path):
    bundles = sorted(lineage_dir.glob("lab_feedback_e*_a*_*.json"))
    global_latest = run / "laboratory" / "latest_feedback.json"
    shared = list(run.glob("lineages/*/lab_feedback_e*_a*_*.json"))
    path = (bundles[-1] if bundles else global_latest if global_latest.is_file()
            else max(shared, key=lambda item: item.stat().st_mtime)
            if shared else global_latest)
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("kind") != "visible_autoresearch_feedback":
        raise ValueError(f"unexpected laboratory bundle kind: {path}")
    return path, value


def _prepare(run: Path, lineage_id: str):
    campaign, line = _line(run, lineage_id)
    active_regime = (campaign.fitness_regime(line.get("task", TASK))
                     if hasattr(campaign, "fitness_regime") else "initial")
    out = run / "lineages" / lineage_id
    out.mkdir(parents=True, exist_ok=True)
    decoder = campaign.current_decoder()
    research_snapshot = out / (
        f"shared_research_v{int(decoder['version']):04d}.md")
    research_snapshot.write_text(decoder["artifact"], encoding="utf-8")
    # A continuation needs its own measured trajectory as well as population
    # memory. Keep this as an ordinary task-adapter snapshot: Finch owns the
    # record, while the evolver decides what the evidence means. In particular,
    # audit outcomes that arrive after the original shared finding must not be
    # hidden from the lineage's next experiment.
    trajectory_snapshot = out / (
        f"trajectory_before_e{len(line['reports']) + 1:04d}.json")
    trajectory_reports = []
    for report in line["reports"]:
        trajectory_reports.append({
            key: report.get(key) for key in (
                "index", "summary", "score", "source", "evidence", "kept",
                "claimed_score", "artifact", "artifact_sha256",
                "experiment_id", "voided", "decoder_version",
                "fitness_regime", "audit_status", "audit_history")
        })
    _write_json(trajectory_snapshot, {
        "lineage": lineage_id,
        "idea": line.get("idea"),
        "status": line.get("status"),
        "best_score": line.get("best_score"),
        "best_report": line.get("best_report"),
        "fitness_regime": active_regime,
        "regime_seed_report": line.get("regime_seed_report"),
        "reports": trajectory_reports,
    })
    if line["parents"]:
        seed = Path(line["parents"][0]["artifact"])
        parent_score = line["parents"][0]["score"]
    elif line["best_report"] is not None:
        report = line["reports"][line["best_report"]]
        seed = Path(report["artifact"])
        parent_score = report["score"]
    elif line.get("regime_seed_report") is not None:
        report = line["reports"][int(line["regime_seed_report"])]
        seed = Path(report["artifact"])
        # The organism crosses the evaluator boundary; its old number does not.
        parent_score = None
    else:
        seed = run / "starter_harness.json"
        control = run / "controls" / "selection-baseline.json"
        parent_score = None
        if control.is_file():
            control_record = json.loads(control.read_text(encoding="utf-8"))
            score = float(control_record.get("fitness_score", 0.0))
            if math.isfinite(score):
                parent_score = score
    shutil.copy2(seed, out / "seed_candidate.json")
    shutil.copy2(seed, out / "candidate.json")
    for index, parent in enumerate(line["parents"], 1):
        shutil.copy2(parent["artifact"], out / f"parent_{index}.json")
    # A voided report still consumed its experiment id.  Reusing that id makes
    # the campaign server reject a repaired rescore as a duplicate submission.
    experiment = len(line["reports"]) + 1
    parent_context = ""
    if line["parents"]:
        rows = [
            f"- parent_{index}.json: "
            f"{report_citation(parent['lineage'], parent['report'], parent['artifact_sha256'])}, "
            f"selection chrF++ {parent['score']}, audit {parent['audit_status']}"
            for index, parent in enumerate(line["parents"], 1)]
        parent_context = (
            "This is crossover. Treat each parent as scored evidence, not shared "
            "truth. Implement one coherent causal interaction; do not concatenate "
            "their prompts or tools.\n" + "\n".join(rows) + "\n")
    continuation = ""
    if experiment > 1:
        continuation = (
            "This is a continuation. Read log.md and take the mechanism to its "
            "next logical step. If the assumption looks exhausted, make a creative "
            "departure instead of polishing constants.\n")
    lab_feedback = _latest_lab_feedback(run, out)
    lab_context = ""
    lab_feedback_sha256 = None
    if lab_feedback is not None:
        lab_path, lab_value = lab_feedback
        lab_feedback_sha256 = hashlib.sha256(lab_path.read_bytes()).hexdigest()
        source_lineage = lab_value.get("source_lineage")
        lab_context = f"""
# Approved visible laboratory feedback

Read {lab_path}. This bundle is deliberately visible training data from verses
that are disjoint from every protected task. It contains the source, your prior
candidate output, the frozen laboratory-baseline output, the reference,
sentence-level metrics, observed commands, and `.runner` artifacts.

First diagnose the actual errors sentence by sentence. Check whether each
declared tool really ran, produced its intended artifact, and affected Luna's
decision. Then implement one causal next step. A laboratory gain is useful
research evidence but is never Finch fitness; protected development and
confirmation references remain sealed.

The bundle was produced by {source_lineage or lineage_id}. If that is another
lineage, treat its candidate behavior as shared empirical evidence, not as your
own prior mechanism.

Latest laboratory mean paired chrF++ delta:
{lab_value.get('mean_sentence_paired_chrf_delta')}
"""
    prompt = f"""You are the GPT-5.6 Sol research agent for Finch lineage {lineage_id}, experiment {experiment}.

Read:
- campaign law: {run / 'CONTRACT.md'}
- current shared research: {research_snapshot}
- your own scored trajectory and audits: {trajectory_snapshot}
- starting individual: {out / 'seed_candidate.json'}
- public evaluator project fixture: {run / 'public_fixture'}
{('- visible laboratory feedback: ' + str(lab_feedback[0])) if lab_feedback else ''}

Assigned direction:
{line.get('idea') or 'invent a distinct, falsifiable Luna harness capability'}

{parent_context}{continuation}Edit only:
- {out / 'candidate.json'}
- {out / 'hypothesis.md'}
- {out / 'log.md'}

{lab_context}
The candidate is a harness for downstream native Codex GPT-5.6 Luna translation
agents. You are not the translator. Do not call any model API and do not inspect
protected evaluator records, protected references, other lineages, or run secrets.
The specifically named visible laboratory bundle above is the sole reference-data
exception and may be used freely for research.
The protected evaluator assigns fitness after you finish. Separate judge agents
later compare your research trajectory with exactly one peer under frozen criteria.
The active protected regime evaluates one submitted harness once on 12 distinct
verses in each of eight languages (96 translations) against one frozen reusable
baseline. Small or uncertain results do not automatically kill your research
line: diagnose them and decide whether the mechanism deserves another logical
step, a creative departure, or abandonment.

Make one consequential experiment. You may invent generic local Python tools,
prompt structure, evidence representations, multi-pass reasoning within the fixed
two turns, critics, planners, or corpus-derived statistical analyses. All tools
must be embedded as source strings in candidate.json and use only supplied data.
Every candidate must state a mechanism, causal prediction, activation condition,
fallback, and falsifier. Be creative; a strange but testable capability is welcome.

If you use schema_version 2, orchestration contains exactly these keys:
`topology`, `research_phase_prompt`, `final_phase_prompt`, and
`compute_envelope`. `topology` is `single_session_two_turn` and
`compute_envelope` is exactly:
`{{"model":"gpt-5.6-luna","sessions_per_language":1,"turns_per_session":2,"tasks_per_language":3}}`.

Write hypothesis.md as concise Markdown. If the mechanism materially adopts,
extends, contrasts, or combines something from shared research or a parent, preserve
the exact citation token beside that claim (for example
[L0003#2@8ddf8b41]). This report text is the lineage record from which Finch
draws the Tree of Life. Cite genuine inspiration, not merely everything read.

Before finishing, materialize each embedded tool against the public fixture and
run its documented command. The evaluator uses that exact line-aligned file
layout; a tool that assumes a combined JSON corpus is not evaluable.
Every generated file—including provisional drafts—must live under `.runner/`;
writing elsewhere invalidates an otherwise successful Luna session.

Validate before stopping:
{sys.executable} {HERE / 'protected_evaluator.py'} {out / 'candidate.json'} --factory-root PLACEHOLDER --corpus-dir PLACEHOLDER --codex PLACEHOLDER --work-root PLACEHOLDER

Do not run that command: its paths deliberately point at the protected evaluator.
Instead, validate the JSON syntax locally and report the exact idea implemented.
"""
    (out / "prompt.md").write_text(prompt, encoding="utf-8")
    _write_json(out / "experiment.json", {
        "lineage": lineage_id, "experiment": experiment,
        "seed": str(seed), "seed_score": parent_score,
        "decoder_version": campaign.decoder_version,
        "decoder_sha256": decoder["artifact_sha256"],
        "shared_research": str(research_snapshot),
        "trajectory": str(trajectory_snapshot),
        "lab_feedback": None if lab_feedback is None else str(lab_feedback[0]),
        "lab_feedback_sha256": lab_feedback_sha256,
    })
    return {"lineage": lineage_id, "experiment": experiment,
            "directory": str(out), "prompt": str(out / "prompt.md"),
            "candidate": str(out / "candidate.json")}


def found(args):
    run = Path(args.run).resolve()
    idea = args.idea
    if idea is None:
        summary = _call(run, "summary")
        idea = FOUNDER_ANGLES[summary["lineages"] % len(FOUNDER_ANGLES)]
    body = {"task": TASK, "rationale": args.rationale, "idea": idea,
            "kind": "crossover" if args.parent else args.kind,
            "worker": args.worker}
    if args.parent:
        body["parents"] = args.parent
    line = _call(run, "found", body)
    prepared = _prepare(run, line["id"])
    print(json.dumps({"lineage": line, "experiment": prepared},
                     ensure_ascii=False))


def prepare(args):
    print(json.dumps(_prepare(Path(args.run).resolve(), args.lineage),
                     ensure_ascii=False))


def _native_sol_command(codex: Path, lineage_dir: Path) -> list[str]:
    """Build the fixed native Codex invocation for one prepared evolver."""
    return [
        str(codex), "exec", "-m", "gpt-5.6-sol",
        "-c", 'model_reasoning_effort="low"',
        "-c", 'web_search="disabled"',
        "--disable", "multi_agent", "--sandbox", "workspace-write",
        "--ephemeral", "--skip-git-repo-check",
        "-C", str(lineage_dir), "-",
    ]


def _normalize_v2_packaging(candidate_path: Path) -> bool:
    """Repair one unambiguous schema-v2 spelling mistake, not research.

    Early native workers independently converged on the descriptive keys
    ``research_phase``/``final_phase`` and a prose compute envelope.  The
    evaluator contract has exact field names and owns the fixed compute
    object. Mapping those three packaging values changes no mechanism.
    """
    value = json.loads(candidate_path.read_text(encoding="utf-8"))
    orchestration = value.get("orchestration")
    legacy = {"topology", "research_phase", "final_phase",
              "compute_envelope"}
    if value.get("schema_version") != 2 or not isinstance(
            orchestration, dict) or set(orchestration) != legacy:
        return False
    value["orchestration"] = {
        "topology": orchestration["topology"],
        "research_phase_prompt": orchestration["research_phase"],
        "final_phase_prompt": orchestration["final_phase"],
        "compute_envelope": dict(FIXED_COMPUTE_ENVELOPE),
    }
    _write_json(candidate_path, value)
    return True


def _queue_once(run: Path, *, key: str, kind: str, lane: str, name: str,
                argv: list[str], timeout: float, metadata: dict,
                max_attempts: int = 1):
    """Task-level idempotency; the generic Finch queue stays policy-free."""
    queue = JobQueue(run)
    for job in queue.status()["jobs"]:
        if job.get("metadata", {}).get("job_key") == key:
            return job
    return queue.enqueue(
        kind=kind, lane=lane, name=name, argv=argv, cwd=ROOT,
        timeout=timeout, metadata={**metadata, "job_key": key},
        max_attempts=max_attempts)


def _score_job(run: Path, lineage_id: str, *, workers: int,
               timeout: float, max_attempts: int = 1):
    settings = _settings(run)
    out = run / "lineages" / lineage_id
    metadata = json.loads(
        (out / "experiment.json").read_text(encoding="utf-8"))
    candidate_hash = hashlib.sha256(
        (out / "candidate.json").read_bytes()).hexdigest()
    experiment = int(metadata["experiment"])
    regime = str(settings.get("evaluation_regime", "initial"))
    key = f"score:{regime}:{lineage_id}:{experiment}:{candidate_hash}"
    argv = [
        settings["evaluator_python"], str(HERE / "native_control.py"),
        "score", "--run", str(run), "--lineage", lineage_id,
        "--workers", str(workers), "--expected-sha256", candidate_hash,
    ]
    return _queue_once(
        run, key=key, kind="protected-score", lane="evaluator",
        name=f"{lineage_id} protected Luna score", argv=argv,
        timeout=timeout, max_attempts=max_attempts,
        metadata={"lineage": lineage_id, "experiment": experiment,
                  "candidate_sha256": candidate_hash,
                  "fitness_regime": regime})


def _lab_job(run: Path, lineage_id: str, *, workers: int,
             timeout: float, max_attempts: int = 1):
    settings = _settings(run)
    out = run / "lineages" / lineage_id
    metadata = json.loads(
        (out / "experiment.json").read_text(encoding="utf-8"))
    candidate_hash = hashlib.sha256(
        (out / "candidate.json").read_bytes()).hexdigest()
    snapshots = run / "laboratory" / "candidates"
    snapshots.mkdir(parents=True, exist_ok=True)
    snapshot = snapshots / f"{candidate_hash}.json"
    if not snapshot.is_file():
        shutil.copy2(out / "candidate.json", snapshot)
    elif hashlib.sha256(snapshot.read_bytes()).hexdigest() != candidate_hash:
        raise ValueError("immutable laboratory candidate snapshot mismatch")
    experiment = int(metadata["experiment"])
    key = f"laboratory:{lineage_id}:{experiment}:{candidate_hash}"
    argv = [
        settings["evaluator_python"], str(HERE / "native_control.py"),
        "lab", "--run", str(run), "--lineage", lineage_id,
        "--candidate", str(snapshot), "--workers", str(workers),
        "--expected-sha256", candidate_hash,
    ]
    return _queue_once(
        run, key=key, kind="visible-laboratory", lane="laboratory",
        name=f"{lineage_id} visible Luna laboratory", argv=argv,
        timeout=timeout, max_attempts=max_attempts,
        metadata={"lineage": lineage_id, "experiment": experiment,
                  "candidate_sha256": candidate_hash,
                  "fitness": False, "references_visible_after_attempt": True})


def _reflection_job(run: Path, lineage_id: str, report_index: int, *,
                    agent_timeout: float = 900, job_timeout: float = 960,
                    max_attempts: int = 2):
    """Queue the scored lineage itself to author shared scientific memory."""
    campaign, line = _line(run, lineage_id)
    try:
        report = line["reports"][int(report_index)]
    except (IndexError, TypeError, ValueError):
        raise ValueError(f"unknown report {lineage_id}#{report_index}")
    if report.get("score") is None or not report.get("artifact_sha256"):
        raise ValueError("reflection requires trusted fitness and an artifact")
    citation = campaign.citation(lineage_id, report_index)
    key = f"reflect:{lineage_id}:{int(report_index)}:{report['artifact_sha256']}"
    argv = [
        sys.executable, str(HERE / "native_control.py"), "reflect-native",
        "--run", str(run), "--lineage", lineage_id,
        "--report", str(int(report_index)), "--timeout", str(agent_timeout),
    ]
    return _queue_once(
        run, key=key, kind="sol-reflection", lane="evolver",
        name=f"{citation} Sol shared-finding reflection", argv=argv,
        timeout=job_timeout, max_attempts=max_attempts,
        metadata={"lineage": lineage_id, "report": int(report_index),
                  "citation": citation,
                  "artifact_sha256": report["artifact_sha256"]})


def _reflection_prompt(run: Path, lineage_id: str, report_index: int,
                       citation: str, research_path: Path,
                       input_path: Path, finding_path: Path) -> str:
    return f"""You are the GPT-5.6 Sol research agent for Finch lineage {lineage_id}.

Your experiment has now been evaluated. This is scientific reflection, not a
new harness experiment and not a request to defend your idea.

Read:
- current shared findings: {research_path}
- your trusted public result: {input_path}

Write exactly one concise Markdown finding to:
{finding_path}

The finding must contain your exact checkpoint citation {citation}. State what
mechanism was tested, what the measured result actually supports, important
cross-language variation or failures, and the most useful implication for a
later researcher. A loss is publishable evidence. A positive screen is not an
audited default. Do not claim that a tool caused the result unless the supplied
evidence shows that it ran and affected decisions. Preserve any earlier inline
citations only when they genuinely explain intellectual inheritance.

Use a short descriptive heading and ordinary prose. This fragment will be
appended directly to the population's one shared Decoder.md; there is no second
findings database. Do not edit candidate.json, hypothesis.md, log.md, reports,
or the shared file itself. Finch will validate citations and serialize the
append after you finish.
"""


def reflect_native(args):
    """Let one scored evolver interpret its result and publish plain Markdown."""
    run = Path(args.run).resolve()
    campaign, line = _line(run, args.lineage)
    try:
        report = line["reports"][args.report]
    except IndexError:
        raise SystemExit("unknown report")
    if report.get("score") is None or not report.get("artifact_sha256"):
        raise SystemExit("reflection requires trusted fitness and an artifact")
    citation = campaign.citation(args.lineage, args.report)
    out = run / "lineages" / args.lineage
    finding_path = out / f"finding_r{args.report:04d}.md"
    decoder = campaign.current_decoder()
    research_path = out / (
        f"reflection_research_r{args.report:04d}_v{decoder['version']:04d}.md")
    research_path.write_text(decoder["artifact"], encoding="utf-8")
    input_path = out / f"reflection_input_r{args.report:04d}.json"
    _write_json(input_path, {
        "citation": citation,
        "shared_research_version": decoder["version"],
        "lineage": {"id": line["id"], "idea": line.get("idea")},
        "report": {
            "index": report["index"], "summary": report["summary"],
            "score": report["score"], "source": report["source"],
            "evidence": report.get("evidence"), "kept": report.get("kept"),
            "audit_status": report.get("audit_status"),
            "artifact": report.get("artifact"),
            "artifact_sha256": report.get("artifact_sha256"),
            "decoder_version_used": report.get("decoder_version", 0),
        },
    })

    def valid_existing():
        if not finding_path.is_file():
            return None
        text = finding_path.read_text(encoding="utf-8").strip()
        if citation not in text:
            return None
        campaign.resolve_citations(text, require_citations=True)
        return text

    fragment = valid_existing()
    if fragment is None:
        settings = _settings(run)
        codex_value = args.codex or settings["codex_binary"]
        codex = Path(codex_value).resolve()
        if not codex.is_file():
            raise SystemExit(f"Codex CLI not found: {codex}")
        protected = {}
        for name in ("candidate.json", "seed_candidate.json", "hypothesis.md",
                     "log.md", f"report_{args.report:04d}.json"):
            path = out / name
            if path.is_file():
                protected[path] = path.read_bytes()
        prompt = _reflection_prompt(
            run, args.lineage, args.report, citation, research_path,
            input_path, finding_path)
        prompt_path = out / f"reflection_prompt_r{args.report:04d}.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        try:
            completed = subprocess.run(
                _native_sol_command(codex, out), input=prompt, text=True,
                capture_output=True, check=False, timeout=args.timeout)
        except subprocess.TimeoutExpired as exc:
            raise SystemExit(
                f"native Sol reflection timed out after {args.timeout}s") from exc
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout).strip()[-2000:]
            raise SystemExit(
                f"native Sol reflection exited {completed.returncode}: {detail}")
        changed = [path for path, payload in protected.items()
                   if not path.is_file() or path.read_bytes() != payload]
        if changed:
            for path in changed:
                path.write_bytes(protected[path])
            raise SystemExit(
                "reflection tried to modify lineage state: "
                + ", ".join(path.name for path in changed))
        fragment = valid_existing()
        if fragment is None:
            raise SystemExit(
                f"reflection did not write a valid cited finding: {finding_path}")

    if len(fragment) > 6000:
        raise SystemExit("shared finding exceeds the 6000-character limit")
    resolved = campaign.resolve_citations(fragment, require_citations=True)
    if not any(item["token"] == citation for item in resolved):
        raise SystemExit(f"shared finding must cite its own report as {citation}")
    published = _call(run, "finding", {
        "fragment": fragment,
        "rationale": (
            f"{args.lineage} Sol evolver publishes its interpretation of "
            f"trusted checkpoint {citation} after evaluation."),
    })
    receipt = out / f"finding_receipt_r{args.report:04d}.json"
    _write_json(receipt, {
        "citation": citation, "decoder_version": published["version"],
        "decoder_sha256": published["artifact_sha256"],
        "idempotent": published.get("idempotent", False),
    })
    print(json.dumps({
        "lineage": args.lineage, "report": args.report,
        "citation": citation, "decoder_version": published["version"],
        "finding": str(finding_path),
        "idempotent": published.get("idempotent", False),
    }, ensure_ascii=False))


def evolve_native(args):
    """Run native Sol once on an already prepared lineage, then validate it."""
    run = Path(args.run).resolve()
    lineage_dir = run / "lineages" / args.lineage
    prompt_path = lineage_dir / "prompt.md"
    candidate_path = lineage_dir / "candidate.json"
    if not prompt_path.is_file():
        raise SystemExit(
            f"prepared lineage prompt not found: {prompt_path}; run prepare first")
    if not candidate_path.is_file():
        raise SystemExit(f"prepared candidate not found: {candidate_path}")
    codex_value = args.codex
    if codex_value is None:
        manifest_path = run / "manifest.json"
        if not manifest_path.is_file():
            raise SystemExit("--codex is required when manifest.json is absent")
        codex_value = json.loads(
            manifest_path.read_text(encoding="utf-8"))["codex_binary"]
    codex = Path(codex_value).resolve()
    if not codex.is_file():
        raise SystemExit(f"Codex CLI not found: {codex}")
    prompt = prompt_path.read_text(encoding="utf-8")
    try:
        completed = subprocess.run(
            _native_sol_command(codex, lineage_dir),
            input=prompt, text=True, capture_output=True, check=False,
            timeout=args.timeout)
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            f"native Sol evolver timed out after {args.timeout}s") from exc
    except OSError as exc:
        raise SystemExit(f"native Sol evolver could not start: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        if len(detail) > 2000:
            detail = detail[-2000:]
        suffix = f": {detail}" if detail else ""
        raise SystemExit(
            f"native Sol evolver exited {completed.returncode}{suffix}")
    packaging_repaired = _normalize_v2_packaging(candidate_path)
    try:
        validated = validate_harness(candidate_path)
    except Exception as exc:
        raise SystemExit(
            f"native Sol completed but candidate validation failed: {exc}") from exc
    if (getattr(args, "enqueue_score", False)
            and getattr(args, "enqueue_lab", False)):
        raise SystemExit("choose at most one of --enqueue-lab and --enqueue-score")
    queued_score = None
    queued_lab = None
    if getattr(args, "enqueue_score", False):
        queued_score = _score_job(
            run, args.lineage,
            workers=getattr(args, "score_workers", 2),
            timeout=getattr(args, "score_timeout", 7200),
            max_attempts=getattr(args, "score_attempts", 1))
    elif getattr(args, "enqueue_lab", False):
        queued_lab = _lab_job(
            run, args.lineage,
            workers=getattr(args, "lab_workers", 2),
            timeout=getattr(args, "lab_timeout", 7200),
            max_attempts=getattr(args, "lab_attempts", 1))
    print(json.dumps({
        "status": ("queued_for_laboratory" if queued_lab is not None
                   else "queued_for_scoring" if queued_score is not None
                   else "ready_for_evaluation"),
        "lineage": args.lineage,
        "candidate": str(candidate_path),
        "name": validated["name"],
        "packaging_repaired": packaging_repaired,
        "queued_lab": None if queued_lab is None else queued_lab["id"],
        "queued_score": None if queued_score is None else queued_score["id"],
    }, ensure_ascii=False))


def repair_packaging(args):
    """Normalize already-produced v2 packaging and queue its real score."""
    run = Path(args.run).resolve()
    results = []
    for lineage_id in args.lineage:
        candidate = run / "lineages" / lineage_id / "candidate.json"
        repaired = _normalize_v2_packaging(candidate)
        validate_harness(candidate)
        job = None
        if args.enqueue_score:
            job = _score_job(
                run, lineage_id, workers=args.score_workers,
                timeout=args.score_timeout,
                max_attempts=args.score_attempts)
        results.append({"lineage": lineage_id, "repaired": repaired,
                        "score_job": None if job is None else job["id"]})
    print(json.dumps(results, ensure_ascii=False))


def queue_evolver(args):
    """Queue one prepared native Sol researcher; Finch launches it later."""
    run = Path(args.run).resolve()
    _, line = _line(run, args.lineage)
    out = run / "lineages" / args.lineage
    if not (out / "prompt.md").is_file():
        _prepare(run, args.lineage)
    metadata = json.loads(
        (out / "experiment.json").read_text(encoding="utf-8"))
    experiment = int(metadata["experiment"])
    feedback_key = str(metadata.get("lab_feedback_sha256") or "founder")[:16]
    key = f"evolve:{args.lineage}:{experiment}:{feedback_key}"
    argv = [
        sys.executable, str(HERE / "native_control.py"), "evolve-native",
        "--run", str(run), "--lineage", args.lineage,
        "--timeout", str(args.agent_timeout),
    ]
    if args.after == "score":
        argv.extend([
            "--enqueue-score", "--score-workers", str(args.score_workers),
            "--score-timeout", str(args.score_timeout),
            "--score-attempts", str(args.score_attempts)])
    elif args.after == "lab":
        argv.extend([
            "--enqueue-lab", "--lab-workers", str(args.lab_workers),
            "--lab-timeout", str(args.lab_timeout),
            "--lab-attempts", str(args.lab_attempts)])
    job = _queue_once(
        run, key=key, kind="sol-evolver", lane="evolver",
        name=f"{args.lineage} experiment {experiment} Sol evolver",
        argv=argv, timeout=args.job_timeout,
        max_attempts=args.agent_attempts,
        metadata={"lineage": args.lineage, "experiment": experiment,
                  "idea": line.get("idea"), "after": args.after,
                  "lab_feedback_sha256": metadata.get("lab_feedback_sha256")})
    print(json.dumps(job, ensure_ascii=False))


def queue_score(args):
    """Queue protected Luna scoring for an already-generated candidate."""
    run = Path(args.run).resolve()
    validate_harness(run / "lineages" / args.lineage / "candidate.json")
    job = _score_job(
        run, args.lineage, workers=args.workers, timeout=args.timeout,
        max_attempts=args.max_attempts)
    print(json.dumps(job, ensure_ascii=False))


def queue_reflection(args):
    """Queue post-evaluation interpretation into the shared Markdown file."""
    run = Path(args.run).resolve()
    job = _reflection_job(
        run, args.lineage, args.report,
        agent_timeout=args.agent_timeout, job_timeout=args.job_timeout,
        max_attempts=args.max_attempts)
    print(json.dumps(job, ensure_ascii=False))


def queue_missing_reflections(args):
    """Queue findings for trusted reports that lack publication receipts.

    This recovers campaigns that acquired reports before post-score reflection
    existed, or whose evaluator died between reporting and queueing reflection.
    Durable job keys make repeated invocations safe.
    """
    run = Path(args.run).resolve()
    campaign = Campaign.load(run / "state.json")
    requested = set(args.lineage or campaign.lineages)
    unknown = requested - set(campaign.lineages)
    if unknown:
        raise SystemExit(
            "unknown lineage(s): " + ", ".join(sorted(unknown)))
    queued = []
    already_published = []
    ineligible = []
    for lineage_id in sorted(requested):
        line = campaign.lineages[lineage_id]
        out = run / "lineages" / lineage_id
        for report in line["reports"]:
            index = int(report["index"])
            label = f"{lineage_id}#{index}"
            if (report.get("voided") or report.get("score") is None
                    or not report.get("artifact_sha256")):
                ineligible.append(label)
                continue
            if (out / f"finding_receipt_r{index:04d}.json").is_file():
                already_published.append(label)
                continue
            job = _reflection_job(
                run, lineage_id, index,
                agent_timeout=args.agent_timeout,
                job_timeout=args.job_timeout,
                max_attempts=args.max_attempts)
            queued.append({"report": label, "job": job["id"],
                           "status": job["status"]})
    print(json.dumps({
        "queued": queued,
        "already_published": already_published,
        "ineligible": ineligible,
    }, ensure_ascii=False))


def laboratory(args):
    """Produce direct, research-visible error feedback without Finch fitness."""
    run = Path(args.run).resolve()
    _line(run, args.lineage)
    out = run / "lineages" / args.lineage
    candidate = (Path(args.candidate).resolve() if args.candidate
                 else out / "candidate.json")
    validate_harness(candidate)
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    if args.expected_sha256 and candidate_hash != args.expected_sha256:
        raise SystemExit(
            "queued laboratory candidate changed before evaluation: expected "
            f"{args.expected_sha256}, observed {candidate_hash}")
    metadata = json.loads(
        (out / "experiment.json").read_text(encoding="utf-8"))
    experiment = int(metadata["experiment"])
    settings = _settings(run)
    _call(run, "media", {
        "name": "active laboratory", "kind": "text",
        "data": (f"Running research-visible prediction/reference feedback for "
                 f"{args.lineage}; this is not Finch fitness.")})
    feedback = evaluate_lab_pair(
        candidate, run / "starter_harness.json", run=run,
        factory_root=Path(settings["factory_root"]),
        corpus_dir=Path(settings["corpus_dir"]),
        codex=Path(settings["codex_binary"]),
        work_root=Path(settings["work_root"]), workers=args.workers)
    feedback["source_lineage"] = args.lineage
    feedback["source_experiment"] = experiment
    attempts = list(out.glob(f"lab_feedback_e{experiment:04d}_a*_*.json"))
    attempt = len(attempts) + 1
    destination = out / (
        f"lab_feedback_e{experiment:04d}_a{attempt:02d}_{candidate_hash[:8]}.json")
    _write_json(destination, feedback)
    _write_json(run / "laboratory" / "latest_feedback.json", feedback)
    _call(run, "media", {
        "name": "active laboratory", "kind": "text",
        "data": (f"{args.lineage} visible laboratory complete: paired sentence "
                 f"chrF++ delta={feedback['mean_sentence_paired_chrf_delta']:.5f}. "
                 "The full prediction/reference bundle is available to its next evolver.")})
    print(json.dumps({
        "lineage": args.lineage, "experiment": experiment,
        "candidate_sha256": candidate_hash,
        "mean_sentence_paired_chrf_delta":
            feedback["mean_sentence_paired_chrf_delta"],
        "candidate_failures": feedback["candidate_failures"],
        "fitness": None, "feedback": str(destination),
    }, ensure_ascii=False))


def queue_lab(args):
    """Queue an immutable candidate in the visible laboratory lane."""
    run = Path(args.run).resolve()
    validate_harness(run / "lineages" / args.lineage / "candidate.json")
    job = _lab_job(
        run, args.lineage, workers=args.workers, timeout=args.timeout,
        max_attempts=args.max_attempts)
    print(json.dumps(job, ensure_ascii=False))


def queue_judge(args):
    """Queue one named, rating-blind Sol judge for an existing match."""
    run = Path(args.run).resolve()
    settings = _settings(run)
    assignment = _call(run, f"match?id={args.match}")
    assigned = assignment.get("judge")
    if assigned is not None and assigned != args.judge:
        raise SystemExit(
            f"match {args.match} is assigned to {assigned!r}, not {args.judge!r}")
    key = f"judge:{args.match}:{args.judge}"
    argv = [
        sys.executable, str(HERE / "judge_worker.py"),
        "--run", str(run), "--match", args.match,
        "--judge", args.judge, "--codex", settings["codex_binary"],
        "--timeout", str(args.agent_timeout),
    ]
    job = _queue_once(
        run, key=key, kind="sol-judge", lane="judge",
        name=f"{args.match} named Sol judge", argv=argv,
        timeout=args.job_timeout, max_attempts=args.max_attempts,
        metadata={"match": args.match, "judge": args.judge})
    print(json.dumps(job, ensure_ascii=False))


def _settings(run: Path):
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))


def _large_panel_path(settings: dict) -> Path:
    value = settings.get("large_development_panel")
    if not value:
        raise RuntimeError("large development panel is not configured")
    path = Path(value)
    if not path.is_file():
        raise RuntimeError(f"large development panel is missing: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != settings.get("large_development_panel_sha256"):
        raise RuntimeError("large development panel identity changed")
    return path


def _large_baseline_path(settings: dict) -> Path:
    value = settings.get("large_baseline_cache")
    if not value:
        raise RuntimeError("large frozen baseline is not configured")
    return Path(value)


def _large_decision(result: dict) -> dict:
    """Make only high-confidence automatic decisions; agents own the rest."""
    mean = float(result["mean_delta"])
    interval = result["paired_bootstrap"]
    lower = float(interval["lower_95"])
    upper = float(interval["upper_95"])
    failures = int(result["candidate_failures"])
    if failures >= 4:
        outcome = "clear_loss"
    elif mean <= -1.0 and upper < 0.0:
        outcome = "clear_loss"
    elif mean >= 1.0 and lower > 0.0:
        outcome = "win"
    else:
        outcome = "inconclusive"
    return {
        "outcome": outcome, "mean_delta": mean,
        "win_rate": float(result["win_rate"]),
        "wins": int(result["wins"]),
        "comparisons": int(result["comparisons"]),
        "candidate_failures": failures,
        "lower_95": lower, "upper_95": upper,
    }


def _evaluate_large_candidate(run: Path, candidate: Path, settings: dict,
                              *, workers: int) -> dict:
    panel_path = _large_panel_path(settings)
    panel_hash = settings["large_development_panel_sha256"]
    baseline_path = _large_baseline_path(settings)
    if not baseline_path.is_file():
        raise RuntimeError(
            "the one-time 96-item baseline is not ready; queue large-control first")
    baseline_cache = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_hash = hashlib.sha256(
        (run / "starter_harness.json").read_bytes()).hexdigest()
    if (baseline_cache.get("schema_version") != 1
            or baseline_cache.get("panel_sha256") != panel_hash
            or baseline_cache.get("baseline_sha256") != baseline_hash
            or len(baseline_cache.get("shards") or []) != 4):
        raise RuntimeError("large frozen baseline cache identity mismatch")
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    key = {
        "candidate_sha256": candidate_hash,
        "baseline_sha256": baseline_hash,
        "panel_sha256": panel_hash,
        "fitness_regime": LARGE_PANEL_VERSION,
        "verification": False,
    }
    consumed = _consumed_match(run, key)
    if consumed is not None:
        return consumed["result"]

    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    candidate_shards = []
    private_record_roots = []
    for shard in panel["shards"]:
        result = evaluate(
            candidate,
            factory_root=Path(settings["factory_root"]),
            corpus_dir=Path(settings["corpus_dir"]),
            codex=Path(settings["codex_binary"]),
            work_root=Path(settings["work_root"]),
            panel=LARGE_PANEL_NAME, workers=workers,
            panel_spec=panel_path, shard_id=shard["id"])
        candidate_shards.append(result)
        private_record_roots.append(result["private_record_root"])
    summary = large_paired_summary(
        candidate_shards, baseline_cache["shards"],
        panel_sha256=panel_hash)
    summary["decision"] = _large_decision(summary)
    summary["fitness_regime"] = LARGE_PANEL_VERSION
    summary["private_record_roots"] = private_record_roots
    public = {key: value for key, value in summary.items()
              if key != "private_record_roots"}
    _append_match(run, {**key, "result": public})
    return summary


def _build_large_baseline(run: Path, settings: dict, *, workers: int) -> dict:
    panel_path = _large_panel_path(settings)
    panel_hash = settings["large_development_panel_sha256"]
    destination = _large_baseline_path(settings)
    if destination.is_file():
        cached = json.loads(destination.read_text(encoding="utf-8"))
        if cached.get("panel_sha256") != panel_hash:
            raise RuntimeError("existing large baseline belongs to another panel")
        return cached
    starter = run / "starter_harness.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    shards = []
    for shard in panel["shards"]:
        shards.append(evaluate(
            starter,
            factory_root=Path(settings["factory_root"]),
            corpus_dir=Path(settings["corpus_dir"]),
            codex=Path(settings["codex_binary"]),
            work_root=Path(settings["work_root"]),
            panel=LARGE_PANEL_NAME, workers=workers,
            panel_spec=panel_path, shard_id=shard["id"]))
    cache = {
        "schema_version": 1,
        "fitness_regime": LARGE_PANEL_VERSION,
        "panel_sha256": panel_hash,
        "baseline_sha256": hashlib.sha256(starter.read_bytes()).hexdigest(),
        "translation_items": LARGE_TRANSLATION_ITEMS,
        "shards": shards,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    temporary.chmod(0o600)
    temporary.replace(destination)
    return cache


def _keeps_candidate(result: dict, previous) -> bool:
    """Strict local hill climb; failed evaluations never become organisms."""
    return (not result["failures"]
            and (previous is None or result["score"] > previous))


def _tournament_decision(blocks: list[dict], *, final: bool = False) -> dict:
    """Conservative, predeclared sequential decision from paired languages."""
    pairs = [pair for block in blocks for pair in block["pairs"]]
    deltas = [float(pair["delta"]) for pair in pairs]
    mean_delta = sum(deltas) / len(deltas)
    wins = sum(delta > 0 for delta in deltas)
    win_rate = wins / len(deltas)
    failures = sum(block["candidate_failures"] for block in blocks)
    # A broken candidate or a broad first-block loss cannot be rescued by
    # spending another hidden block. A genuinely large, broad win may advance
    # immediately; smaller effects must survive both independent blocks.
    if failures:
        outcome = "clear_loss"
    elif len(blocks) == 1 and mean_delta <= -1.0 and win_rate <= 0.25:
        outcome = "clear_loss"
    elif len(blocks) == 1 and mean_delta >= 2.5 and win_rate >= 0.75:
        outcome = "large_win"
    elif not final:
        outcome = "continue"
    elif mean_delta > 0.0 and win_rate >= 0.625:
        outcome = "win"
    else:
        outcome = "loss"
    return {"outcome": outcome, "mean_delta": mean_delta,
            "win_rate": win_rate, "wins": wins, "comparisons": len(deltas),
            "candidate_failures": failures}


def _search_ledger(run: Path) -> dict:
    path = run / "tournaments.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"matches": []}


def _consumed_match(run: Path, key: dict):
    """Return an exact completed block under the writer's process lock."""
    lock_path = run / "tournaments.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for row in _search_ledger(run)["matches"]:
            if (not row.get("voided")
                    and all(row.get(k) == v for k, v in key.items())):
                return row
        return None


def _append_match(run: Path, row: dict) -> None:
    """Preserve every result when distinct lineages evaluate concurrently."""
    lock_path = run / "tournaments.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        ledger = _search_ledger(run)
        ledger["matches"].append(row)
        _write_json(run / "tournaments.json", ledger)


def _run_tournament(run: Path, candidate: Path, baseline: Path, settings: dict,
                    *, workers: int, verification: bool = False) -> dict:
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    baseline_hash = hashlib.sha256(baseline.read_bytes()).hexdigest()
    blocks = []
    private_record_roots = []
    terminal = None
    for index, block in enumerate(DEVELOPMENT_BLOCKS):
        key = {"candidate_sha256": candidate_hash, "baseline_sha256": baseline_hash,
               "block": block, "verification": verification}
        consumed = None if verification else _consumed_match(run, key)
        if consumed is not None:
            # A worker/server interruption can occur after the protected block
            # is durably ledgered but before its Finch report is submitted.
            # Resume from that immutable public result; never spend the same
            # hidden block again merely to repair bookkeeping.
            public = consumed["result"]
            private_record_roots.append(None)
        else:
            result = evaluate_pair(
                candidate, baseline, block=block,
                factory_root=Path(settings["factory_root"]),
                corpus_dir=Path(settings["corpus_dir"]),
                codex=Path(settings["codex_binary"]),
                work_root=Path(settings["work_root"]), workers=workers,
                candidate_first=(index % 2 == 0))
            public = {k: v for k, v in result.items()
                      if k != "private_record_roots"}
            private_record_roots.append(result["private_record_roots"])
            _append_match(run, {**key, "result": public})
        blocks.append(public)
        decision = _tournament_decision(blocks, final=False)
        if decision["outcome"] in {"clear_loss", "large_win"}:
            terminal = decision
            break
    decision = terminal or _tournament_decision(blocks, final=True)
    return {"blocks": blocks, "decision": decision,
            "candidate_sha256": candidate_hash, "baseline_sha256": baseline_hash,
            "verification": verification,
            "private_record_roots": private_record_roots}


def score(args):
    run = Path(args.run).resolve()
    campaign, line = _line(run, args.lineage)
    out = run / "lineages" / args.lineage
    candidate = (Path(args.candidate).resolve() if args.candidate
                 else out / "candidate.json")
    validate_harness(candidate)
    metadata = json.loads((out / "experiment.json").read_text(encoding="utf-8"))
    experiment_id = f"experiment-{int(metadata['experiment']):04d}"
    candidate_hash = hashlib.sha256(candidate.read_bytes()).hexdigest()
    expected_hash = getattr(args, "expected_sha256", None)
    if expected_hash is not None and candidate_hash != expected_hash:
        raise SystemExit(
            "queued candidate changed before evaluation: expected "
            f"{expected_hash}, observed {candidate_hash}; enqueue the current "
            "immutable candidate explicitly")
    summary_path = out / "hypothesis.md"
    summary = (summary_path.read_text(encoding="utf-8").strip()
               if summary_path.exists() else validate_harness(candidate)["hypothesis"])
    # Fail before spending Luna sessions when report prose names an unknown or
    # malformed local checkpoint. Cross-campaign history stays plain prose
    # unless it has been explicitly imported into this Campaign.
    campaign.resolve_citations(summary)
    for existing in line["reports"]:
        if (not existing.get("voided")
                and existing.get("experiment_id") == experiment_id
                and existing.get("artifact_sha256") == candidate_hash):
            reflection = _reflection_job(
                run, args.lineage, existing["index"],
                agent_timeout=getattr(args, "reflect_agent_timeout", 900),
                job_timeout=getattr(args, "reflect_job_timeout", 960),
                max_attempts=getattr(args, "reflect_attempts", 2))
            print(json.dumps({"lineage": args.lineage,
                              "report": existing["index"],
                              "score": existing["score"],
                              "idempotent_retry": True,
                              "reflection_job": reflection["id"]}))
            return
    settings = _settings(run)
    baseline = out / "seed_candidate.json"
    if args.result_json:
        result = json.loads(Path(args.result_json).resolve().read_text(
            encoding="utf-8"))
        if "blocks" not in result or "decision" not in result:
            raise SystemExit("recovered result is not a matched tournament")
        if result.get("candidate_sha256") != candidate_hash:
            raise SystemExit("recovered result does not match candidate artifact")
    else:
        _call(run, "media", {"name": "active evaluation", "kind": "text",
              "data": (f"Running the 96-item one-shot evaluation for {args.lineage}."
                       if settings.get("evaluation_regime") == LARGE_PANEL_VERSION
                       else f"Running a matched hidden-block tournament for {args.lineage}.")})
        if settings.get("evaluation_regime") == LARGE_PANEL_VERSION:
            result = _evaluate_large_candidate(
                run, candidate, settings, workers=args.workers)
        else:
            result = _run_tournament(run, candidate, baseline, settings,
                                     workers=args.workers)
    previous = (line["best_score"] if line["best_score"] is not None
                else metadata.get("seed_score"))
    outcome = result["decision"]["outcome"]
    kept = (True if outcome in {"win", "large_win"}
            else False if outcome == "clear_loss" else None)
    if settings.get("evaluation_regime") == LARGE_PANEL_VERSION:
        candidate_chrf = float(result["candidate_score"])
        source = (
            "protected native Codex Luna large-96-v1; one-shot 96-item paired "
            "chrF++ delta over one frozen reusable baseline")
    else:
        candidate_scores = [block["candidate_score"] for block in result["blocks"]]
        candidate_chrf = sum(candidate_scores) / len(candidate_scores)
        source = (
            "protected native Codex Luna matched sequential tournament v3; "
            "paired chrF++ point delta")
    score_value = float(result["decision"]["mean_delta"])
    evidence = {key: value for key, value in result.items()
                if key != "private_record_roots"}
    evidence["candidate_sha256_pre_submit"] = candidate_hash
    evidence["evaluator_sha256"] = settings["evaluator_sha256"]
    report = _call(run, "report", {
        "lineage": args.lineage, "summary": summary[:4000],
        "score": float(score_value),
        "source": source,
        "evidence": evidence, "kept": kept,
        "artifact": str(candidate), "experiment_id": experiment_id})
    _write_json(out / f"report_{report['index']:04d}.json", {
        "report": report, "evidence": evidence,
        "private_record_roots": result.get("private_record_roots", [])})
    if kept is False:
        shutil.copy2(out / "seed_candidate.json", candidate)
    reflection = _reflection_job(
        run, args.lineage, report["index"],
        agent_timeout=getattr(args, "reflect_agent_timeout", 900),
        job_timeout=getattr(args, "reflect_job_timeout", 960),
        max_attempts=getattr(args, "reflect_attempts", 2))
    _call(run, "media", {"name": "active evaluation", "kind": "text",
          "data": f"{args.lineage} evaluation: {result['decision']['outcome']}; paired delta={result['decision']['mean_delta']:.5f}."})
    print(json.dumps({
        "lineage": args.lineage, "report": report["index"],
        "fitness_paired_chrf_delta": score_value,
        "candidate_chrf": candidate_chrf, "previous": previous,
        "kept": kept, "decision": result["decision"],
        "artifact": report.get("artifact"),
        "reflection_job": reflection["id"]}, ensure_ascii=False))


def audit(args):
    run = Path(args.run).resolve()
    _, line = _line(run, args.lineage)
    try:
        report = line["reports"][args.report]
    except IndexError:
        raise SystemExit("unknown report")
    if not report.get("artifact"):
        raise SystemExit("report has no immutable artifact")
    settings = _settings(run)
    if settings.get("evaluation_regime") == LARGE_PANEL_VERSION:
        raise SystemExit(
            "large-96-v1 does not repeat search items; reserve a disjoint final "
            "audit panel after the allocator freezes a winner")
    # Audit is the explicitly reserved repeat-verification stage. It is still
    # matched against the frozen pre-GAR staged genome on each development
    # block; the untouched confirmation languages are never entered here.
    result = _run_tournament(
        run, Path(report["artifact"]), run / "starter_harness.json", settings,
        workers=args.workers, verification=True)
    decision = result["decision"]
    outcome = ("passed" if decision["outcome"] in {"win", "large_win"}
               else "failed" if decision["outcome"] in {"loss", "clear_loss"}
               else "inconclusive")
    evidence = {key: value for key, value in result.items()
                if key != "private_record_roots"}
    evidence["selection_score"] = report["score"]
    _call(run, "audit", {
        "lineage": args.lineage, "report": args.report,
        "outcome": outcome,
        "rationale": args.rationale or
        f"native Luna matched repeat verification audit: {outcome}",
        "evidence": evidence})
    print(json.dumps({"lineage": args.lineage, "report": args.report,
                      "outcome": outcome,
                      "paired_delta_vs_pre_gar_baseline": decision["mean_delta"],
                      "win_rate": decision["win_rate"],
                      "selection_chrf": report["score"]}, ensure_ascii=False))


def control(args):
    run = Path(args.run).resolve()
    settings = _settings(run)
    candidate = run / "starter_harness.json"
    destination = run / "controls" / f"{args.panel}-baseline.json"
    if destination.is_file():
        raise SystemExit(
            f"pre-GAR baseline already exists at {destination}; exact control repeats "
            "are reserved for matched verification")
    _call(run, "media", {"name": "active evaluation", "kind": "text",
          "data": f"Scoring the pre-GAR staged native-Luna baseline on the fixed {args.panel} panel."})
    result = evaluate(
        candidate, factory_root=Path(settings["factory_root"]),
        corpus_dir=Path(settings["corpus_dir"]),
        codex=Path(settings["codex_binary"]),
        work_root=Path(settings["work_root"]), panel=args.panel,
        workers=args.workers)
    public = {key: value for key, value in result.items()
              if key != "private_record_root"}
    public["fitness_score"] = 0.0
    public["fitness_metric"] = "paired chrF++ point delta over this baseline"
    controls = run / "controls"
    controls.mkdir(exist_ok=True)
    _write_json(destination, public)
    if args.panel == "selection":
        _call(run, "baseline", {
            "task": TASK,
            "score": 0.0,
            "source": "pre-GAR staged harness; paired chrF++ point delta",
            "rationale": "Freeze zero paired improvement as the search baseline; preserve its absolute chrF++ in evidence.",
            "evidence": public,
        })
    _call(run, "media", {"name": "active evaluation", "kind": "text",
          "data": f"Pre-GAR baseline {args.panel} complete: {result['score']:.5f} chrF++; failures={result['failures']}."})
    print(json.dumps({"panel": args.panel, "score": result["score"],
                      "failures": result["failures"],
                      "path": str(destination)},
                     ensure_ascii=False))


def migrate_large(args):
    """Move an active campaign to the 96-item regime without erasing history."""
    run = Path(args.run).resolve()
    settings = _settings(run)
    summary = _call(run, "summary")
    campaign_regime = (summary.get("fitness_regime") or {}).get(TASK)
    manifest_migrated = (
        settings.get("evaluation_regime") == LARGE_PANEL_VERSION)
    campaign_migrated = campaign_regime == LARGE_PANEL_VERSION
    if manifest_migrated and campaign_migrated:
        print(json.dumps({"status": "already_migrated",
                          "fitness_regime": LARGE_PANEL_VERSION}))
        return
    queue = JobQueue(run).status()["jobs"]
    active_scores = [job["id"] for job in queue
                     if job["kind"] == "protected-score"
                     and job["status"] in {"queued", "running"}]
    if active_scores:
        raise SystemExit(
            "finish or cancel old-regime protected scores before migration: "
            + ", ".join(active_scores))
    large_regime = _large_regime_files(
        Path(settings["factory_root"]), Path(settings["corpus_dir"]),
        Path(settings["work_root"]), settings["starter_sha256"])
    panel_hash = large_regime["large_development_panel_sha256"]
    baseline_path = Path(large_regime["large_baseline_cache"])
    if not manifest_migrated:
        settings["legacy_evaluation_regime"] = settings.get(
            "evaluation_regime", "initial")
    settings.update(large_regime)
    settings["evaluator_sha256"] = hashlib.sha256(
        (HERE / "protected_evaluator.py").read_bytes()).hexdigest()
    _write_json(run / "manifest.json", settings)
    shutil.copy2(HERE / "CONTRACT.md", run / "CONTRACT.md")
    if campaign_migrated:
        regime = {"status": "already_active", "name": LARGE_PANEL_VERSION}
    else:
        regime = _call(run, "regime", {
            "task": TASK,
            "name": LARGE_PANEL_VERSION,
            "rationale": (
                "Replace the noisy 12/24-item sequential screen with 96 "
                "distinct paired translations while preserving every "
                "research trajectory."),
            "evidence": {
                "translation_items": LARGE_TRANSLATION_ITEMS,
                "verses_per_language": 12,
                "languages": 8,
                "panel_sha256": panel_hash,
                "old_reports": (
                    "historical low-sample evidence; not comparable"),
            },
        })
    _call(run, "media", {
        "name": "evaluation regime", "kind": "text",
        "data": (
            "large-96-v1 active: 12 distinct verses × 8 languages; frozen "
            "baseline pending; old scores retained as historical evidence."),
    })
    print(json.dumps({
        "status": "migrated", "regime": regime,
        "panel_sha256": panel_hash,
        "translation_items": LARGE_TRANSLATION_ITEMS,
        "baseline_cache": str(baseline_path),
    }, ensure_ascii=False))


def large_control(args):
    """Create the single reusable baseline for the active 96-item panel."""
    run = Path(args.run).resolve()
    settings = _settings(run)
    if settings.get("evaluation_regime") != LARGE_PANEL_VERSION:
        raise SystemExit("run migrate-large before building the large baseline")
    _call(run, "media", {
        "name": "active evaluation", "kind": "text",
        "data": "Building the one-time frozen 96-item baseline."})
    cache = _build_large_baseline(run, settings, workers=args.workers)
    session_rows = [session for shard in cache["shards"]
                    for session in shard["sessions"]]
    values = [score for session in session_rows
              for score in session["task_scores"]]
    score_value = sum(values) / len(values)
    failures = sum(int(shard["failures"]) for shard in cache["shards"])
    evidence = {
        "fitness_regime": LARGE_PANEL_VERSION,
        "panel_sha256": cache["panel_sha256"],
        "baseline_sha256": cache["baseline_sha256"],
        "translation_items": len(values),
        "absolute_chrf": score_value,
        "failed_sessions": failures,
    }
    _call(run, "baseline", {
        "task": TASK, "score": 0.0,
        "source": "one frozen native-Luna 96-item baseline",
        "rationale": (
            "Reuse this exact control for every candidate so research compute "
            "buys new candidate evidence instead of baseline reruns."),
        "evidence": evidence,
    })
    _call(run, "media", {
        "name": "active evaluation", "kind": "text",
        "data": (f"Frozen 96-item baseline ready: {score_value:.4f} chrF++; "
                 f"failed sessions={failures}.")})
    print(json.dumps(evidence, ensure_ascii=False))


def queue_large_control(args):
    run = Path(args.run).resolve()
    settings = _settings(run)
    panel = _large_panel_path(settings)
    panel_hash = hashlib.sha256(panel.read_bytes()).hexdigest()
    key = f"large-control:{panel_hash}:{settings['starter_sha256']}"
    argv = [
        settings["evaluator_python"], str(HERE / "native_control.py"),
        "large-control", "--run", str(run), "--workers", str(args.workers),
    ]
    job = _queue_once(
        run, key=key, kind="protected-baseline", lane="evaluator",
        name="one-time 96-item frozen baseline", argv=argv,
        timeout=args.timeout, max_attempts=args.max_attempts,
        metadata={"fitness_regime": LARGE_PANEL_VERSION,
                  "panel_sha256": panel_hash})
    print(json.dumps(job, ensure_ascii=False))


def status(args):
    run = Path(args.run).resolve()
    print(json.dumps({"summary": _call(run, "summary"),
                      "lineages": _call(run, "lineages"),
                      "decoder": _call(run, "decoder")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("init")
    command.add_argument("--run", required=True)
    command.add_argument("--factory-root", required=True)
    command.add_argument("--factory-repo", required=True)
    command.add_argument("--corpus-dir", required=True)
    command.add_argument("--codex", required=True)
    command.add_argument("--evaluator-python")
    command.add_argument("--work-root", required=True)
    command.set_defaults(function=initialize)

    command = commands.add_parser("server")
    command.add_argument("--run", required=True)
    command.add_argument("--port", type=int, default=8770)
    command.set_defaults(function=run_server)

    command = commands.add_parser("found")
    command.add_argument("--run", required=True)
    command.add_argument("--idea")
    command.add_argument("--rationale", required=True)
    command.add_argument("--kind", choices=("found", "inject"), default="found")
    command.add_argument("--worker")
    command.add_argument("--parent", action="append", type=_report_ref)
    command.set_defaults(function=found)

    command = commands.add_parser("prepare")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", required=True)
    command.set_defaults(function=prepare)

    command = commands.add_parser("evolve-native")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", required=True)
    command.add_argument("--codex")
    command.add_argument("--timeout", type=float, default=1800)
    command.add_argument("--enqueue-score", action="store_true")
    command.add_argument("--enqueue-lab", action="store_true")
    command.add_argument("--lab-workers", type=int, default=2)
    command.add_argument("--lab-timeout", type=float, default=7200)
    command.add_argument("--lab-attempts", type=int, default=1)
    command.add_argument("--score-workers", type=int, default=2)
    command.add_argument("--score-timeout", type=float, default=7200)
    command.add_argument("--score-attempts", type=int, default=1)
    command.set_defaults(function=evolve_native)

    command = commands.add_parser("repair-packaging")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", action="append", required=True)
    command.add_argument("--enqueue-score", action="store_true")
    command.add_argument("--score-workers", type=int, default=2)
    command.add_argument("--score-timeout", type=float, default=7200)
    command.add_argument("--score-attempts", type=int, default=1)
    command.set_defaults(function=repair_packaging)

    command = commands.add_parser("queue-evolver")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", required=True)
    command.add_argument("--agent-timeout", type=float, default=1800)
    command.add_argument("--job-timeout", type=float, default=1900)
    command.add_argument("--agent-attempts", type=int, default=2)
    command.add_argument(
        "--after", choices=("lab", "score", "none"), default="lab",
        help="what Finch queues after Sol materializes the candidate")
    command.add_argument("--lab-workers", type=int, default=2)
    command.add_argument("--lab-timeout", type=float, default=7200)
    command.add_argument("--lab-attempts", type=int, default=1)
    command.add_argument("--score-workers", type=int, default=2)
    command.add_argument("--score-timeout", type=float, default=7200)
    command.add_argument("--score-attempts", type=int, default=1)
    command.set_defaults(function=queue_evolver)

    command = commands.add_parser("queue-score")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", required=True)
    command.add_argument("--workers", type=int, default=2)
    command.add_argument("--timeout", type=float, default=7200)
    command.add_argument("--max-attempts", type=int, default=1)
    command.set_defaults(function=queue_score)

    command = commands.add_parser("reflect-native")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", required=True)
    command.add_argument("--report", required=True, type=int)
    command.add_argument("--codex")
    command.add_argument("--timeout", type=float, default=900)
    command.set_defaults(function=reflect_native)

    command = commands.add_parser("queue-reflection")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", required=True)
    command.add_argument("--report", required=True, type=int)
    command.add_argument("--agent-timeout", type=float, default=900)
    command.add_argument("--job-timeout", type=float, default=960)
    command.add_argument("--max-attempts", type=int, default=2)
    command.set_defaults(function=queue_reflection)

    command = commands.add_parser("queue-missing-reflections")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", action="append")
    command.add_argument("--agent-timeout", type=float, default=900)
    command.add_argument("--job-timeout", type=float, default=960)
    command.add_argument("--max-attempts", type=int, default=2)
    command.set_defaults(function=queue_missing_reflections)

    command = commands.add_parser("lab")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", required=True)
    command.add_argument("--candidate")
    command.add_argument("--workers", type=int, default=2)
    command.add_argument("--expected-sha256")
    command.set_defaults(function=laboratory)

    command = commands.add_parser("queue-lab")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", required=True)
    command.add_argument("--workers", type=int, default=2)
    command.add_argument("--timeout", type=float, default=7200)
    command.add_argument("--max-attempts", type=int, default=1)
    command.set_defaults(function=queue_lab)

    command = commands.add_parser("queue-judge")
    command.add_argument("--run", required=True)
    command.add_argument("--match", required=True)
    command.add_argument("--judge", required=True)
    command.add_argument("--agent-timeout", type=float, default=600)
    command.add_argument("--job-timeout", type=float, default=660)
    command.add_argument("--max-attempts", type=int, default=2)
    command.set_defaults(function=queue_judge)

    command = commands.add_parser("score")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", required=True)
    command.add_argument("--candidate")
    command.add_argument("--result-json", help="record a completed evaluator result after a bookkeeping failure")
    command.add_argument("--workers", type=int, default=2)
    command.add_argument("--expected-sha256")
    command.add_argument("--reflect-agent-timeout", type=float, default=900)
    command.add_argument("--reflect-job-timeout", type=float, default=960)
    command.add_argument("--reflect-attempts", type=int, default=2)
    command.set_defaults(function=score)

    command = commands.add_parser("audit")
    command.add_argument("--run", required=True)
    command.add_argument("--lineage", required=True)
    command.add_argument("--report", required=True, type=int)
    command.add_argument("--rationale")
    command.add_argument("--workers", type=int, default=2)
    command.set_defaults(function=audit)

    command = commands.add_parser("migrate-large")
    command.add_argument("--run", required=True)
    command.set_defaults(function=migrate_large)

    command = commands.add_parser("large-control")
    command.add_argument("--run", required=True)
    command.add_argument("--workers", type=int, default=4)
    command.set_defaults(function=large_control)

    command = commands.add_parser("queue-large-control")
    command.add_argument("--run", required=True)
    command.add_argument("--workers", type=int, default=4)
    command.add_argument("--timeout", type=float, default=14400)
    command.add_argument("--max-attempts", type=int, default=1)
    command.set_defaults(function=queue_large_control)

    command = commands.add_parser("control")
    command.add_argument("--run", required=True)
    command.add_argument("--panel", choices=("selection", "generalization"),
                         required=True)
    command.add_argument("--workers", type=int, default=2)
    command.set_defaults(function=control)

    command = commands.add_parser("status")
    command.add_argument("--run", required=True)
    command.set_defaults(function=status)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
