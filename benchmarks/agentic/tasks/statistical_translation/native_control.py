#!/usr/bin/env python3
"""Protected control helpers for the high-agent statistical-translation run.

This module does not schedule evolution. The root agent founds lineages,
dispatches native subagents, and decides what to audit/cross/incorporate.
These commands only prepare isolated experiments, assign protected fitness,
and record those decisions through Finch's high-agent HTTP API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finch4.agentic import Campaign, report_citation
from finch4.serve import serve

from campaign import (ANGLES, _load_or_create_secrets, _write_json,
                      build_partitions)
from checkpoint_policy import (checkpoint_eligible, keep_decision, seed_for)
from evaluator import (dataset_hash, evaluate, paired_bootstrap_delta,
                       public_result)


HERE = Path(__file__).resolve().parent
TASK = "statistical_translation"
COMPOSITION_MARGIN = 3.0
ORDINARY_NONINFERIORITY_MARGIN = 1.0
CELL_LOSS_LIMIT = 3.0


def _call(run: Path, name: str, body=None):
    server_info = json.loads(
        (run / "server.json").read_text(encoding="utf-8"))
    url = f"http://127.0.0.1:{server_info['port']}/{name}"
    request = (urllib.request.Request(url) if body is None else
               urllib.request.Request(
                   url, data=json.dumps(body).encode(), method="POST",
                   headers={"Content-Type": "application/json"}))
    with urllib.request.urlopen(request) as response:
        return json.loads(response.read())


def _partitions(run: Path, corpus_dir: str):
    return build_partitions(run, corpus_dir)


def _composition_eligibility(result, baseline):
    cell_deltas = {
        name: cell["correct_chrf"] - baseline["cells"][name]["correct_chrf"]
        for name, cell in result.get("cells", {}).items()
        if name in baseline.get("cells", {})}
    reasons = []
    if not result.get("admissible") or result.get("failures"):
        reasons.append("inadmissible or failed composition batch")
    if result.get("score", -100) < baseline.get("score", 0) + COMPOSITION_MARGIN:
        reasons.append("composition chrF did not clear frozen-copy margin")
    if result.get("exact_training_copy_rate", 1.0) > .05:
        reasons.append("composition outputs copy complete training targets")
    if cell_deltas and min(cell_deltas.values()) < 0:
        reasons.append("a language/condition composition cell lost to control")
    return not reasons, {
        "passed": not reasons, "reasons": reasons,
        "margin_required": COMPOSITION_MARGIN, "cell_deltas": cell_deltas}


def initialize(args):
    run = Path(args.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "state.json").exists() or (run / "manifest.json").exists():
        raise SystemExit("run already initialized; choose a fresh directory")
    shutil.copy2(HERE / "CONTRACT.md", run / "CONTRACT.md")
    # The initial decoder is methodology only. In particular, it does not
    # install retrieval-copy or any old champion as the shared default.
    shutil.copy2(HERE / "base_playbook.md", run / "Decoder.md")
    practice, selection, generalization, confirmation = _partitions(
        run, args.corpus_dir)
    _write_json(run / "practice.json", practice)

    controls = {}
    for label, artifact in (("retrieval_copy", HERE / "baseline_candidate.py"),
                            ("neutral_starter", HERE / "starter_candidate.py")):
        controls[label] = {
            "selection": public_result(evaluate(
                artifact, selection, python=args.python)),
            "selection_composition": public_result(evaluate(
                artifact, selection, python=args.python,
                query_group="composition_queries")),
            "generalization": public_result(evaluate(
                artifact, generalization, python=args.python,
                diagnostics=True)),
        }
    _write_json(run / "controls.json", controls)

    secrets_value = _load_or_create_secrets(run)
    manifest = {
        "contract": "v3-high-agent-report-stream",
        "orchestrator": "root Codex native subagents",
        "worker_model_requested": "gpt-5.6-luna",
        "evolution_policy": "high-agent judgment; Finch records only",
        "initial_lineage_target": 12,
        "initial_lineage_ceiling": 25,
        "fitness_cadence": "one protected score per evaluable experiment",
        "fitness": "equal-language equal-condition macro sentence chrF++",
        "breeding_gate": "trusted score plus immutable artifact; disclose audit",
        "shared_methodology_gate": "exact report must pass generalization audit",
        "example_limit": None,
        "practice_hash": dataset_hash(practice),
        "selection_hash": dataset_hash(selection),
        "generalization_hash": dataset_hash(generalization),
        "confirmation_hash": dataset_hash(confirmation),
        "evaluator_sha256": hashlib.sha256(
            (HERE / "evaluator.py").read_bytes()).hexdigest(),
        "starter_sha256": hashlib.sha256(
            (HERE / "starter_candidate.py").read_bytes()).hexdigest(),
        "retrieval_control_sha256": hashlib.sha256(
            (HERE / "baseline_candidate.py").read_bytes()).hexdigest(),
        "seed_commitments": {
            key: hashlib.sha256(str(value).encode()).hexdigest()
            for key, value in secrets_value.items()},
        "confirmation_status": "UNTOUCHED",
    }
    _write_json(run / "manifest.json", manifest)
    print(json.dumps({
        "run": str(run),
        "neutral_selection": controls["neutral_starter"]["selection"]["score"],
        "retrieval_control_selection": controls["retrieval_copy"]["selection"]["score"],
        "retrieval_control_composition": controls["retrieval_copy"][
            "selection_composition"]["score"],
        "confirmation_status": "UNTOUCHED"}))


def run_server(args):
    run = Path(args.run).resolve()
    server = serve(run, port=args.port, tasks=[TASK])
    print(json.dumps({
        "port": server.server_address[1],
        "progress": f"http://127.0.0.1:{server.server_address[1]}/progress",
    }), flush=True)
    server.serve_forever()


def _report_ref(value: str):
    try:
        lineage, report = value.split(":", 1)
        return [lineage, int(report)]
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("parent must be LINEAGE:REPORT")


def _line(run: Path, lineage_id: str):
    campaign = Campaign.load(run / "state.json")
    try:
        return campaign, campaign.lineages[lineage_id]
    except KeyError:
        raise SystemExit(f"unknown lineage {lineage_id}")


def _prepare(run: Path, lineage_id: str, python: str):
    campaign, line = _line(run, lineage_id)
    out = run / "lineages" / lineage_id
    out.mkdir(parents=True, exist_ok=True)
    seed, seed_score = seed_for(
        campaign, line, HERE / "starter_candidate.py")
    shutil.copy2(seed, out / "seed_candidate.py")
    shutil.copy2(seed, out / "candidate.py")
    for index, parent in enumerate(line["parents"], 1):
        shutil.copy2(parent["artifact"], out / f"parent_{index}.py")
    experiment = sum(not report.get("voided")
                     for report in line["reports"]) + 1
    angle = line.get("idea") or "invent a distinct, testable statistical mechanism"
    continuation = ""
    if experiment > 1:
        continuation = (
            f"This is a continuation. Your checkpoint's trusted selection "
            f"score is {seed_score}. Read your existing log.md, take the "
            "mechanism to its next logical step, and do not repeat a rejected "
            "move. A creative departure is welcome when the current assumption "
            "looks exhausted.\n")
    parent_context = ""
    if line["parents"]:
        parent_lines = []
        for index, parent in enumerate(line["parents"], 1):
            parent_report = campaign.lineages[parent["lineage"]][
                "reports"][parent["report"]]
            checkpoint = ("eligible complete checkpoint" if
                          checkpoint_eligible(parent_report) else
                          "INELIGIBLE as a complete checkpoint; gene donor only")
            audit = parent.get("audit") or {}
            # New records store a compact audit brief. Accept older full
            # audit snapshots so an in-flight campaign remains readable.
            evidence = audit.get("evidence") or audit
            reasons = evidence.get("reasons") or []
            delta = evidence.get("paired_bootstrap") or {}
            warning = "; ".join(map(str, reasons)) or "none recorded"
            interval = delta.get("ci95")
            parent_lines.append(
                f"- parent_{index}.py = "
                f"{report_citation(parent['lineage'], parent['report'], parent['artifact_sha256'])}, "
                f"protected chrF {parent['score']}, "
                f"audit {parent.get('audit_status', 'none')}; "
                f"composition status: {checkpoint}; "
                f"failure warnings: {warning}; "
                f"generalization delta CI: {interval}")
        parent_context = (
            "This is crossover. Inspect every parent_N.py as an exact, "
            "independently scored checkpoint. A parent is evidence, not "
            "shared truth: failed or inconclusive audits below are explicit "
            "constraints for the child to repair. Implement one coherent "
            "interaction; do not merely concatenate their code.\n"
            + "\n".join(parent_lines) + "\n")
    prompt = f"""You are the continuous research worker for Finch lineage {lineage_id}, experiment {experiment}.

Read these files first:
- campaign law: {run / 'CONTRACT.md'}
- current shared methodology: {run / 'Decoder.md'}
- your starting checkpoint: {out / 'seed_candidate.py'}
- public practice data: {run / 'practice.json'}
- protected scorer implementation (read-only): {HERE / 'evaluator.py'}

Assigned research direction:
{angle}

{parent_context}{continuation}Perform ONE consequential experiment. You may use every training pair supplied to the candidate; there is no ten-example limit. The submitted translator must be deterministic, non-neural, and derive every tokenizer, lexicon, embedding-like representation, and parameter solely from the current payload. Retrieval-copy is a comparison control, not the shared default or a sufficient contribution.

Be creative. Prefer a change to representation, bilingual induction, segmentation, candidate generation, ordering, coverage, uncertainty, or decoding over a coefficient tweak. Strange but falsifiable mechanisms are welcome. You are not required to remain inside the inherited mechanism: if you see a better framing, make the conceptual leap and test it. At a plateau, question one assumption instead of polishing constants. State a causal prediction, an activation condition, and a falsifier before coding.

Write hypothesis.md as concise Markdown. If the mechanism materially adopts,
extends, contrasts, or combines something from Decoder.md or a parent, preserve
the exact citation token beside that claim (for example
[L0003#2@8ddf8b41]). Finch parses that report text into the Tree of Life.
Cite genuine inspiration, not everything you happened to read.

Edit only {out / 'candidate.py'}, {out / 'hypothesis.md'}, and {out / 'log.md'}. Start candidate.py from seed_candidate.py. You may run public practice at most four times:
{python} {HERE / 'public_score.py'} --dataset {run / 'practice.json'} --python {python} {out / 'candidate.py'}

Private selection/generalization/confirmation data and their seeds are unavailable. Do not inspect other run directories or lineages, use network access, or contact Finch. The orchestrator will independently score your candidate and decide keep/revert. Finish with the implemented mechanism and public score; never claim a protected score.
"""
    (out / "prompt.md").write_text(prompt, encoding="utf-8")
    _write_json(out / "experiment.json", {
        "lineage": lineage_id, "experiment": experiment,
        "seed_artifact": str(seed), "seed_score": seed_score,
        "decoder_version": campaign.decoder_version})
    return {
        "lineage": lineage_id, "experiment": experiment,
        "directory": str(out), "prompt": str(out / "prompt.md"),
        "candidate": str(out / "candidate.py"), "seed_score": seed_score}


def found(args):
    run = Path(args.run).resolve()
    idea = args.idea
    if idea is None:
        summary = _call(run, "summary")
        idea = ANGLES[summary["lineages"] % len(ANGLES)]
    body = {"task": TASK, "rationale": args.rationale,
            "idea": idea, "kind": args.kind, "worker": args.worker}
    if args.parent:
        body["parents"] = args.parent
        body["kind"] = "crossover"
    line = _call(run, "found", body)
    prepared = _prepare(run, line["id"], args.python)
    print(json.dumps({
        "lineage": {
            "id": line["id"], "kind": line["kind"],
            "worker": line.get("worker"),
            "parents": [{"lineage": parent["lineage"],
                         "report": parent["report"],
                         "score": parent["score"],
                         "audit_status": parent.get("audit_status")}
                        for parent in line["parents"]]},
        "experiment": prepared}, ensure_ascii=False))


def prepare(args):
    print(json.dumps(_prepare(Path(args.run).resolve(), args.lineage,
                              args.python), ensure_ascii=False))


def score(args):
    run = Path(args.run).resolve()
    campaign, line = _line(run, args.lineage)
    out = run / "lineages" / args.lineage
    candidate = Path(args.candidate).resolve() if args.candidate else (
        out / "candidate.py")
    metadata_path = out / "experiment.json"
    metadata = (json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata_path.exists() else {})
    experiment = int(metadata.get("experiment", sum(
        not report.get("voided") for report in line["reports"]) + 1))
    experiment_id = f"experiment-{experiment:04d}"
    candidate_sha256 = (hashlib.sha256(candidate.read_bytes()).hexdigest()
                        if candidate.is_file() else None)
    for existing in line["reports"]:
        same_id = existing.get("experiment_id") == experiment_id
        # Schema-4 reports created before experiment IDs were introduced:
        # a retry of the current experiment is also recognized by content.
        legacy_retry = (
            existing.get("experiment_id") is None
            and experiment <= sum(not report.get("voided")
                                  for report in line["reports"])
            and candidate_sha256 is not None
            and existing.get("artifact_sha256") == candidate_sha256)
        if not existing.get("voided") and (same_id or legacy_retry):
            print(json.dumps({
                "lineage": args.lineage, "report": existing["index"],
                "protected_chrf": existing.get("score"),
                "idempotent_retry": True}))
            return
    _, selection, _, _ = _partitions(run, args.corpus_dir)
    protected = evaluate(candidate, selection, python=args.python)
    protected_composition = evaluate(
        candidate, selection, python=args.python,
        query_group="composition_queries")
    controls = json.loads((run / "controls.json").read_text(encoding="utf-8"))
    eligible, gate = _composition_eligibility(
        protected_composition,
        controls["retrieval_copy"]["selection_composition"])
    # A crossover's inherited artifact is its first local champion even
    # though it belongs to a parent report. Compare the first child against
    # that actual checkpoint, not against an empty report stream.
    inherited = metadata.get("seed_score")
    previous = (line["best_score"] if line["best_score"] is not None
                else inherited)
    kept, improved, keep_reason = keep_decision(
        previous, protected["score"], eligible)
    hypothesis_path = out / "hypothesis.md"
    summary = (hypothesis_path.read_text(encoding="utf-8").strip()
               if hypothesis_path.exists()
               else f"lineage {args.lineage} experiment")[:4000]
    evidence = {
        "protected": public_result(protected),
        "protected_composition": public_result(protected_composition),
        "composition_eligibility": gate,
        "selection_metric_only": "macro sentence chrF++",
        "checkpoint_decision": {
            "kept": kept, "improved": improved,
            "eligible": eligible, "reason": keep_reason},
    }
    report = _call(run, "report", {
        "lineage": args.lineage, "summary": summary,
        "score": float(protected["score"]),
        "source": "protected translation evaluator v3",
        "evidence": evidence, "kept": kept,
        "claimed_score": args.claimed_score,
        "artifact": str(candidate) if candidate.is_file() else None,
        "experiment_id": experiment_id})
    if not kept:
        seed_path = out / "seed_candidate.py"
        if seed_path.is_file() and candidate != seed_path:
            shutil.copy2(seed_path, candidate)
    _write_json(out / f"report_{report['index']:04d}.json", {
        "report": report, "evidence": evidence})
    print(json.dumps({
        "lineage": args.lineage, "report": report["index"],
        "protected_chrf": protected["score"],
        "previous_lineage_best": previous, "kept": kept,
        "keep_reason": keep_reason,
        "composition_chrf": protected_composition["score"],
        "composition_eligible": eligible,
        "eligibility_reasons": gate["reasons"],
        "claimed_public_chrf": args.claimed_score,
        "artifact": report.get("artifact")}, ensure_ascii=False))


def audit(args):
    run = Path(args.run).resolve()
    campaign, line = _line(run, args.lineage)
    if args.report < 0:
        raise SystemExit("report index must be non-negative")
    try:
        report = line["reports"][args.report]
    except IndexError:
        raise SystemExit("unknown report")
    if report.get("score") is None or not report.get("artifact"):
        raise SystemExit("audit requires a trusted score and artifact")
    _, selection, generalization, _ = _partitions(run, args.corpus_dir)
    controls = json.loads((run / "controls.json").read_text(encoding="utf-8"))
    candidate = Path(report["artifact"])
    reproduced = evaluate(candidate, selection, python=args.python)
    general = evaluate(
        candidate, generalization, python=args.python, diagnostics=True)
    baseline_general = evaluate(
        HERE / "baseline_candidate.py", generalization,
        python=args.python, diagnostics=True)
    general_composition = evaluate(
        candidate, generalization, python=args.python,
        query_group="composition_queries")
    baseline_composition = evaluate(
        HERE / "baseline_candidate.py", generalization,
        python=args.python, query_group="composition_queries")
    composition_passed, composition_gate = _composition_eligibility(
        general_composition, baseline_composition)
    delta = paired_bootstrap_delta(
        general, baseline_general,
        seed=_load_or_create_secrets(run)["bootstrap_seed"]
        + report["arrival"])
    cell_deltas = {
        name: cell["correct_chrf"]
        - baseline_general["cells"][name]["correct_chrf"]
        for name, cell in general.get("cells", {}).items()}
    hard_reasons = []
    if not reproduced.get("admissible") or reproduced.get("failures"):
        hard_reasons.append("selection evaluation failed or became inadmissible")
    if abs(reproduced.get("score", -100) - report["score"]) > 1e-9:
        hard_reasons.append("selection score did not reproduce")
    if any(value < -CELL_LOSS_LIMIT for value in cell_deltas.values()):
        hard_reasons.append("a generalization cell exceeded the loss limit")
    if not composition_passed:
        hard_reasons.append("generalization composition gate failed")
    ordinary_passed = delta["ci95"][0] >= -ORDINARY_NONINFERIORITY_MARGIN
    if delta["delta"] < -ORDINARY_NONINFERIORITY_MARGIN:
        hard_reasons.append("ordinary generalization exceeded loss margin")
    if hard_reasons:
        outcome = "failed"
    elif ordinary_passed:
        outcome = "passed"
    else:
        outcome = "inconclusive"
    evidence = {
        "reproduced_selection": public_result(reproduced),
        "generalization": public_result(general),
        "retrieval_control_generalization": public_result(baseline_general),
        "paired_bootstrap": delta, "cell_deltas": cell_deltas,
        "generalization_composition": public_result(general_composition),
        "retrieval_control_composition": public_result(baseline_composition),
        "composition_eligibility": composition_gate,
        "ordinary_noninferiority": {
            "passed": ordinary_passed,
            "margin": ORDINARY_NONINFERIORITY_MARGIN},
        "reasons": hard_reasons,
        "confirmation_status": controls and "UNTOUCHED",
    }
    decision = _call(run, "audit", {
        "lineage": args.lineage, "report": args.report,
        "outcome": outcome,
        "rationale": args.rationale or (
            f"protected reproduction and generalization audit: {outcome}"),
        "evidence": evidence})
    audit_dir = run / "audits"
    audit_dir.mkdir(exist_ok=True)
    _write_json(audit_dir / f"{args.lineage}-r{args.report:04d}.json",
                {"decision": decision, "evidence": evidence})
    print(json.dumps({
        "lineage": args.lineage, "report": args.report,
        "outcome": outcome, "selection": report["score"],
        "generalization": general["score"],
        "retrieval_control_generalization": baseline_general["score"],
        "delta_ci95": delta["ci95"],
        "composition": general_composition["score"],
        "reasons": hard_reasons}, ensure_ascii=False))


def status(args):
    run = Path(args.run).resolve()
    print(json.dumps({
        "summary": _call(run, "summary"),
        "lineages": _call(run, "lineages"),
        "decoder": _call(run, "decoder")}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", required=True)
    common.add_argument("--corpus-dir", required=True)
    common.add_argument("--python", required=True)

    command = sub.add_parser("init", parents=[common])
    command.set_defaults(function=initialize)

    command = sub.add_parser("server")
    command.add_argument("--run", required=True)
    command.add_argument("--port", type=int, default=8769)
    command.set_defaults(function=run_server)

    command = sub.add_parser("found")
    command.add_argument("--run", required=True)
    command.add_argument("--python", required=True)
    command.add_argument("--idea")
    command.add_argument("--rationale", required=True)
    command.add_argument("--kind", choices=("found", "inject"),
                         default="found")
    command.add_argument("--worker")
    command.add_argument("--parent", action="append", type=_report_ref)
    command.set_defaults(function=found)

    command = sub.add_parser("prepare")
    command.add_argument("--run", required=True)
    command.add_argument("--python", required=True)
    command.add_argument("--lineage", required=True)
    command.set_defaults(function=prepare)

    command = sub.add_parser("score", parents=[common])
    command.add_argument("--lineage", required=True)
    command.add_argument("--candidate")
    command.add_argument("--claimed-score", type=float)
    command.set_defaults(function=score)

    command = sub.add_parser("audit", parents=[common])
    command.add_argument("--lineage", required=True)
    command.add_argument("--report", required=True, type=int)
    command.add_argument("--rationale")
    command.set_defaults(function=audit)

    command = sub.add_parser("status")
    command.add_argument("--run", required=True)
    command.set_defaults(function=status)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
