#!/usr/bin/env python3
"""Run the protected statistical-translation GAR v2 compositional campaign."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from finch4.serve import serve

from evaluator import (build_dataset, dataset_hash, dataset_source_keys,
                       evaluate, paired_bootstrap_delta, public_result)


HERE = Path(__file__).resolve().parent
PRACTICE_LANGUAGES = ["French", "Swahili"]
SELECTION_LANGUAGES = ["French", "Swahili"]
GENERALIZATION_LANGUAGES = ["Tzotzil", "Limbu"]
CONFIRMATION_LANGUAGES = ["Assyrian Neo-Aramaic", "Gamo"]
TRAINING_SIZE = 1200
SELECTION_QUERIES = 32
GENERALIZATION_QUERIES = 50
CONFIRMATION_QUERIES = 50
PRACTICE_COMPOSITION_QUERIES = 8
SELECTION_COMPOSITION_QUERIES = 24
GENERALIZATION_COMPOSITION_QUERIES = 32
CONFIRMATION_COMPOSITION_QUERIES = 32

ANGLES = [
    "derive an IBM-Model-1-like lexical channel with EM from the full training set",
    "induce reusable phrase correspondences and compose rather than copy outputs",
    "use relative positions to distinguish translation from co-occurrence",
    "build a character-substring channel for unseen word forms",
    "use a variable-order Markov or PPM target model with principled backoff",
    "formulate translation as noisy-channel beam search with explicit coverage",
    "infer fertility, insertion, and deletion behavior from aligned lengths",
    "learn reordering templates from monotonic and inverted fragments",
    "combine word and character evidence without language-specific rules",
    "learn when retrieval is useful and when the full corpus should train a model",
    "build an analogical translation-memory editor rather than copying one target",
    "infer a sparse lexicon using contrastive co-occurrence across all pairs",
    "use held-out-within-training reliability to weight translation rules",
    "construct a finite-state or dynamic-programming decoder with coverage",
    "model target order from pairwise precedence rather than adjacent bigrams",
    "jointly segment source and target into reusable variable-length units",
    "estimate uncertainty and avoid unsupported target insertions",
    "exploit repeated function-word patterns while remaining valid under codes",
    "ensemble several cheap statistical induction mechanisms causally",
    "invent a distinct classical statistical translation family not listed above",
]


def _write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2),
                    encoding="utf-8")


def _secret_path(run_dir: Path) -> Path:
    label = hashlib.sha256(str(run_dir).encode()).hexdigest()[:20]
    directory = Path("/private/tmp/finch4-statistical-translation-v1-secrets")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory / f"{label}.json"


def _load_or_create_secrets(run_dir: Path):
    path = _secret_path(run_dir)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    value = {
        "selection_seed": secrets.randbits(60),
        "generalization_seed": secrets.randbits(60),
        "confirmation_seed": secrets.randbits(60),
        "bootstrap_seed": secrets.randbits(60),
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)
    return value


def build_partitions(run_dir: Path, corpus_dir: str):
    secrets_value = _load_or_create_secrets(run_dir)
    practice = build_dataset(
        corpus_dir, PRACTICE_LANGUAGES, seed=2026073101,
        training_size=300, queries_per_language=8,
        composition_queries_per_language=PRACTICE_COMPOSITION_QUERIES)
    excluded = dataset_source_keys(practice)
    selection = build_dataset(
        corpus_dir, SELECTION_LANGUAGES,
        seed=secrets_value["selection_seed"],
        training_size=TRAINING_SIZE,
        queries_per_language=SELECTION_QUERIES,
        composition_queries_per_language=SELECTION_COMPOSITION_QUERIES,
        exclude_sources=excluded)
    excluded.update(dataset_source_keys(selection))
    generalization = build_dataset(
        corpus_dir, GENERALIZATION_LANGUAGES,
        seed=secrets_value["generalization_seed"],
        training_size=TRAINING_SIZE,
        queries_per_language=GENERALIZATION_QUERIES,
        composition_queries_per_language=GENERALIZATION_COMPOSITION_QUERIES,
        exclude_sources=excluded)
    excluded.update(dataset_source_keys(generalization))
    confirmation = build_dataset(
        corpus_dir, CONFIRMATION_LANGUAGES,
        seed=secrets_value["confirmation_seed"],
        training_size=TRAINING_SIZE,
        queries_per_language=CONFIRMATION_QUERIES,
        composition_queries_per_language=CONFIRMATION_COMPOSITION_QUERIES,
        exclude_sources=excluded)
    return practice, selection, generalization, confirmation


def _prompt(job, out_dir: Path, practice_path: Path, public_score: Path,
            python: str, base_playbook: Path,
            max_crossover_branches: int = 4) -> str:
    parent_text = ""
    kind = job["kind"]
    if kind in ("found", "inject"):
        if kind == "found":
            number = int(job["job_id"][1:])
            assignment = (
                "Create an independent lineage from this research direction:\n"
                f"{ANGLES[number % len(ANGLES)]}\n")
        else:
            assignment = (
                "Realize this unscheduled orchestrator hypothesis as an "
                "independent lineage:\n"
                f"{json.dumps(job['idea'], ensure_ascii=False)}\n")
        outputs = "candidate.py, hypothesis.md, and log.md"
    elif kind in ("advance", "explore"):
        parent = job["parents"][0]
        parent_source = Path(parent["artifact"])
        shutil.copy2(parent_source, out_dir / "parent_candidate.py")
        shutil.copy2(parent_source, out_dir / "candidate.py")
        assignment = ((
            "Choose one consequential research move. Name one parent assumption "
            "you preserve and one you challenge. Prefer a change to representation, "
            "candidate generation, alignment, decoding, uncertainty, or training "
            "signal; creative leaps are explicitly welcome. A coefficient-only "
            "change needs a falsifiable calibration hypothesis.\n")
            if kind == "advance" else (
            "Use parent_candidate.py only as context and invent a genuinely "
            "different testable statistical mechanism. Change what bilingual "
            "structure is induced, how candidates are generated, how order or "
            "length is decided, or what evidence permits composition. Be creative.\n"))
        parent_text = (
            f"Parent protected chrF++: {parent['score']:.6f}\n"
            f"Parent proposal: {parent['proposal']}\n")
        outputs = "candidate.py, hypothesis.md, and log.md"
    elif kind == "crossover_plan":
        first, second = job["parents"]
        shutil.copy2(first["artifact"], out_dir / "parent_a.py")
        shutil.copy2(second["artifact"], out_dir / "parent_b.py")
        assignment = (
            f"Inspect both implemented parents and propose zero to "
            f"{max_crossover_branches} causal, non-additive interactions. Each "
            "branch must specify activation_condition, shared_decision, ablation, "
            "and expected_nonadditivity. Do not implement or "
            "score a candidate. Zero is valid.\n")
        parent_text = (
            f"Parent A chrF++ {first['score']:.6f}: {first['proposal']}\n"
            f"Parent B chrF++ {second['score']:.6f}: {second['proposal']}\n")
        outputs = ("branches.json: a JSON list whose items contain title, "
                   "hypothesis, mechanism, inherited_from_a, inherited_from_b, "
                   "interaction, activation_condition, shared_decision, ablation, "
                   "expected_nonadditivity, and main_risk")
    elif kind == "crossover":
        first, second = job["parents"]
        shutil.copy2(first["artifact"], out_dir / "parent_a.py")
        shutil.copy2(second["artifact"], out_dir / "parent_b.py")
        shutil.copy2(first["artifact"], out_dir / "candidate.py")
        assignment = (
            "Implement this accepted interaction as one coherent candidate:\n"
            f"{json.dumps(job['branch'], ensure_ascii=False)}\n")
        parent_text = (
            f"Parent A chrF++ {first['score']:.6f}: {first['proposal']}\n"
            f"Parent B chrF++ {second['score']:.6f}: {second['proposal']}\n")
        outputs = "candidate.py, hypothesis.md, and log.md"
    else:
        raise ValueError(f"unknown GAR job kind {kind!r}")

    if kind == "crossover_plan":
        practice = "Do not run the scorer for this planning-only job."
        finish = "Validate branches.json and stop."
    else:
        practice = f"""Public practice command:
{python} {public_score} --dataset {practice_path} --python {python} {out_dir / 'candidate.py'}
You may run it at most four times."""
        finish = (
            "Run public practice at least once unless execution is broken. The "
            "orchestrator, not you, assigns protected fitness.")

    return f"""You are a research worker for Finch GAR job {job['job_id']}.

Author one executable, deterministic, generic, non-neural statistical
translation mechanism. Read:
- Campaign laws: {HERE / 'CONTRACT.md'}
- Current shared methodology: {base_playbook}
- Frozen retrieval-copy control (comparison only; never the default): {HERE / 'baseline_candidate.py'}

{assignment}{parent_text}
Work only in {out_dir}. Required output: {outputs}.

The candidate protocol is `python candidate.py translate`. It receives the
complete approved parallel training set plus a batch of held-out source
queries and returns one translation per query. It may use all supplied pairs,
train once for the batch, or retrieve internally. There is no ten-example
limit. Seek an actual translation mechanism, not merely a cosmetic variation.
Retrieval-copy is a safety fallback, not a completed research contribution.
The public scorer includes novel-conjunction composition episodes: each query
combines source material whose target evidence lives in separate training
pairs. A complete copied training target cannot pass eligibility.

Before coding, state in hypothesis.md: (1) the mechanism, (2) a causal
prediction describing when it changes an output, (3) its fallback guardrail,
and (4) a falsifier. The mechanism must affect a reachable decision path.
Do not merely attach an inert low-weight term. Creative, unconventional, and
new statistical model families are welcome when they obey the data boundary.

{practice}

Private selection, generalization, and confirmation data are unavailable. Do
not search for seeds, corpora, other run directories, or references. Do not use
web/network access, pretrained tokenizers, dictionaries, embeddings, model
files, neural inference/training, hard-coded translations, or persistent
external state. All language-dependent state must come from the current JSON
payload. Imports are mechanically allowlisted.

Write hypothesis.md to describe the implemented mechanism and log.md to record
experiments and failures honestly. Log whether the mechanism changed at least
one practice output relative to its fallback; if it changed none, treat that as
a failed research move unless you can give a concrete protected activation
condition. {finish} Do not contact Finch directly.
"""


def _worker_command(codex: str, out_dir: Path):
    return [
        codex, "exec", "-m", "gpt-5.6-luna",
        "-c", 'model_reasoning_effort="medium"',
        "-c", 'web_search="disabled"',
        "--disable", "multi_agent", "--sandbox", "workspace-write",
        "--ephemeral", "--skip-git-repo-check", "-C", str(out_dir), "-",
    ]


def _run_job(server, job, args, practice, selection):
    out_dir = Path(args.run) / "individuals" / job["job_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    if job["kind"] in ("found", "inject"):
        seed_artifact = (job.get("idea", {}).get("seed_artifact")
                         if isinstance(job.get("idea"), dict) else None)
        shutil.copy2(seed_artifact or HERE / "starter_candidate.py",
                     out_dir / "candidate.py")
    prompt = _prompt(
        job, out_dir, Path(args.run) / "practice.json",
        HERE / "public_score.py", args.python,
        Path(args.run) / "base_playbook.md", args.max_crossover_branches)
    (out_dir / "prompt.md").write_text(prompt, encoding="utf-8")
    started = time.time()
    returncode = None
    error = None
    with open(out_dir / "agent.log", "w", encoding="utf-8") as log:
        try:
            completed = subprocess.run(
                _worker_command(args.codex, out_dir), input=prompt, text=True,
                stdout=log, stderr=subprocess.STDOUT,
                timeout=args.agent_timeout, check=False)
            returncode = completed.returncode
        except subprocess.TimeoutExpired:
            error = f"worker timed out after {args.agent_timeout}s"
        except OSError as exc:
            error = repr(exc)

    if job["kind"] == "crossover_plan":
        try:
            branches = json.loads(
                (out_dir / "branches.json").read_text(encoding="utf-8"))
            if not isinstance(branches, list):
                raise TypeError("branches.json is not a list")
            children = server.service.handle("expand", {
                "job_id": job["job_id"],
                "branches": branches[:args.max_crossover_branches]})
            print(f"[campaign] {job['job_id']} -> {len(children)} children",
                  flush=True)
        except Exception as exc:
            server.service.handle("abandon", {
                "job_id": job["job_id"], "reason": repr(exc)})
        return

    candidate = out_dir / "candidate.py"
    try:
        public_score = evaluate(candidate, practice, python=args.python)
        protected = evaluate(candidate, selection, python=args.python)
    except Exception as exc:
        public_score = {"score": -100.0, "error": repr(exc)}
        protected = {"score": -100.0, "error": repr(exc),
                     "admissible": False, "failures": 1}
    _write_json(out_dir / "public_score.json", public_result(public_score))
    _write_json(out_dir / "protected_score.json", public_result(protected))
    hypothesis_path = out_dir / "hypothesis.md"
    proposal = (hypothesis_path.read_text(encoding="utf-8").strip()
                if hypothesis_path.exists()
                else f"{job['kind']} candidate {job['job_id']}")[:4000]
    log_path = out_dir / "log.md"
    log_text = (log_path.read_text(encoding="utf-8")[-12000:]
                if log_path.exists() else error or "no worker log")
    submitted = server.service.handle("submit", {
        "job_id": job["job_id"], "proposal": proposal,
        "artifact": str(candidate.resolve()) if candidate.exists() else None,
        "reported_score": public_score.get("score"), "log": log_text})
    evidence = {
        "worker_model": "gpt-5.6-luna", "worker_reasoning": "medium",
        "worker_returncode": returncode, "worker_error": error,
        "worker_seconds": time.time() - started,
        "public": public_result(public_score),
        "protected": public_result(protected),
    }
    server.service.handle("fitness", {
        "id": submitted["id"], "score": float(protected.get("score", -100)),
        "source": "private_batch_chrf_v1", "evidence": evidence,
        "verified": bool(protected.get("admissible", False)
                         and protected.get("failures", 1) == 0)})
    print(f"[campaign] {job['job_id']} {job['kind']} "
          f"public={public_score.get('score', -100):.3f} "
          f"protected={protected.get('score', -100):.3f}", flush=True)


def audit_candidate(candidate, selection_score, selection, generalization,
                    baseline_generalization, *, python, bootstrap_seed):
    reproduced = evaluate(candidate, selection, python=python)
    result = evaluate(candidate, generalization, python=python,
                      diagnostics=True)
    delta = paired_bootstrap_delta(
        result, baseline_generalization, seed=bootstrap_seed)
    baseline_cells = baseline_generalization["cells"]
    cell_deltas = {
        name: cell["correct_chrf"] - baseline_cells[name]["correct_chrf"]
        for name, cell in result["cells"].items()}
    reasons = []
    if not result.get("admissible") or result.get("failures"):
        reasons.append("inadmissible or failed batch")
    if abs(reproduced.get("score", -100) - selection_score) > 1e-9:
        reasons.append("selection score did not reproduce")
    if delta["ci95"][0] <= 0:
        reasons.append("generalization chrF++ CI includes zero")
    if any(value < -3.0 for value in cell_deltas.values()):
        reasons.append("catastrophic language/condition cell loss")
    evidence = {
        "reproduced_selection": public_result(reproduced),
        "generalization": public_result(result),
        "baseline_generalization": public_result(baseline_generalization),
        "paired_bootstrap": delta, "cell_deltas": cell_deltas,
        "pass_reasons": reasons,
    }
    return not reasons, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--founders", type=int, default=20)
    parser.add_argument("--events-per-round", type=int, default=8)
    parser.add_argument("--population-cap", type=int, default=25)
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--max-crossover-branches", type=int, default=3)
    parser.add_argument("--agent-timeout", type=float, default=1800)
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.population_cap > 25:
        raise SystemExit("population cap may not exceed 25")
    run_dir = Path(args.run).resolve()
    args.run = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("base_playbook.md", "CONTRACT.md"):
        if not (run_dir / name).exists():
            shutil.copy2(HERE / name, run_dir / name)
    secrets_value = _load_or_create_secrets(run_dir)
    practice, selection, generalization, confirmation = build_partitions(
        run_dir, args.corpus_dir)
    _write_json(run_dir / "practice.json", practice)

    baseline_selection = evaluate(
        HERE / "baseline_candidate.py", selection, python=args.python)
    baseline_generalization = evaluate(
        HERE / "baseline_candidate.py", generalization, python=args.python,
        diagnostics=True)
    _write_json(run_dir / "baseline.json", {
        "selection": public_result(baseline_selection),
        "generalization": public_result(baseline_generalization)})
    manifest = {
        "contract": "v1-batch-continuation",
        "worker_model": "gpt-5.6-luna", "worker_reasoning": "medium",
        "founders": args.founders,
        "events_per_round": args.events_per_round,
        "population_cap": args.population_cap,
        "fitness": "equal-language equal-condition macro sentence chrF++",
        "training_size": TRAINING_SIZE,
        "selection_queries_per_language": SELECTION_QUERIES,
        "generalization_queries_per_language": GENERALIZATION_QUERIES,
        "confirmation_queries_per_language": CONFIRMATION_QUERIES,
        "practice_hash": dataset_hash(practice),
        "selection_hash": dataset_hash(selection),
        "generalization_hash": dataset_hash(generalization),
        "confirmation_hash": dataset_hash(confirmation),
        "evaluator_sha256": hashlib.sha256(
            (HERE / "evaluator.py").read_bytes()).hexdigest(),
        "baseline_sha256": hashlib.sha256(
            (HERE / "baseline_candidate.py").read_bytes()).hexdigest(),
        "seed_commitments": {
            key: hashlib.sha256(str(value).encode()).hexdigest()
            for key, value in secrets_value.items()},
        "confirmation_status": "UNTOUCHED",
    }
    _write_json(run_dir / "manifest.json", manifest)
    print(f"[campaign] baseline selection={baseline_selection['score']:.3f} "
          f"generalization={baseline_generalization['score']:.3f}",
          flush=True)
    if args.validate_only:
        return

    server = serve(
        run_dir, port=args.port, tasks=["statistical_translation"],
        founders=args.founders, events_per_round=args.events_per_round,
        population_cap=args.population_cap, crossover_rate=.25,
        explore_rate=.25, outcross_rate=.15, seed=args.seed)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[campaign] dashboard http://127.0.0.1:"
          f"{server.server_address[1]}/progress", flush=True)
    for round_number in range(1, args.rounds + 1):
        jobs = server.service.handle("ask", {})
        while jobs:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=args.max_parallel) as executor:
                futures = {executor.submit(
                    _run_job, server, job, args, practice, selection): job
                    for job in jobs}
                for future in concurrent.futures.as_completed(futures):
                    job = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        if any(open_job["job_id"] == job["job_id"]
                               for open_job in server.service.ga.open_jobs()):
                            server.service.handle("abandon", {
                                "job_id": job["job_id"],
                                "reason": repr(exc)})
            jobs = server.service.ga.open_jobs()
        best = server.service.ga.best_record()
        passed, evidence = audit_candidate(
            best["artifact"], best["score"], selection, generalization,
            baseline_generalization, python=args.python,
            bootstrap_seed=secrets_value["bootstrap_seed"] + round_number)
        server.service.handle("audit", {
            "id": best["id"], "passed": passed, "evidence": evidence})
        _write_json(run_dir / f"generalization_round_{round_number}.json", {
            "individual": best["id"], "selection": best["score"],
            "passed": passed, "evidence": evidence})
        print(f"[campaign] round {round_number}: {server.service.ga.summary()}",
              flush=True)


if __name__ == "__main__":
    main()
