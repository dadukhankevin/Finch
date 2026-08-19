#!/usr/bin/env python3
"""Quick method-blind agentic audit of the existing large Luna panel.

The translator arms all see the same frozen 96-item development panel.  One
native Sol judge receives every matched item in one assignment, with harness
identity and chrF hidden and the output labels shuffled independently per item.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import statistics
import subprocess

import protected_evaluator as evaluator


AUDIT_NAME = "blind-agentic-large96-v1"
ARM_ORDER = ("neutral", "dual", "aquilla", "hybrid", "dual_100")
GENERATED_ARMS = ("neutral", "dual", "aquilla", "hybrid")
LABELS = tuple("ABCDE")
EXPECTED_HASHES = {
    "neutral": "bcdb089ad0a8b74a00bfbe497dba1f6d4222856dc4ef4ba3401ff52302bccdd4",
    "dual": "3655a4f1d8b9e3c416fff8c0d97090ca95c0c339d039650d29ae8862827b8aa5",
    "aquilla": "87ff046063c0d979435dcbf9a6d8cb805983bf65b1cce26c4ca32fe99818f772",
    "hybrid": "38e198b5428dad638e987902466db2051213607c61fb8e8dd193a14a482b09a1",
    "dual_100": "3655a4f1d8b9e3c416fff8c0d97090ca95c0c339d039650d29ae8862827b8aa5",
}
ERROR_CATEGORIES = {
    "omission", "addition", "mistranslation", "polarity",
    "participant_role", "number_quantity", "tense_aspect", "named_entity",
    "grammar_word_order", "malformed",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value, *, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600 if private else 0o644)
    temporary.replace(path)


def status(run: Path, **values) -> None:
    write_json(
        run / "media" / "blind_agentic_audit_status.json",
        {"audit": AUDIT_NAME, **values},
        private=False,
    )


def valid_translation_result(value: dict, *, artifact_hash: str,
                             panel_hash: str, shard: str,
                             few_shot_examples: int) -> bool:
    sessions = value.get("sessions") or []
    return (
        value.get("candidate_sha256") == artifact_hash
        and value.get("panel_sha256") == panel_hash
        and value.get("shard") == shard
        and value.get("few_shot_examples_per_task") == few_shot_examples
        and len(sessions) == len(evaluator.DEVELOPMENT_LANGUAGES)
        and int(value.get("failures", 0)) == 0
        and all(not row.get("failure") and not row.get("policy_violations")
                for row in sessions)
        and all(len(row.get("translations") or []) == evaluator.TASKS_PER_SHARD
                for row in sessions)
        and all(
            all(str(translation.get("translated_text") or "").strip()
                for translation in row.get("translations") or [])
            for row in sessions
        )
    )


def evaluate_arm(*, arm: str, artifact: Path, settings: dict,
                 panel_path: Path, panel: dict, panel_hash: str,
                 private_root: Path, workers: int) -> dict[str, dict]:
    artifact_hash = sha256(artifact)
    results = {}
    for shard_row in panel["shards"]:
        shard = shard_row["id"]
        cache = private_root / "translations" / f"{shard}-{arm}.json"
        if cache.is_file():
            value = read_json(cache)
        else:
            value = evaluator.evaluate(
                artifact,
                factory_root=Path(settings["factory_root"]),
                corpus_dir=Path(settings["corpus_dir"]),
                codex=Path(settings["codex_binary"]),
                work_root=Path(settings["work_root"]),
                panel=evaluator.LARGE_PANEL_NAME,
                workers=workers,
                panel_spec=panel_path,
                shard_id=shard,
                model="gpt-5.6-luna",
                few_shot_examples=10,
                retain_translations=True,
            )
            write_json(cache, value, private=True)
        if not valid_translation_result(
            value,
            artifact_hash=artifact_hash,
            panel_hash=panel_hash,
            shard=shard,
            few_shot_examples=10,
        ):
            raise RuntimeError(f"incomplete or mismatched result: {arm}/{shard}")
        results[shard] = value
    return results


def _last_agent_message(path: Path) -> dict:
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        item = event.get("item") or {}
        if event.get("type") == "item.completed" and item.get("type") == "agent_message":
            last = item.get("text")
    if not last:
        raise RuntimeError(f"missing final translation message: {path}")
    value = json.loads(last)
    if not isinstance(value.get("translations"), list):
        raise RuntimeError(f"invalid final translation message: {path}")
    return value


def load_existing_dual_100(*, settings: dict, panel: dict,
                           panel_hash: str) -> dict[str, dict]:
    """Recover the completed k=100 run from its retained native traces."""
    work_root = Path(settings["work_root"])
    found = {}
    for result_path in work_root.glob("finch-luna-development-large-*/result.json"):
        result = read_json(result_path)
        if (
            result.get("candidate_sha256") == EXPECTED_HASHES["dual_100"]
            and result.get("panel_sha256") == panel_hash
            and result.get("few_shot_examples_per_task") == 100
            and result.get("shard") in {row["id"] for row in panel["shards"]}
        ):
            found[result["shard"]] = (result_path.parent, result)
    expected_shards = {row["id"] for row in panel["shards"]}
    if set(found) != expected_shards:
        raise RuntimeError("the completed dual-100 trace set is incomplete")

    recovered = {}
    for shard, (record_root, summary) in found.items():
        task_cache = work_root / (
            f"_protected_{panel_hash[:16]}_{shard}_k100_tasks.json")
        tasks = read_json(task_cache)
        summary_sessions = {row["language"]: row for row in summary["sessions"]}
        sessions = []
        for language in evaluator.DEVELOPMENT_LANGUAGES:
            trace = record_root / "records" / "traces" / f"agent_luna-{language}.jsonl"
            generated = _last_agent_message(trace)["translations"]
            generated_by_line = {int(row["line_number"]): row for row in generated}
            task_rows = tasks[language]
            score_row = summary_sessions[language]
            translations = []
            for task, task_score in zip(task_rows, score_row["task_scores"]):
                generated_row = generated_by_line[int(task["line_number"])]
                translations.append({
                    "line_number": int(task["line_number"]),
                    "verse_reference": task["verse_reference"],
                    "source_text": task["source_text"],
                    "reference_text": task["reference_text"],
                    "translated_text": generated_row["translation"],
                    "metrics": {"chrf": float(task_score)},
                })
            sessions.append({**score_row, "translations": translations})
        recovered[shard] = {**summary, "sessions": sessions}
    return recovered


def _arm_session(arm_results: dict, arm: str, shard: str,
                 language: str) -> dict:
    return next(
        row for row in arm_results[arm][shard]["sessions"]
        if row["language"] == language
    )


def build_blind_assignment(*, panel: dict, panel_hash: str,
                           arm_results: dict) -> tuple[dict, dict]:
    language_names = panel.get("language_names") or {
        code: code for code in evaluator.DEVELOPMENT_LANGUAGES
    }
    items = []
    mapping = {}
    for shard_row in panel["shards"]:
        shard = shard_row["id"]
        for language in evaluator.DEVELOPMENT_LANGUAGES:
            translations = {}
            for arm in ARM_ORDER:
                session = _arm_session(arm_results, arm, shard, language)
                translations[arm] = {
                    int(row["line_number"]): row
                    for row in session["translations"]
                }
            line_sets = {tuple(sorted(rows)) for rows in translations.values()}
            if len(line_sets) != 1:
                raise RuntimeError(f"arms contain different tasks: {shard}/{language}")
            for line_number in sorted(translations[ARM_ORDER[0]]):
                rows = {arm: translations[arm][line_number] for arm in ARM_ORDER}
                first = rows[ARM_ORDER[0]]
                item_id = hashlib.sha256(
                    f"{panel_hash}|{language}|{line_number}".encode()
                ).hexdigest()[:20]
                shuffled = list(ARM_ORDER)
                seed = int.from_bytes(hashlib.sha256(
                    f"{AUDIT_NAME}|{item_id}".encode()).digest()[:8], "big")
                random.Random(seed).shuffle(shuffled)
                labels = dict(zip(LABELS, shuffled))
                items.append({
                    "item_id": item_id,
                    "language": language_names.get(language, language),
                    "source": first["source_text"],
                    "reference": first["reference_text"],
                    "outputs": {
                        label: rows[arm]["translated_text"]
                        for label, arm in labels.items()
                    },
                })
                mapping[item_id] = {
                    "language": language,
                    "line_number": line_number,
                    "verse_reference": first.get("verse_reference"),
                    "labels": labels,
                }
    assignment = {
        "schema_version": 1,
        "assignment_id": hashlib.sha256(
            f"{AUDIT_NAME}|{panel_hash}".encode()).hexdigest()[:20],
        "items": items,
        "instructions": {
            "primary": "meaning preservation relative to the English source",
            "reference": "target-language evidence, not a mandatory wording template",
            "secondary": "natural and grammatical target-language expression",
        },
    }
    if len(items) != 96:
        raise RuntimeError(f"blind audit expected 96 items, found {len(items)}")
    visible = json.dumps(assignment, ensure_ascii=False).casefold()
    for forbidden in (
        "neutral", "dual", "aquilla", "hybrid", "chrF", "candidate_sha256",
        "harness", "artifact", "prompt", "trace",
    ):
        if forbidden.casefold() in visible:
            raise RuntimeError(f"blind assignment leaked forbidden marker: {forbidden}")
    return assignment, mapping


def judge_prompt(assignment: dict) -> str:
    return """You are the sole independent evaluator of a blinded machine-translation comparison.

All systems translated exactly the same 96 items. Their identities are hidden,
and labels A-E are reshuffled independently for every item. Do not infer system
identity or reward a recurring label. Judge only the supplied English source,
target-language reference, and five candidate outputs.

Faithfulness to the English source is primary: check propositions, participants
and roles, negation, quantities, time/aspect, names, quotations, and discourse
relations. Treat the reference as strong evidence about target vocabulary,
grammar, orthography, and naturalness, but not as an infallible wording template.
A candidate may legitimately preserve an explicit source distinction that the
reference paraphrases. Penalize unsupported additions, omissions, role swaps,
and malformed output. If the evidence cannot separate candidates, tie them.
Do not use tools, files, web search, outside data, lexical-overlap formulas, or
any metric. Evaluate all 96 items before drawing an overall conclusion.

Return exactly one JSON object and no markdown:
{"assignment_id":"...","verdicts":[
 {"item_id":"...","ranking":[["A"],["B","C"],["D"],["E"]],
  "ratings":{"A":{"adequacy":1,"naturalness":1,"major_errors":0,"minor_errors":0},
             "B":{"adequacy":1,"naturalness":1,"major_errors":0,"minor_errors":0},
             "C":{"adequacy":1,"naturalness":1,"major_errors":0,"minor_errors":0},
             "D":{"adequacy":1,"naturalness":1,"major_errors":0,"minor_errors":0},
             "E":{"adequacy":1,"naturalness":1,"major_errors":0,"minor_errors":0}},
  "key_error":"short concrete description or none","confidence":0.0}
],"overall_observations":["method-blind pattern visible across items"]}

Every label must appear exactly once across the ordered ranking tiers. Ratings
are integers; adequacy and naturalness are 1-5; error counts are nonnegative.
Keep key_error concise. Here is the blinded assignment:
""" + json.dumps(assignment, ensure_ascii=False)


def validate_judgment(value: dict, assignment: dict) -> None:
    if set(value) != {"assignment_id", "verdicts", "overall_observations"}:
        raise ValueError("judge response has the wrong top-level shape")
    if value["assignment_id"] != assignment["assignment_id"]:
        raise ValueError("judge response has the wrong assignment id")
    expected = {item["item_id"] for item in assignment["items"]}
    verdicts = value["verdicts"]
    if len(verdicts) != len(expected) or {row.get("item_id") for row in verdicts} != expected:
        raise ValueError("judge did not return exactly one verdict per item")
    for row in verdicts:
        if set(row) != {"item_id", "ranking", "ratings", "key_error", "confidence"}:
            raise ValueError("invalid verdict shape")
        tiers = row["ranking"]
        flattened = [label for tier in tiers for label in tier]
        if (not tiers or any(not isinstance(tier, list) or not tier for tier in tiers)
                or len(flattened) != len(LABELS) or set(flattened) != set(LABELS)):
            raise ValueError("ranking must contain every label exactly once")
        if set(row["ratings"]) != set(LABELS):
            raise ValueError("ratings changed the output labels")
        for rating in row["ratings"].values():
            if set(rating) != {"adequacy", "naturalness", "major_errors", "minor_errors"}:
                raise ValueError("invalid rating shape")
            if any(not isinstance(rating[key], int) or not 1 <= rating[key] <= 5
                   for key in ("adequacy", "naturalness")):
                raise ValueError("adequacy/naturalness must be integers 1-5")
            if any(not isinstance(rating[key], int) or rating[key] < 0
                   for key in ("major_errors", "minor_errors")):
                raise ValueError("error counts must be nonnegative integers")
        if not isinstance(row["key_error"], str):
            raise ValueError("key_error must be text")
        if (not isinstance(row["confidence"], (int, float))
                or isinstance(row["confidence"], bool)
                or not 0 <= float(row["confidence"]) <= 1):
            raise ValueError("confidence must be between zero and one")


def run_one_judge(*, assignment: dict, codex: Path, cwd: Path,
                  cache: Path, timeout: float) -> dict:
    assignment_hash = hashlib.sha256(
        json.dumps(assignment, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()
    if cache.is_file():
        cached = read_json(cache)
        if cached.get("assignment_sha256") != assignment_hash:
            raise RuntimeError("judge cache belongs to another assignment")
        validate_judgment(cached["judgment"], assignment)
        return cached["judgment"]
    command = [
        str(codex), "exec", "-m", "gpt-5.6-sol",
        "-c", 'model_reasoning_effort="low"',
        "-c", 'web_search="disabled"',
        "--disable", "multi_agent", "--sandbox", "read-only",
        "--ephemeral", "--skip-git-repo-check", "-C", str(cwd), "-",
    ]
    completed = subprocess.run(
        command,
        input=judge_prompt(assignment),
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    if completed.returncode:
        raise RuntimeError(
            f"native Sol judge failed ({completed.returncode}): {completed.stderr[-1500:]}")
    judgment = json.loads(completed.stdout)
    validate_judgment(judgment, assignment)
    write_json(cache, {
        "assignment_sha256": assignment_hash,
        "assignment": assignment,
        "judgment": judgment,
    }, private=True)
    return judgment


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def hierarchical_interval(matrix: list[list[float]], *, seed: int,
                          replicates: int = 4000) -> dict:
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        languages = [rng.randrange(len(matrix)) for _ in matrix]
        verses = [rng.randrange(len(matrix[0])) for _ in matrix[0]]
        values = [matrix[language][verse]
                  for language in languages for verse in verses]
        samples.append(statistics.fmean(values))
    return {
        "lower_95": _percentile(samples, .025),
        "median": _percentile(samples, .5),
        "upper_95": _percentile(samples, .975),
        "replicates": replicates,
    }


def aggregate(*, judgment: dict, mapping: dict, arm_results: dict,
              panel: dict) -> dict:
    observations = {arm: [] for arm in ARM_ORDER}
    ranks = {}
    for verdict in judgment["verdicts"]:
        private = mapping[verdict["item_id"]]
        tier_by_label = {
            label: tier_index
            for tier_index, tier in enumerate(verdict["ranking"])
            for label in tier
        }
        for label, arm in private["labels"].items():
            rating = verdict["ratings"][label]
            observations[arm].append({
                "item_id": verdict["item_id"],
                "language": private["language"],
                "line_number": private["line_number"],
                "tier": tier_by_label[label],
                **rating,
            })
        ranks[verdict["item_id"]] = {
            private["labels"][label]: tier_by_label[label]
            for label in LABELS
        }

    summaries = {}
    for arm, rows in observations.items():
        summaries[arm] = {
            "mean_rank_tier": statistics.fmean(row["tier"] for row in rows),
            "first_place_or_tied": sum(row["tier"] == 0 for row in rows),
            "mean_adequacy": statistics.fmean(row["adequacy"] for row in rows),
            "mean_naturalness": statistics.fmean(row["naturalness"] for row in rows),
            "mean_major_errors": statistics.fmean(row["major_errors"] for row in rows),
            "mean_minor_errors": statistics.fmean(row["minor_errors"] for row in rows),
        }

    languages = list(evaluator.DEVELOPMENT_LANGUAGES)
    lines = [index + 1 for shard in panel["shards"] for index in shard["target_indices"]]
    pairwise = {}
    for left_index, left in enumerate(ARM_ORDER):
        for right in ARM_ORDER[left_index + 1:]:
            cells = {}
            for item_id, arm_ranks in ranks.items():
                private = mapping[item_id]
                left_rank = arm_ranks[left]
                right_rank = arm_ranks[right]
                cells[(private["language"], private["line_number"])] = (
                    1 if left_rank < right_rank else -1 if left_rank > right_rank else 0)
            matrix = [[cells[(language, line)] for line in lines]
                      for language in languages]
            flat = [value for row in matrix for value in row]
            name = f"{left}_vs_{right}"
            interval = hierarchical_interval(
                matrix,
                seed=int(hashlib.sha256(name.encode()).hexdigest()[:16], 16),
            )
            pairwise[name] = {
                "focus": left,
                "other": right,
                "wins": sum(value > 0 for value in flat),
                "ties": sum(value == 0 for value in flat),
                "losses": sum(value < 0 for value in flat),
                "preference_margin": statistics.fmean(flat),
                "hierarchical_bootstrap": interval,
                "conclusion": (
                    "better" if interval["lower_95"] > 0
                    else "worse" if interval["upper_95"] < 0
                    else "inconclusive"
                ),
            }

    secondary_chrf = {}
    for arm in ARM_ORDER:
        scores = []
        for shard in panel["shards"]:
            for session in arm_results[arm][shard["id"]]["sessions"]:
                scores.extend(float(value) for value in session["task_scores"])
        secondary_chrf[arm] = statistics.fmean(scores)

    return {
        "schema_version": 1,
        "kind": "single_agent_method_blind_translation_audit",
        "audit": AUDIT_NAME,
        "panel": {
            "name": panel["name"],
            "items": 96,
            "languages": languages,
            "shared_items_per_language": 12,
            "confirmation_languages": "untouched",
        },
        "translator": "native Codex gpt-5.6-luna",
        "judge": {
            "model": "native Codex gpt-5.6-sol",
            "agents": 1,
            "method_blind": True,
            "metric_blind": True,
            "per_item_label_randomization": True,
        },
        "arms": {
            "neutral": {"description": "plain agent plus 10 supplied examples", "sha256": EXPECTED_HASHES["neutral"]},
            "dual": {"description": "best evolved dual-derivation harness, 10 examples", "sha256": EXPECTED_HASHES["dual"]},
            "aquilla": {"description": "Aquilla analyzer-performer-verifier harness, 10 examples", "sha256": EXPECTED_HASHES["aquilla"]},
            "hybrid": {"description": "Aquilla plus context-gated dual derivation, 10 examples", "sha256": EXPECTED_HASHES["hybrid"]},
            "dual_100": {"description": "same evolved dual harness with 100 supplied examples", "sha256": EXPECTED_HASHES["dual_100"]},
        },
        "agentic_summary": summaries,
        "agentic_pairwise": pairwise,
        "judge_record_validation": {
            "assignment_items": len(mapping),
            "structured_verdicts": len(judgment["verdicts"]),
            "unique_verdict_ids": len({row["item_id"] for row in judgment["verdicts"]}),
            "note": (
                "The judge's free-text observation says 72 items, but the "
                "machine-validated assignment and verdict arrays contain all 96."
            ),
        },
        "judge_overall_observations": judgment["overall_observations"],
        "secondary_chrf": secondary_chrf,
    }


def benchmark(args: argparse.Namespace) -> dict:
    run = args.run.resolve()
    settings = read_json(run / "manifest.json")
    if settings.get("confirmation_status") != "UNTOUCHED":
        raise RuntimeError("confirmation status is not UNTOUCHED")
    panel_path = Path(settings["large_development_panel"])
    panel_hash = sha256(panel_path)
    if panel_hash != settings["large_development_panel_sha256"]:
        raise RuntimeError("frozen large panel identity changed")
    panel = evaluator._load_large_panel(panel_path)

    artifacts = {
        "neutral": args.neutral.resolve(),
        "dual": args.dual.resolve(),
        "aquilla": args.aquilla.resolve(),
        "hybrid": args.hybrid.resolve(),
    }
    for arm, artifact in artifacts.items():
        evaluator.validate_harness(artifact)
        if sha256(artifact) != EXPECTED_HASHES[arm]:
            raise RuntimeError(f"frozen {arm} harness identity changed")

    private_root = args.private_root.resolve()
    private_root.mkdir(parents=True, exist_ok=True)
    private_root.chmod(0o700)
    write_json(private_root / "manifest.json", {
        "audit": AUDIT_NAME,
        "panel_sha256": panel_hash,
        "artifacts": {arm: {"path": str(path), "sha256": sha256(path)}
                      for arm, path in artifacts.items()},
        "dual_100": "reused completed raw native traces; no translation rerun",
    }, private=True)

    status(run, status="translating", completed_arms=0,
           total_generated_arms=len(GENERATED_ARMS),
           note="four harness arms run concurrently; completed dual-100 traces are reused")
    arm_results = {
        "dual_100": load_existing_dual_100(
            settings=settings, panel=panel, panel_hash=panel_hash)
    }
    # Each evaluator temporarily patches an upstream project-builder hook.
    # Isolate arms in spawned processes so concurrent harnesses cannot race on
    # that hook and silently build one another's workspaces.
    with ProcessPoolExecutor(
        max_workers=len(GENERATED_ARMS),
        mp_context=multiprocessing.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(
                evaluate_arm,
                arm=arm,
                artifact=artifacts[arm],
                settings=settings,
                panel_path=panel_path,
                panel=panel,
                panel_hash=panel_hash,
                private_root=private_root,
                workers=args.translation_workers,
            ): arm
            for arm in GENERATED_ARMS
        }
        for completed, future in enumerate(as_completed(futures), 1):
            arm = futures[future]
            arm_results[arm] = future.result()
            status(run, status="translating", completed_arms=completed,
                   total_generated_arms=len(GENERATED_ARMS), last_completed=arm)

    assignment, mapping = build_blind_assignment(
        panel=panel, panel_hash=panel_hash, arm_results=arm_results)
    write_json(private_root / "blind_assignment.json", assignment, private=True)
    write_json(private_root / "private_mapping.json", mapping, private=True)
    status(run, status="judging", judge_agents=1, items=96,
           note="one Sol agent sees all randomized A-E outputs and no method or chrF metadata")
    judge_cwd = private_root / "empty-judge-workspace"
    judge_cwd.mkdir(exist_ok=True)
    judgment = run_one_judge(
        assignment=assignment,
        codex=Path(settings["codex_binary"]),
        cwd=judge_cwd,
        cache=private_root / "judge.json",
        timeout=args.judge_timeout,
    )
    result = aggregate(
        judgment=judgment,
        mapping=mapping,
        arm_results=arm_results,
        panel=panel,
    )
    result["panel"]["sha256"] = panel_hash
    result["private_record_root"] = str(private_root)
    output = args.output.resolve()
    write_json(output, result, private=False)
    ordered = sorted(
        result["agentic_summary"],
        key=lambda arm: result["agentic_summary"][arm]["mean_rank_tier"],
    )
    status(run, status="complete", result=str(output), agentic_order=ordered)
    return result


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    experiment = here / "experiments" / "aquilla_ethos"
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--neutral", type=Path, default=experiment / "neutral_harness.json")
    parser.add_argument("--dual", type=Path, required=True)
    parser.add_argument("--aquilla", type=Path, default=experiment / "aquilla_ethos_harness.json")
    parser.add_argument("--hybrid", type=Path, default=experiment / "context_gated_dual_harness.json")
    parser.add_argument("--translation-workers", type=int, default=2)
    parser.add_argument("--judge-timeout", type=float, default=1800)
    parser.add_argument(
        "--private-root", type=Path,
        default=Path("/private/tmp/finch-luna-blind-agentic-large96-v1"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/luna-harness-gar-v3-elo/controls/blind-agentic-large96-v1.json"),
    )
    args = parser.parse_args(argv)
    result = benchmark(args)
    print(json.dumps({
        "agentic_summary": result["agentic_summary"],
        "agentic_pairwise": result["agentic_pairwise"],
        "secondary_chrf": result["secondary_chrf"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
