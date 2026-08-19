#!/usr/bin/env python3
"""Run only the 100-example arm of the best saved Luna harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from protected_evaluator import (
    DEVELOPMENT_LANGUAGES,
    LARGE_PANEL_NAME,
    create_large_development_panel,
    evaluate,
    validate_harness,
)
from three_arm_benchmark import _sha256, _write_json


EXAMPLE_COUNT = 100
BEST_SAVED_SCORE = 43.047572118120264
BEST_SAVED_HASH = "3655a4f1d8b9e3c416fff8c0d97090ca95c0c339d039650d29ae8862827b8aa5"


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _saved_result(run: Path, candidate_hash: str, panel_hash: str) -> dict:
    tournament_file = json.loads(
        (run / "tournaments.json").read_text(encoding="utf-8"))
    tournaments = tournament_file.get("matches") or []
    matches = []
    for row in tournaments:
        result = row.get("result") or {}
        if (result.get("candidate_sha256") == candidate_hash
                and result.get("panel_sha256") == panel_hash
                and result.get("comparisons") == 96):
            matches.append(result)
    if len(matches) != 1:
        raise RuntimeError(f"expected one saved best result, found {len(matches)}")
    result = matches[0]
    if abs(float(result["candidate_score"]) - BEST_SAVED_SCORE) > 1e-12:
        raise RuntimeError("saved best score changed")
    return result


def _cached_shard(*, cache: Path, artifact: Path, shard: str,
                  settings: dict, panel: Path, workers: int) -> dict:
    identity = {
        "candidate_sha256": _sha256(artifact),
        "panel_sha256": _sha256(panel),
        "shard": shard,
        "few_shot_examples_per_task": EXAMPLE_COUNT,
    }
    if cache.is_file():
        value = json.loads(cache.read_text(encoding="utf-8"))
        observed = {key: value.get(key) for key in identity}
        sessions = value.get("sessions") or []
        if (observed == identity and sessions
                and int(value.get("failures", 0)) < len(sessions)):
            return value
        if observed != identity:
            raise RuntimeError(f"evaluation cache identity mismatch: {cache}")
    value = evaluate(
        artifact,
        factory_root=Path(settings["factory_root"]),
        corpus_dir=Path(settings["corpus_dir"]),
        codex=Path(settings["codex_binary"]),
        work_root=Path(settings["work_root"]),
        panel=LARGE_PANEL_NAME,
        workers=workers,
        panel_spec=panel,
        shard_id=shard,
        model="gpt-5.6-luna",
        few_shot_examples=EXAMPLE_COUNT,
    )
    sessions = value.get("sessions") or []
    if sessions and int(value.get("failures", 0)) == len(sessions):
        reasons = sorted({str(row.get("failure") or "unknown") for row in sessions})
        raise RuntimeError(
            f"all native Luna sessions failed for {shard}: " + "; ".join(reasons))
    _write_json(cache, value, private=True)
    return value


def _aggregate(shards: list[dict], saved: dict, panel_hash: str) -> dict:
    values = []
    by_language = {code: [] for code in DEVELOPMENT_LANGUAGES}
    failures = 0
    for shard in shards:
        if shard.get("panel_sha256") != panel_hash:
            raise RuntimeError("large-panel identity changed")
        failures += int(shard.get("failures", 0))
        for session in shard["sessions"]:
            scores = [float(value) for value in session["task_scores"]]
            by_language[session["language"]].extend(scores)
            values.extend(scores)
    language_scores = {
        code: sum(scores) / len(scores) for code, scores in by_language.items()
    }
    saved_languages = {
        row["language"]: float(row["candidate_score"]) for row in saved["pairs"]
    }
    language_deltas = {
        code: language_scores[code] - saved_languages[code]
        for code in DEVELOPMENT_LANGUAGES
    }
    rng = random.Random(int(hashlib.sha256(
        f"{BEST_SAVED_HASH}:{panel_hash}:k{EXAMPLE_COUNT}".encode()
    ).hexdigest()[:16], 16))
    codes = list(DEVELOPMENT_LANGUAGES)
    samples = []
    for _ in range(10000):
        drawn = [codes[rng.randrange(len(codes))] for _ in codes]
        samples.append(sum(language_deltas[code] for code in drawn) / len(drawn))
    score = sum(values) / len(values)
    return {
        "candidate_score": score,
        "saved_top_10_score": float(saved["candidate_score"]),
        "mean_delta": score - float(saved["candidate_score"]),
        "translation_items": len(values),
        "failures": failures,
        "language_scores": language_scores,
        "saved_top_10_language_scores": saved_languages,
        "language_deltas": language_deltas,
        "language_wins": sum(delta > 0 for delta in language_deltas.values()),
        "language_ties": sum(delta == 0 for delta in language_deltas.values()),
        "language_paired_bootstrap": {
            "method": "paired bootstrap over eight saved language means",
            "replicates": len(samples),
            "lower_95": _percentile(samples, 0.025),
            "median": _percentile(samples, 0.5),
            "upper_95": _percentile(samples, 0.975),
        },
    }


def benchmark(args: argparse.Namespace) -> dict:
    run = args.run.resolve()
    settings = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    artifact = args.candidate.resolve()
    candidate = validate_harness(artifact)
    candidate_hash = _sha256(artifact)
    if candidate_hash != BEST_SAVED_HASH:
        raise RuntimeError("candidate is not the saved best dual-derivation harness")

    work_root = Path(settings["work_root"])
    work_root.mkdir(parents=True, exist_ok=True)
    panel = Path(settings["large_development_panel"])
    if not panel.is_file():
        create_large_development_panel(
            factory_root=Path(settings["factory_root"]),
            corpus_dir=Path(settings["corpus_dir"]),
            destination=panel,
        )
    panel_hash = _sha256(panel)
    if panel_hash != settings["large_development_panel_sha256"]:
        raise RuntimeError("recreated frozen panel identity changed")
    saved = _saved_result(run, candidate_hash, panel_hash)

    condition_hash = hashlib.sha256(
        f"{candidate_hash}:{panel_hash}:k{EXAMPLE_COUNT}".encode()).hexdigest()
    private_root = work_root / f"_protected_cardinality_{condition_hash[:20]}"
    private_root.mkdir(parents=True, exist_ok=True)
    private_root.chmod(0o700)
    panel_value = json.loads(panel.read_text(encoding="utf-8"))
    shards = []
    completed = []
    _write_json(args.status.resolve(), {
        "status": "running",
        "candidate": candidate["name"],
        "few_shot_examples_per_task": EXAMPLE_COUNT,
        "completed": completed,
        "remaining_shards": 4,
    })
    for shard_row in panel_value["shards"]:
        shard = shard_row["id"]
        value = _cached_shard(
            cache=private_root / f"{shard}-k{EXAMPLE_COUNT}.json",
            artifact=artifact,
            shard=shard,
            settings=settings,
            panel=panel,
            workers=args.workers,
        )
        shards.append(value)
        completed.append({"shard": shard, "score": value["score"]})
        _write_json(args.status.resolve(), {
            "status": "running",
            "candidate": candidate["name"],
            "few_shot_examples_per_task": EXAMPLE_COUNT,
            "completed": completed,
            "remaining_shards": 4 - len(completed),
        })

    comparison = _aggregate(shards, saved, panel_hash)
    record = {
        "schema_version": 1,
        "kind": "protected_native_luna_example_cardinality_benchmark",
        "model": "gpt-5.6-luna",
        "panel": "large-96-v1",
        "panel_sha256": panel_hash,
        "candidate": {
            "name": candidate["name"],
            "path": str(artifact),
            "sha256": candidate_hash,
        },
        "change_from_saved_run": "10 to 100 preloaded Branching-BM25 examples per task",
        "old_arm_reused": True,
        "comparison": comparison,
        "private_cache": str(private_root),
    }
    _write_json(args.output.resolve(), record)
    _write_json(args.status.resolve(), {
        "status": "complete",
        "candidate": candidate["name"],
        "few_shot_examples_per_task": EXAMPLE_COUNT,
        "completed": completed,
        "result": str(args.output.resolve()),
    })
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = benchmark(parser.parse_args())
    print(json.dumps(result["comparison"], ensure_ascii=False))


if __name__ == "__main__":
    main()
