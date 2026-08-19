#!/usr/bin/env python3
"""Evaluate one harness on the frozen 96-item panel and known comparison arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from protected_evaluator import LARGE_PANEL_NAME, large_paired_summary, validate_harness
from three_arm_benchmark import _cached_evaluation, _load_staged_baseline, _sha256, _write_json


def _load_comparison_shards(result_path: Path, arm: str) -> list[dict]:
    record = json.loads(result_path.read_text(encoding="utf-8"))
    cache = Path(record["private_cache"])
    shards = []
    for index in range(1, 5):
        path = cache / f"large-{index}-{arm}.json"
        if not path.is_file():
            raise RuntimeError(f"missing {arm} comparison shard: {path}")
        shards.append(json.loads(path.read_text(encoding="utf-8")))
    return shards


def benchmark(args: argparse.Namespace) -> dict:
    run = args.run.resolve()
    settings = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    panel = Path(settings["large_development_panel"])
    panel_hash = _sha256(panel)
    if panel_hash != settings.get("large_development_panel_sha256"):
        raise RuntimeError("frozen development panel identity changed")

    candidate = args.candidate.resolve()
    staged = args.staged.resolve()
    validate_harness(candidate)
    validate_harness(staged)
    candidate_hash = _sha256(candidate)

    private_root = Path(settings["work_root"]) / (
        f"_protected_candidate_{panel_hash[:16]}_{candidate_hash[:16]}")
    private_root.mkdir(parents=True, exist_ok=True)
    private_root.chmod(0o700)

    baseline = {
        "staged": _load_staged_baseline(settings, staged, panel_hash),
        "neutral": _load_comparison_shards(args.comparison_result.resolve(), "neutral"),
        "aquilla": _load_comparison_shards(args.comparison_result.resolve(), "aquilla"),
    }
    panel_value = json.loads(panel.read_text(encoding="utf-8"))
    candidate_shards = []
    completed = []
    _write_json(args.status.resolve(), {
        "status": "running",
        "model": "gpt-5.6-luna",
        "panel": "large-96-v1",
        "completed": completed,
        "remaining_candidate_shards": 4,
        "private_cache": str(private_root),
    })
    for shard_row in panel_value["shards"]:
        shard = shard_row["id"]
        result = _cached_evaluation(
            cache=private_root / f"{shard}-candidate.json",
            artifact=candidate,
            shard=shard,
            settings=settings,
            panel=panel,
            workers=args.workers,
        )
        candidate_shards.append(result)
        completed.append({"shard": shard, "score": result["score"]})
        _write_json(args.status.resolve(), {
            "status": "running",
            "model": "gpt-5.6-luna",
            "panel": "large-96-v1",
            "completed": completed,
            "remaining_candidate_shards": 4 - len(completed),
            "private_cache": str(private_root),
        })

    pairwise = {
        f"candidate_vs_{name}": large_paired_summary(
            candidate_shards, shards, panel_sha256=panel_hash)
        for name, shards in baseline.items()
    }
    record = {
        "schema_version": 1,
        "kind": "protected_native_luna_frozen_candidate_benchmark",
        "model": "gpt-5.6-luna",
        "panel": "large-96-v1",
        "panel_sha256": panel_hash,
        "translation_items": 96,
        "candidate": {"path": str(candidate), "sha256": candidate_hash},
        "staged_arm_reuses_frozen_baseline": True,
        "neutral_and_aquilla_reuse_completed_frozen_panel_runs": True,
        "pairwise": pairwise,
        "private_cache": str(private_root),
    }
    _write_json(args.output.resolve(), record)
    _write_json(args.status.resolve(), {
        "status": "complete",
        "model": "gpt-5.6-luna",
        "panel": "large-96-v1",
        "completed": completed,
        "result": str(args.output.resolve()),
    })
    return record


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--staged", type=Path, default=here / "starter_harness.json")
    parser.add_argument("--comparison-result", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    result = benchmark(parser.parse_args())
    print(json.dumps({
        name: {
            "candidate_score": row["candidate_score"],
            "baseline_score": row["baseline_score"],
            "mean_delta": row["mean_delta"],
            "lower_95": row["paired_bootstrap"]["lower_95"],
            "upper_95": row["paired_bootstrap"]["upper_95"],
        }
        for name, row in result["pairwise"].items()
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
