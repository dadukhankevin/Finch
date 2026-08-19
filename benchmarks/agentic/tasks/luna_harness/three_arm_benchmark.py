#!/usr/bin/env python3
"""Resumable neutral/staged/Aquilla comparison on the frozen Luna panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from protected_evaluator import (
    LARGE_PANEL_NAME,
    evaluate,
    large_paired_summary,
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


def _cached_evaluation(
    *,
    cache: Path,
    artifact: Path,
    shard: str,
    settings: dict,
    panel: Path,
    workers: int,
) -> dict:
    artifact_hash = _sha256(artifact)
    panel_hash = _sha256(panel)
    if cache.is_file():
        value = json.loads(cache.read_text(encoding="utf-8"))
        expected = {
            "candidate_sha256": artifact_hash,
            "panel_sha256": panel_hash,
            "shard": shard,
        }
        observed = {
            "candidate_sha256": value.get("candidate_sha256"),
            "panel_sha256": value.get("panel_sha256"),
            "shard": value.get("shard"),
        }
        if observed != expected:
            raise RuntimeError(f"evaluation cache identity mismatch: {cache}")
        sessions = value.get("sessions") or []
        # A terminal-launch failure is plumbing, not experimental evidence.
        # Never preserve an all-failed shard as a reusable scientific cache.
        if sessions and int(value.get("failures", 0)) < len(sessions):
            return value
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
    )
    sessions = value.get("sessions") or []
    if sessions and int(value.get("failures", 0)) == len(sessions):
        reasons = sorted({str(row.get("failure") or "unknown") for row in sessions})
        raise RuntimeError(
            f"all native Luna sessions failed for {shard}: " + "; ".join(reasons))
    _write_json(cache, value, private=True)
    return value


def _load_staged_baseline(settings: dict, staged: Path, panel_hash: str) -> list[dict]:
    cache_path = Path(settings["large_baseline_cache"])
    if not cache_path.is_file():
        raise RuntimeError(f"frozen staged baseline is missing: {cache_path}")
    value = json.loads(cache_path.read_text(encoding="utf-8"))
    if value.get("panel_sha256") != panel_hash:
        raise RuntimeError("frozen staged baseline belongs to another panel")
    if value.get("baseline_sha256") != _sha256(staged):
        raise RuntimeError("frozen staged baseline belongs to another artifact")
    shards = value.get("shards") or []
    if len(shards) != 4:
        raise RuntimeError("frozen staged baseline does not contain four shards")
    return shards


def benchmark(args: argparse.Namespace) -> dict:
    run = args.run.resolve()
    settings = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    panel = Path(settings["large_development_panel"])
    panel_hash = _sha256(panel)
    if panel_hash != settings.get("large_development_panel_sha256"):
        raise RuntimeError("frozen development panel identity changed")

    artifacts = {
        "neutral": args.neutral.resolve(),
        "staged": args.staged.resolve(),
        "aquilla": args.aquilla.resolve(),
    }
    for artifact in artifacts.values():
        validate_harness(artifact)
    hashes = {name: _sha256(path) for name, path in artifacts.items()}

    private_root = Path(settings["work_root"]) / (
        f"_protected_aquilla_ethos_ab_{panel_hash[:16]}_"
        f"{hashes['neutral'][:8]}_{hashes['aquilla'][:8]}"
    )
    private_root.mkdir(parents=True, exist_ok=True)
    private_root.chmod(0o700)

    panel_value = json.loads(panel.read_text(encoding="utf-8"))
    results = {
        "neutral": [],
        "staged": _load_staged_baseline(settings, artifacts["staged"], panel_hash),
        "aquilla": [],
    }
    completed: list[dict] = []
    _write_json(args.status.resolve(), {
        "status": "running",
        "model": "gpt-5.6-luna",
        "panel": "large-96-v1",
        "completed": completed,
        "remaining_new_arm_shards": 8,
        "private_cache": str(private_root),
    })
    for index, shard_row in enumerate(panel_value["shards"]):
        shard = shard_row["id"]
        order = ("neutral", "aquilla") if index % 2 == 0 else ("aquilla", "neutral")
        for arm in order:
            cache = private_root / f"{shard}-{arm}.json"
            result = _cached_evaluation(
                cache=cache,
                artifact=artifacts[arm],
                shard=shard,
                settings=settings,
                panel=panel,
                workers=args.workers,
            )
            results[arm].append(result)
            completed.append({"shard": shard, "arm": arm, "score": result["score"]})
            _write_json(args.status.resolve(), {
                "status": "running",
                "model": "gpt-5.6-luna",
                "panel": "large-96-v1",
                "completed": completed,
                "remaining_new_arm_shards": 8 - len(completed),
                "private_cache": str(private_root),
            })

    pairwise = {
        "aquilla_vs_neutral": large_paired_summary(
            results["aquilla"], results["neutral"], panel_sha256=panel_hash),
        "aquilla_vs_staged": large_paired_summary(
            results["aquilla"], results["staged"], panel_sha256=panel_hash),
        "staged_vs_neutral": large_paired_summary(
            results["staged"], results["neutral"], panel_sha256=panel_hash),
    }
    record = {
        "schema_version": 1,
        "kind": "protected_native_luna_three_arm_benchmark",
        "model": "gpt-5.6-luna",
        "panel": "large-96-v1",
        "panel_sha256": panel_hash,
        "translation_items_per_arm": 96,
        "artifacts": {
            name: {"path": str(path), "sha256": hashes[name]}
            for name, path in artifacts.items()
        },
        "staged_arm_reuses_frozen_baseline": True,
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
    experiment = here / "experiments" / "aquilla_ethos"
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument(
        "--neutral", type=Path, default=experiment / "neutral_harness.json")
    parser.add_argument(
        "--staged", type=Path, default=here / "starter_harness.json")
    parser.add_argument(
        "--aquilla", type=Path, default=experiment / "aquilla_ethos_harness.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(args)
    print(json.dumps({
        name: {
            "mean_delta": row["mean_delta"],
            "lower_95": row["paired_bootstrap"]["lower_95"],
            "upper_95": row["paired_bootstrap"]["upper_95"],
        }
        for name, row in result["pairwise"].items()
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
