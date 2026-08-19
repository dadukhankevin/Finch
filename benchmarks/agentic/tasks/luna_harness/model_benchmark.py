#!/usr/bin/env python3
"""Run a model-only counterfactual without admitting it as Finch fitness.

The panel, harness artifacts, data, prompts, two-turn topology, and scoring stay
fixed.  Only the downstream native Codex model changes.  Results are cached per
model/artifact/shard so an interrupted side study can resume without repeating
completed sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from protected_evaluator import (
    LARGE_PANEL_NAME, SUPPORTED_AGENT_MODELS, evaluate,
    large_paired_summary, validate_harness,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    if private:
        temporary.chmod(0o600)
    temporary.replace(path)


def _cached_evaluation(*, cache: Path, artifact: Path, shard: str,
                       settings: dict, panel: Path, model: str,
                       workers: int) -> dict:
    artifact_hash = _sha256(artifact)
    panel_hash = _sha256(panel)
    if cache.is_file():
        value = json.loads(cache.read_text(encoding="utf-8"))
        if (value.get("candidate_sha256") != artifact_hash
                or value.get("panel_sha256") != panel_hash
                or value.get("shard") != shard
                or value.get("model") != model):
            raise RuntimeError(f"model-benchmark cache identity mismatch: {cache}")
        return value
    value = evaluate(
        artifact,
        factory_root=Path(settings["factory_root"]),
        corpus_dir=Path(settings["corpus_dir"]),
        codex=Path(settings["codex_binary"]),
        work_root=Path(settings["work_root"]),
        panel=LARGE_PANEL_NAME, workers=workers,
        panel_spec=panel, shard_id=shard, model=model)
    _write_json(cache, value, private=True)
    return value


def _luna_reference(run: Path, candidate_hash: str):
    state = json.loads((run / "state.json").read_text(encoding="utf-8"))
    matches = []
    for lineage in state.get("lineages", {}).values():
        for report in lineage.get("reports", []):
            if (report.get("fitness_regime") == "large-96-v1"
                    and report.get("artifact_sha256") == candidate_hash
                    and not report.get("voided")):
                matches.append({
                    "lineage": lineage["id"], "report": report["index"],
                    "paired_chrf_delta": report["score"],
                })
    return matches[-1] if matches else None


def benchmark(args) -> dict:
    run = args.run.resolve()
    settings = json.loads(
        (run / "manifest.json").read_text(encoding="utf-8"))
    panel = Path(settings["large_development_panel"])
    if _sha256(panel) != settings["large_development_panel_sha256"]:
        raise RuntimeError("large development panel identity changed")
    panel_value = json.loads(panel.read_text(encoding="utf-8"))
    baseline = run / "starter_harness.json"
    candidate = args.candidate.resolve()
    validate_harness(baseline)
    validate_harness(candidate)
    baseline_hash = _sha256(baseline)
    candidate_hash = _sha256(candidate)
    model_key = args.model.removeprefix("gpt-5.6-")
    private_root = Path(settings["work_root"]) / (
        f"_protected_{model_key}_model_benchmark_"
        f"{settings['large_development_panel_sha256'][:16]}_"
        f"{candidate_hash[:16]}")
    private_root.mkdir(parents=True, exist_ok=True)
    private_root.chmod(0o700)

    results = {"baseline": [], "candidate": []}
    execution_order = []
    for index, shard_row in enumerate(panel_value["shards"]):
        shard = shard_row["id"]
        order = (("baseline", baseline), ("candidate", candidate))
        if index % 2:
            order = tuple(reversed(order))
        execution_order.append({"shard": shard,
                                "order": [role for role, _ in order]})
        for role, artifact in order:
            cache = private_root / f"{shard}-{role}.json"
            results[role].append(_cached_evaluation(
                cache=cache, artifact=artifact, shard=shard,
                settings=settings, panel=panel, model=args.model,
                workers=args.workers))

    paired = large_paired_summary(
        results["candidate"], results["baseline"],
        panel_sha256=settings["large_development_panel_sha256"])
    luna = _luna_reference(run, candidate_hash)
    record = {
        "schema_version": 1,
        "kind": "protected_native_model_counterfactual",
        "admitted_as_finch_fitness": False,
        "model": args.model,
        "panel": LARGE_PANEL_NAME,
        "panel_sha256": settings["large_development_panel_sha256"],
        "translation_items_per_artifact": paired["comparisons"],
        "baseline_sha256": baseline_hash,
        "candidate_sha256": candidate_hash,
        "execution_order": execution_order,
        "paired": paired,
        "luna_reference": luna,
        "terra_minus_luna_effect": (
            None if luna is None else
            paired["mean_delta"] - float(luna["paired_chrf_delta"])),
        "private_cache": str(private_root),
    }
    _write_json(args.output.resolve(), record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--model", choices=SUPPORTED_AGENT_MODELS,
                        default="gpt-5.6-terra")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = benchmark(args)
    paired = result["paired"]
    print(json.dumps({
        "model": result["model"],
        "candidate_score": paired["candidate_score"],
        "baseline_score": paired["baseline_score"],
        "mean_delta": paired["mean_delta"],
        "lower_95": paired["paired_bootstrap"]["lower_95"],
        "upper_95": paired["paired_bootstrap"]["upper_95"],
        "output": str(args.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
