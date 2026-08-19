"""Canonical fitness for translation anomaly detection.

The artifact must define:

    def flag(source, candidate, examples) -> bool
        source: English
        candidate: a string in some target language (human gold or a model draft)
        examples: list of {source, gold} memory pairs from the same prompt

Score is flag_rate(drafts) - flag_rate(gold). Higher is better.

Default CLI is a smoke check. The driver evaluates sealed practice
(keep) and holdout (observation only) in its own process.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

CELLS = Path(__file__).with_name("cells.json")
LOCKED = frozenset({"--practice", "--sealed", "--holdout"})

# One public example of the call shape. Invented text, not an evaluation language.
SMOKE = [
    {
        "kind": "draft",
        "source": "The child sat down.",
        "candidate": "xxx child sit down sit.",
        "examples": [
            {"source": "The dog ran.", "gold": "dog-run."},
            {"source": "The woman stood.", "gold": "woman-stand."},
        ],
    },
    {
        "kind": "gold",
        "source": "The child sat down.",
        "candidate": "child-sit.",
        "examples": [
            {"source": "The dog ran.", "gold": "dog-run."},
            {"source": "The woman stood.", "gold": "woman-stand."},
        ],
    },
]


def load_cells(split: str) -> list[dict]:
    if split == "smoke":
        return list(SMOKE)
    if os.environ.get("FINCH_SEALED_OK") != "1":
        raise PermissionError("real evaluation cells are allocator-only")
    payload = json.loads(CELLS.read_text(encoding="utf-8"))
    want = "holdout" if split == "holdout" else "practice"
    return [cell for cell in payload["cells"] if cell["split"] == want]


def decide(flag, cell) -> bool:
    return bool(flag(cell["source"], cell["candidate"], cell["examples"]))


def evaluate(flag, split: str) -> dict:
    drafts, golds, errors = [], [], []
    for index, cell in enumerate(load_cells(split)):
        try:
            flagged = decide(flag, cell)
        except Exception as exc:
            flagged = False
            errors.append(f"cell {index}: {exc!r}")
        (drafts if cell["kind"] == "draft" else golds).append(flagged)
    should = sum(drafts) / len(drafts) if drafts else 0.0
    should_not = sum(golds) / len(golds) if golds else 0.0
    return {
        "task": "translation_anomaly",
        "split": split,
        "score": should - should_not,
        "should_flag": should,
        "should_not_flag": should_not,
        "n_draft": len(drafts),
        "n_gold": len(golds),
        "errors": errors,
    }


def load_flag(artifact: str):
    namespace: dict = {}
    exec(compile(Path(artifact).read_text(encoding="utf-8"), artifact, "exec"), namespace)
    return namespace["flag"]


def main() -> None:
    artifact = sys.argv[1]
    flags = set(sys.argv[2:])
    if flags & LOCKED and os.environ.get("FINCH_SEALED_OK") != "1":
        print(json.dumps({
            "task": "translation_anomaly",
            "split": "denied",
            "score": 0.0,
            "errors": ["real splits are allocator-only; run with no flags for the smoke example"],
        }))
        return
    if "--holdout" in flags:
        split = "holdout"
    elif flags & {"--practice", "--sealed"}:
        split = "practice"
    else:
        split = "smoke"
    try:
        flag = load_flag(artifact)
    except Exception as exc:
        print(json.dumps({
            "task": "translation_anomaly",
            "split": split,
            "score": 0.0,
            "errors": [f"load: {exc!r}"],
        }))
        return
    print(json.dumps(evaluate(flag, split)))


if __name__ == "__main__":
    main()
