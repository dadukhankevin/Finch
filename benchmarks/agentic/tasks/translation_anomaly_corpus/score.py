"""Canonical fitness for translation anomaly detection with a corpus.

The artifact must define:

    def flag(source, candidate, examples, corpus) -> bool
        source: English
        candidate: a string in some target language (human gold or a model draft)
        examples: list of {source, gold} memory pairs from the same prompt
        corpus: list of {source, gold} from the rest of that language
                (evaluation verses stripped; no verse ids)

Score is flag_rate(drafts) - flag_rate(gold). Higher is better.
Counts (`draft_flags`, `draft_missed`, `gold_flags`, `gold_clean`)
are the same quantities as integers.

Default CLI is a smoke check. The driver evaluates sealed practice
(keep) and holdout (observation only) in its own process.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path

CELLS = Path(__file__).with_name("cells.json")
MANIFEST = Path(__file__).with_name("corpus_manifest.json")
LOCKED = frozenset({"--practice", "--sealed", "--holdout"})

SMOKE_CORPUS = (
    {"source": "The man walked.", "gold": "man-walk."},
    {"source": "The bird flew.", "gold": "bird-fly."},
    {"source": "The fish swam.", "gold": "fish-swim."},
    {"source": "The fire burned.", "gold": "fire-burn."},
)

SMOKE = [
    {
        "kind": "draft",
        "source": "The child sat down.",
        "candidate": "xxx child sit down sit.",
        "examples": [
            {"source": "The dog ran.", "gold": "dog-run."},
            {"source": "The woman stood.", "gold": "woman-stand."},
        ],
        "corpus": list(SMOKE_CORPUS),
    },
    {
        "kind": "gold",
        "source": "The child sat down.",
        "candidate": "child-sit.",
        "examples": [
            {"source": "The dog ran.", "gold": "dog-run."},
            {"source": "The woman stood.", "gold": "woman-stand."},
        ],
        "corpus": list(SMOKE_CORPUS),
    },
]


def blocked_verse_ids(payload: dict) -> frozenset[str]:
    ids = set(payload.get("practice_verses") or [])
    ids.update(payload.get("holdout_verses") or [])
    for cell in payload.get("cells") or []:
        if cell.get("verse_id"):
            ids.add(cell["verse_id"])
    return frozenset(ids)


@lru_cache(maxsize=8)
def corpus_for(language: str, blocked: frozenset[str]) -> tuple[dict, ...]:
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    source = Path(man["source_file"]).read_text(encoding="utf-8").splitlines()
    target = Path(man["targets"][language]).read_text(encoding="utf-8").splitlines()
    vrefs = Path(man["vref_file"]).read_text(encoding="utf-8").splitlines()
    n = min(len(source), len(target), len(vrefs))
    rows = []
    for i in range(n):
        if vrefs[i].strip() in blocked:
            continue
        src, gold = source[i], target[i]
        if not src.strip() or not gold.strip():
            continue
        rows.append({"source": src, "gold": gold})
    return tuple(rows)


def load_cells(split: str) -> list[dict]:
    if split == "smoke":
        return list(SMOKE)
    if os.environ.get("FINCH_SEALED_OK") != "1":
        raise PermissionError("real evaluation cells are allocator-only")
    payload = json.loads(CELLS.read_text(encoding="utf-8"))
    want = "holdout" if split == "holdout" else "practice"
    blocked = blocked_verse_ids(payload)
    cells = []
    for cell in payload["cells"]:
        if cell["split"] != want:
            continue
        view = dict(cell)
        view["corpus"] = corpus_for(cell["language"], blocked)
        cells.append(view)
    return cells


def decide(flag, cell) -> bool:
    return bool(
        flag(cell["source"], cell["candidate"], cell["examples"], cell["corpus"])
    )


def evaluate(flag, split: str) -> dict:
    drafts, golds, errors = [], [], []
    for index, cell in enumerate(load_cells(split)):
        try:
            flagged = decide(flag, cell)
        except Exception as exc:
            flagged = False
            errors.append(f"cell {index}: {exc!r}")
        (drafts if cell["kind"] == "draft" else golds).append(flagged)
    draft_flags = sum(drafts)
    gold_flags = sum(golds)
    n_draft = len(drafts)
    n_gold = len(golds)
    should = draft_flags / n_draft if n_draft else 0.0
    should_not = gold_flags / n_gold if n_gold else 0.0
    return {
        "task": "translation_anomaly_corpus",
        "split": split,
        "score": should - should_not,
        "should_flag": should,
        "should_not_flag": should_not,
        "n_draft": n_draft,
        "n_gold": n_gold,
        "draft_flags": draft_flags,
        "draft_missed": n_draft - draft_flags,
        "gold_flags": gold_flags,
        "gold_clean": n_gold - gold_flags,
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
            "task": "translation_anomaly_corpus",
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
            "task": "translation_anomaly_corpus",
            "split": split,
            "score": 0.0,
            "errors": [f"load: {exc!r}"],
        }))
        return
    print(json.dumps(evaluate(flag, split)))


if __name__ == "__main__":
    main()
