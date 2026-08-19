"""Canonical fitness for gold-corruption detection.

The artifact must define:

    def flag(source, candidate, examples, corpus) -> bool

should_flag is false on the human verse and true on a broken copy.
Score is (accuracy on humans + mean accuracy on each break type) / 2.

Default CLI is a smoke check. The driver evaluates sealed practice
(keep) and holdout (observation only) in its own process.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

CELLS = Path(__file__).with_name("cells.json")
MANIFEST = Path(__file__).with_name("corpus_manifest.json")
LOCKED = frozenset({"--practice", "--sealed", "--holdout"})
HUMAN = "human"
BREAK_TYPES = (
    "wrong_verse",
    "omit",
    "neighbor_graft",
    "name_number",
    "char_edit",
)

SMOKE_CORPUS = (
    {"source": "The man walked.", "gold": "man-walk."},
    {"source": "The bird flew.", "gold": "bird-fly."},
    {"source": "The fish swam.", "gold": "fish-swim."},
    {"source": "The fire burned.", "gold": "fire-burn."},
)

SMOKE_EXAMPLES = [
    {"source": "The dog ran.", "gold": "dog-run."},
    {"source": "The woman stood.", "gold": "woman-stand."},
]

SMOKE = [
    {
        "break_type": HUMAN,
        "should_flag": False,
        "source": "The child sat down.",
        "candidate": "child-sit.",
        "examples": list(SMOKE_EXAMPLES),
        "corpus": list(SMOKE_CORPUS),
    },
    {
        "break_type": "wrong_verse",
        "should_flag": True,
        "source": "The child sat down.",
        "candidate": "man-walk.",
        "examples": list(SMOKE_EXAMPLES),
        "corpus": list(SMOKE_CORPUS),
    },
    {
        "break_type": "omit",
        "should_flag": True,
        "source": "The child sat down.",
        "candidate": "child.",
        "examples": list(SMOKE_EXAMPLES),
        "corpus": list(SMOKE_CORPUS),
    },
    {
        "break_type": "char_edit",
        "should_flag": True,
        "source": "The child sat down.",
        "candidate": "chxld-sit.",
        "examples": list(SMOKE_EXAMPLES),
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


@lru_cache(maxsize=32)
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
    try:
        return bool(
            flag(cell["source"], cell["candidate"], cell["examples"], cell["corpus"])
        )
    except Exception:
        return False


def _accuracy(flags: list[bool], should: list[bool]) -> float:
    if not flags:
        return 0.0
    return sum(hit == want for hit, want in zip(flags, should)) / len(flags)


def evaluate(flag, split: str) -> dict:
    groups: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    errors = []
    for index, cell in enumerate(load_cells(split)):
        try:
            flagged = decide(flag, cell)
        except Exception as exc:
            flagged = False
            errors.append(f"cell {index}: {type(exc).__name__}")
        groups[cell["break_type"]].append((flagged, bool(cell["should_flag"])))
    humans = groups.get(HUMAN, [])
    broken_types = [name for name in BREAK_TYPES if name in groups]
    right_on_humans = (
        _accuracy([row[0] for row in humans], [row[1] for row in humans])
        if humans
        else 0.0
    )
    right_on_breaks = 0.0
    by_type = {HUMAN: right_on_humans}
    if broken_types:
        scores = []
        for name in broken_types:
            rows = groups[name]
            acc = _accuracy([row[0] for row in rows], [row[1] for row in rows])
            by_type[name] = acc
            scores.append(acc)
        right_on_breaks = sum(scores) / len(scores)
    if humans and broken_types:
        score = (right_on_humans + right_on_breaks) / 2
    else:
        score = _accuracy(
            [hit for rows in groups.values() for hit, _ in rows],
            [want for rows in groups.values() for _, want in rows],
        )
    return {
        "task": "gold_corruption",
        "split": split,
        "score": score,
        "right_on_humans": right_on_humans,
        "right_on_breaks": right_on_breaks,
        "accuracy_by_break_type": by_type,
        "n_by_break_type": {name: len(rows) for name, rows in groups.items()},
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
            "task": "gold_corruption",
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
            "task": "gold_corruption",
            "split": split,
            "score": 0.0,
            "errors": [f"load: {type(exc).__name__}"],
        }))
        return
    print(json.dumps(evaluate(flag, split)))


if __name__ == "__main__":
    main()
