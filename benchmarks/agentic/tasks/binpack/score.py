"""Canonical fitness for the binpack task — online bin packing.

The artifact must define:

    def priority(item, capacities):
        '''item: float in (0,1). capacities: np.ndarray of remaining
        capacity per open bin. Return an array of scores, one per bin;
        the item goes into the FEASIBLE bin with the highest score, or a
        new bin if none is feasible.'''

Usage:  python3 score.py artifact.py [--holdout]

Prints one JSON line: {"task", "score", "cases", "errors"}. Score is the
mean over instances of lower_bound / bins_used (higher is better, 1.0 is
a perfect packing). Failures score 0 on that instance and are listed in
"errors" — the script always exits 0 so callers can parse.

This file is CANONICAL: agents run it, never edit it. Audits re-run it
(and --holdout) on the shipped artifact; the numbers must reproduce.
"""
import json
import sys

import numpy as np

BIN_CAP = 1.0
N_ITEMS = 120
TRAIN_SEEDS = [101, 102, 103, 104, 105]
HOLDOUT_SEEDS = [201, 202, 203, 204, 205]


def pack(priority, items):
    caps = []
    for item in items:
        feasible = [i for i, c in enumerate(caps) if c >= item - 1e-9]
        if not feasible:
            caps.append(BIN_CAP - item)
            continue
        scores = np.asarray(priority(float(item), np.asarray(caps)),
                            dtype=np.float64)
        if scores.shape != (len(caps),):
            raise ValueError(f"priority returned shape {scores.shape}, "
                             f"expected ({len(caps)},)")
        best = max(feasible, key=lambda i: scores[i])
        caps[best] -= item
    return len(caps)


def main():
    artifact = sys.argv[1]
    seeds = HOLDOUT_SEEDS if "--holdout" in sys.argv else TRAIN_SEEDS
    ns = {"np": np, "numpy": np}
    cases, errors = [], []
    try:
        with open(artifact) as f:
            exec(compile(f.read(), artifact, "exec"), ns)
        priority = ns["priority"]
    except Exception as e:
        print(json.dumps({"task": "binpack", "score": 0.0, "cases": [],
                          "errors": [f"load: {e!r}"]}))
        return
    for seed in seeds:
        rng = np.random.default_rng(seed)
        items = rng.uniform(0.1, 0.7, N_ITEMS)
        lower_bound = int(np.ceil(items.sum() / BIN_CAP))
        try:
            used = pack(priority, items)
            cases.append(lower_bound / used)
        except Exception as e:
            cases.append(0.0)
            errors.append(f"seed {seed}: {e!r}")
    print(json.dumps({"task": "binpack",
                      "score": float(np.mean(cases)) if cases else 0.0,
                      "cases": [round(c, 6) for c in cases],
                      "errors": errors}))


if __name__ == "__main__":
    main()
