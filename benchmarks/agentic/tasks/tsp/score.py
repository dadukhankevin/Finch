"""Canonical fitness for the tsp task — greedy tour construction.

The artifact must define:

    def next_city(current, unvisited, coords):
        '''current: int city index. unvisited: np.ndarray of remaining
        city indices. coords: (n, 2) np.ndarray of all city positions.
        Return ONE element of unvisited — the city to visit next.'''

Usage:  python3 score.py artifact.py [--holdout]

The tour starts at city 0, visits every city via next_city, and returns
to 0. Score is the mean over instances of nn_length / tour_length, where
nn is the nearest-neighbor baseline: 1.0 ties nearest-neighbor, above
1.0 beats it (higher is better). Failures score 0 on that instance.

This file is CANONICAL: agents run it, never edit it. Audits re-run it
(and --holdout) on the shipped artifact; the numbers must reproduce.
"""
import json
import sys

import numpy as np

N_CITIES = 60
TRAIN_SEEDS = [301, 302, 303, 304, 305]
HOLDOUT_SEEDS = [401, 402, 403, 404, 405]


def tour_length(order, coords):
    pts = coords[np.asarray(order)]
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()
                 + np.linalg.norm(pts[-1] - pts[0]))


def nearest_neighbor(coords):
    order, unvisited = [0], set(range(1, len(coords)))
    while unvisited:
        cur = coords[order[-1]]
        nxt = min(unvisited,
                  key=lambda i: float(np.linalg.norm(coords[i] - cur)))
        order.append(nxt)
        unvisited.remove(nxt)
    return tour_length(order, coords)


def build_tour(next_city, coords):
    order = [0]
    unvisited = np.arange(1, len(coords))
    while len(unvisited):
        step = int(next_city(order[-1], unvisited.copy(), coords))
        if step not in set(unvisited.tolist()):
            raise ValueError(f"next_city returned {step}, "
                             "which is not an unvisited city")
        order.append(step)
        unvisited = unvisited[unvisited != step]
    return tour_length(order, coords)


def main():
    artifact = sys.argv[1]
    seeds = HOLDOUT_SEEDS if "--holdout" in sys.argv else TRAIN_SEEDS
    ns = {"np": np, "numpy": np}
    cases, errors = [], []
    try:
        with open(artifact) as f:
            exec(compile(f.read(), artifact, "exec"), ns)
        next_city = ns["next_city"]
    except Exception as e:
        print(json.dumps({"task": "tsp", "score": 0.0, "cases": [],
                          "errors": [f"load: {e!r}"]}))
        return
    for seed in seeds:
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0.0, 1.0, (N_CITIES, 2))
        try:
            cases.append(nearest_neighbor(coords)
                         / build_tour(next_city, coords))
        except Exception as e:
            cases.append(0.0)
            errors.append(f"seed {seed}: {e!r}")
    print(json.dumps({"task": "tsp",
                      "score": float(np.mean(cases)) if cases else 0.0,
                      "cases": [round(c, 6) for c in cases],
                      "errors": errors}))


if __name__ == "__main__":
    main()
