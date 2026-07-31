"""0/1 knapsack — pick items to maximize value under a weight limit.
The genome is a bitmask over items; infeasible packs score negative by
how far over the limit they are, so evolution walks back inside.

    python3 examples/classics/knapsack.py
    python3 -m finch4.hub        # http://127.0.0.1:8800
"""
import random

from finch4 import (Breed, CapPopulation, Environment, Evaluate, Mutate,
                    Populate, SortByFitness)

rng0 = random.Random(7)
N = 40
WEIGHTS = [rng0.uniform(1, 10) for _ in range(N)]
VALUES = [rng0.uniform(1, 20) for _ in range(N)]
CAPACITY = sum(WEIGHTS) * 0.25


def random_pack(rng):
    return [rng.randint(0, 1) for _ in range(N)]


def one_point(a, b, rng):
    cut = rng.randrange(1, N)
    return a[:cut] + b[cut:]


def toggle(genome, rng):
    out = list(genome)
    i = rng.randrange(N)
    out[i] = 1 - out[i]
    return out


def pack_value(genome):
    weight = sum(w for g, w in zip(genome, WEIGHTS) if g)
    value = sum(v for g, v in zip(genome, VALUES) if g)
    return value if weight <= CAPACITY else CAPACITY - weight


env = Environment([
    Populate(random_pack, 80),
    Breed(one_point, children=60),
    Mutate(toggle, rate=0.9),
    Evaluate(pack_value, name="value"),
    SortByFitness(),
    CapPopulation(80),
], name="knapsack", live=True, seed=2)
env.evolve(generations=150)

best = env.best_ever
taken = [i for i, g in enumerate(best["genome"]) if g]
weight = sum(WEIGHTS[i] for i in taken)
print(f"best value: {best['fitness']:.2f} with {len(taken)} items, "
      f"weight {weight:.1f}/{CAPACITY:.1f} "
      f"({env.state['evaluations']} evaluations)")
env.report_media("best pack", text=(
    f"{len(taken)} items, value {best['fitness']:.2f}, "
    f"weight {weight:.1f} of {CAPACITY:.1f}\nitems: {taken}"))
