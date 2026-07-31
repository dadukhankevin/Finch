"""Classic Finch layers: a traditional GA as a stack, over plain
genomes (lists, permutations, strings, arrays — any object).

This is the "normal Finch" wing of Finch 4 — no decoder, no agents,
just the textbook loop composed from layers:

    env = Environment([
        Populate(factory, 100),
        Breed(order_crossover, children=60),
        Mutate(inversion, rate=0.3),
        Evaluate(fitness),
        SortByFitness(),
        CapPopulation(100),
    ], seed=0)
    env.evolve(generations=200)

Individuals are dicts {"genome": ..., "fitness": float|None}; fitness
is HIGHER-better everywhere in this library. These layers are free
recompositions, not the vetted engines — per the ledger rule they earn
any performance claims by measurement, and their first outing is the
TSP demo in benchmarks/finch_tsp_demo.py."""
from __future__ import annotations

from .layers import Layer


def individual(genome):
    return {"genome": genome, "fitness": None}


class Populate(Layer):
    """Fill the population to n with factory(rng) -> genome."""

    def __init__(self, factory, n):
        self.factory = factory
        self.n = int(n)

    def __call__(self, env):
        pop = env.state.setdefault("population", [])
        rng = env.state["rng"]
        while len(pop) < self.n:
            pop.append(individual(self.factory(rng)))


class Breed(Layer):
    """Tournament-select parents, append children (unscored).
    crossover(genome_a, genome_b, rng) -> child genome."""

    def __init__(self, crossover, children, tournament=3):
        self.crossover = crossover
        self.children = int(children)
        self.tournament = int(tournament)

    def _pick(self, pop, rng):
        pool = [pop[rng.randrange(len(pop))]
                for _ in range(self.tournament)]
        scored = [i for i in pool if i["fitness"] is not None]
        return max(scored or pool,
                   key=lambda i: (i["fitness"] is not None,
                                  i["fitness"] or 0.0))

    def __call__(self, env):
        pop = env.state["population"]
        rng = env.state["rng"]
        if not pop:
            return
        for _ in range(self.children):
            a, b = self._pick(pop, rng), self._pick(pop, rng)
            pop.append(individual(self.crossover(a["genome"],
                                                 b["genome"], rng)))


class Mutate(Layer):
    """mutation(genome, rng) -> genome, applied to each UNSCORED
    individual with probability rate (parents stay untouched — their
    fitness is already paid for)."""

    def __init__(self, mutation, rate=0.5):
        self.mutation = mutation
        self.rate = float(rate)

    def __call__(self, env):
        rng = env.state["rng"]
        for ind in env.state["population"]:
            if ind["fitness"] is None and rng.random() < self.rate:
                ind["genome"] = self.mutation(ind["genome"], rng)


class Evaluate(Layer):
    """Score every unscored individual with fitness(genome) -> float
    (higher better). Counts evaluations and tracks best-ever."""

    def __init__(self, fitness, name="fitness"):
        self.fitness = fitness
        self.metric = name

    def __call__(self, env):
        for ind in env.state["population"]:
            if ind["fitness"] is None:
                ind["fitness"] = float(self.fitness(ind["genome"]))
                env.state["evaluations"] += 1
                best = env.state.get("best_ever")
                if best is None or ind["fitness"] > best["fitness"]:
                    env.state["best_ever"] = dict(ind)
        if env.state.get("best_ever") is not None:
            env.state.setdefault("best", {})[self.metric] = \
                env.state["best_ever"]["fitness"]


class SortByFitness(Layer):
    def __call__(self, env):
        env.state["population"].sort(
            key=lambda i: (i["fitness"] is not None, i["fitness"] or 0.0),
            reverse=True)


class CapPopulation(Layer):
    def __init__(self, n):
        self.n = int(n)

    def __call__(self, env):
        del env.state["population"][self.n:]


# ------------------------------------------------- permutation operators

def order_crossover(a, b, rng):
    """OX1: keep a random slice of parent a, fill the rest in parent
    b's order — the classic permutation crossover."""
    n = len(a)
    i, j = sorted(rng.sample(range(n), 2))
    hold = a[i:j + 1]
    held = set(hold)
    rest = [g for g in b if g not in held]
    return rest[:i] + hold + rest[i:]


def inversion(genome, rng):
    """Reverse a random segment — 2-opt's mutation-shaped cousin."""
    n = len(genome)
    i, j = sorted(rng.sample(range(n), 2))
    return genome[:i] + genome[i:j + 1][::-1] + genome[j + 1:]


def swap(genome, rng):
    i, j = rng.sample(range(len(genome)), 2)
    out = list(genome)
    out[i], out[j] = out[j], out[i]
    return out
